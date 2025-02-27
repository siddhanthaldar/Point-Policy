import einops
import numpy as np
from collections import deque

import torch
from torch import nn

from torchvision import transforms as T

import utils
from agent.networks.rgb_modules import ResnetEncoder
from agent.networks.policy_head import (
    DeterministicHead,
    DiffusionHead,
)
from agent.networks.gpt import GPT, GPTConfig
from agent.networks.mlp import MLP


class Actor(nn.Module):
    def __init__(
        self,
        repr_dim,
        act_dim,
        hidden_dim,
        policy_head="deterministic",
        num_feat_per_step=1,
        device="cuda",
    ):
        super().__init__()

        self._policy_head = policy_head
        self._repr_dim = repr_dim
        self._act_dim = act_dim
        self._num_feat_per_step = num_feat_per_step

        self._action_token = nn.Parameter(torch.randn(1, 1, 1, repr_dim))

        # GPT model
        self._policy = GPT(
            GPTConfig(
                block_size=65,
                input_dim=repr_dim,
                output_dim=hidden_dim,
                n_layer=8,
                n_head=4,
                n_embd=hidden_dim,
                dropout=0.1,
            )
        )

        if policy_head == "deterministic":
            self._action_head = DeterministicHead(
                hidden_dim, self._act_dim, hidden_size=hidden_dim, num_layers=2
            )
        elif policy_head == "diffusion":
            self._action_head = DiffusionHead(
                input_size=hidden_dim,
                output_size=self._act_dim,
                obs_horizon=1,
                pred_horizon=1,
                hidden_size=hidden_dim,
                num_layers=2,
                device=device,
            )

        self.apply(utils.weight_init)

    def forward(self, obs, stddev, action=None):
        B, T, D = obs.shape

        # insert action token at each self._num_feat_per_step interval
        obs = obs.view(B, -1, self._num_feat_per_step, obs.shape[-1])
        action_token = self._action_token.repeat(B, obs.shape[1], 1, 1)
        obs = torch.cat([obs, action_token], dim=-2).view(B, -1, D)

        # get action features
        features = self._policy(obs)
        num_feat_per_step = self._num_feat_per_step + 1  # +1 for action token
        features = features[:, num_feat_per_step - 1 :: num_feat_per_step]

        # action head
        pred_action = self._action_head(
            features,
            stddev,
            **{"action_seq": action},
        )

        if action is None:
            return pred_action
        else:
            loss = self._action_head.loss_fn(
                pred_action,
                action,
                reduction="mean",
            )
            return pred_action, loss[0] if isinstance(loss, tuple) else loss


class BCAgent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        hidden_dim,
        stddev_schedule,
        use_tb,
        policy_head,
        pixel_keys,
        proprio_key,
        use_proprio,
        history,
        history_len,
        eval_history_len,
        temporal_agg,
        max_episode_len,
        num_queries,
        use_depth,
    ):
        self.device = device
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.stddev_schedule = stddev_schedule
        self.use_tb = use_tb
        self.policy_head = policy_head
        self.use_proprio = use_proprio
        self.history_len = history_len if history else 1
        self.eval_history_len = eval_history_len if history else 1
        self.use_depth = use_depth

        # actor parameters
        # Action dim was 6 with 3D position and 3D rotation.
        # Now it is 9D with 3D position and 6D rotation.
        self._act_dim = action_shape[0] + 3

        # keys
        self.pixel_keys = pixel_keys
        self.proprio_key = proprio_key

        # action chunking params
        self.temporal_agg = temporal_agg
        self.max_episode_len = max_episode_len
        self.num_queries = num_queries if self.temporal_agg else 1

        # number of inputs per time step
        num_feat_per_step = len(self.pixel_keys)
        if use_proprio:
            num_feat_per_step += 1
        if self.use_depth:
            num_feat_per_step += len(self.pixel_keys)

        # observation params
        if use_proprio:
            proprio_shape = obs_shape[self.proprio_key]
        obs_shape = obs_shape[self.pixel_keys[0]]

        # Track model size
        model_size = 0

        # encoder
        self.encoder = ResnetEncoder(
            obs_shape,
            512,
        ).to(device)
        model_size += sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        if self.use_depth:
            self.depth_encoder = ResnetEncoder(
                (1, *obs_shape[1:]),
                512,
            ).to(device)
            model_size += sum(
                p.numel() for p in self.depth_encoder.parameters() if p.requires_grad
            )
        self.repr_dim = 512

        # projector for proprioceptive features
        if use_proprio:
            self.proprio_projector = MLP(
                proprio_shape[0], hidden_channels=[self.repr_dim, self.repr_dim]
            ).to(device)
            self.proprio_projector.apply(utils.weight_init)
            model_size += sum(
                p.numel()
                for p in self.proprio_projector.parameters()
                if p.requires_grad
            )

        # actor
        action_dim = (
            self._act_dim * self.num_queries if self.temporal_agg else self._act_dim
        )
        self.actor = Actor(
            self.repr_dim,
            action_dim,
            hidden_dim,
            self.policy_head,
            num_feat_per_step,
            device,
        ).to(device)
        model_size += sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        print("Total parameter count: %.2fM" % (model_size / 1e6,))

        # optimizers
        # encoder
        params = list(self.encoder.parameters())
        if self.use_depth:
            params += list(self.depth_encoder.parameters())
        self.encoder_opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        # proprio
        if self.use_proprio:
            self.proprio_opt = torch.optim.AdamW(
                self.proprio_projector.parameters(), lr=lr, weight_decay=1e-4
            )
        # actor
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=lr, weight_decay=1e-4
        )

        # scaling for images at inference
        self.test_aug = T.Compose([T.ToPILImage(), T.ToTensor()])

        self.train()
        self.buffer_reset()

    def __repr__(self):
        return "bc"

    def train(self, training=True):
        self.training = training
        if training:
            self.encoder.train(training)
            if self.use_proprio:
                self.proprio_projector.train(training)
            self.actor.train(training)
        else:
            self.encoder.eval()
            if self.use_proprio:
                self.proprio_projector.eval()
            self.actor.eval()

    def buffer_reset(self):
        self.observation_buffer = {}
        for key in self.pixel_keys:
            self.observation_buffer[key] = deque(maxlen=self.eval_history_len)
            if self.use_depth:
                self.observation_buffer[f"depth_{key}"] = deque(
                    maxlen=self.eval_history_len
                )
        if self.use_proprio:
            self.proprio_buffer = deque(maxlen=self.eval_history_len)

        # temporal aggregation
        if self.temporal_agg:
            self.all_time_actions = torch.zeros(
                [
                    self.max_episode_len,
                    self.max_episode_len + self.num_queries,
                    self._act_dim,
                ]
            ).to(self.device)

    def clear_buffers(self):
        del self.observation_buffer
        if self.use_proprio:
            del self.proprio_buffer
        if self.temporal_agg:
            del self.all_time_actions

    def act(self, obs, norm_stats, step, global_step, eval_mode=False):
        if norm_stats is not None:
            pre_process = {
                self.proprio_key: lambda s_qpos: (
                    s_qpos - norm_stats[self.proprio_key]["min"]
                )
                / (
                    norm_stats[self.proprio_key]["max"]
                    - norm_stats[self.proprio_key]["min"]
                    + 1e-5
                ),
                "depth": lambda d: (d - norm_stats["depth"]["min"])
                / (norm_stats["depth"]["max"] - norm_stats["depth"]["min"] + 1e-5),
            }
            post_process = (
                lambda a: a
                * (norm_stats["actions"]["max"] - norm_stats["actions"]["min"])
                + norm_stats["actions"]["min"]
            )

        # add to buffer
        features = []
        for key in self.pixel_keys:
            self.observation_buffer[key].append(
                self.test_aug(obs[key].transpose(1, 2, 0)).numpy()
            )
            pixels = torch.as_tensor(
                np.array(self.observation_buffer[key]), device=self.device
            ).float()
            # encoder
            pixels = self.encoder(pixels)
            features.append(pixels)
            if self.use_depth:
                depth = pre_process["depth"](obs[f"depth_{key}"])[None]
                self.observation_buffer[f"depth_{key}"].append(depth)
                depth = torch.as_tensor(
                    np.array(self.observation_buffer[f"depth_{key}"]),
                    device=self.device,
                ).float()
                depth = self.depth_encoder(depth)
                features.append(depth)

        if self.use_proprio:
            obs[self.proprio_key] = pre_process[self.proprio_key](obs[self.proprio_key])
            self.proprio_buffer.append(obs[self.proprio_key])
            proprio = torch.as_tensor(
                np.array(self.proprio_buffer), device=self.device
            ).float()
            proprio = self.proprio_projector(proprio)
            features.append(proprio)
        features = torch.cat(features, dim=-1).view(-1, self.repr_dim)

        stddev = utils.schedule(self.stddev_schedule, global_step)
        action = self.actor(features.unsqueeze(0), stddev)

        if self.policy_head == "deterministic":
            action = action.mean
        if self.policy_head == "diffusion":
            action = action[0]
        # else:
        #     if eval_mode:
        #         action = action.mean
        #     else:
        #         action = action.sample()

        if self.temporal_agg:
            action = action.view(-1, self.num_queries, self._act_dim)
            self.all_time_actions[[step], step : step + self.num_queries] = action[-1:]
            actions_for_curr_step = self.all_time_actions[:, step]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
            action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            if norm_stats is not None:
                return post_process(action.cpu().numpy()[0])
            return action.cpu().numpy()[0]
        else:
            if norm_stats is not None:
                return post_process(action.cpu().numpy()[0, -1])
            return action.cpu().numpy()[0, -1, :]

    def update(self, expert_replay_iter, step):
        metrics = dict()

        batch = next(expert_replay_iter)
        data = utils.to_torch(batch, self.device)
        action = data["actions"].float()

        # features
        features = []
        for key in self.pixel_keys:
            pixel = data[key].float()
            shape = pixel.shape
            # rearrange
            pixel = einops.rearrange(pixel, "b t c h w -> (b t) c h w")
            # encode
            pixel = self.encoder(pixel)
            pixel = einops.rearrange(pixel, "(b t) d -> b t d", t=shape[1])
            features.append(pixel)

            if self.use_depth:
                depth = data[f"depth_{key}"].float()
                shape = depth.shape
                # rearrange
                depth = einops.rearrange(depth, "b t c h w -> (b t) c h w")
                # encode
                depth = self.depth_encoder(depth)
                depth = einops.rearrange(depth, "(b t) d -> b t d", t=shape[1])
                features.append(depth)

        if self.use_proprio:
            proprio = data[self.proprio_key].float()
            proprio = self.proprio_projector(proprio)
            features.append(proprio)

        # concatenate
        features = torch.cat(features, dim=-1).view(
            action.shape[0], -1, self.repr_dim
        )  # (B, T * num_feat_per_step, D)

        # rearrange action
        if self.temporal_agg:
            action = einops.rearrange(action, "b t1 t2 d -> b t1 (t2 d)")

        # actor loss
        stddev = utils.schedule(self.stddev_schedule, step)
        _, actor_loss = self.actor(features, stddev, action)

        # optimizer step
        self.encoder_opt.zero_grad(set_to_none=True)
        if self.use_proprio:
            self.proprio_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss["actor_loss"].backward()
        self.encoder_opt.step()
        if self.use_proprio:
            self.proprio_opt.step()
        self.actor_opt.step()

        if self.policy_head == "diffusion" and step % 10 == 0:
            self.actor._action_head.net.ema_step()

        if self.use_tb:
            for key, value in actor_loss.items():
                metrics[key] = value.item()

        return metrics

    def save_snapshot(self):
        model_keys = ["actor", "encoder"]
        opt_keys = ["actor_opt", "encoder_opt"]
        if self.use_proprio:
            model_keys += ["proprio_projector"]
            opt_keys += ["proprio_opt"]

        # models
        payload = {k: self.__dict__[k].state_dict() for k in model_keys}
        # optimizers
        payload.update({k: self.__dict__[k] for k in opt_keys})

        others = [
            "use_proprio",
            "max_episode_len",
        ]
        payload.update({k: self.__dict__[k] for k in others})
        return payload

    def load_snapshot(self, payload, eval=True):
        # models
        model_keys = ["actor", "encoder"]
        if self.use_proprio:
            model_keys += ["proprio_projector"]
        for k in model_keys:
            self.__dict__[k].load_state_dict(payload[k])

        if eval:
            self.train(False)
        return
