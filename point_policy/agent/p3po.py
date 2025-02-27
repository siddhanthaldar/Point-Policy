"""
Model that divides image into patches and performs cross attention with patches around points
to predict future point tracks.
"""

import einops
import numpy as np
from collections import deque

import torch
from torch import nn

import utils
from agent.networks.policy_head import (
    DeterministicHead,
    DiffusionHead,
)

from agent.networks.mlp import MLP
from agent.networks.gpt import GPT, GPTConfig


class Actor(nn.Module):
    def __init__(
        self,
        repr_dim,
        act_dim,
        history_len,
        hidden_dim,
        policy_head="deterministic",
        device="cuda",
    ):
        super().__init__()

        self._policy_head = policy_head
        self._repr_dim = repr_dim
        self._act_dim = act_dim

        self._policy = GPT(
            GPTConfig(
                block_size=20,
                input_dim=repr_dim,
                output_dim=hidden_dim,
                n_layer=4,
                n_head=2,
                n_embd=hidden_dim,
                dropout=0.1,
                causal=True,
            )
        )

        if policy_head == "deterministic":
            self._action_head = DeterministicHead(
                hidden_dim, self._act_dim, hidden_size=hidden_dim, num_layers=2
            )
        elif policy_head == "diffusion":
            obs_horizon = history_len
            pred_horizon = history_len
            self._action_head = DiffusionHead(
                input_size=hidden_dim,
                output_size=self._act_dim,
                obs_horizon=obs_horizon,
                pred_horizon=pred_horizon,
                hidden_size=hidden_dim,
                num_layers=2,
                device=device,
            )

        self.apply(utils.weight_init)

    def forward(
        self,
        past_tracks,
        stddev,
        action=None,
    ):
        features = self._policy(past_tracks)

        pred_action = self._action_head(
            features,
            stddev,
            **{
                "action_seq": action if action is not None else None,
            },
        )

        if action is None:
            return pred_action
        else:
            mask = torch.ones(action.shape[0], action.shape[1]).to(action.device)
            loss = self._action_head.loss_fn(
                pred_action,
                action,
                mask,
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
        history,
        history_len,
        eval_history_len,
        temporal_agg,
        max_episode_len,
        num_queries,
        use_robot_points,
        num_robot_points,
        use_object_points,
        num_object_points,
        point_dim,
    ):
        assert point_dim in [2, 3], "Only 2D or 3D points are supported"

        self.device = device
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.stddev_schedule = stddev_schedule
        self.use_tb = use_tb
        self.policy_head = policy_head
        self.history_len = history_len if history else 1
        self.eval_history_len = eval_history_len if history else 1

        self._use_robot_points = use_robot_points
        self._num_robot_points = num_robot_points
        self._use_object_points = use_object_points
        self._num_object_points = num_object_points
        self.num_track_points = (num_robot_points if use_robot_points else 0) + (
            num_object_points if use_object_points else 0
        )

        # keys
        self.pixel_keys = pixel_keys

        # action chunking params
        self.temporal_agg = temporal_agg
        self.max_episode_len = max_episode_len
        self.num_queries = num_queries if self.temporal_agg else 1

        # observation params
        self._obs_dim = point_dim * self.num_track_points
        self.repr_dim = 512
        obs_shape = obs_shape[self.pixel_keys[0]]

        # actor parameters
        # Action dim was 6 with 3D position and 3D rotation.
        # Now it is 9D with 3D position and 6D rotation.
        self._act_dim = action_shape[0] + 3

        # Track model size
        model_size = 0

        # projector for points and patches
        self.point_projector = MLP(self._obs_dim, hidden_channels=[self.repr_dim]).to(
            device
        )
        self.point_projector.apply(utils.weight_init)
        model_size += sum(
            p.numel() for p in self.point_projector.parameters() if p.requires_grad
        )

        # actor
        action_dim = (
            self._act_dim * self.num_queries if self.temporal_agg else self._act_dim
        )
        self.actor = Actor(
            self.repr_dim,
            action_dim,
            self.history_len,
            hidden_dim,
            self.policy_head,
            device,
        ).to(device)
        model_size += sum(p.numel() for p in self.actor.parameters() if p.requires_grad)

        # optimizers
        # point projector
        params = list(self.point_projector.parameters())
        self.point_opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        # actor
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=lr, weight_decay=1e-4
        )

        self.train()
        self.buffer_reset()

    def __repr__(self):
        return "bc"

    def train(self, training=True):
        self.training = training
        if training:
            self.point_projector.train(training)
            self.actor.train(training)
        else:
            self.point_projector.eval()
            self.actor.eval()

    def buffer_reset(self):
        self.observation_buffer = {}
        for key in self.pixel_keys:
            self.observation_buffer[f"past_tracks_{key}"] = deque(
                maxlen=self.eval_history_len
            )  # since point track history concatenated
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
        if self.temporal_agg:
            del self.all_time_actions

    def act(self, obs, norm_stats, step, global_step, eval_mode=False, **kwargs):
        if norm_stats is not None:
            preprocess = {
                "past_tracks": lambda x: (x - norm_stats["past_tracks"]["min"])
                / (
                    norm_stats["past_tracks"]["max"]
                    - norm_stats["past_tracks"]["min"]
                    + 1e-5
                ),
            }
            post_process = {
                "actions": lambda a: a
                * (norm_stats["actions"]["max"] - norm_stats["actions"]["min"])
                + norm_stats["actions"]["min"],
            }

        past_tracks = []
        for key in self.pixel_keys:
            point_tracks = preprocess["past_tracks"](obs[f"point_tracks_{key}"])
            self.observation_buffer[f"past_tracks_{key}"].append(point_tracks)
            while len(self.observation_buffer[f"past_tracks_{key}"]) < self.history_len:
                self.observation_buffer[f"past_tracks_{key}"].append(point_tracks)
            past_tracks.append(
                np.stack(self.observation_buffer[f"past_tracks_{key}"], axis=0)
            )

        # convert to tensor
        past_tracks = torch.as_tensor(np.array(past_tracks), device=self.device).float()

        # reshape past_tracks
        past_tracks = einops.rearrange(past_tracks, "n t p d-> n t (p d)")

        # encode past tracks
        past_tracks = self.point_projector(past_tracks)

        stddev = 0.1
        action = self.actor(past_tracks, stddev)

        if self.policy_head == "deterministic":
            action = action.mean
        elif self.policy_head == "diffusion":
            action = action[0]

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
                return post_process["actions"](action.cpu().numpy()[0])
            return action.cpu().numpy()[0]
        else:
            if norm_stats is not None:
                return post_process["actions"](action.cpu().numpy()[0, -1])
            return action.cpu().numpy()[0, -1, :]

    def update(self, expert_replay_iter, step, **kwargs):
        metrics = dict()

        batch = next(expert_replay_iter)
        data = utils.to_torch(batch, self.device)

        past_tracks = data["past_tracks"].float()
        action = data["actions"].float()

        # reshape for training
        past_tracks = einops.rearrange(past_tracks, "n t p d-> n t (p d)")

        # encode past tracks
        past_tracks = self.point_projector(past_tracks)

        # rearrange action
        if self.temporal_agg:
            action = einops.rearrange(action, "b t1 t2 d -> b t1 (t2 d)")

        # actor loss
        stddev = utils.schedule(self.stddev_schedule, step)
        pred_action, actor_loss = self.actor(past_tracks, stddev, action, **kwargs)

        # optimize
        self.point_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss["actor_loss"].backward()
        self.point_opt.step()
        self.actor_opt.step()

        if self.policy_head == "diffusion" and step % 10 == 0:
            self.actor._action_head.net.ema_step()

        if self.use_tb:
            for key, value in actor_loss.items():
                metrics[key] = value.item()

        return metrics

    def save_snapshot(self):
        model_keys = ["actor", "point_projector"]
        opt_keys = ["actor_opt", "point_opt"]
        # models
        payload = {k: self.__dict__[k].state_dict() for k in model_keys}
        # optimizers
        payload.update({k: self.__dict__[k] for k in opt_keys})

        others = ["max_episode_len"]
        payload.update({k: self.__dict__[k] for k in others})
        return payload

    def load_snapshot(self, payload, eval=False):
        # models
        model_keys = ["actor", "point_projector"]
        for k in model_keys:
            self.__dict__[k].load_state_dict(payload[k])

        if eval:
            self.train(False)
            return
