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
        num_track_points,
        hidden_dim,
        policy_head="deterministic",
        device="cuda",
        pred_gripper=True,
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
            )
        )

        if policy_head == "deterministic":
            self._action_head = DeterministicHead(
                hidden_dim, self._act_dim, hidden_size=hidden_dim, num_layers=2
            )
        elif policy_head == "diffusion":
            obs_horizon = num_track_points if not pred_gripper else num_track_points + 1
            pred_horizon = (
                num_track_points if not pred_gripper else num_track_points + 1
            )
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
        target=None,
        mask=None,
    ):
        features = self._policy(past_tracks)

        pred_action = self._action_head(
            features,
            stddev,
            **{
                "action_seq": target if target is not None else None,
            },
        )

        if target is None:
            return pred_action
        else:
            loss = self._action_head.loss_fn(
                pred_action,
                target,
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
        pred_gripper,
    ):
        self.device = device
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.stddev_schedule = stddev_schedule
        self.use_tb = use_tb
        self.policy_head = policy_head
        self.history_len = history_len if history else 1
        self.eval_history_len = eval_history_len if history else 1
        self.pred_gripper = pred_gripper

        self._use_robot_points = use_robot_points
        self._num_robot_points = num_robot_points
        self._use_object_points = use_object_points
        self._num_object_points = num_object_points
        self.num_track_points = (num_robot_points if use_robot_points else 0) + (
            num_object_points if use_object_points else 0
        )

        # actor parameters
        self._act_dim = point_dim
        assert self._act_dim in [2, 3], "Only 2D or 3D actions are supported"

        # keys
        self.pixel_keys = pixel_keys

        # action chunking params
        self.temporal_agg = temporal_agg
        self.max_episode_len = max_episode_len
        self.num_queries = num_queries if self.temporal_agg else 1

        # observation params
        self.repr_dim = 512  # representation dim for transformer input
        obs_shape = obs_shape[self.pixel_keys[0]]

        # Track model size
        model_size = 0

        # projector for points and patches
        self.point_projector = MLP(
            self._act_dim * self.history_len, hidden_channels=[self.repr_dim]
        ).to(device)
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
            self.num_track_points,
            hidden_dim,
            self.policy_head,
            device,
            self.pred_gripper,
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
        if self.pred_gripper:
            self.observation_buffer["past_gripper_states"] = deque(
                maxlen=self.eval_history_len
            )

        # temporal aggregation
        if self.temporal_agg:
            self.all_time_actions = {}
            for pixel_key in self.pixel_keys:
                gripper_points = 1 if self.pred_gripper else 0
                self.all_time_actions[pixel_key] = torch.zeros(
                    [
                        self.max_episode_len,
                        self.max_episode_len + self.num_queries,
                        self._act_dim * (self._num_robot_points + gripper_points),
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
                "gripper_states": lambda x: (x - norm_stats["gripper_states"]["min"])
                / (
                    norm_stats["gripper_states"]["max"]
                    - norm_stats["gripper_states"]["min"]
                    + 1e-5
                ),
            }
            post_process = {
                "future_tracks": lambda x: x
                * (norm_stats["past_tracks"]["max"] - norm_stats["past_tracks"]["min"])
                + norm_stats["past_tracks"]["min"],
                "gripper_states": lambda x: x
                * (
                    norm_stats["gripper_states"]["max"]
                    - norm_stats["gripper_states"]["min"]
                )
                + norm_stats["gripper_states"]["min"],
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
        if self.pred_gripper:
            gripper_state = preprocess["gripper_states"](obs["features"][-1])
            self.observation_buffer["past_gripper_states"].append(gripper_state)
            while (
                len(self.observation_buffer["past_gripper_states"]) < self.history_len
            ):
                self.observation_buffer["past_gripper_states"].append(gripper_state)
            past_gripper_states = np.stack(
                self.observation_buffer["past_gripper_states"], axis=0
            )

        # convert to tensor
        past_tracks = torch.as_tensor(np.array(past_tracks), device=self.device).float()
        if self.pred_gripper:
            past_gripper_states = torch.as_tensor(
                np.array(past_gripper_states), device=self.device
            ).float()

        # reshape past_tracks
        shape = past_tracks.shape
        past_tracks = past_tracks.transpose(1, 2).reshape(shape[0], shape[2], -1)
        if self.pred_gripper:
            past_gripper_states = past_gripper_states[None, None].repeat(
                past_tracks.shape[0], 1, self._act_dim
            )
            past_tracks = torch.cat([past_tracks, past_gripper_states], dim=1)

        # encode past tracks
        past_tracks = self.point_projector(past_tracks)

        stddev = 0.1
        future_tracks = self.actor(past_tracks, stddev)

        if self.policy_head == "deterministic":
            future_tracks = future_tracks.mean  # .cpu().numpy() # for deterministic

        # extract robot and gripper points
        robot_points = future_tracks[:, : self._num_robot_points]
        if self.pred_gripper:
            gripper_points = future_tracks[:, -1:]
            robot_points = torch.cat([robot_points, gripper_points], dim=1)
        future_tracks = robot_points

        return_dict = {}
        if not self.temporal_agg:
            for idx in range(len(future_tracks)):
                return_dict[f"future_tracks_{self.pixel_keys[idx]}"] = post_process[
                    "future_tracks"
                ](
                    future_tracks[idx, : self._num_robot_points, : self._act_dim]
                    .cpu()
                    .numpy()
                )
                return_dict["future_gripper_states"] = post_process["gripper_states"](
                    future_tracks[idx, -1:, :1].cpu().numpy()
                )
        else:
            for idx in range(len(future_tracks)):
                pixel_key = self.pixel_keys[idx]
                track = future_tracks[idx]
                track = track.view(-1, self.num_queries, self._act_dim)
                # consider only robot points
                start_idx = 0
                end_idx = (
                    start_idx + self._num_robot_points + (1 if self.pred_gripper else 0)
                )
                track = track[start_idx:end_idx]
                # convert to proper shape
                track = track.transpose(0, 1).reshape(self.num_queries, -1)[None]
                self.all_time_actions[pixel_key][
                    [step],
                    step : step + self.num_queries,
                ] = track[-1:]
                tracks_for_curr_step = self.all_time_actions[pixel_key][:, step]
                tracks_populated = torch.all(tracks_for_curr_step != 0.0, dim=-1)
                tracks_for_curr_step = tracks_for_curr_step[tracks_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(tracks_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = (
                    torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
                )
                track = (tracks_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                track = track.cpu().numpy()[0].reshape(-1, self._act_dim)
                return_dict[f"future_tracks_{pixel_key}"] = post_process[
                    "future_tracks"
                ](track[: self._num_robot_points])
                return_dict["gripper"] = post_process["gripper_states"](
                    future_tracks[idx, -1:, :1].cpu().numpy()
                )

        return return_dict

    def update(self, expert_replay_iter, step, **kwargs):
        metrics = dict()

        batch = next(expert_replay_iter)
        data = utils.to_torch(batch, self.device)

        past_tracks = data["past_tracks"].float()
        future_tracks = data["future_tracks"].float()
        action_masks = data["action_mask"].float()
        if self.pred_gripper:
            past_gripper_states = data["past_gripper_states"].float()
            future_gripper_states = data["future_gripper_states"].float()
            # Add a dimension to action masks
            gripper_mask = torch.ones_like(action_masks)[:, :1]
            action_masks = torch.cat([action_masks, gripper_mask], dim=1)

        # reshape for training
        shape = past_tracks.shape
        past_tracks = past_tracks.transpose(1, 2).reshape(shape[0], shape[2], -1)
        future_tracks = future_tracks[:, 0]

        if self.pred_gripper:
            past_gripper_states = past_gripper_states[:, None]
            future_gripper_states = future_gripper_states[:, :1]

            # Make last dim of gripper_states same as that of tracks
            past_gripper_states = past_gripper_states.repeat(1, 1, self._act_dim)
            future_gripper_states = future_gripper_states.repeat(1, 1, self._act_dim)

            # add gripper states as (n+1)-th track point
            past_tracks = torch.cat([past_tracks, past_gripper_states], dim=1)
            future_tracks = torch.cat([future_tracks, future_gripper_states], dim=1)

        # encode past tracks
        past_tracks = self.point_projector(past_tracks)

        # actor loss
        stddev = utils.schedule(self.stddev_schedule, step)
        pred_action, actor_loss = self.actor(
            # features,
            past_tracks,
            stddev,
            future_tracks,
            action_masks,
            **kwargs,
        )

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
        payload = {
            k: self.__dict__[k].state_dict() for k in model_keys if k != "encoder"
        }
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
