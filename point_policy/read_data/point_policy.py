import einops
import random
import numpy as np
import pickle as pkl
from pathlib import Path

from torch.utils.data import IterableDataset
from scipy.spatial.transform import Rotation as R


class BCDataset(IterableDataset):
    def __init__(
        self,
        path,
        tasks,
        num_demos_per_task,
        history,
        history_len,
        temporal_agg,
        num_queries,
        img_size,
        action_after_steps,
        use_robot_points,
        num_robot_points,
        use_object_points,
        num_object_points,
        point_dim,
        pixel_keys,
        subsample,
        skip_first_n,
        gt_depth,
    ):
        tasks = [tasks]  # NOTE: single task for now

        self._history = history
        self._history_len = history_len if history else 1
        self._img_size = np.array(img_size)
        self._action_after_steps = action_after_steps
        self._pixel_keys = pixel_keys
        self._subsample = subsample

        # track points
        self._use_robot_points = use_robot_points
        self._num_robot_points = num_robot_points
        self._use_object_points = use_object_points
        self._num_object_points = num_object_points
        self._point_dim = point_dim
        assert self._point_dim in [2, 3], "Point dimension must be 2 or 3"
        self._robot_points_key = (
            "robot_tracks" if self._point_dim == 2 else "robot_tracks_3d"
        )
        self._object_points_key = (
            "object_tracks" if self._point_dim == 2 else "object_tracks_3d"
        )

        # temporal aggregation
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries if temporal_agg else 1

        # get data paths
        self._paths = []
        for task in tasks:
            if gt_depth:
                self._paths.extend([Path(path) / f"{task}_gt_depth.pkl"])
            else:
                self._paths.extend([Path(path) / f"{task}.pkl"])

        paths = {}
        idx = 0
        for path, task in zip(self._paths, tasks):
            paths[idx] = path
            idx += 1
        del self._paths
        self._paths = paths

        # read data
        self._episodes = {}
        self._num_demos = {}
        self._max_episode_len = 0
        self._max_state_dim = 0
        self._num_samples = 0
        min_track, max_track = None, None
        for _path_idx in self._paths:
            print(f"Loading {str(self._paths[_path_idx])}")
            # read
            data = pkl.load(open(str(self._paths[_path_idx]), "rb"))
            observations = data["observations"]

            # store
            self._episodes[_path_idx] = []
            self._num_demos[_path_idx] = min(num_demos_per_task, len(observations))
            for i in range(min(num_demos_per_task, len(observations))):
                # skip first n
                if skip_first_n is not None:
                    for key in observations[i].keys():
                        observations[i][key] = observations[i][key][skip_first_n:]

                # Repeat last dimension of each observation for history_len times
                for key in observations[i].keys():
                    observations[i][key] = np.concatenate(
                        [
                            observations[i][key],
                            [observations[i][key][-1]] * self._history_len,
                        ],
                        axis=0,
                    )

                # store
                episode = dict(
                    observation=observations[i],
                )
                self._episodes[_path_idx].append(episode)
                self._max_episode_len = max(
                    self._max_episode_len,
                    (
                        len(observations[i])
                        if not isinstance(observations[i], dict)
                        else len(observations[i][self._pixel_keys[0]])
                    ),
                )
                self._max_state_dim = self._num_robot_points * self._point_dim
                self._num_samples += len(observations[i][self._pixel_keys[0]])

                # min, max track
                for pixel_key in self._pixel_keys:
                    if self._use_robot_points:
                        track_key = f"{self._robot_points_key}_{pixel_key}"
                        track = observations[i][track_key]
                        track = einops.rearrange(track, "t n d -> (t n) d")
                        min_track = (
                            np.minimum(min_track, np.min(track, axis=0))
                            if min_track is not None
                            else np.min(track, axis=0)
                        )
                        max_track = (
                            np.maximum(max_track, np.max(track, axis=0))
                            if max_track is not None
                            else np.max(track, axis=0)
                        )
                    if self._use_object_points:
                        track_key = f"{self._object_points_key}_{pixel_key}"
                        track = observations[i][track_key]
                        track = einops.rearrange(track, "t n d -> (t n) d")
                        min_track = (
                            np.minimum(min_track, np.min(track, axis=0))
                            if min_track is not None
                            else np.min(track, axis=0)
                        )
                        max_track = (
                            np.maximum(max_track, np.max(track, axis=0))
                            if max_track is not None
                            else np.max(track, axis=0)
                        )

        self.stats = {
            "past_tracks": {
                "min": min_track,
                "max": max_track,
            },
            "future_tracks": {
                "min": np.concatenate(
                    [min_track for _ in range(self._num_queries)], axis=0
                ),
                "max": np.concatenate(
                    [max_track for _ in range(self._num_queries)], axis=0
                ),
            },
            "gripper_states": {
                "min": -2.0,
                "max": 2.0,
            },
        }

        self.preprocess = {
            "past_tracks": lambda x: (x - self.stats["past_tracks"]["min"])
            / (
                self.stats["past_tracks"]["max"]
                - self.stats["past_tracks"]["min"]
                + 1e-5
            ),
            "future_tracks": lambda x: (x - self.stats["future_tracks"]["min"])
            / (
                self.stats["future_tracks"]["max"]
                - self.stats["future_tracks"]["min"]
                + 1e-5
            ),
            "gripper_states": lambda x: (x - self.stats["gripper_states"]["min"])
            / (
                self.stats["gripper_states"]["max"]
                - self.stats["gripper_states"]["min"]
                + 1e-5
            ),
        }

        # Samples from envs
        self.envs_till_idx = len(self._episodes)

    def _sample_episode(self, env_idx=None):
        if env_idx is not None:
            idx = env_idx
        else:
            idx = np.random.choice(list(self._episodes.keys()))

        episode = random.choice(self._episodes[idx])
        return (episode, idx) if env_idx is None else episode

    def _sample(self):
        episodes, env_idx = self._sample_episode()
        observations = episodes["observation"]
        traj_len = len(observations[self._pixel_keys[0]])

        # Sample obs, action
        sample_idx = np.random.randint(
            0, len(observations[self._pixel_keys[0]]) - self._history_len
        )
        pixel_key = np.random.choice(self._pixel_keys)

        # action mask to only apply loss for robot or hand points
        action_mask = []

        past_tracks = []

        if self._use_robot_points:
            track_key = f"{self._robot_points_key}_{pixel_key}"
            num_points = self._num_robot_points
            robot_points = observations[track_key][
                max(
                    0,
                    sample_idx - self._history_len * self._subsample + self._subsample,
                ) : sample_idx
                + 1 : self._subsample
            ][:, -num_points:]
            if len(robot_points) < self._history_len:
                prior = np.array(
                    [robot_points[0]] * (self._history_len - len(robot_points))
                )
                robot_points = np.concatenate([prior, robot_points], axis=0)
            past_tracks.append(robot_points)
            action_mask.extend([1] * num_points)

        if self._use_object_points:
            object_points = observations[f"{self._object_points_key}_{pixel_key}"][
                max(
                    0,
                    sample_idx
                    - self._history_len * self._subsample
                    + self._subsample,  # 1
                ) : sample_idx
                + 1 : self._subsample
            ]
            if len(object_points) < self._history_len:
                prior = np.array(
                    [object_points[0]] * (self._history_len - len(object_points))
                )
                object_points = np.concatenate([prior, object_points], axis=0)
            past_tracks.append(object_points)
            action_mask.extend([0] * self._num_object_points)

        past_tracks = np.concatenate(past_tracks, axis=1)
        action_mask = np.array(action_mask)

        # past gripper_states
        past_gripper_states = observations[f"gripper_states"][
            max(
                0,
                sample_idx - self._history_len * self._subsample + self._subsample,  # 1
            ) : sample_idx
            + 1 : self._subsample
        ]
        if len(past_gripper_states) < self._history_len:
            prior = np.array(
                [past_gripper_states[0]]
                * (self._history_len - len(past_gripper_states))
            )
            past_gripper_states = np.concatenate([prior, past_gripper_states], axis=0)

        future_tracks = []
        num_future_tracks = self._history_len + self._num_queries - 1

        # for action sampling
        start_idx = min(sample_idx + 1, traj_len - 1)
        end_idx = min(start_idx + num_future_tracks * self._subsample, traj_len)

        if self._use_robot_points:
            track_key = f"{self._robot_points_key}_{pixel_key}"
            num_points = self._num_robot_points
            ft = observations[track_key][start_idx : end_idx : self._subsample][
                :, -num_points:
            ]
            if len(ft) < num_future_tracks:
                post = np.array([ft[-1]] * (num_future_tracks - len(ft)))
                ft = np.concatenate([ft, post], axis=0)
            # ft is of shape (T, N, D)
            ft = ft.transpose(
                1, 0, 2
            )  # (N, T, D) where T=history_len+num_queries-1=H+Q-1
            ft = np.lib.stride_tricks.sliding_window_view(
                ft, self._num_queries, 1
            )  # (N, H, D, Q)
            ft = ft.transpose(1, 0, 3, 2)  # (H, N, Q, D)
            ft = einops.rearrange(ft, "h n q d -> h n (q d)")
            future_tracks.append(ft)

        if self._use_object_points:
            ft = observations[f"{self._object_points_key}_{pixel_key}"][
                start_idx : end_idx : self._subsample
            ]
            if len(ft) < num_future_tracks:
                post = np.array([ft[-1]] * (num_future_tracks - len(ft)))
                ft = np.concatenate([ft, post], axis=0)
            # ft is of shape (T, N, D)
            ft = ft.transpose(
                1, 0, 2
            )  # (N, T, D) where T=history_len+num_queries-1=H+Q-1
            ft = np.lib.stride_tricks.sliding_window_view(
                ft, self._num_queries, 1
            )  # (N, H, D, Q)
            ft = ft.transpose(1, 0, 3, 2)  # (H, N, Q, D)
            ft = einops.rearrange(ft, "h n q d -> h n (q d)")
            future_tracks.append(ft)

        future_tracks = np.concatenate(future_tracks, axis=1)

        # future gripper_states
        future_gripper_states = observations[f"gripper_states"][
            start_idx : end_idx : self._subsample
        ]
        if len(future_gripper_states) < num_future_tracks:
            post = np.array(
                [future_gripper_states[-1]]
                * (num_future_tracks - len(future_gripper_states))
            )
            future_gripper_states = np.concatenate(
                [future_gripper_states, post], axis=0
            )
        future_gripper_states = future_gripper_states.reshape(
            future_gripper_states.shape[0]
        )
        future_gripper_states = np.lib.stride_tricks.sliding_window_view(
            future_gripper_states, self._num_queries
        )

        return_dict = {
            "past_tracks": self.preprocess["past_tracks"](past_tracks),
            "past_gripper_states": self.preprocess["gripper_states"](
                past_gripper_states
            ),
            "future_tracks": self.preprocess["future_tracks"](future_tracks),
            "future_gripper_states": self.preprocess["gripper_states"](
                future_gripper_states
            ),
            "action_mask": action_mask,
        }

        return return_dict

    def sample_actions(self, env_idx):
        episode = self._sample_episode(env_idx)
        actions = []
        for i in range(
            0,
            len(episode["observation"][f"point_tracks_{self._pixel_keys[0]}"]),
            self._subsample,
        ):
            action = {}
            for pixel_key in self._pixel_keys:
                action[f"future_tracks_{pixel_key}"] = episode["observation"][
                    f"point_tracks_{pixel_key}"
                ][i]
            actions.append(action)
        return actions

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples
