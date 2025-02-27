import random
import einops
import numpy as np
import pickle as pkl
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset
from scipy.spatial.transform import Rotation as R


def get_relative_action(actions, action_after_steps):
    """
    Convert absolute axis angle actions to relative axis angle actions
    Action has both position and orientation. Convert to transformation matrix, get
    relative transformation matrix, convert back to axis angle
    """

    relative_actions = []
    for i in range(len(actions)):
        ####### Get relative transformation matrix #######
        # previous pose
        pos_prev = actions[i, :3]
        ori_prev = actions[i, 3:6]
        r_prev = R.from_rotvec(ori_prev).as_matrix()
        matrix_prev = np.eye(4)
        matrix_prev[:3, :3] = r_prev
        matrix_prev[:3, 3] = pos_prev
        # current pose
        next_idx = min(i + action_after_steps, len(actions) - 1)
        pos = actions[next_idx, :3]
        ori = actions[next_idx, 3:6]
        gripper = actions[next_idx, 6:]
        r = R.from_rotvec(ori).as_matrix()
        matrix = np.eye(4)
        matrix[:3, :3] = r
        matrix[:3, 3] = pos
        # relative transformation
        matrix_rel = np.linalg.inv(matrix_prev) @ matrix
        # relative pose
        pos_rel = pos - pos_prev
        r_rel = R.from_matrix(matrix_rel[:3, :3]).as_rotvec()
        # add to list
        relative_actions.append(np.concatenate([pos_rel, r_rel, gripper]))

    # last action
    last_action = np.zeros_like(actions[-1])
    last_action[-1] = actions[-1][-1]
    while len(relative_actions) < len(actions):
        relative_actions.append(last_action)
    return np.array(relative_actions, dtype=np.float32)


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
        self._paths.extend([Path(path) / f"{task}.pkl" for task in tasks])

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
                    track_key = f"{self._robot_points_key}_{pixel_key}"
                    track = observations[i][track_key]
                    track = einops.rearrange(track, "t n d -> (t n) d")
                    if min_track is None:
                        min_track = np.min(track, axis=0)
                        max_track = np.max(track, axis=0)
                    else:
                        min_track = np.minimum(min_track, np.min(track, axis=0))
                        max_track = np.maximum(max_track, np.max(track, axis=0))
                    if self._use_object_points:
                        track_key = f"{self._object_points_key}_{pixel_key}"
                        track = observations[i][track_key]
                        track = einops.rearrange(track, "t n d -> (t n) d")
                        min_track = np.minimum(min_track, np.min(track, axis=0))
                        max_track = np.maximum(max_track, np.max(track, axis=0))

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

        # augmentation
        self.aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(self._img_size, padding=4),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
            ]
        )

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
        episodes, _ = self._sample_episode()
        observations = episodes["observation"]
        traj_len = len(observations[self._pixel_keys[0]])

        # Sample obs, action
        sample_idx = np.random.randint(
            0, len(observations[self._pixel_keys[0]]) - self._history_len
        )
        pixel_key = np.random.choice(self._pixel_keys)

        pixel = self.aug(observations[pixel_key][sample_idx])

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
                    sample_idx - self._history_len * self._subsample + self._subsample,
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
                sample_idx - self._history_len * self._subsample + self._subsample,
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
            ft = ft.transpose(1, 0, 2)
            ft = ft.reshape(num_points, -1)
            ft = np.lib.stride_tricks.sliding_window_view(
                ft, self._num_queries * self._point_dim, 1
            )[:, :: self._point_dim]
            ft = ft.transpose(1, 0, 2)
            future_tracks.append(ft)

        if self._use_object_points:
            ft = observations[f"{self._object_points_key}_{pixel_key}"][
                start_idx : end_idx : self._subsample
            ]
            if len(ft) < num_future_tracks:
                post = np.array([ft[-1]] * (num_future_tracks - len(ft)))
                ft = np.concatenate([ft, post], axis=0)
            ft = ft.transpose(1, 0, 2)
            ft = ft.reshape(ft.shape[0], -1)
            ft = np.lib.stride_tricks.sliding_window_view(
                ft, self._num_queries * self._point_dim, 1
            )[:, :: self._point_dim]
            ft = ft.transpose(1, 0, 2)
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
            "pixels": pixel,
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
