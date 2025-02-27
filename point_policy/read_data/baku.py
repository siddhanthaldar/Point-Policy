import numpy as np
import pickle as pkl
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset
from scipy.spatial.transform import Rotation as R

from robot_utils.franka.utils import matrix_to_rotation_6d


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


def get_quaternion_orientation(cartesian):
    """
    Get quaternion orientation from axis angle representation
    """
    new_cartesian = []
    for i in range(len(cartesian)):
        pos = cartesian[i, :3]
        ori = cartesian[i, 3:]
        quat = R.from_rotvec(ori).as_quat()
        new_cartesian.append(np.concatenate([pos, quat], axis=-1))
    return np.array(new_cartesian, dtype=np.float32)


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
        pixel_keys,
        subsample,
        skip_first_n,
        action_type,
        gt_depth,
    ):
        tasks = [tasks]  # NOTE: single task for now

        self._history = history
        self._history_len = history_len if history else 1
        self._img_size = img_size
        self._action_after_steps = action_after_steps
        self._pixel_keys = pixel_keys
        self._action_type = action_type
        self._gt_depth = gt_depth

        # temporal aggregation
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries

        # get data paths
        self._paths = []
        for task in tasks:
            if self._gt_depth:
                self._paths.extend([Path(path) / f"{task}_gt_depth.pkl"])
            else:
                self._paths.extend([Path(path) / f"{task}.pkl"])

        paths = {}
        idx = 0
        for path in self._paths:
            paths[idx] = path
            idx += 1
        del self._paths
        self._paths = paths

        # read data
        self._episodes = {}
        self._max_episode_len = 0
        self._max_state_dim = 7
        self._num_samples = 0
        min_stat, max_stat = None, None
        min_act, max_act = None, None
        if self._gt_depth:
            min_depth, max_depth = None, None
        for _path_idx in self._paths:
            print(f"Loading {str(self._paths[_path_idx])}")
            # read
            data = pkl.load(open(str(self._paths[_path_idx]), "rb"))
            observations = data["observations"]

            # store
            self._episodes[_path_idx] = []
            for i in range(min(num_demos_per_task, len(observations))):
                # compute actions
                # absolute actions
                actions = np.concatenate(
                    [
                        observations[i]["human_poses"],
                        observations[i]["gripper_states"][:, None],
                    ],
                    axis=1,
                )
                if len(actions) == 0:
                    continue
                # skip first n
                if skip_first_n is not None:
                    for key in observations[i].keys():
                        observations[i][key] = observations[i][key][skip_first_n:]
                    actions = actions[skip_first_n:]
                # subsample
                if subsample is not None:
                    for key in observations[i].keys():
                        observations[i][key] = observations[i][key][::subsample]
                    actions = actions[::subsample]
                # action after steps
                if action_type == "absolute":
                    actions = actions[self._action_after_steps :]
                else:
                    actions = get_relative_action(actions, self._action_after_steps)

                # convert orientation to quarternions
                if self._action_type == "absolute":
                    gripper = actions[:, -1:]
                    # orientaion represented at 6D rotations
                    pos = actions[:, :3]
                    rot = actions[:, 3:6]
                    rot = [R.from_rotvec(rot[i]).as_matrix() for i in range(len(rot))]
                    rot = np.array(rot)
                    rot = matrix_to_rotation_6d(rot)
                    actions = np.concatenate([pos, rot], axis=-1)
                    actions = np.concatenate([actions, gripper], axis=-1)

                # Convert cartesian states to quaternion orientation
                observations[i]["cartesian_states"] = get_quaternion_orientation(
                    observations[i]["cartesian_states"]
                )
                # Repeat last dimension of each observation for history_len times
                for key in observations[i].keys():
                    observations[i][key] = np.concatenate(
                        [
                            observations[i][key],
                            [observations[i][key][-1]] * self._history_len,
                        ],
                        axis=0,
                    )
                # Repeat last action for history_len times
                remaining_actions = actions[-1]
                if action_type != "absolute":
                    pos = remaining_actions[:-1]
                    ori_gripper = remaining_actions[-1:]
                    remaining_actions = np.concatenate(
                        [np.zeros_like(pos), ori_gripper]
                    )
                actions = np.concatenate(
                    [
                        actions,
                        [remaining_actions] * self._history_len,
                    ],
                    axis=0,
                )

                # store
                episode = dict(
                    observation=observations[i],
                    action=actions,
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
                self._num_samples += len(observations[i][self._pixel_keys[0]])

                # max, min action
                if min_act is None:
                    min_act = np.min(actions, axis=0)
                    max_act = np.max(actions, axis=0)
                else:
                    min_act = np.minimum(min_act, np.min(actions, axis=0))
                    max_act = np.maximum(max_act, np.max(actions, axis=0))

                if self._gt_depth:
                    for key in self._pixel_keys:
                        mean = np.min(observations[i][f"depth_{key}"])
                        std = np.max(observations[i][f"depth_{key}"])
                        if min_depth is None:
                            min_depth = mean - 3 * std
                            max_depth = mean + 3 * std
                        else:
                            min_depth = np.minimum(min_depth, mean - 3 * std)
                            max_depth = np.maximum(max_depth, mean + 3 * std)

            # keep record of max and min stat
            max_cartesian = data["max_cartesian"]
            min_cartesian = data["min_cartesian"]
            max_cartesian = np.concatenate(
                [data["max_cartesian"][:3], [1] * 4]
            )  # for quaternion
            min_cartesian = np.concatenate(
                [data["min_cartesian"][:3], [-1] * 4]
            )  # for quaternion
            max_gripper = data["max_gripper"]
            min_gripper = data["min_gripper"]
            max_val = np.concatenate([max_cartesian, max_gripper[None]], axis=0)
            min_val = np.concatenate([min_cartesian, min_gripper[None]], axis=0)
            if max_stat is None:
                max_stat = max_val
                min_stat = min_val
            else:
                max_stat = np.maximum(max_stat, max_val)
                min_stat = np.minimum(min_stat, min_val)

        self.stats = {
            "actions": {
                "min": min_act,
                "max": max_act,
            },
            "proprioceptive": {
                "min": min_stat,
                "max": max_stat,
            },
        }
        self.preprocess = {
            "actions": lambda x: (x - self.stats["actions"]["min"])
            / (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5),
            "proprioceptive": lambda x: (x - self.stats["proprioceptive"]["min"])
            / (
                self.stats["proprioceptive"]["max"]
                - self.stats["proprioceptive"]["min"]
                + 1e-5
            ),
        }
        if self._gt_depth:
            self.stats["depth"] = {"min": min_depth, "max": max_depth}
            self.preprocess["depth"] = lambda x: (x - self.stats["depth"]["min"]) / (
                self.stats["depth"]["max"] - self.stats["depth"]["min"] + 1e-5
            )

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
        idx = (
            np.random.choice(list(self._episodes.keys()))
            if env_idx is None
            else env_idx
        )
        episode = np.random.choice(self._episodes[idx])
        return (episode, idx) if env_idx is None else episode

    def _sample(self):
        episodes, env_idx = self._sample_episode()
        observations = episodes["observation"]
        actions = episodes["action"]

        # Sample obs, action
        sample_idx = np.random.randint(
            0, len(observations[self._pixel_keys[0]]) - self._history_len
        )
        sampled_pixel = {}
        if self._gt_depth:
            sampled_depth = {}
        for key in self._pixel_keys:
            sampled_pixel[key] = observations[key][
                sample_idx : sample_idx + self._history_len
            ]
            sampled_pixel[key] = torch.stack(
                [
                    self.aug(sampled_pixel[key][i])
                    for i in range(len(sampled_pixel[key]))
                ]
            )
            if self._gt_depth:
                sampled_depth[key] = observations[f"depth_{key}"][
                    sample_idx : sample_idx + self._history_len
                ]
                sampled_depth[key] = torch.stack(
                    [
                        torch.tensor(self.preprocess["depth"](sampled_depth[key][i]))[
                            None
                        ]
                        for i in range(len(sampled_depth[key]))
                    ]
                )

        sampled_proprioceptive_state = np.concatenate(
            [
                observations["cartesian_states"][
                    sample_idx : sample_idx + self._history_len
                ],
                observations["gripper_states"][
                    sample_idx : sample_idx + self._history_len
                ][:, None],
            ],
            axis=1,
        )

        if self._temporal_agg:
            # arrange sampled action to be of shape (history_len, num_queries, action_dim)
            sampled_action = np.zeros(
                (self._history_len, self._num_queries, actions.shape[-1])
            )
            num_actions = (
                self._history_len + self._num_queries - 1
            )  # -1 since its num_queries including the last action of the history
            act = np.zeros((num_actions, actions.shape[-1]))
            act[: min(len(actions), sample_idx + num_actions) - sample_idx] = actions[
                sample_idx : sample_idx + num_actions
            ]
            if len(actions) < sample_idx + num_actions:
                act[len(actions) - sample_idx :] = actions[-1]
            sampled_action = np.lib.stride_tricks.sliding_window_view(
                act, (self._num_queries, actions.shape[-1])
            )
            sampled_action = sampled_action[:, 0]
        else:
            sampled_action = actions[sample_idx : sample_idx + self._history_len]

        return_dict = {}
        for key in self._pixel_keys:
            return_dict[key] = sampled_pixel[key]
            if self._gt_depth:
                return_dict[f"depth_{key}"] = sampled_depth[key]
        return_dict["proprioceptive"] = self.preprocess["proprioceptive"](
            sampled_proprioceptive_state
        )
        return_dict["actions"] = self.preprocess["actions"](sampled_action)

        return return_dict

    def sample_actions(self, env_idx):
        episodes = self._sample_episode(env_idx)
        actions = episodes["action"]
        return actions

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples
