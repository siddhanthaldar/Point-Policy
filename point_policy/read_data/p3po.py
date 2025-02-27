import einops
import random
import numpy as np
import pickle as pkl
from pathlib import Path

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
        action_type,
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
        self._action_type = action_type
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
        if self._use_object_points:
            self._object_pt_paths = {}

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
        min_act, max_act = None, None
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
                # absolute actions
                action_key = (
                    "human_poses"
                    if "human_poses" in observations[i].keys()
                    else "cartesian_states"
                )
                actions = np.concatenate(
                    [
                        observations[i][action_key],
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

                # action after steps
                if self._action_type == "absolute":
                    actions = actions[self._action_after_steps :]
                else:
                    actions = get_relative_action(actions, self._subsample)

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
                if self._action_type == "relative":
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
                self._max_state_dim = self._num_robot_points * self._point_dim
                self._num_samples += len(observations[i][self._pixel_keys[0]])

                # max, min action
                if min_act is None:
                    min_act = np.min(actions, axis=0)
                    max_act = np.max(actions, axis=0)
                else:
                    min_act = np.minimum(min_act, np.min(actions, axis=0))
                    max_act = np.maximum(max_act, np.max(actions, axis=0))

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
            "actions": {
                "min": min_act,
                "max": max_act,
            },
            "past_tracks": {
                "min": min_track,
                "max": max_track,
            },
        }

        self.preprocess = {
            "actions": lambda x: (x - self.stats["actions"]["min"])
            / (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5),
            "past_tracks": lambda x: (x - self.stats["past_tracks"]["min"])
            / (
                self.stats["past_tracks"]["max"]
                - self.stats["past_tracks"]["min"]
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
        actions = episodes["action"]

        # Sample obs, action
        sample_idx = np.random.randint(
            0, len(observations[self._pixel_keys[0]]) - self._history_len
        )
        pixel_key = np.random.choice(self._pixel_keys)

        if self._temporal_agg:
            # arrange sampled action to be of shape (history_len, num_queries, action_dim)
            action = np.zeros((self._history_len, self._num_queries, actions.shape[-1]))
            num_actions = (
                self._history_len + self._num_queries - 1
            )  # -1 since its num_queries including the last action of the history
            act = np.zeros((num_actions, actions.shape[-1]))
            act[: min(len(actions), sample_idx + num_actions) - sample_idx] = actions[
                sample_idx : sample_idx + num_actions
            ]
            if len(actions) < sample_idx + num_actions:
                act[len(actions) - sample_idx :] = actions[-1]
            action = np.lib.stride_tricks.sliding_window_view(
                act, (self._num_queries, actions.shape[-1])
            )
            action = action[:, 0]
        else:
            action = actions[sample_idx : sample_idx + self._history_len]

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

        past_tracks = np.concatenate(past_tracks, axis=1)

        return_dict = {
            "past_tracks": self.preprocess["past_tracks"](past_tracks),
            "actions": self.preprocess["actions"](action),
        }

        return return_dict

    def sample_actions(self, env_idx):
        episode = self._sample_episode(env_idx)
        return episode["action"]

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples
