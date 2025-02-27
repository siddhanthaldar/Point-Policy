from collections import deque
from typing import Any, NamedTuple

import gym
from gym import spaces

import franka_env
import dm_env
import numpy as np
from dm_env import StepType, specs, TimeStep

import cv2
import torch
from scipy.spatial.transform import Rotation as R

from robot_utils.franka.gripper_points import extrapoints, Tshift
from robot_utils.franka.utils import (
    triangulate_points,
    pixelkey2camera,
    rigid_transform_3D,
)

crop_h, crop_w = (0.0, 1.0), (0.0, 1.0)


class RGBArrayAsObservationWrapper(dm_env.Environment):
    """
    Use env.render(rgb_array) as observation
    rather than the observation environment provides

    From: https://github.com/hill-a/stable-baselines/issues/915
    """

    def __init__(
        self,
        env,
        task_name,
        object_labels,
        calib_path,
        width=256,
        height=256,
        calib_height=480,
        calib_width=640,
        use_robot=False,
        max_episode_len=300,
        max_state_dim=100,
        pixel_keys=["pixels0"],
        use_robot_points=True,
        num_robot_points=9,
        use_object_points=True,
        num_object_points=7,
        points_cfg=None,
    ):
        """
        Actions are always absolute actions
        """

        self._env = env
        self._task_name = task_name
        self._object_labels = object_labels
        self._height, self._width = height, width
        self._calib_height, self._calib_width = calib_height, calib_width
        self._use_robot = use_robot
        self._max_episode_len = max_episode_len
        self._max_state_dim = max_state_dim
        self._pixel_keys = pixel_keys
        self._device = "cpu"

        # track vars
        self._use_robot_points = use_robot_points
        self._num_robot_points = num_robot_points
        self._use_object_points = use_object_points
        self._num_object_points = num_object_points

        if self._use_robot and self._use_object_points:
            # init points class if using object points
            from point_utils.points_class import PointsClass

            points_cfg["task_name"] = task_name
            points_cfg["pixel_keys"] = self._pixel_keys
            points_cfg["object_labels"] = object_labels
            self._points_class = PointsClass(**points_cfg)

        # calibration data
        assert self._height == self._width
        assert calib_path is not None
        self.calibration_data = np.load(calib_path, allow_pickle=True).item()
        self._camera_names = list(self.calibration_data.keys())
        self.camera_projections = {}
        for camera_name in self._camera_names:
            intrinsic = self.calibration_data[camera_name]["int"]
            intrinsic = np.concatenate((intrinsic, np.zeros((3, 1))), axis=1)
            extrinsic = self.calibration_data[camera_name]["ext"]
            self.camera_projections[camera_name] = intrinsic @ extrinsic

        obs = self._env.reset()
        if self._use_robot:
            pixels = obs[self._pixel_keys[0]]
            self.observation_space = spaces.Box(
                low=0, high=255, shape=pixels.shape, dtype=pixels.dtype
            )

            # Action spec
            action_spec = self._env.action_space
            self._action_spec = specs.Array(
                shape=action_spec.shape, dtype=action_spec.dtype, name="action"
            )
            # Observation spec
            robot_state = obs["features"]
            self._obs_spec = {}
            for pixel_key in self._pixel_keys:
                self._obs_spec[pixel_key] = specs.BoundedArray(
                    # shape=obs[pixel_key].shape,
                    shape=(self._height, self._width, 3),
                    dtype=np.uint8,
                    minimum=0,
                    maximum=255,
                    name=pixel_key,
                )
            self._obs_spec["proprioceptive"] = specs.BoundedArray(
                shape=robot_state.shape,
                dtype=np.float32,
                minimum=-np.inf,
                maximum=np.inf,
                name="proprioceptive",
            )
        else:
            pixels, features = obs["pixels"], obs["features"]
            self.observation_space = spaces.Box(
                low=0, high=255, shape=pixels.shape, dtype=pixels.dtype
            )

            # Action spec
            action_spec = self._env.action_space
            self._action_spec = specs.Array(
                shape=action_spec.shape, dtype=action_spec.dtype, name="action"
            )

            # Observation spec
            self._obs_spec = {}
            for pixel_key in self._pixel_keys:
                self._obs_spec[pixel_key] = specs.BoundedArray(
                    # shape=pixels.shape,
                    shape=(self._height, self._width, 3),
                    dtype=np.uint8,
                    minimum=0,
                    maximum=255,
                    name=pixel_key,
                )
            self._obs_spec["proprioceptive"] = specs.BoundedArray(
                shape=features.shape,
                dtype=np.float32,
                minimum=-np.inf,
                maximum=np.inf,
                name="proprioceptive",
            )
        self._obs_spec["features"] = specs.BoundedArray(
            shape=(self._max_state_dim,),
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="features",
        )

        self.render_image = None
        self.prev_gripper_points = None

        # amount for shifting the points in robot base frame
        self.Tshift = Tshift

    def reset(self, **kwargs):
        self._step = 0
        obs = self._env.reset(**kwargs)
        self.prev_gripper_state = -1  # Default open gripper

        self._current_pose = obs["features"]

        observation = {}

        # point tracker init
        self.init_track_points(obs)
        # pixels and point tracks
        for pixel_key in self._pixel_keys:
            img = obs[pixel_key]
            img = cv2.resize(img, (self._width, self._height))
            img = img.transpose(2, 0, 1)
            observation[pixel_key] = img
            observation[f"point_tracks_{pixel_key}"] = (
                self._track_pts[pixel_key].cpu().numpy()
            )
        # proprioceptive
        observation["proprioceptive"] = self._current_pose
        # others
        observation["features"] = self._current_pose
        observation["goal_achieved"] = False
        self.observation = observation
        return observation

    def step(self, action):
        self._step += 1

        robot_action = self.point2action(action)
        print("Robot action:", robot_action)
        obs, reward, done, info = self._env.step(robot_action)

        self._current_pose = obs["features"]

        observation = {}
        for pixel_key in self._pixel_keys:
            img = obs[pixel_key]
            img = cv2.resize(img, (self._width, self._height))
            img = img.transpose(2, 0, 1)
            observation[pixel_key] = img
        observation["proprioceptive"] = self._current_pose

        # robot points
        robot_points, robot_points_3d = self.get_pixel_on_robot()
        self.prev_gripper_points = robot_points_3d
        for pixel_key in self._pixel_keys:
            robot_point = robot_points[pixel_key]
            current_track = robot_point

            if self._use_object_points:
                self._points_class.add_to_image_list(
                    obs[pixel_key][:, :, ::-1], pixel_key
                )
                self._points_class.track_points(pixel_key)
                object_pts = self._points_class.get_points_on_image(pixel_key).numpy()[
                    0
                ]
                current_track = np.concatenate([current_track, object_pts], axis=0)

            self._track_pts[pixel_key] = current_track
            observation[f"point_tracks_{pixel_key}"] = current_track

        observation["features"] = self._current_pose
        observation["goal_achieved"] = done

        if self._step >= self._max_episode_len:
            done = True
        done = done | observation["goal_achieved"]

        self.observation = observation

        return observation, reward, done, info

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def render(self, mode="rgb_array", width=256, height=256):
        return cv2.resize(self._env.render("rgb_array"), (width, height))

    def get_pixel_on_robot(self):
        # get current gripper pose in robot base frame
        pos = self._current_pose[:3]
        ori = self._current_pose[3:7]  # in quat
        T_g_b = np.eye(4)
        T_g_b[:3, :3] = R.from_quat(ori).as_matrix()
        T_g_b[:3, 3] = pos

        # shift the points in robot base frame
        T_g_b = T_g_b @ self.Tshift

        # add extra points
        points3d = [T_g_b[:3, 3]]
        gripper_state = self._current_pose[-1]
        for idx, Tp in enumerate(extrapoints):
            if gripper_state == 1 and idx in [0, 1]:
                Tp = Tp.copy()
                Tp[1, 3] = 0.015 if idx == 0 else -0.015
            pt = T_g_b @ Tp
            pt = pt[:3, 3]
            points3d.append(pt[:3])
        points3d = np.array(points3d)

        pixel_poses = {}
        for pixel_key in self._pixel_keys:
            if pixel_key == "pixels51":
                continue

            camera_name = pixelkey2camera[pixel_key]

            P = self.calibration_data[camera_name]["ext"]
            K = self.calibration_data[camera_name]["int"]
            D = self.calibration_data[camera_name]["dist_coeff"]

            r, t = P[:3, :3], P[:3, 3]
            r, _ = cv2.Rodrigues(r)
            points2d, _ = cv2.projectPoints(points3d, r, t, K, D)
            points2d = points2d[:, 0]
            pixel_poses[pixel_key] = points2d

        return pixel_poses, points3d

    def init_track_points(self, obs):
        robot_points, robot_points_3d = self.get_pixel_on_robot()
        self.prev_gripper_points = robot_points_3d

        self.base_robot_points = np.array(robot_points_3d)
        # orientation of the robot at the 0th step
        self.robot_base_orientation = R.from_rotvec([np.pi, 0, 0]).as_matrix()

        self._track_pts = {}
        for pixel_key in self._pixel_keys:
            points = []

            robot_pts = torch.tensor(
                robot_points[pixel_key], device=self._device
            ).float()[None]
            if self._use_robot_points:
                points.append(robot_pts)
            else:
                points[0][:, -len(robot_pts[0]) :] = robot_pts

            if self._use_object_points:
                frame = obs[pixel_key]
                self._points_class.reset_episode()
                self._points_class.add_to_image_list(frame[:, :, ::-1], pixel_key)
                for object_label in self._object_labels:
                    self._points_class.find_semantic_similar_points(
                        pixel_key, object_label
                    )
                self._points_class.track_points(pixel_key, is_first_step=True)
                self._points_class.track_points(pixel_key)
                object_pts = self._points_class.get_points_on_image(pixel_key)
                points.append(object_pts)

            self._track_pts[pixel_key] = torch.cat(points, dim=1)[0]

    def point2action(self, action):
        points, projection_matrices = [], []
        for pixel_key in self._pixel_keys:
            if pixel_key == "pixels51":
                continue
            robot_pts_end_idx = self._num_robot_points if self._use_robot_points else 0
            future_tracks = action[f"future_tracks_{pixel_key}"][:robot_pts_end_idx, :2]

            camera_name = pixelkey2camera[pixel_key]
            extrinsic = self.calibration_data[camera_name]["ext"]
            intrinsic = self.calibration_data[camera_name]["int"]
            intrinsic = np.concatenate([intrinsic, np.zeros((3, 1))], axis=1)
            projection_matrix = intrinsic @ extrinsic

            points.append(future_tracks)
            projection_matrices.append(projection_matrix)

        points3d = triangulate_points(projection_matrices, points)[:, :3]

        robot_pos, ori = self.compute_action_from_3dpoints(points3d)
        gripper_state = self.compute_gripper(action)
        robot_action = self.compute_robot_action(robot_pos, ori, gripper_state)

        return robot_action

    def compute_action_from_3dpoints(self, points3d):
        robot_pos = points3d[0]
        ori, _ = rigid_transform_3D(self.base_robot_points, points3d)
        ori = ori @ self.robot_base_orientation
        return robot_pos, ori

    def compute_gripper(self, action):
        gripper_state = action["gripper"][:1]

        if self.prev_gripper_state == -1 and gripper_state > -0.63:
            gripper_state = 1
        elif self.prev_gripper_state == 1 and gripper_state < 0.6:
            gripper_state = -1
        else:
            gripper_state = self.prev_gripper_state
        self.prev_gripper_state = gripper_state

        gripper_state = np.array([gripper_state])
        return gripper_state

    def compute_robot_action(self, target_position, target_orientation, gripper):
        """
        Return absolute actions
        """
        T_target = np.eye(4)
        T_target[:3, :3] = target_orientation
        T_target[:3, 3] = target_position

        # T_target = T_eef @ Tshift -> get T_eef
        T_eef = T_target @ np.linalg.inv(self.Tshift)
        target_position = T_eef[:3, 3]
        target_orientation = T_eef[:3, :3]

        # convert orientation from rotation matrix to quaternion
        target_orientation = R.from_matrix(target_orientation).as_quat()

        return np.concatenate([target_position, target_orientation, gripper])

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixel_keys):
        self._env = env
        self._num_frames = num_frames
        self._pixel_keys = pixel_keys

        self._frames, self._track_pts = {}, {}
        for key in self._pixel_keys:
            self._frames[key] = deque([], maxlen=num_frames)
            self._track_pts[key] = deque([], maxlen=num_frames)

        wrapped_obs_spec = env.observation_spec()[self._pixel_keys[0]]
        pixels_shape = wrapped_obs_spec.shape
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]

        self._obs_spec = {}
        for key in self._pixel_keys:
            self._obs_spec[key] = specs.BoundedArray(
                shape=np.concatenate(
                    [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
                ),
                dtype=np.uint8,
                minimum=0,
                maximum=255,
                name=key,
            )
        self._obs_spec["features"] = self._env.observation_spec()["features"]
        self._obs_spec["proprioceptive"] = self._env.observation_spec()[
            "proprioceptive"
        ]

    def _transform_observation(self, time_step):
        obs = {}
        for key in self._pixel_keys:
            assert len(self._frames[key]) == self._num_frames
            assert len(self._track_pts[key]) == self._num_frames
            obs[key] = np.concatenate(list(self._frames[key]), axis=0)
            obs[f"point_tracks_{key}"] = np.concatenate(
                list(self._track_pts[key]), axis=0
            )
        obs["features"] = time_step.observation["features"]
        obs["proprioceptive"] = time_step.observation["proprioceptive"]
        obs["goal_achieved"] = time_step.observation["goal_achieved"]
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step, key):
        pixels = time_step.observation[key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.copy()

    def reset(self):
        time_step = self._env.reset()
        for key in self._pixel_keys:
            pixels = self._extract_pixels(time_step, key)
            for _ in range(self._num_frames):
                self._frames[key].append(pixels)
                self._track_pts[key].append(
                    time_step.observation[f"point_tracks_{key}"]
                )
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        for key in self._pixel_keys:
            pixels = self._extract_pixels(time_step, key)
            self._frames[key].append(pixels)
            self._track_pts[key].append(time_step.observation[f"point_tracks_{key}"])
        return self._transform_observation(time_step)

    def point2action(self, action):
        return self._env.point2action(action)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._discount = 1.0

        # Action spec
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.Array(
            shape=wrapped_action_spec.shape, dtype=dtype, name="action"
        )

    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        step_type = StepType.LAST if done else StepType.MID

        return TimeStep(
            step_type=step_type,
            reward=reward,
            discount=self._discount,
            observation=observation,
        )

    def point2action(self, action):
        return self._env.point2action(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        return TimeStep(
            step_type=StepType.FIRST, reward=0, discount=self._discount, observation=obs
        )

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self, **kwargs):
        time_step = self._env.reset(**kwargs)
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def _replace(
        self, time_step, observation=None, action=None, reward=None, discount=None
    ):
        if observation is None:
            observation = time_step.observation
        if action is None:
            action = time_step.action
        if reward is None:
            reward = time_step.reward
        if discount is None:
            discount = time_step.discount
        return ExtendedTimeStep(
            observation=observation,
            step_type=time_step.step_type,
            action=action,
            reward=reward,
            discount=discount,
        )

    def point2action(self, action):
        return self._env.point2action(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(
    task_name,
    object_labels,
    action_repeat,
    height,
    width,
    calib_height,
    calib_width,
    max_episode_len,
    max_state_dim,
    calib_path,
    eval,  # True means use_robot=True
    pixel_keys,
    use_robot_points,
    num_robot_points,
    use_object_points,
    num_object_points,
    points_cfg,
):
    env = gym.make(
        "Franka-v1",
        height=calib_height,
        width=calib_width,
        use_robot=eval,
    )

    # apply wrappers
    env = RGBArrayAsObservationWrapper(
        env,
        task_name=task_name,
        object_labels=object_labels,
        calib_path=calib_path,
        height=height,
        width=width,
        calib_height=calib_height,
        calib_width=calib_width,
        use_robot=eval,
        max_episode_len=max_episode_len,
        max_state_dim=max_state_dim,
        pixel_keys=pixel_keys,
        use_robot_points=use_robot_points,
        num_robot_points=num_robot_points,
        use_object_points=use_object_points,
        num_object_points=num_object_points,
        points_cfg=points_cfg,
    )
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FrameStackWrapper(env, 1, pixel_keys)
    env = ExtendedTimeStepWrapper(env)

    return [env], [task_name]
