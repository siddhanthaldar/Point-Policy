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
    pixelkey2camera,
    pixel2d_to_3d,
    triangulate_points,
    rotation_6d_to_matrix,
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
        use_robot=False,
        max_episode_len=300,
        max_state_dim=100,
        pixel_keys=["pixels0"],
        use_robot_points=True,
        num_robot_points=9,
        use_object_points=True,
        num_object_points=8,
        action_type="absolute",  # absolute, delta
        points_cfg=None,
        use_gt_depth=False,
        point_dim=3,
    ):
        """
        Actions are always absolute actions
        """

        self._env = env
        self._task_name = task_name
        self._object_labels = object_labels
        self._height, self._width = height, width
        self.use_robot = use_robot
        self._max_episode_len = max_episode_len
        self._max_state_dim = max_state_dim
        self._pixel_keys = pixel_keys
        self._device = "cpu"
        self._action_type = action_type
        self._use_gt_depth = use_gt_depth
        self._point_dim = point_dim

        # track vars
        self._use_robot_points = use_robot_points
        self._num_robot_points = num_robot_points
        self._use_object_points = use_object_points
        self._num_object_points = num_object_points

        if self.use_robot and self._use_object_points:
            # init points class if using object points
            from point_utils.points_class import PointsClass

            points_cfg["task_name"] = task_name
            points_cfg["pixel_keys"] = self._pixel_keys
            points_cfg["object_labels"] = object_labels
            self._points_class = PointsClass(**points_cfg)

        # calibration data
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
        if self.use_robot:
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
                    shape=obs[pixel_key].shape,
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
                    shape=pixels.shape,
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
        robot_points, robot_points_3d = self.get_pixel_on_robot()
        self.init_track_points(obs, robot_points, robot_points_3d)

        for pixel_key in self._pixel_keys:
            observation[pixel_key] = obs[pixel_key]

        if self._point_dim == 2:
            # pixels and point tracks
            for pixel_key in self._pixel_keys:
                observation[f"point_tracks_{pixel_key}"] = self._track_pts[pixel_key]
        elif self._point_dim == 3:
            if not self._use_gt_depth:
                P, pts = [], []
                for pixel_key in self._pixel_keys:
                    camera_name = pixelkey2camera[pixel_key]
                    P.append(self.camera_projections[camera_name])
                    pt2d = self._track_pts[pixel_key]
                    pts.append(pt2d)

                pts3d = triangulate_points(P, pts)[:, :3]
                pts3d[: self._num_robot_points] = robot_points_3d
                for pixel_key in self._pixel_keys:
                    observation[f"point_tracks_{pixel_key}"] = np.array(pts3d)
            else:
                for pixel_key in self._pixel_keys:
                    camera_name = pixelkey2camera[pixel_key]
                    pt2d = self._track_pts[pixel_key]
                    depth_key = f"depth{pixel_key[-1]}"
                    depth = obs[depth_key]
                    # compute depth for each points
                    depths = []
                    for pt in pt2d:
                        x, y = pt.astype(int)
                        depths.append(depth[y, x])
                    depths = np.array(depths) / 1000.0  # convert to meters
                    extr = self.calibration_data[camera_name]["ext"]
                    intr = self.calibration_data[camera_name]["int"]
                    pt3d = pixel2d_to_3d(pt2d, depths, intr, extr)
                    observation[f"point_tracks_{pixel_key}"] = pt3d

        observation["features"] = self._current_pose
        observation["goal_achieved"] = False
        self.observation = observation
        return observation

    def step(self, action):
        self._step += 1

        gripper = action[-1]
        if self.prev_gripper_state == -1 and gripper > -0.3:
            gripper = 1
        elif self.prev_gripper_state == 1 and gripper < 0.6:
            gripper = -1
        else:
            gripper = self.prev_gripper_state
        self.prev_gripper_state = gripper

        # convert action to quarternion before sending
        # Incoming action is in the rotation 6D format.
        pos, rot = action[:3], action[3:9]
        rot = rotation_6d_to_matrix(rot)
        rot = R.from_matrix(rot).as_quat()
        action = np.concatenate([pos, rot, [gripper]])

        print(f"Robot action: {action}")
        obs, reward, done, info = self._env.step(action)

        self._current_pose = obs["features"]

        observation = {}
        for pixel_key in self._pixel_keys:
            observation[pixel_key] = obs[pixel_key]

        # robot points
        robot_points, robot_points_3d = self.get_pixel_on_robot()
        self.prev_gripper_points = robot_points_3d
        for pixel_key in self._pixel_keys:
            current_track = robot_points[pixel_key]

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

        # Get 3d points from 3D depth or 2D triangulation
        if self._point_dim == 3:
            if not self._use_gt_depth:
                P, pts = [], []
                for pixel_key in self._pixel_keys:
                    camera_name = pixelkey2camera[pixel_key]
                    P.append(self.camera_projections[camera_name])
                    pt2d = self._track_pts[pixel_key]
                    pts.append(pt2d)

                pts3d = triangulate_points(P, pts)[:, :3]
                pts3d[: self._num_robot_points] = robot_points_3d
                for pixel_key in self._pixel_keys:
                    observation[f"point_tracks_{pixel_key}"] = np.array(pts3d)
            else:
                for pixel_key in self._pixel_keys:
                    camera_name = pixelkey2camera[pixel_key]
                    pt2d = self._track_pts[pixel_key]
                    depth_key = f"depth{pixel_key[-1]}"
                    depth = obs[depth_key]
                    # compute depth for each points
                    depths = []
                    for pt in pt2d:
                        x, y = pt.astype(int)
                        depths.append(depth[y, x])
                    depths = np.array(depths) / 1000.0  # convert to meters
                    extr = self.calibration_data[camera_name]["ext"]
                    intr = self.calibration_data[camera_name]["int"]
                    pt3d = pixel2d_to_3d(pt2d, depths, intr, extr)
                    observation[f"point_tracks_{pixel_key}"] = pt3d

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
                # ignore egocentric camera
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

    def init_track_points(self, obs, robot_points, robot_points_3d):
        self.prev_gripper_points = robot_points_3d

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

            self._track_pts[pixel_key] = torch.cat(points, dim=1)[0].numpy()

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
    max_episode_len,
    max_state_dim,
    calib_path,
    eval,  # True means use_robot=True
    pixel_keys,
    use_robot_points,
    num_robot_points,
    use_object_points,
    num_object_points,
    action_type,
    points_cfg,
    use_gt_depth,
    point_dim,
):
    env = gym.make(
        "Franka-v1",
        height=height,
        width=width,
        use_robot=eval,
        use_gt_depth=use_gt_depth,
    )

    # apply wrappers
    env = RGBArrayAsObservationWrapper(
        env,
        task_name,
        object_labels,
        calib_path=calib_path,
        height=height,
        width=width,
        use_robot=eval,
        max_episode_len=max_episode_len,
        max_state_dim=max_state_dim,
        pixel_keys=pixel_keys,
        use_robot_points=use_robot_points,
        num_robot_points=num_robot_points,
        use_object_points=use_object_points,
        num_object_points=num_object_points,
        action_type=action_type,
        points_cfg=points_cfg,
        use_gt_depth=use_gt_depth,
        point_dim=point_dim,
    )
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = ExtendedTimeStepWrapper(env)

    return [env], [task_name]
