from collections import deque
from typing import Any, NamedTuple

import gym
from gym import spaces

import franka_env
import dm_env
import numpy as np
from dm_env import StepType, specs, TimeStep

import cv2
from scipy.spatial.transform import Rotation as R
from robot_utils.franka.utils import rotation_6d_to_matrix

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
        max_episode_len=300,
        max_state_dim=100,
        pixel_keys=["pixels1"],
        use_robot=True,
        action_type="delta",
        use_gt_depth=False,
    ):
        self._env = env
        self._max_episode_len = max_episode_len
        self._max_state_dim = max_state_dim
        self.pixel_keys = pixel_keys
        self.use_robot = use_robot
        self.action_type = action_type
        self._use_gt_depth = use_gt_depth

        obs = self._env.reset()
        if self.use_robot:
            pixels = obs[pixel_keys[0]]
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
            for key in pixel_keys:
                self._obs_spec[key] = specs.BoundedArray(
                    shape=obs[key].shape,
                    dtype=np.uint8,
                    minimum=0,
                    maximum=255,
                    name=key,
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
            for key in pixel_keys:
                self._obs_spec[key] = specs.BoundedArray(
                    shape=pixels.shape,
                    dtype=np.uint8,
                    minimum=0,
                    maximum=255,
                    name=key,
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

    def reset(self, **kwargs):
        self._step = 0
        obs = self._env.reset(**kwargs)
        self.prev_gripper_state = -1  # Default open gripper

        observation = {}
        for key in self.pixel_keys:
            observation[key] = obs[key]
            if self._use_gt_depth:
                depth_key = f"depth_{key}"
                observation[depth_key] = obs[depth_key]
        observation["proprioceptive"] = obs["features"]
        observation["features"] = obs["features"]
        observation["goal_achieved"] = False
        return observation

    def get_state(self):
        return self._env.get_state()

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

        # Convert action to quarternion before sending.
        # Incoming action is in the rotation 6D format.
        pos, rot = action[:3], action[3:9]
        rot = rotation_6d_to_matrix(rot)
        rot = R.from_matrix(rot).as_quat()
        action = np.concatenate([pos, rot, [gripper]])

        if self.action_type == "delta":
            current_state = self.get_state()

            pos_curr = current_state.pos
            ori_curr = current_state.quat
            r_curr = R.from_quat(ori_curr).as_matrix()
            matrix_curr = np.eye(4)
            matrix_curr[:3, :3] = r_curr
            matrix_curr[:3, 3] = pos_curr

            # find transformation matrix
            pos_delta = action[:3]
            ori_delta = action[3:6]
            r_delta = R.from_rotvec(ori_delta).as_matrix()
            matrix_delta = np.eye(4)
            matrix_delta[:3, :3] = r_delta
            matrix_delta[:3, 3] = pos_delta

            # find desired matrix
            matrix_desired = matrix_curr @ matrix_delta

            pos_desired = pos_curr + pos_delta
            r_desired = matrix_desired[:3, :3]
            ori_desired = R.from_matrix(r_desired).as_quat()
            desired_cartesian_pose = np.concatenate([pos_desired, ori_desired])

            action = np.concatenate([desired_cartesian_pose, action[6:]])

        obs, reward, done, info = self._env.step(action)
        franka_state = self.get_state()
        obs["features"] = np.concatenate(
            (franka_state.pos, franka_state.quat, [franka_state.gripper])
        )
        observation = {}
        for key in self.pixel_keys:
            observation[key] = obs[key]
            if self._use_gt_depth:
                depth_key = f"depth_{key}"
                observation[depth_key] = obs[depth_key]
        observation["proprioceptive"] = obs["features"]
        observation["features"] = obs["features"]
        observation["goal_achieved"] = done
        return observation, reward, done, info

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def render(self, mode="rgb_array", width=256, height=256):
        return cv2.resize(self._env.render("rgb_array"), (width, height))

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
    def __init__(self, env, num_frames):
        self._env = env
        self._num_frames = num_frames

        self.pixel_keys = [
            keys for keys in env.observation_spec().keys() if "pixels" in keys
        ]
        wrapped_obs_spec = env.observation_spec()[self.pixel_keys[0]]

        # frames lists
        self._frames = {}
        for key in self.pixel_keys:
            self._frames[key] = deque([], maxlen=num_frames)

        pixels_shape = wrapped_obs_spec.shape
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = {}
        self._obs_spec["features"] = self._env.observation_spec()["features"]
        self._obs_spec["proprioceptive"] = self._env.observation_spec()[
            "proprioceptive"
        ]
        for key in self.pixel_keys:
            self._obs_spec[key] = specs.BoundedArray(
                shape=np.concatenate(
                    [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
                ),
                dtype=np.uint8,
                minimum=0,
                maximum=255,
                name=key,
            )

    def _transform_observation(self, time_step):
        for key in self.pixel_keys:
            assert len(self._frames[key]) == self._num_frames
        obs = {}
        obs["features"] = time_step.observation["features"]
        for key in self.pixel_keys:
            obs[key] = np.concatenate(list(self._frames[key]), axis=0)
        obs["proprioceptive"] = time_step.observation["proprioceptive"]
        obs["goal_achieved"] = time_step.observation["goal_achieved"]
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = {}
        for key in self.pixel_keys:
            pixels[key] = time_step.observation[key]
            if len(pixels[key].shape) == 4:
                pixels[key] = pixels[key][0]
            pixels[key] = pixels[key].transpose(2, 0, 1)
        return pixels

    def reset(self, **kwargs):
        time_step = self._env.reset(**kwargs)
        pixels = self._extract_pixels(time_step)
        for key in self.pixel_keys:
            for _ in range(self._num_frames):
                self._frames[key].append(pixels[key])
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        for key in self.pixel_keys:
            self._frames[key].append(pixels[key])
        return self._transform_observation(time_step)

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
        action = action.astype(self._env.action_spec().dtype)
        # Make time step for action space
        observation, reward, done, info = self._env.step(action)
        step_type = StepType.LAST if done else StepType.MID

        return TimeStep(
            step_type=step_type,
            reward=reward,
            discount=self._discount,
            observation=observation,
        )

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

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(
    action_repeat,
    seed,
    height,
    width,
    max_episode_len,
    max_state_dim,
    pixel_keys,
    eval,  # True means use_robot=True
    action_type,
    use_gt_depth,
):
    env = gym.make(
        "Franka-v1",
        height=height,
        width=width,
        use_robot=eval,
        crop_h=crop_h,
        crop_w=crop_w,
        use_gt_depth=use_gt_depth,
    )
    env.seed(seed)

    # apply wrappers
    env = RGBArrayAsObservationWrapper(
        env,
        max_episode_len=max_episode_len,
        max_state_dim=max_state_dim,
        pixel_keys=pixel_keys,
        use_robot=eval,
        action_type=action_type,
        use_gt_depth=use_gt_depth,
    )
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FrameStackWrapper(env, 1)
    env = ExtendedTimeStepWrapper(env)

    return [env], ["franka_env"]
