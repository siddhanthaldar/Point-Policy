import cv2
import gym
import numpy as np
import time
import pickle

from frankateach.constants import (
    CAM_PORT,
    GRIPPER_OPEN,
    HOST,
    CONTROL_PORT,
)
from frankateach.messages import FrankaAction, FrankaState
from frankateach.network import (
    ZMQCameraSubscriber,
    create_request_socket,
)


class FrankaEnv(gym.Env):
    def __init__(
        self,
        width=640,
        height=480,
        use_robot=True,
        use_gt_depth=False,
        crop_h=None,
        crop_w=None,
    ):
        super(FrankaEnv, self).__init__()
        self.width = width
        self.height = height
        self.crop_h = crop_h
        self.crop_w = crop_w

        self.channels = 3
        self.feature_dim = 8
        self.action_dim = 7  # (pos, axis angle, gripper)

        self.use_robot = use_robot
        self.use_gt_depth = use_gt_depth

        self.n_channels = 3
        self.reward = 0

        self.franka_state = None
        self.curr_images = None

        self.action_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(self.action_dim,)
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, self.n_channels), dtype=np.uint8
        )

        if self.use_robot:
            self.cam_ids = [1, 2]
            self.image_subscribers = {}
            if self.use_gt_depth:
                self.depth_subscribers = {}
            for cam_idx in self.cam_ids:
                port = CAM_PORT + cam_idx
                self.image_subscribers[cam_idx] = ZMQCameraSubscriber(
                    host=HOST,
                    port=port,
                    topic_type="RGB",
                )

                if self.use_gt_depth:
                    depth_port = CAM_PORT + cam_idx + 1000  # depth offset =1000
                    self.depth_subscribers[cam_idx] = ZMQCameraSubscriber(
                        host=HOST,
                        port=depth_port,
                        topic_type="Depth",
                    )

            self.action_request_socket = create_request_socket(HOST, CONTROL_PORT)

    def get_state(self):
        self.action_request_socket.send(b"get_state")
        franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())
        self.franka_state = franka_state
        return franka_state

    def step(self, abs_action):
        """
        Step the environment with an absolute action.

        Args:
            abs_action (_type_): absolute action in the format (position, orientation in quaternion, gripper)

        Returns:
            obs (dict): observation dictionary containing features and images
            reward (float): reward
            done (bool): whether the episode is done
            info (dict): additional information
        """
        pos = abs_action[:3]
        quat = abs_action[3:7]
        gripper = abs_action[-1]

        # Send action to the robot
        franka_action = FrankaAction(
            pos=pos,
            quat=quat,
            gripper=gripper,
            reset=False,
            timestamp=time.time(),
        )

        self.action_request_socket.send(bytes(pickle.dumps(franka_action, protocol=-1)))
        franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())
        self.franka_state = franka_state

        image_list = {}
        for cam_idx, subscriber in self.image_subscribers.items():
            image, _ = subscriber.recv_rgb_image()

            # crop the image
            if self.crop_h is not None and self.crop_w is not None:
                h, w, _ = image.shape
                image = image[
                    int(h * self.crop_h[0]) : int(h * self.crop_h[1]),
                    int(w * self.crop_w[0]) : int(w * self.crop_w[1]),
                ]

            image_list[cam_idx] = image

        if self.use_gt_depth:
            depth_list = {}
            for cam_idx, subscriber in self.depth_subscribers.items():
                depth, _ = subscriber.recv_depth_image()

                if self.crop_h is not None and self.crop_w is not None:
                    h, w = depth.shape
                    depth = depth[
                        int(h * self.crop_h[0]) : int(h * self.crop_h[1]),
                        int(w * self.crop_w[0]) : int(w * self.crop_w[1]),
                    ]

                depth_list[cam_idx] = depth

        self.curr_images = image_list

        obs = {
            "features": np.concatenate(
                (franka_state.pos, franka_state.quat, [franka_state.gripper])
            ),
        }

        for cam_idx, image in image_list.items():
            obs[f"pixels{cam_idx}"] = cv2.resize(image, (self.width, self.height))
        if self.use_gt_depth:
            for cam_idx, depth in depth_list.items():
                obs[f"depth{cam_idx}"] = cv2.resize(depth, (self.width, self.height))

        return obs, self.reward, False, None

    def reset(self):
        if self.use_robot:
            print("resetting")
            franka_action = FrankaAction(
                pos=np.zeros(3),
                quat=np.zeros(4),
                gripper=GRIPPER_OPEN,
                reset=True,
                timestamp=time.time(),
            )

            self.action_request_socket.send(
                bytes(pickle.dumps(franka_action, protocol=-1))
            )
            franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())
            self.franka_state = franka_state
            print("reset done: ", franka_state)

            image_list = {}
            for cam_idx, subscriber in self.image_subscribers.items():
                image, _ = subscriber.recv_rgb_image()

                # crop the image
                if self.crop_h is not None and self.crop_w is not None:
                    h, w, _ = image.shape
                    image = image[
                        int(h * self.crop_h[0]) : int(h * self.crop_h[1]),
                        int(w * self.crop_w[0]) : int(w * self.crop_w[1]),
                    ]

                image_list[cam_idx] = image

            if self.use_gt_depth:
                depth_list = {}
                for cam_idx, subscriber in self.depth_subscribers.items():
                    depth, _ = subscriber.recv_depth_image()

                    if self.crop_h is not None and self.crop_w is not None:
                        h, w = depth.shape
                        depth = depth[
                            int(h * self.crop_h[0]) : int(h * self.crop_h[1]),
                            int(w * self.crop_w[0]) : int(w * self.crop_w[1]),
                        ]

                    depth_list[cam_idx] = depth

            self.curr_images = image_list

            obs = {
                "features": np.concatenate(
                    (franka_state.pos, franka_state.quat, [franka_state.gripper])
                ),
            }
            for cam_idx, image in image_list.items():
                obs[f"pixels{cam_idx}"] = cv2.resize(image, (self.width, self.height))
            if self.use_gt_depth:
                for cam_idx, depth in depth_list.items():
                    obs[f"depth{cam_idx}"] = cv2.resize(
                        depth, (self.width, self.height)
                    )

            return obs

        else:
            obs = {}
            obs["features"] = np.zeros(self.feature_dim)
            obs["pixels"] = np.zeros((self.height, self.width, self.n_channels))
            if self.use_gt_depth:
                obs["depth"] = np.zeros((self.height, self.width))

            return obs

    def render(self, mode="rgb_array", width=640, height=480):
        assert self.curr_images is not None, "Must call reset() before render()"
        if mode == "rgb_array":
            image_list = []
            for key, im in self.curr_images.items():
                image_list.append(cv2.resize(im, (width, height)))

            return np.concatenate(image_list, axis=1)
        else:
            raise NotImplementedError
