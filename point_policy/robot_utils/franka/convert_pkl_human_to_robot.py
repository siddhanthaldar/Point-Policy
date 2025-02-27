import cv2
import argparse
import numpy as np
import pickle as pkl
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import zoom

from gripper_points import extrapoints, Tshift
from utils import camera2pixelkey, rigid_transform_3D


def resize_depth_image(depth_image, new_size):
    # Calculate zoom factors
    zoom_factors = (
        new_size[0] / depth_image.shape[0],
        new_size[1] / depth_image.shape[1],
    )
    # Use scipy's zoom function with order=1 for bilinear interpolation
    resized_depth = zoom(depth_image, zoom_factors, order=1)
    return resized_depth


# Create the parser
parser = argparse.ArgumentParser(
    description="Convert human key points in pkl file to robot key points"
)

# Add the arguments
parser.add_argument("--data_dir", type=str, help="Path to the data directory")
parser.add_argument("--calib_path", type=str, help="Path to the calibration file")
parser.add_argument("--task_name", type=str, help="List of task names")
parser.add_argument(
    "--use_gt_depth", type=bool, default=False, help="Use ground truth depth"
)

args = parser.parse_args()
DATA_DIR = Path(args.data_dir)
CALIB_PATH = Path(args.calib_path)
task_name = args.task_name
use_gt_depth = args.use_gt_depth

camera_indices = [1, 2]
image_size = (640, 480)
save_image_size = (256, 256)
num_hand_points = 9
index_finger_indices = [3, 4]
thumb_indices = [7, 8]

if use_gt_depth:
    task_name += "_gt_depth"

# orientation of the robot at the 0th step
robot_base_orientation = R.from_rotvec([np.pi, 0, 0]).as_matrix()

DATA_DIR = DATA_DIR / "processed_data_pkl"
SAVE_DIR = DATA_DIR / "expert_demos" / "franka_env"

calibration_data = np.load(CALIB_PATH, allow_pickle=True).item()
DATA = pkl.load(open(DATA_DIR / f"{task_name}.pkl", "rb"))

# make sure SAVE_DIR exists
SAVE_DIR.mkdir(parents=True, exist_ok=True)

observations = DATA["observations"]

# find all pairs of indices with first index as index_finger and second index as thumb
index_finger_thumb_pairs = [
    (idx1, idx2) for idx1 in index_finger_indices for idx2 in thumb_indices
]

observations = []
for obs_idx, observation in enumerate(DATA["observations"]):
    print(f"Processing observation {obs_idx}")

    for cam_idx in camera_indices:
        camera_name = f"cam_{cam_idx}"
        pixel_key = camera2pixelkey[camera_name]

        pixels = observation[pixel_key]
        pixels = [cv2.resize(p, save_image_size) for p in pixels]
        observation[pixel_key] = np.array(pixels)

        if use_gt_depth:
            depth = observation[f"depth_{pixel_key}"]
            depth = [resize_depth_image(d, save_image_size) for d in depth]
            observation[f"depth_{pixel_key}"] = np.array(depth)

        human_tracks_3d = observation[f"human_tracks_3d_{pixel_key}"]

        hand_points = human_tracks_3d[:, :num_hand_points]
        object_points = human_tracks_3d[:, num_hand_points:]

        robot_points, gripper_states = [], []
        human_poses = []
        for idx, hand_point in enumerate(hand_points):
            index_finger_thumb_dists = [
                np.linalg.norm(hand_point[idx1] - hand_point[idx2])
                for idx1, idx2 in index_finger_thumb_pairs
            ]
            index_finger_thumb_mindist = np.min(index_finger_thumb_dists)
            index_finger_thumb_mindist_idx = np.argmin(index_finger_thumb_dists)
            index_finger_idx, thumb_idx = index_finger_thumb_pairs[
                index_finger_thumb_mindist_idx
            ]
            robot_pos = (hand_point[index_finger_idx] + hand_point[thumb_idx]) / 2

            if idx == 0:
                robot_ori = robot_base_orientation
                base_hand_points = hand_point.copy()
            else:
                current_hand_points = hand_point.copy()
                # find the rotation matrix between the base hand points and the current hand points
                rot, pos = rigid_transform_3D(base_hand_points, current_hand_points)

                robot_ori = rot @ robot_base_orientation

            # store human pose
            human_poses.append(
                np.concatenate([robot_pos, R.from_matrix(robot_ori).as_rotvec()])
            )

            # pos and orientation of gripper in robot base frame
            T_g_b = np.eye(4)
            T_g_b[:3, :3] = robot_ori
            T_g_b[:3, 3] = robot_pos

            # shift the point
            T_g_b = T_g_b @ Tshift

            # add extra points
            points3d = [T_g_b[:3, 3]]
            gripper_state = -1  # -1: open, 1: closed
            for idx, Tp in enumerate(extrapoints):
                if index_finger_thumb_mindist < 0.07 and idx in [0, 1]:
                    Tp = Tp.copy()
                    Tp[1, 3] = 0.015 if idx == 0 else -0.015
                    gripper_state = 1
                pt = T_g_b @ Tp
                pt = pt[:3, 3]
                points3d.append(pt)
            points3d = np.array(points3d)

            robot_points.append(points3d)
            gripper_states.append(gripper_state)

        observation[f"robot_tracks_3d_{pixel_key}"] = np.array(robot_points)
        observation[f"object_tracks_3d_{pixel_key}"] = np.array(object_points)
        observation[f"gripper_states"] = np.array(gripper_states)
        observation[f"human_poses"] = np.array(human_poses)

        # get 2d robot tracks from 3d robot tracks
        P = calibration_data[camera_name]["ext"]
        K = calibration_data[camera_name]["int"]
        D = calibration_data[camera_name]["dist_coeff"]
        r, t = P[:3, :3], P[:3, 3]
        r, _ = cv2.Rodrigues(r)
        # robot points
        robot_points_2d = []
        for points3d in robot_points:
            points3d = points3d[:, :3]
            points2d = cv2.projectPoints(points3d, r, t, K, D)[0].squeeze()
            robot_points_2d.append(points2d)
        robot_points_2d = np.array(robot_points_2d)
        observation[f"robot_tracks_{pixel_key}"] = robot_points_2d
        # object points
        object_points_2d = []
        for points3d in object_points:
            points3d = points3d[:, :3]
            points2d = cv2.projectPoints(points3d, r, t, K, D)[0].squeeze()
            object_points_2d.append(points2d)
        object_points_2d = np.array(object_points_2d)
        observation[f"object_tracks_{pixel_key}"] = object_points_2d

    observations.append(observation)

DATA["observations"] = observations

# save data
pkl.dump(DATA, open(SAVE_DIR / f"{task_name}.pkl", "wb"))
