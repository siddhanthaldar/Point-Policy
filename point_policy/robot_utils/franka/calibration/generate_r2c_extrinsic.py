"""
A script which given camera intrinsics computes te robot to camera transformation
for each camera and uses that as extrinsics to save in a calib.pkl file
"""

import cv2
from cv2 import aruco
import numpy as np
import pickle as pkl
from pathlib import Path
from scipy.spatial.transform import Rotation as R

PATH_DATA_PKL = Path("/path/to/data/processed_data_pkl/calib.pkl")
PATH_INTRINSICS = None
SAVE_DIR = Path("../../../calib")
PATH_SAVE_CALIB = SAVE_DIR / "calib.npy"
CAM_IDS = [1, 2]
R2C_TRAJ_IDX = 0
FRAME_FREQ = 1  # consider every Nth frame


with open(PATH_DATA_PKL, "rb") as f:
    observations = pkl.load(f)["observations"]

if PATH_INTRINSICS is not None and PATH_INTRINSICS.exists():
    print("Using intrinsics from file")
    with open(PATH_INTRINSICS, "rb") as f:
        intrinsics = pkl.load(f)
else:
    print("Using intrinsics from constants")
    from constants import CAMERA_MATRICES, DISTORTION_COEFFICIENTS

    intrinsics = {
        "camera_matrices": {},
        "distortion_coefficients": {},
    }
    for cam_id in CAM_IDS:
        intrinsics["camera_matrices"][f"cam_{cam_id}"] = CAMERA_MATRICES[
            f"cam_{cam_id}"
        ]
        intrinsics["distortion_coefficients"][
            f"cam_{cam_id}"
        ] = DISTORTION_COEFFICIENTS[f"cam_{cam_id}"]

SAVE_DIR.mkdir(exist_ok=True, parents=True)

################################# compute the robot to camera transformation #################################

T_ci_b = {}

for cam_id in CAM_IDS:
    pixels = observations[R2C_TRAJ_IDX][f"pixels{cam_id}"][..., ::-1]

    # object point transformations
    object_pos = observations[R2C_TRAJ_IDX]["cartesian_states"][:, :3]
    object_aa = observations[R2C_TRAJ_IDX]["cartesian_states"][:, 3:]
    object_rot_mat = R.from_rotvec(object_aa).as_matrix()
    object_trans = np.zeros(
        (object_pos.shape[0], 4, 4)
    )  # pose of gripper in robot base frame
    object_trans[:, :3, :3] = object_rot_mat
    object_trans[:, :3, 3] = object_pos

    # compute object points
    T_a_g = np.array([[1, 0, 0, 0.0025], [0, 1, 0, 0], [0, 0, 1, 0.0625], [0, 0, 0, 1]])

    # Franka axis is rotated by 45 degrees
    # See pg 52 of https://download.franka.de/documents/Product%20Manual%20Franka%20Research%203_R02210_1.0_EN.pdf
    angle = -np.pi / 4
    R_z = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    T_a_g[:3, :3] = R_z @ T_a_g[:3, :3]

    object_pts = [T @ T_a_g for T in object_trans]
    object_pts = np.array(object_pts)
    object_points = object_pts[:, :3, 3]

    # Aruco marker detection with Cv2 on pixels
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    image_points = []
    invalid_indices = []
    idx = 0
    for i in range(0, len(pixels), FRAME_FREQ):
        corners, ids, rejectedImgPoints = detector.detectMarkers(pixels[i])
        if corners:
            center_img = corners[0].mean(axis=1).flatten()
            image_points.append(center_img)
        else:
            invalid_indices.append(idx)
            print("error")
        idx += 1

    # remove invalid indices from subsampled object points
    object_points = object_points[::FRAME_FREQ]
    object_points = [
        object_points[i] for i in range(len(object_points)) if i not in invalid_indices
    ]

    # convert to numpy float arrays
    object_points = np.array(object_points).astype(np.float32)
    image_points = np.array(image_points).astype(np.float32)

    # get T_ci_b
    camera_matrix = intrinsics["camera_matrices"][f"cam_{cam_id}"]
    dist_coeffs = intrinsics["distortion_coefficients"][f"cam_{cam_id}"]
    ret, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs
    )
    rot = cv2.Rodrigues(rvec)[0]
    T_ci_b[f"cam_{cam_id}"] = np.eye(4)
    T_ci_b[f"cam_{cam_id}"][:3, :3] = rot
    T_ci_b[f"cam_{cam_id}"][
        :3, 3
    ] = tvec.flatten()  # these are extrinsics (world in camera frame)

# save intrinsics and extrinsics in a dictionary
calibration_dict = {}
for cam_id in CAM_IDS:
    calibration_dict[f"cam_{cam_id}"] = {
        "int": intrinsics["camera_matrices"][f"cam_{cam_id}"],
        "dist_coeff": intrinsics["distortion_coefficients"][f"cam_{cam_id}"],
        "ext": T_ci_b[f"cam_{cam_id}"],
    }
np.save(PATH_SAVE_CALIB, calibration_dict)
