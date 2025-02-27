import os
import re
import shutil
import argparse
import subprocess
import cv2
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# Create the parser
parser = argparse.ArgumentParser(
    description="Script for processing human demonstrations"
)

# Add the arguments
parser.add_argument("--data_dir", type=str, help="Path to the data directory")
parser.add_argument("--task_names", nargs="+", type=str, help="List of task names")
parser.add_argument(
    "--process_depth", type=bool, default=True, help="Whether to process depth data"
)
parser.add_argument(
    "--num_demos", type=int, default=None, help="Number of demonstrations to process"
)

args = parser.parse_args()
DATA_DIR = Path(args.data_dir)
task_names = args.task_names
process_depth = args.process_depth
num_demos = args.num_demos

DATA_PATH = Path(DATA_DIR) / "extracted_data"
SAVE_PATH = Path(DATA_DIR) / "processed_data"
cam_indices = {
    1: "rgb",
    2: "rgb",
}
states_file_name = "states"

if task_names is None:
    task_names = [f.name for f in DATA_PATH.iterdir() if f.is_dir()]

# Create the save path
SAVE_PATH.mkdir(parents=True, exist_ok=True)
done_flag = True
skip_cam_processing = False
process_type = "cont"

for TASK_NAME in task_names:
    print(f"#################### Processing task {TASK_NAME} ####################")

    # Check if previous demos from this task exist
    if Path(f"{SAVE_PATH}/{TASK_NAME}").exists():
        num_prev_demos = len(
            [f for f in (SAVE_PATH / TASK_NAME).iterdir() if f.is_dir()]
        )
        if num_prev_demos > 0:
            cont_check = input(
                f"Previous demonstrations from task {TASK_NAME} exist. Continue from existing demos? y/n."
            )
            if cont_check == "n":
                ow_check = input(
                    f"Overwrite existing demonstrations from task {TASK_NAME}? y/n."
                )
                if ow_check == "y":
                    num_prev_demos = 0
                else:
                    print("Appending new demonstrations to the existing ones.")
                    process_type = "append"
            elif cont_check == "y":
                num_prev_demos -= 1  # overwrite the last demo

    else:
        num_prev_demos = 0
        (SAVE_PATH / TASK_NAME).mkdir(parents=True, exist_ok=True)

    # demo directories
    DEMO_DIRS = [
        f
        for f in (DATA_PATH / TASK_NAME).iterdir()
        if f.is_dir() and "fail" not in f.name and "ignore" not in f.name
    ]
    if num_demos is not None:
        DEMO_DIRS = DEMO_DIRS[:num_demos]

    for num, demo_dir in enumerate(sorted(DEMO_DIRS)):
        if process_type == "cont" and num < num_prev_demos:
            print(f"Skipping demonstration {demo_dir.name}")
            continue
        if process_type == "append":
            demo_id = num + num_prev_demos
        elif process_type == "cont":
            demo_id = int(demo_dir.name.split("_")[-1])
        print("Processing demonstration", demo_dir.name)
        output_path = f"{SAVE_PATH}/{TASK_NAME}/demonstration_{demo_id}/"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        csv_list = [f for f in os.listdir(output_path) if f.endswith(".csv")]
        for f in csv_list:
            os.remove(os.path.join(output_path, f))
        cam_avis = [
            f"{demo_dir}/cam_{i}_{cam_indices[i]}_video.avi" for i in cam_indices
        ]
        if process_depth:
            DEPTH_FRAMES = {}
            for i in cam_indices:
                if i == 51:
                    continue
                depth_pkl = f"{demo_dir}/cam_{i}_depth.pkl"
                with open(depth_pkl, "rb") as f:
                    depth_frames = pkl.load(f)
                    DEPTH_FRAMES[i] = np.array(depth_frames)

        states_path = Path(f"{demo_dir}/states.pkl")
        if not states_path.exists():
            continue
        states = pkl.load(open(f"{demo_dir}/states.pkl", "rb"))

        state_positions = [state.pos for state in states]

        state_orientations = [R.from_quat(state.quat).as_rotvec() for state in states]
        gripper_positions = [state.gripper for state in states]
        state_timestamps = [state.timestamp for state in states]
        state_start_teleop = [state.start_teleop for state in states]

        state_positions = np.array(state_positions)
        state_orientations = np.array(state_orientations)
        state_positions = np.concatenate([state_positions, state_orientations], axis=1)
        gripper_positions = np.array(gripper_positions)
        gripper_positions = gripper_positions.reshape(-1, 1)

        state_timestamps = np.array(state_timestamps)

        # Find indices of timestamps where the robot moves
        static_timestamps = []
        static = False
        start, end = None, None
        for i in range(1, len(state_positions) - 1):
            if state_start_teleop[i] == False and static == False:
                static = True
                start = i
            elif state_start_teleop[i] == True and static == True:
                static = False
                end = i
                static_timestamps.append((start, end))
        if static:
            static_timestamps.append((start, len(state_positions) - 1))

        # read metadata file
        CAM_TIMESTAMPS = []
        CAM_VALID_LENS = []
        skip = False
        for idx in cam_indices:
            cam_meta_file_path = (
                f"{demo_dir}/cam_{idx}_{cam_indices[idx]}_video.metadata"
            )
            with open(cam_meta_file_path, "rb") as f:
                image_metadata = pkl.load(f)
                image_timestamps = np.asarray(image_metadata["timestamps"]) / 1000.0

                cam_timestamps = dict(timestamps=image_timestamps)
            # convert to numpy array
            cam_timestamps = np.array(cam_timestamps["timestamps"])

            # # Fish eye cam timestamps are divided by 1000
            if max(cam_timestamps) < state_timestamps[static_timestamps[0][1]]:
                cam_timestamps *= 1000
            elif min(cam_timestamps) > state_timestamps[static_timestamps[-1][0]]:
                cam_timestamps /= 1000

            valid_indices = []
            if not process_depth or idx == 51:
                for k in range(len(static_timestamps) - 1):
                    start_idx = sum(
                        cam_timestamps < state_timestamps[static_timestamps[k][1]]
                    )
                    end_idx = sum(
                        cam_timestamps < state_timestamps[static_timestamps[k + 1][0]]
                    )
                    print(start_idx, end_idx)
                    valid_indices.extend([i for i in range(start_idx, end_idx)])
                cam_timestamps = cam_timestamps[valid_indices]
            else:
                depth_meta_file_path = f"{demo_dir}/cam_{idx}_depth.metadata"
                with open(cam_meta_file_path, "rb") as f:
                    depth_metadata = pkl.load(f)
                    depth_timestamps = np.asarray(depth_metadata["timestamps"]) / 1000.0

                for k in range(len(static_timestamps) - 1):
                    start_idx = sum(
                        depth_timestamps < state_timestamps[static_timestamps[k][1]]
                    )
                    end_idx = sum(
                        depth_timestamps < state_timestamps[static_timestamps[k + 1][0]]
                    )
                    valid_indices.extend([i for i in range(start_idx, end_idx)])
                depth_timestamps = depth_timestamps[valid_indices]
                DEPTH_FRAMES[idx] = DEPTH_FRAMES[idx][valid_indices] / 1000.0

                valid_indices = []
                for j, k in enumerate(cam_timestamps):
                    if k in depth_timestamps:
                        valid_indices.append(j)
                cam_timestamps = cam_timestamps[valid_indices]
                print(len(cam_timestamps), len(depth_timestamps))

            # if no valid timestamps, skip
            if len(cam_timestamps) == 0:
                skip = True
                break

            CAM_VALID_LENS.append(valid_indices)
            CAM_TIMESTAMPS.append(cam_timestamps)

        if skip:
            continue
        # cam frames
        if not skip_cam_processing:
            CAM_FRAMES = []
            for idx in range(len(cam_avis)):
                cam_avi = cam_avis[idx]
                cam_frames = []
                cap_cap = cv2.VideoCapture(cam_avi)
                while cap_cap.isOpened():
                    ret, frame = cap_cap.read()
                    if ret == False:
                        break
                    cam_frames.append(frame)
                cap_cap.release()

                # save frames
                cam_frames = np.array(cam_frames)
                cam_frames = cam_frames[CAM_VALID_LENS[idx]]
                CAM_FRAMES.append(cam_frames)

            rgb_frames = CAM_FRAMES
        timestamps = CAM_TIMESTAMPS
        timestamps.append(state_timestamps)
        min_time_index = np.argmin([len(timestamp) for timestamp in timestamps])
        reference_timestamps = timestamps[min_time_index]
        align = []
        index = []
        for i in range(len(timestamps)):
            # aligning frames
            if i == min_time_index:
                align.append(timestamps[i])
                index.append(np.arange(len(timestamps[i])))
                continue
            curindex = []
            currrlist = []
            for j in range(len(reference_timestamps)):
                curlist = []
                for k in range(len(timestamps[i])):
                    curlist.append(abs(timestamps[i][k] - reference_timestamps[j]))
                min_index = curlist.index(min(curlist))
                currrlist.append(timestamps[i][min_index])
                curindex.append(min_index)
            align.append(currrlist)
            index.append(curindex)

        index = np.array(index)

        # convert left_state_timestamps and left_state_positions to a csv file with header "created timestamp", "pose_aa", "gripper_state"
        state_timestamps_test = pd.DataFrame(state_timestamps)
        # convert each pose_aa to a list
        state_positions_test = state_positions
        for i in range(len(state_positions_test)):
            state_positions_test[i] = np.array(state_positions_test[i])
        state_positions_test = pd.DataFrame(
            {"column": [list(row) for row in state_positions_test]}
        )
        # convert left_gripper to True and False
        gripper_positions_test = pd.DataFrame(gripper_positions)

        state_test = pd.concat(
            [state_timestamps_test, state_positions_test, gripper_positions_test],
            axis=1,
        )
        with open(output_path + f"big_{states_file_name}.csv", "a") as f:
            state_test.to_csv(
                f,
                header=["created timestamp", "pose_aa", "gripper_state"],
                index=False,
            )

        df = pd.read_csv(output_path + f"big_{states_file_name}.csv")
        for i in range(len(reference_timestamps)):
            curlist = []
            for j in range(len(state_timestamps)):
                curlist.append(abs(state_timestamps[j] - reference_timestamps[i]))
            min_index = curlist.index(min(curlist))
            min_df = df.iloc[min_index]
            min_df = min_df.to_frame().transpose()
            with open(output_path + f"{states_file_name}.csv", "a") as f:
                min_df.to_csv(f, header=f.tell() == 0, index=False)

        # Create folders for each camera if they don't exist
        output_folder = output_path + "videos"
        os.makedirs(output_folder, exist_ok=True)
        camera_folders = [f"camera{i}" for i in cam_indices]
        for folder in camera_folders:
            os.makedirs(os.path.join(output_folder, folder), exist_ok=True)
        if process_depth:
            depth_output_folder = output_path + "depth"
            os.makedirs(depth_output_folder, exist_ok=True)
            depth_folders = {}
            for i in cam_indices:
                depth_folders[i] = f"depth{i}"
            for folder in depth_folders:
                os.makedirs(
                    os.path.join(depth_output_folder, depth_folders[folder]),
                    exist_ok=True,
                )

        # Iterate over each camera and extract the frames based on the indexes
        if not skip_cam_processing:
            for camera_index, frames in enumerate(rgb_frames):
                camera_folder = camera_folders[camera_index]
                print(f"Extracting frames for {camera_folder}...")
                indexes = index[camera_index]

                # Iterate over the indexes and save the corresponding frames
                for i, indexx in enumerate(indexes):
                    if i % 100 == 0:
                        print(f"Extracting frame {i}...")
                    frame = frames[indexx]
                    # name frame with its timestamp
                    image_output_path = os.path.join(
                        output_folder,
                        camera_folder,
                        f"frame_{i}_{timestamps[camera_index][indexx]}.jpg",
                    )
                    cv2.imwrite(image_output_path, frame)

        csv_file = os.path.join(output_path, f"{states_file_name}.csv")

        def get_timestamp_from_filename(filename):
            # Extract the timestamp from the filename using regular expression
            timestamp_match = re.search(r"\d+\.\d+", filename)
            if timestamp_match:
                return float(timestamp_match.group())
            else:
                return None

        # add desired gripper states
        for file in [csv_file]:
            df = pd.read_csv(file)
            df["desired_gripper_state"] = df["gripper_state"].shift(-1)
            df.loc[df.index[-1], "desired_gripper_state"] = df.loc[
                df.index[-2], "gripper_state"
            ]
            df.to_csv(file, index=False)

        def save_only_videos(base_folder_path, depth=False):
            base_folder_path = os.path.join(
                base_folder_path, "videos" if not depth else "depth"
            )
            # Iterate over each camera folder
            for cam in cam_indices:
                if depth and cam == 51:
                    continue
                cam_folder = f"camera{cam}" if not depth else f"depth{cam}"
                full_folder_path = os.path.join(base_folder_path, cam_folder)

                # Check if the folder exists
                if os.path.exists(full_folder_path):
                    # List all jpg files
                    all_files = [
                        f for f in os.listdir(full_folder_path) if f.endswith(".jpg")
                    ]

                    # Sort files based on the floating-point number in their name
                    sorted_files = sorted(all_files, key=get_timestamp_from_filename)

                    # Write filenames to a temp file
                    temp_list_filename = os.path.join(base_folder_path, "temp_list.txt")
                    with open(temp_list_filename, "w") as f:
                        for filename in sorted_files:
                            f.write(
                                f"file '{os.path.join(full_folder_path, filename)}'\n"
                            )

                    # Use ffmpeg to convert sorted images to video
                    output_video_path = os.path.join(
                        base_folder_path,
                        f"camera{cam}.mp4" if not depth else f"depth{cam}.mp4",
                    )
                    cmd = [
                        "ffmpeg",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        temp_list_filename,
                        "-framerate",
                        "30",
                        "-vcodec",
                        "libx264",
                        "-crf",
                        "18",  # quality, lower means better quality
                        "-pix_fmt",
                        "yuv420p",
                        output_video_path,
                    ]
                    try:
                        subprocess.run(cmd, check=True)
                    except Exception as e:
                        print(f"EXCEPTION: {e}")
                        input("Continue?")

                    # Delete the temporary list file and the image folder
                    os.remove(temp_list_filename)
                    shutil.rmtree(full_folder_path)
                else:
                    print(f"Folder {cam_folder} does not exist!")

        if not skip_cam_processing:
            save_only_videos(output_path)

        if not skip_cam_processing and process_depth:
            for idx, cidx in enumerate(cam_indices):
                if cidx == 51:
                    continue

                depth_folder = depth_folders[cidx]
                print(f"Extracting frames for {depth_folder}...")
                indexes = index[idx]

                for i, indexx in enumerate(indexes):
                    if i % 100 == 0:
                        print(f"Extracting depth frame {i}...")
                    frame = DEPTH_FRAMES[cidx][indexx]
                    min_value, max_value = frame.min(), frame.max()
                    frame = (frame - min_value) / (max_value - min_value)
                    colormap = matplotlib.colormaps["magma_r"]
                    frame = colormap(frame, bytes=True)  # ((1)xhxwx4)
                    frame = frame[:, :, :3]  # Discard alpha component
                    # name frame with its timestamp
                    depth_output_path = os.path.join(
                        depth_output_folder,
                        depth_folder,
                        f"frame_{i}_{timestamps[idx][indexx]}.jpg",
                    )
                    cv2.imwrite(depth_output_path, frame)

                pkl.dump(
                    DEPTH_FRAMES[cidx],
                    open(f"{depth_output_folder}/depth{cidx}.pkl", "wb"),
                )

            save_only_videos(output_path, depth=True)
