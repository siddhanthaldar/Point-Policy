import pickle as pkl
import numpy as np
import imageio
from pathlib import Path
import cv2

DATA_DIR = Path("/path/to/expert_demos")
TASK_NAME = "close_oven"
plot_pts = True

DATA_PATH = DATA_DIR / f"{TASK_NAME}.pkl"
SAVE_DIR = Path(f"./videos/{TASK_NAME}")
pixel_keys = ["pixels1", "pixels2"]
original_image_size = (640, 480)
k = 1  # number of track points to plot per frame
traj_indices = None

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Read data
with open(DATA_PATH, "rb") as f:
    data = pkl.load(f)

if traj_indices is None:
    traj_indices = [i for i in range(len(data["observations"]))]

for traj_idx in traj_indices:
    print(f"Processing traj_idx: {traj_idx}")
    for pixel_key in pixel_keys:
        point_track_key = (
            f"robot_tracks_{pixel_key}"
            if "human" not in TASK_NAME
            else f"human_tracks_{pixel_key}"
        )
        object_track_key = f"object_tracks_{pixel_key}"

        # Extract images and point tracks
        frames = data["observations"][traj_idx][pixel_key]
        frames = np.array(frames)

        if plot_pts and pixel_key != "pixels51":
            point_tracks = data["observations"][traj_idx][point_track_key]
            point_tracks = np.array(point_tracks)
            object_tracks = data["observations"][traj_idx][object_track_key]
            object_tracks = np.array(object_tracks)
            point_tracks = np.concatenate([point_tracks, object_tracks], axis=1)

            # Color for each point
            num_points = point_tracks.shape[1]
            colors = np.zeros((num_points, 3))
            third = num_points // 3
            colors[:third, 0] = 255
            colors[third : 2 * third, 1] = 255
            colors[2 * third :, 2] = 255

        save_frames = []
        for i, frame in enumerate(frames):
            frame = frame[..., [2, 1, 0]].copy()
            if plot_pts and pixel_key != "pixels51":
                for j, points in enumerate(point_tracks[max(0, i - k) : i + 1]):
                    # points = points[3:4]
                    for l, point in enumerate(points):
                        point = point.astype(int)
                        point[0] = int(
                            point[0] * frame.shape[1] / original_image_size[0]
                        )
                        point[1] = int(
                            point[1] * frame.shape[0] / original_image_size[1]
                        )
                        frame = cv2.circle(
                            frame, tuple(point), 2, colors[l].tolist(), -1
                        )
            save_frames.append(frame)

        # Save the video
        save_frames = np.array(save_frames).astype(np.uint8)
        save_path = SAVE_DIR / f"{TASK_NAME}_traj{traj_idx}_{pixel_key}.mp4"
        imageio.mimwrite(save_path, save_frames, fps=20)
