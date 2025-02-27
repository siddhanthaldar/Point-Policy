import numpy as np

# From pyrealsense2
# Command: rs-enumerate-devices -c
# 640x480
CAMERA_MATRICES = {
    "cam_1": np.array([[604.97, 0, 314.83], [0.0, 604.79, 249.03], [0, 0, 1]]),
    "cam_2": np.array([[609.41, 0, 314.85], [0.0, 609.65, 240.52], [0, 0, 1]]),
}

DISTORTION_COEFFICIENTS = {
    "cam_1": np.zeros((5)),
    "cam_2": np.zeros((5)),
}
