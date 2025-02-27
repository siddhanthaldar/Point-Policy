import sys
import torch
import cv2


class Depth:
    def __init__(self, depth_path, device):
        """
        Initialize the Depth class for finding depth maps from images.

        Parameters:
        -----------
        depth_path : str
            The file path to the directory containing the Depth Anything source code.

        device : str
            The device to use for computation, either 'cpu' or 'cuda' (for GPU acceleration).
        """
        sys.path.append(depth_path)
        from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

        # Initialize the Depth model
        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
        }
        encoder = "vitl"
        dataset = "hypersim"
        max_depth = 20
        self.depth = DepthAnythingV2(
            **{**model_configs[encoder], "max_depth": max_depth}
        )
        self.depth.load_state_dict(
            torch.load(
                f"{depth_path}/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth",
                map_location="cpu",
                weights_only=False,
            )
        )
        self.depth = self.depth.to(device).eval()

        self.device = device

    def get_depth(self, image):
        """
        Get the depth map from an image.

        Parameters:
        -----------
        image : np.ndarray
            The image to find the depth map for. The image should be in RGB format.

        Returns:
        --------
        depth : np.ndarray
            The depth map for the image.
        """
        bgr_array = image[:, :, ::-1]
        depth = self.depth.infer_image(bgr_array)
        return depth
