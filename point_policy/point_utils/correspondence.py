import sys
import torch
from torchvision import transforms
import numpy as np
import PIL


class Correspondence:
    def __init__(
        self,
        device,
        dift_path,
        width,
        height,
        image_size_multiplier,
        ensemble_size,
        dift_layer,
        dift_steps,
    ):
        """
        Initialize the Correspondence class.

        Parameters:
        -----------
        device : str
            The device to use for computation, either 'cpu' or 'cuda' (for GPU). If you need to put the dift model on a different device to
            save space you can set this to cuda:1

        dift_path : str
            The file path to the directory containing the DIFT model source code.

        width : int
            The width that should be used in the correspondence model.

        height : int
            The height that should be used in the correspondence model.

        image_size_multiplier : int
            The multiplier to use for the image size in the DIFT model if height and weight are -1.

        ensemble_size : int
            The size of the ensemble for the DIFT model.

        dift_layer : int
            The specific layer of the DIFT model to use for feature extraction.

        dift_steps : int
            The number of steps/iterations for the DIFT model to use in feature extraction.

        """
        sys.path.append(dift_path)
        from src.models.dift_sd import SDFeaturizer

        self.dift = SDFeaturizer(device=device)

        self.device = device
        self.width = width
        self.height = height
        self.image_size_multiplier = image_size_multiplier
        self.ensemble_size = ensemble_size
        self.dift_layer = dift_layer
        self.dift_steps = dift_steps

    # Get the feature map from the DIFT model for the expert image to compare with the first frame of each episode later on
    def set_expert_correspondence(self, expert_image, pixel_key, object_label=""):
        with torch.no_grad():
            # Use a null prompt
            self.prompt = ""
            self.original_size = expert_image.size

            if self.width == -1 or self.height == -1:
                self.width = expert_image.size[0] * self.image_size_multiplier
                self.height = expert_image.size[1] * self.image_size_multiplier

            expert_image = expert_image.resize(
                (self.width, self.height), resample=PIL.Image.BILINEAR
            )
            expert_image = (transforms.PILToTensor()(expert_image) / 255.0 - 0.5) * 2
            expert_image = expert_image.cuda(self.device)

            expert_img_features = self.dift.forward(
                expert_image,
                prompt=self.prompt,
                ensemble_size=self.ensemble_size,
                up_ft_index=self.dift_layer,
                t=self.dift_steps,
            )
            expert_img_features = expert_img_features.to(self.device)

        return expert_img_features

    def find_correspondence(
        self, expert_img_features, current_image, coords, pixel_key, object_label
    ):
        """
        Find the corresponding points between the expert image and the current image

        Parameters:
        -----------
        expert_img_features : torch.Tensor
            The feature map from the DIFT model for the expert image.

        current_image : torch.Tensor
            The current image to compare with the expert image.

        coords : list
            The coordinates of the points to find correspondence between the expert image and the current image.
        """

        with torch.no_grad():
            curr_image_shape = (current_image.shape[2], current_image.shape[1])
            current_image = transforms.Resize((self.height, self.width))(current_image)
            current_image_features = self.dift.forward(
                ((current_image - 0.5) * 2),
                prompt=self.prompt,
                ensemble_size=self.ensemble_size,
                up_ft_index=self.dift_layer,
                t=self.dift_steps,
            )

            ft = torch.cat([expert_img_features, current_image_features])
            num_channel = ft.size(1)
            src_ft = ft[0].unsqueeze(0)
            src_ft = torch.nn.Upsample(
                size=(self.height, self.width), mode="bilinear", align_corners=True
            )(src_ft)

            out_coords = torch.zeros(coords.shape)
            for idx, coord in enumerate(coords):
                x, y = int(coord[1] * self.width / self.original_size[0]), int(
                    coord[2] * self.height / self.original_size[1]
                )

                src_vec = src_ft[0, :, y, x].view(1, num_channel).clone()
                trg_ft = torch.nn.Upsample(
                    size=(self.height, self.width), mode="bilinear", align_corners=True
                )(ft[1:])
                trg_vec = trg_ft.view(1, num_channel, -1)  # N, C, HW

                src_vec = torch.nn.functional.normalize(src_vec)  # 1, C
                trg_vec = torch.nn.functional.normalize(trg_vec)  # N, C, HW
                cos_map = (
                    torch.matmul(src_vec, trg_vec)
                    .view(1, self.height, self.width)
                    .cpu()
                    .numpy()
                )

                max_yx = np.unravel_index(cos_map[0].argmax(), cos_map[0].shape)
                out_coords[idx, 1], out_coords[idx, 2] = int(
                    max_yx[1] * curr_image_shape[0] / self.width
                ), int(max_yx[0] * curr_image_shape[1] / self.height)

            return out_coords
