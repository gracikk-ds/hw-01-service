"""Utility functions for the service."""
from typing import Tuple

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

BASE_SCALING_FACTOR: int = 255


def preprocess_image(
    image: NDArray[np.uint8],
    target_image_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Preprocess an image for ImageNet.

    This function takes an RGB image, normalizes it, resizes it to the target
    image size, and transposes it to meet the input requirements for ImageNet
    models.

    Args:
        image (np.ndarray): The input RGB image.
        target_image_size (Tuple[int, int]): The target image size (height, width)

    Returns:
        torch.Tensor: A batch containing a single preprocessed image.

    """
    processed_image = image.astype(np.float32)
    processed_image /= BASE_SCALING_FACTOR
    processed_image = cv2.resize(processed_image, target_image_size)
    processed_image = np.transpose(processed_image, (2, 0, 1))
    processed_image -= np.array([0.485, 0.456, 0.406])[:, None, None]
    processed_image /= np.array([0.229, 0.224, 0.225])[:, None, None]
    return torch.from_numpy(processed_image)[None]
