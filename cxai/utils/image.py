from torchvision.datasets.folder import pil_loader

from PIL import Image


from nptyping import NDArray

import cv2
import numpy as np


def load_image(path: str) -> Image.Image:
    """load_image is simply an alias of torchvision's pil_loader

    Args:
        path (str): path to file

    Returns:
        Image.Image: loaded image
    """
    return pil_loader(path)


def construct_translation_matrix(delta_x: float, delta_y: float) -> NDArray:
    return np.float32([[1, 0, delta_x], [0, 1, delta_x]])


def inverse_translation_matrix(T: NDArray) -> NDArray:
    T = np.array(T)

    T[:, 2] = -T[:, 2]

    return T


def translate(img: NDArray, translation_matrix: NDArray) -> (NDArray, NDArray):
    return (
        cv2.warpAffine(img, translation_matrix, (img.shape[0], img.shape[1])),
        translation_matrix,
    )


def _generate_dummy_x_image(size=(224, 224)) -> NDArray:
    img = np.zeros(size)

    img[20:-20, 100:105] = 1
    img[100:105, 20:-20] = 1

    return img
