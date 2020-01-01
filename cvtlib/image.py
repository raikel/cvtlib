from typing import Tuple
from base64 import b64encode, b64decode

import numpy as np
import cv2 as cv


def resize(image: np.ndarray, size: int, method: int = cv.INTER_LINEAR):
    h, w = image.shape[:2]
    scale = float(size) / max(w, h)
    resize_to = (int(scale * w), int(scale * h))
    return cv.resize(image, resize_to, interpolation=method), scale


class ImageResizer:
    def __init__(
        self,
        input_size: Tuple[int, int],
        output_size: int = -1,
        scale: Tuple[float, float] = (1.0, 1.0),
        channels: int = 3,
        dtype: int = np.uint8
    ):

        if output_size > 0:
            scale_x = float(output_size) / max(input_size)
            scale = (scale_x, scale_x)

        self.resize_to = (int(scale[0] * input_size[0]), int(scale[1] * input_size[1]))

        if self.resize_to == input_size:
            self.resize = False
        else:
            self.resize = True
            self.frame_resized = np.zeros(
                (self.resize_to[1], self.resize_to[0], channels), dtype
            )

    def resize(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, self.resize_to, self.frame_resized, cv.INTER_LINEAR)


def image_to_base64(image: np.ndarray) -> str:
    return str(b64encode(cv.imencode('.jpg', image)[1]), 'utf-8')


def image_from_base64(image: str) -> np.ndarray:
    return cv.imdecode(b64decode(image), cv.IMREAD_COLOR)
