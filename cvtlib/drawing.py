import cv2
from typing import Tuple
import numpy as np


class Drawer:
    def __init__(
        self,
        box_color: Tuple[int, int, int] = (255, 0, 0),
        box_linewidth: int = 2,
        font_type: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_color: Tuple[int, int, int] = (255, 255, 255),
        font_scale: int = 2,
        font_linewidth: int = 2,
    ):

        self.box_color: tuple = box_color
        self.box_linewidth: int = box_linewidth
        self.font_type: int = font_type
        self.font_color: tuple = font_color
        self.font_scale: int = font_scale
        self.font_linewidth: int = font_linewidth

    def draw_labeled_box(
        self,
        frame: np.ndarray,
        label: str,
        rect: Tuple[int, int, int, int],
        offset: Tuple[int, int] = (0, -10)
    ):
        cv2.rectangle(
            frame,
            (rect[0], rect[1]),
            (rect[2], rect[3]),
            self.box_color,
            self.box_linewidth
        )
        cv2.putText(
            frame,
            label,
            (rect[0] + offset[0], rect[1] + offset[1]),
            self.font_type,
            self.font_scale,
            self.font_color,
            self.font_linewidth,
            cv2.LINE_AA
        )
        return frame

    def draw_text(
        self,
        frame: np.ndarray,
        text: str,
        pos: Tuple[int, int]
    ):
        cv2.putText(
            frame,
            text,
            pos,
            self.font_type,
            self.font_scale,
            self.font_color,
            self.font_linewidth,
            cv2.LINE_AA
        )
        return frame
