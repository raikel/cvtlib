import cv2
from typing import Tuple
import numpy as np


class Drawer:

    TEXT_TOP = 0
    TEXT_BOTTOM = 0

    def __init__(
        self,
        box_color: Tuple[int, int, int] = (255, 0, 0),
        box_linewidth: int = 2,
        font_type: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_color: Tuple[int, int, int] = (255, 255, 255),
        font_scale: int = 2,
        font_linewidth: int = 2,
        text_margins: tuple = (2, 2, 2, 2),
        text_line_sep: float = 1.3,
        text_loc: int = 0,
        text_background: tuple = (255, 0, 0)
    ):

        self.box_color: tuple = box_color
        self.box_linewidth: int = box_linewidth
        self.font_type: int = font_type
        self.font_color: tuple = font_color
        self.font_scale: int = font_scale
        self.font_linewidth: int = font_linewidth
        self.text_margins = text_margins
        self.text_line_sep = text_line_sep
        self.text_loc = text_loc
        self.text_background = text_background

    def draw_labeled_box(
        self,
        frame: np.ndarray,
        label: str,
        rect: Tuple[int, int, int, int]
    ):

        lines = [line.strip() for line in label.split('\n')]
        offsets = []
        current_offset = 0
        text_width = 0

        for line in lines:
            line_size, _ = cv2.getTextSize(
                text=line,
                fontFace=self.font_type,
                fontScale=self.font_scale,
                thickness=self.font_linewidth
            )
            current_offset += int(self.text_line_sep * line_size[1])
            offsets.append(current_offset)
            text_width = max(text_width, line_size[0])

        text_height = current_offset

        if self.text_loc == self.TEXT_TOP:
            text_corner = (
                rect[0] + self.text_margins[3],
                rect[1] - text_height - self.text_margins[2]
            )
        elif self.text_loc == self.TEXT_BOTTOM:
            text_corner = (
                rect[0] + self.text_margins[3],
                rect[3] + self.text_margins[0]
            )
        else:
            raise ValueError(f'Invalid text location "{self.text_loc}"')

        background_rect = (
            text_corner[0] - self.text_margins[3],
            text_corner[1] - self.text_margins[0],
            text_corner[0] + text_width + self.text_margins[1],
            text_corner[1] + text_height + self.text_margins[2],
        )

        cv2.rectangle(
            frame,
            (rect[0], rect[1]),
            (rect[2], rect[3]),
            self.box_color,
            self.box_linewidth
        )

        if self.text_background is not None:
            cv2.rectangle(
                frame,
                (background_rect[0], background_rect[1]),
                (background_rect[2], background_rect[3]),
                self.text_background,
                -1
            )

        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (text_corner[0], text_corner[1] + offsets[i]),
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
