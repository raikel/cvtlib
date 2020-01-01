import cv2
import numpy as np
from time import time
from typing import List


class VideoCapture(object):
    def __init__(self, src):
        self.source = src
        # noinspection PyTypeChecker
        self.capture: cv2.VideoCapture = None
        # noinspection PyTypeChecker
        self.frame: np.ndarray = None
        self.frame_number: int = 0
        self.first_frame_time = 0
        self.measure_framerate = False
        self.capture_rate = 0

    def open(self):
        if self.capture is None:
            self.frame = None
            self.capture = cv2.VideoCapture(self.source)
            if not self.capture.isOpened():
                raise self.InvalidSource(f'Unable to open video source {self.source}.')

    def next_frame(self) -> (np.ndarray, bool):

        ret, self.frame = self.capture.read()

        if ret:
            self.frame_number = self.frame_number + 1
            if self.measure_framerate:
                curr_frame_time = time()
                if self.frame_number == 1:
                    self.first_frame_time = curr_frame_time
                else:
                    self.capture_rate = self.frame_number / (
                        curr_frame_time - self.first_frame_time
                    )
        else:
            self.frame = None

        return self.frame, ret

    def grab_next(self) -> bool:
        ret = self.capture.grab()
        if ret:
            self.frame_number += 1
        return ret

    def is_opened(self) -> bool:
        return self.capture.isOpened()

    def goto_frame(self, frame_number) -> None:
        try:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.frame_number = frame_number
        except:
            raise AttributeError('Goto frame operation not supported.')

    def goto_time(self, timestamp) -> None:
        try:
            self.capture.set(cv2.CAP_PROP_POS_MSEC, 1000 * timestamp)
            self.frame_number = int(timestamp * self.frame_rate)
        except:
            raise AttributeError('Goto time operation not supported.')

    def move(self, seconds: float = None, frames: int = None):
        if seconds is not None:
            self.goto_time(self.timestamp + seconds)
        elif frames is not None:
            self.goto_frame(self.frame_number + frames)

    @property
    def timestamp(self):
        try:
            return self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        except:
            raise AttributeError('Timestamp property not supported.')

    @property
    def duration_seconds(self):
        try:
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            return frame_count / fps
        except:
            raise AttributeError('Duration property not supported.')

    @property
    def frame_rate(self):
        try:
            return self.capture.get(cv2.CAP_PROP_FPS)
        except:
            raise AttributeError('Frame rate property not supported.')

    @property
    def size(self):
        try:
            return (
                int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
        except:
            raise AttributeError('Size rate property not supported.')

    def create_thumbs(self, count=4, size=128) -> List[np.ndarray]:

        frames = []
        video_length = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        if video_length > count:
            frames_ind = np.array(video_length * np.arange(0.1, 0.9, 0.9 / (count - 1)), np.int32)

            for ind in frames_ind:
                self.goto_frame(ind)
                frame, ret = self.next_frame()
                if frame is not None:
                    scale = size / min(frame.shape[0:2])
                    if scale < 1.0:
                        h, w = frame.shape[0:2]
                        frame = cv2.resize(frame, (int(scale * w), int(scale * h)))

                    frames.append(frame)

        return frames

    def release(self):
        self.capture.release()

    class InvalidSource(ValueError):
        pass


class VideoWriter(object):
    def __init__(self, file_path: str, frame_size: tuple, fps: float = 30.0, codec: str = ''):
        if codec == '':
            codec = 'DIB '
        self.writer = cv2.VideoWriter(
            file_path,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            frame_size
        )

    def put_frame(self, frame):
        self.writer.write(frame)

    def is_opened(self):
        return self.writer.isOpened()