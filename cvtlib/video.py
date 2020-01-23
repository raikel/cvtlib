import cv2
import numpy as np
from time import time, sleep
from typing import List
from threading import Thread, Lock


FRAME_RATE_RANGE = [1, 60]
DEFAULT_FRAME_RATE = 25


class GrabThread(Thread):

    def __init__(self, capture: cv2.VideoCapture, interval: float = 0.04):
        super().__init__(daemon=True)
        self._run = False
        self.lock = Lock()
        self.capture = capture
        self.grabbed = False
        self.interval = interval

    def run(self):
        self._run = True
        while self._run:
            sleep(self.interval)
            with self.lock:
                self.grabbed = self.capture.grab()

    def stop(self):
        self._run = False


class VideoCapture(object):
    def __init__(
        self,
        src: str,
        auto_grab: bool = False,
        measure_framerate: bool = False
    ):
        self.source = src
        self.auto_grab: bool = auto_grab
        self.measure_framerate: bool = measure_framerate
        # noinspection PyTypeChecker
        self.capture: cv2.VideoCapture = None
        # noinspection PyTypeChecker
        self.frame: np.ndarray = None
        self.frame_number: int = 0
        self.first_frame_time = 0
        self.capture_rate = 0
        # noinspection PyTypeChecker
        self.grab_thread: GrabThread = None

    def open(self):
        if self.capture is None:
            self.frame = None
            self.capture = cv2.VideoCapture(self.source)

            if not self.capture.isOpened():
                raise ValueError(f'Unable to open video source {self.source}.')

            if self.auto_grab:
                try:
                    fps = self.frame_rate
                except AttributeError:
                    fps = DEFAULT_FRAME_RATE
                if fps < FRAME_RATE_RANGE[0] or fps > FRAME_RATE_RANGE[1]:
                    fps = DEFAULT_FRAME_RATE
                self.grab_thread = GrabThread(self.capture, 1/fps)
                self.grab_thread.start()

    def next_frame(self) -> (np.ndarray, bool):

        ret, self.frame = False, None

        if self.grab_thread is not None:
            with self.grab_thread.lock:
                if self.grab_thread.grabbed:
                    ret, self.frame = self.capture.retrieve()
                else:
                    ret, self.frame = self.capture.read()
        else:
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
            raise AttributeError('Duration in seconds property not supported.')

    @property
    def duration_frames(self):
        try:
            frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            return frame_count
        except:
            raise AttributeError('Duration in frames property not supported.')

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
            pos = np.arange(0.1, 0.9, 0.9 / (count - 1))
            frames_ind = np.array(video_length * pos, np.int32)

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
        if self.grab_thread is not None:
            self.grab_thread.stop()
        self.capture.release()


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