"""Where frames come from: a camera, or a video file standing in for one.

Every source hands out BGR frames with a timestamp in seconds on its own
clock. The tracker uses the timestamps, never a nominal frame rate: phones
record at a variable rate, and a camera under load drops frames.

The Pi camera is behind the same interface so that the rest of the pipeline
runs unchanged on a laptop with recorded footage. picamera2 is imported only
when a Pi camera is actually opened.
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    t: float            # seconds on the source's clock
    index: int          # position in the source's own sequence, gaps included


class VideoFileSource:
    """Frames from a video file.

    realtime=True releases each frame when the file's timestamp says it would
    have arrived, measured from the first read, the way a camera does; a slow
    consumer then misses frames instead of slowing the source down. False
    hands them out as fast as they decode, for tests and offline work.
    """

    def __init__(self, path, realtime=False, loop=False):
        self.path = path
        self.realtime = realtime
        self.loop = loop
        self._cap = None
        self._start = None
        self._offset = 0.0
        self._index = 0
        probe = cv2.VideoCapture(path)
        if not probe.isOpened():
            raise OSError(f"cannot open {path}")
        self.size = (int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)), int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
        probe.release()

    def start(self):
        self._cap = cv2.VideoCapture(self.path)
        self._start = None
        self._offset = 0.0
        self._index = 0

    def read(self):
        """The next frame, or None at the end of the file."""
        ok, image = self._cap.read()
        if not ok:
            if not self.loop:
                return None
            self._offset += self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 + 1.0 / self.fps
            self._cap.release()
            self._cap = cv2.VideoCapture(self.path)
            ok, image = self._cap.read()
            if not ok:
                return None
        t = self._offset + self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if self.realtime:
            now = time.monotonic()
            if self._start is None:
                self._start = now - t
            wait = self._start + t - now
            if wait > 0:
                time.sleep(wait)
        frame = Frame(image, t, self._index)
        self._index += 1
        return frame

    def stop(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class PiCameraSource:
    """Raspberry Pi camera through picamera2.

    The exposure is set by hand and short. A phone at 30 fps exposes for up
    to a thirtieth of a second, and the rim of a plate moving at 2 m/s smears
    over several centimetres; that blur, not the detector, is what limits
    precision on the phone footage. At 80 fps and 2 ms the smear is 4 mm.
    The price is noise, which the gain has to make up and which the rim fit
    averages out far better than it averages out blur.
    """

    def __init__(self, size=(1536, 864), fps=80.0, exposure_us=2000, gain=8.0, buffers=6):
        from picamera2 import Picamera2  # only on a Pi

        self.size = tuple(size)
        self.fps = float(fps)
        self._cam = Picamera2()
        frame_us = int(1e6 / self.fps)
        config = self._cam.create_video_configuration(
            main={"size": self.size, "format": "RGB888"},   # BGR in memory, as OpenCV wants
            buffer_count=buffers,
            controls={"FrameDurationLimits": (frame_us, frame_us),
                      "ExposureTime": int(exposure_us), "AnalogueGain": float(gain),
                      "AeEnable": False, "AwbEnable": True},
        )
        self._cam.configure(config)
        self._t0 = None
        self._last_ns = None
        self._index = 0
        self._period_ns = 1e9 / self.fps

    def start(self):
        self._cam.start()

    def read(self):
        request = self._cam.capture_request()
        try:
            image = request.make_array("main")
            ns = request.get_metadata().get("SensorTimestamp")
        finally:
            request.release()
        if ns is None:
            ns = time.monotonic_ns()
        if self._t0 is None:
            self._t0 = ns
        if self._last_ns is not None:
            # a gap in the sensor clock is a dropped frame: count it
            self._index += max(1, int(round((ns - self._last_ns) / self._period_ns)))
        self._last_ns = ns
        return Frame(image, (ns - self._t0) / 1e9, self._index)

    def stop(self):
        self._cam.stop()
        self._cam.close()
