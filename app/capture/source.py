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

    pause_s, when looping, holds the last frame that long, and then the first
    frame of the next round, so that a set played over and over is watched as
    it would be at the gym: the plate lies still after the set, which ends it,
    and waits to be lifted before the next.
    """

    def __init__(self, path, realtime=False, loop=False, pause_s=0.0):
        self.path = path
        self.realtime = realtime
        self.loop = loop
        self.pause_s = pause_s
        self._held = None
        self._lead = None
        self._last_image = None
        self._cap = None
        self._start = None
        self._offset = 0.0
        self._last_t = 0.0
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
        if self._lead is not None:
            image, first_t = self._lead
            t = self._last_t + 1.0 / self.fps
            if t < first_t - 0.5 / self.fps:
                return self._emit(image.copy(), t)
            self._lead = None
            self._last_image = image
            return self._emit(image, first_t)
        ok, image = self._cap.read()
        if not ok:
            if not self.loop or self._last_image is None:
                return None
            if self._held is None:
                self._held = self._last_t
            if self._last_t - self._held < self.pause_s:
                return self._emit(self._last_image.copy(), self._last_t + 1.0 / self.fps)
            self._held = None
            self._cap.release()
            self._cap = cv2.VideoCapture(self.path)
            ok, image = self._cap.read()
            if not ok:
                return None
            # after the last frame the position reads as zero, so the next
            # round starts one frame after the last time handed out
            self._offset = self._last_t + (1 + round(self.pause_s * self.fps)) / self.fps
            if self.pause_s > 0:
                self._lead = (image, self._offset + self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                return self._emit(image.copy(), self._last_t + 1.0 / self.fps)
        self._last_image = image
        return self._emit(image, self._offset + self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

    def _emit(self, image, t):
        self._last_t = t
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
    averages out far better than it averages out blur. The gain is left to
    whoever runs the camera: app.station adjusts it to the light between sets.

    A Recorder, if given, records every frame to disk with the hardware
    encoder while the camera runs.
    """

    MAX_GAIN = 16.0     # the Camera Module 3's analogue gain range ends here

    def __init__(self, size=(1536, 864), fps=80.0, exposure_us=2000, gain=8.0, buffers=6, recorder=None):
        # With a recorder, 60 fps is the most the Pi 4B keeps up with: at 80 the
        # encoder and the copy into the ring buffer compete for memory and
        # frames are lost. app.station runs it at 60.
        from picamera2 import Picamera2  # only on a Pi

        self.size = tuple(size)
        self.fps = float(fps)
        self.recorder = recorder
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
        self.metadata = {}              # of a recent frame: exposure, gain, focus and so on

    def start(self):
        if self.recorder is not None:
            self.recorder.start(self._cam, self.fps)
        self._cam.start()

    def read(self):
        request = self._cam.capture_request()
        try:
            image = request.make_array("main")
            meta = request.get_metadata()
        finally:
            request.release()
        t = self._clock(meta)
        return Frame(image, t, self._index)

    def read_into(self, push):
        """Hand the next frame to push(image, t) straight from the camera's buffer.

        read() copies the frame out of the camera's buffer and the ring buffer
        copies it again. With the encoder reading every frame as well, the Pi 4's
        memory could not keep up with both copies at 80 fps, so this one is
        spared. push must copy the image: it is gone once push returns.
        """
        from picamera2 import MappedArray

        request = self._cam.capture_request()
        try:
            t = self._clock(request.get_metadata())
            with MappedArray(request, "main") as mapped:
                push(mapped.array[:, : self.size[0]], t)
        finally:
            request.release()
        return t

    def _clock(self, meta):
        """This frame's time in seconds from the first, counting any dropped frames."""
        ns = meta.get("SensorTimestamp")
        if ns is None:
            ns = time.monotonic_ns()
        if self._t0 is None:
            self._t0 = ns
        if self._last_ns is not None:
            # a gap in the sensor clock is a dropped frame: count it
            self._index += max(1, int(round((ns - self._last_ns) / self._period_ns)))
        self._last_ns = ns
        if self._index % 16 == 0:
            self.metadata = meta
        return (ns - self._t0) / 1e9

    def sensor_us(self, t):
        """A time on this source's clock as sensor time in microseconds, as the recorder keeps it."""
        return None if self._t0 is None else int(round(self._t0 / 1000 + t * 1e6))

    def set_controls(self, **controls):
        """Change camera controls while it runs, e.g. ExposureTime, AnalogueGain, LensPosition."""
        self._cam.set_controls(controls)

    def autofocus(self):
        """Focus once on what is in view, then hold that focus."""
        from libcamera import controls

        self._cam.set_controls({"AfMode": controls.AfModeEnum.Auto, "AfTrigger": controls.AfTriggerEnum.Start})

    def stop(self):
        if self.recorder is not None:
            self.recorder.stop()
        self._cam.stop()
        self._cam.close()
