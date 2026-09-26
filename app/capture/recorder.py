"""Continuous recording of the camera, kept for improving detection later.

The sets the watcher misses are the ones worth most for improving it, and
those are only on file if everything is recorded, not only what the watcher
recognised. So the camera is recorded for as long as it runs.

The Pi's hardware H.264 encoder takes the same frames the tracker sees, at
full rate, for almost no CPU: 1536x864 at 80 fps kept up on the Pi 4B. The
stream is cut into segments of a few minutes, and a segment is also closed
right after a set ends, so that the set's clip can be cut from a finished
file within seconds.

Segments are fragmented MP4. The Pi is switched off by pulling the plug, and a
plain MP4 whose index was never written cannot be read at all, where a
fragmented one loses only its last fragment.

Each segment has a JSON file beside it with what the video cannot hold: the
sensor time of its first frame, its keyframes, and how the camera was turned.
Timestamps inside a segment start at zero.
"""

import json
import os
import shutil
import threading
import time
from datetime import datetime

SEGMENT_OPTIONS = {"movflags": "frag_keyframe+empty_moov+default_base_moof"}


def _segment_output_class():
    from picamera2.outputs import PyavOutput  # only on a Pi

    class SegmentOutput(PyavOutput):
        """One segment file, with its timestamps rebased to zero and its keyframes noted."""

        def __init__(self, path, info, origin):
            super().__init__(path, format="mp4", options=SEGMENT_OPTIONS)
            self.path = path
            self.info = info
            self.origin = origin
            self.first_us = None
            self.last_us = None
            self.frames = 0
            self.keyframes_us = []

        def outputframe(self, frame, keyframe=True, timestamp=None, packet=None, audio=False):
            if timestamp is not None:
                if self.first_us is None:
                    self.first_us = timestamp
                self.last_us = timestamp
                if keyframe:
                    self.keyframes_us.append(timestamp - self.first_us)
                self.frames += 1
                timestamp -= self.first_us
            super().outputframe(frame, keyframe, timestamp, packet, audio)

        def stop(self):
            super().stop()
            self.info["frames"] = self.frames
            self.info["sensor_origin_us"] = self.origin()
            if self.first_us is not None:
                self.info["encoder_start_us"] = self.first_us
                self.info["duration_s"] = round((self.last_us - self.first_us) / 1e6, 3)
            self.info["keyframes_us"] = self.keyframes_us
            self.info["complete"] = True
            _write_json(self.path[:-4] + ".json", self.info)

    return SegmentOutput


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)


class Recorder:
    """Records a picamera2 camera into segments in `directory`.

    keyframe_s sets how finely a clip can be cut without re-encoding: a cut
    starts at the keyframe before the wanted time.
    """

    def __init__(self, directory, segment_s=300.0, bitrate=12_000_000, keyframe_s=0.5,
                 min_free_bytes=5_000_000_000, wall_clock=time.time, log=print):
        self.directory = directory
        self.segment_s = segment_s
        self.bitrate = bitrate
        self.keyframe_s = keyframe_s
        self.min_free_bytes = min_free_bytes
        self.wall_clock = wall_clock
        self.log = log
        self.extra = {}                   # written into every segment's JSON, e.g. the rotation
        self._cam = None
        self._encoder = None
        self._splitter = None
        self._current = None
        self._opened_at = 0.0
        self._split_wanted = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self.error = None

    # --- lifecycle ------------------------------------------------------------------

    def start(self, cam, fps):
        from picamera2.encoders import H264Encoder
        from picamera2.outputs import SplittableOutput

        os.makedirs(self.directory, exist_ok=True)
        self._cam = cam
        self._segment_cls = _segment_output_class()
        self._encoder = H264Encoder(bitrate=self.bitrate, iperiod=max(1, int(round(fps * self.keyframe_s))),
                                    framerate=fps)
        self._current = self._new_segment()
        self._splitter = SplittableOutput(self._current)
        cam.start_encoder(self._encoder, self._splitter, name="main")
        self._opened_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._cam is not None and self._encoder is not None:
            try:
                self._cam.stop_encoder(self._encoder)
            except Exception as e:           # the camera may already be gone
                self.log(f"recorder: stopping the encoder failed: {e!r}")
        self._cam = None
        with self._lock:
            self._current = None

    def split(self):
        """Close the current segment at the next keyframe, without waiting for it."""
        self._split_wanted.set()

    # --- where things are -----------------------------------------------------------

    def encoder_origin_us(self):
        """Sensor time, in microseconds, that the encoder's timestamps count from."""
        return None if self._encoder is None else self._encoder.firsttimestamp

    def current(self):
        with self._lock:
            return None if self._current is None else self._current.path

    def segments(self):
        """Every segment on disk, oldest first, with its JSON if it has one."""
        out = []
        if not os.path.isdir(self.directory):
            return out
        for name in sorted(os.listdir(self.directory)):
            if not name.endswith(".mp4"):
                continue
            path = os.path.join(self.directory, name)
            info = {}
            try:
                with open(path[:-4] + ".json") as fh:
                    info = json.load(fh)
            except (OSError, ValueError):
                pass
            info.update(name=name, path=path, bytes=os.path.getsize(path),
                        recording=(path == self.current()))
            out.append(info)
        return out

    def find(self, sensor_start_us, sensor_end_us):
        """The finished segment that holds this span of sensor time, or None."""
        for seg in self.segments():
            if not seg.get("complete") or seg.get("sensor_origin_us") is None:
                continue
            start = seg["sensor_origin_us"] + seg.get("encoder_start_us", 0)
            end = start + 1e6 * seg.get("duration_s", 0)
            if start <= sensor_start_us and sensor_end_us <= end:
                return seg
        return None

    # --- internals ------------------------------------------------------------------

    def _new_segment(self):
        stamp = datetime.fromtimestamp(self.wall_clock()).strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.directory, f"{stamp}.mp4")
        n = 1
        while os.path.exists(path):
            path = os.path.join(self.directory, f"{stamp}-{n}.mp4")
            n += 1
        info = {"started_at": datetime.fromtimestamp(self.wall_clock()).isoformat(timespec="seconds"),
                "complete": False, **self.extra}
        _write_json(path[:-4] + ".json", info)
        return self._segment_cls(path, info, self.encoder_origin_us)

    def _run(self):
        while not self._stop.is_set():
            wanted = self._split_wanted.wait(1.0)
            due = time.monotonic() - self._opened_at > self.segment_s
            if not (wanted or due):
                continue
            self._split_wanted.clear()
            try:
                self._free_space()
                seg = self._new_segment()
                self._splitter.split_output(seg)     # returns once the old one is closed
                with self._lock:
                    self._current = seg
                self._opened_at = time.monotonic()
            except Exception as e:
                self.error = repr(e)
                self.log(f"recorder: split failed: {e!r}")

    def _free_space(self):
        """Delete the oldest finished segments while the disk is nearly full."""
        while shutil.disk_usage(self.directory).free < self.min_free_bytes:
            done = [s for s in self.segments() if not s["recording"]]
            if not done:
                return
            oldest = done[0]["path"]
            self.log(f"recorder: disk nearly full, deleting {os.path.basename(oldest)}")
            for p in (oldest, oldest[:-4] + ".json"):
                try:
                    os.remove(p)
                except OSError:
                    pass
