"""A capture session: camera, ring buffer, watcher and analysis, running together.

    camera thread     source -> ring buffer, never waiting for anyone
    watcher thread    ring buffer -> live tracker -> a recording per set
    analysis thread   recording -> app.vision tracker -> result JSON on disk

The analysis gets its own thread so that a set being analysed does not stop
the next one from being noticed. Results go to out_dir/<start time>/result.json.

The camera may stand turned. The live stage does not care which way is up, a
plate is round, so frames are kept as the camera gives them. Only the crops of
a set are turned upright, before the analysis, which needs to know where
gravity points.
"""

import json
import os
import queue
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np

from app.analysis.report import summarise
from app.capture.live import LiveConfig, LiveTracker
from app.capture.ring import RingBuffer
from app.vision import tracker
from app.vision.frames import Crops


_TURNS = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}


def turn_point(x, y, size, quarter_turns):
    """A pixel position in a frame of size (w, h) after turning the frame clockwise."""
    w, h = size
    k = quarter_turns % 4
    if k == 1:
        return h - 1 - y, x
    if k == 2:
        return w - 1 - x, h - 1 - y
    if k == 3:
        return y, w - 1 - x
    return x, y


def turn_recording(rec, quarter_turns):
    """Turn a set's crops, their origins and the live centres clockwise, in place."""
    k = quarter_turns % 4
    if k == 0:
        return rec
    w, h = rec.frame_size
    for i, (crop, (ox, oy)) in enumerate(zip(rec.crops, rec.origins)):
        ch, cw = crop.shape[:2]
        # the crop's far corner lands on the new top-left
        corner = {1: (ox, oy + ch - 1), 2: (ox + cw - 1, oy + ch - 1), 3: (ox + cw - 1, oy)}[k]
        rec.origins[i] = turn_point(*corner, (w, h), k)
        rec.crops[i] = cv2.rotate(crop, _TURNS[k])
    rec.centres = [None if c is None else turn_point(*c, (w, h), k) for c in rec.centres]
    rec.frame_size = (h, w) if k % 2 else (w, h)
    return rec


def analyse(rec, quarter_turns=0):
    """A live recording to a result dict; None if nothing could be measured.

    quarter_turns: how far the camera's image must be turned clockwise to be
    upright.
    """
    if len(rec.times) < 10:
        return None
    turn_recording(rec, quarter_turns)
    frames = Crops(rec.crops, rec.origins, rec.times, rec.frame_size)
    st = tracker.measure(frames, rec.centres, rec.radius)
    if not st.accepted:
        return None
    started = datetime.fromtimestamp(rec.started_wall, timezone.utc).isoformat(timespec="seconds")
    return summarise(st, frame_count=len(rec.times), started_at=started,
                     extra={"ended": rec.ended, "lost_frames": rec.lost_frames,
                            "quarter_turns": quarter_turns % 4,
                            "source_span_s": [round(rec.times[0], 4), round(rec.times[-1], 4)],
                            "origin": "image centre, y up, in the plane of the bar"})


class Session:
    def __init__(self, source, out_dir, ring_seconds=3.0, config: LiveConfig | None = None,
                 keep_frames=False, log=print, on_result=None, clock=time.time, quarter_turns=0,
                 analyser=analyse):
        self.source = source
        self.out_dir = out_dir
        self.ring_seconds = ring_seconds
        self.config = config or LiveConfig()
        self.keep_frames = keep_frames
        self.log = log
        self.on_result = on_result          # called with (recording, result) after each set
        # analyse, or app.capture.worker.ProcessAnalyser, which uses up the crops
        self.analyser = analyser
        self.clock = clock
        self.quarter_turns = quarter_turns  # may be changed while running; a set keeps its own
        self.results = []
        self.frames_in = 0
        self.analysing = 0                  # sets waiting for or in analysis
        self._stop = threading.Event()
        self._sets = queue.Queue()
        self.ring = None
        self.live = None

    def stop(self):
        self._stop.set()

    def _camera(self):
        direct = getattr(self.source, "read_into", None)
        try:
            while not self._stop.is_set():
                if direct is not None:
                    direct(self.ring.push)
                else:
                    frame = self.source.read()
                    if frame is None:
                        break
                    self.ring.push(frame.image, frame.t)
                self.frames_in += 1
        finally:
            self.ring.close()

    def _analyser(self):
        while True:
            rec = self._sets.get()
            if rec is None:
                return
            try:
                self._analyse_one(rec)
            finally:
                self.analysing -= 1

    def _queue(self, rec):
        rec.quarter_turns = self.quarter_turns
        self.analysing += 1
        self._sets.put(rec)

    def _analyse_one(self, rec):
        t0 = time.monotonic()
        try:
            result = self.analyser(rec, rec.quarter_turns)
        except Exception as e:          # a bad set must not take the session down
            self.log(f"analysis failed: {e!r}")
            return
        if result is None:
            self.log("set discarded: no plate could be measured")
            return
        result["analysis_s"] = round(time.monotonic() - t0, 2)
        self._save(rec, result)
        self.results.append(result)
        if self.on_result is not None:
            try:
                self.on_result(rec, result)
            except Exception as e:
                self.log(f"storing the set failed: {e!r}")
        rises = [r for mv in result["movements"] for r in mv["rises"]]
        self.log(f"set analysed in {result['analysis_s']} s: {result['measured']}/{result['frames']} "
                 f"frames measured, {len(rises)} rises" +
                 "".join(f"\n  rise {r['height_mm']:.0f} mm, mean {r['mean_concentric_velocity_mps']:.2f} "
                         f"m/s, peak {r['peak_velocity_mps']:.2f} m/s" for r in rises))

    def _save(self, rec, result):
        stamp = datetime.fromtimestamp(rec.started_wall).strftime("%Y%m%d-%H%M%S")
        directory = os.path.join(self.out_dir, stamp)
        result["set_id"] = stamp
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "result.json"), "w") as fh:
            json.dump(result, fh, indent=1)
        if self.keep_frames:
            # crops differ in size at the frame's edge, so they are kept one by one
            np.savez_compressed(os.path.join(directory, "frames.npz"),
                                times=np.array(rec.times), origins=np.array(rec.origins),
                                radius=rec.radius, frame_size=np.array(rec.frame_size),
                                **{f"crop{i:05d}": c for i, c in enumerate(rec.crops)})
        result["path"] = directory

    def run(self, duration_s=None):
        """Run until the source ends, duration_s passes, or stop() is called."""
        self.source.start()
        first = self.source.read()
        if first is None:
            raise OSError("the source gave no frames")
        h, w = first.image.shape[:2]
        fps = getattr(self.source, "fps", 30.0) or 30.0
        capacity = max(16, int(self.ring_seconds * fps))
        self.ring = RingBuffer(capacity, first.image.shape, first.image.dtype)
        self.ring.push(first.image, first.t)
        self.log(f"camera {w}x{h} at {fps:.0f} fps; ring buffer {capacity} frames, "
                 f"{self.ring.nbytes / 1e6:.0f} MB")
        self.live = LiveTracker(self.ring, (w, h), self.config, on_set=self._queue, log=self.log,
                                clock=self.clock)
        threads = [threading.Thread(target=self._camera, name="camera", daemon=True),
                   threading.Thread(target=self.live.run, args=(self._stop,), name="watcher", daemon=True),
                   threading.Thread(target=self._analyser, name="analysis", daemon=True)]
        for th in threads:
            th.start()
        started = time.monotonic()
        try:
            while threads[0].is_alive() or threads[1].is_alive():
                if duration_s is not None and time.monotonic() - started > duration_s:
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.log("stopping")
        finally:
            self._stop.set()
            threads[0].join(timeout=2)
            threads[1].join(timeout=5)
            self._sets.put(None)
            threads[2].join()
            self.source.stop()
        return self.results
