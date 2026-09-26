"""Tests of the capture side: the ring buffer, and a whole session on a
synthetic set played at camera pace.

The session test takes a few seconds of wall time on purpose. The watcher
only ever looks at the newest frame while idle, as it must on a camera, so a
source that hands out frames as fast as it can would be over before the
watcher saw the plate move.
"""

import threading
import time

import numpy as np
import pytest

import synthetic as syn
from app.capture.live import LiveConfig, SetRecording
from app.capture.ring import RingBuffer
from app.capture.session import Session, turn_point, turn_recording
from app.capture.source import Frame, VideoFileSource
from app.capture.worker import ProcessAnalyser


# --- ring buffer ---------------------------------------------------------------------

def _pushed(capacity, count):
    ring = RingBuffer(capacity, (4, 6, 3))
    for i in range(count):
        ring.push(np.full((4, 6, 3), i, np.uint8), 0.1 * i)
    return ring


def test_ring_keeps_only_the_newest_frames():
    ring = _pushed(4, 6)
    assert (ring.oldest, ring.newest) == (2, 5)
    assert ring.read(1) is None                     # overwritten
    assert ring.read(6) is None                     # not yet written
    image, t = ring.read(3)
    assert image[0, 0, 0] == 3 and t == pytest.approx(0.3)
    assert RingBuffer(4, (4, 6, 3)).read(-1) is None     # "newest" of an empty ring


def test_ring_reads_a_crop_as_a_copy():
    ring = _pushed(4, 3)
    crop, _ = ring.read(2, (1, 1, 4, 3))
    assert crop.shape == (2, 3, 3)
    crop[:] = 99
    assert ring.read(2)[0][1, 1, 0] == 2


def test_ring_wait_ends_when_the_camera_stops():
    ring = _pushed(4, 1)
    assert ring.wait(0, timeout=0.01)
    result = []
    waiter = threading.Thread(target=lambda: result.append(ring.wait(5, timeout=5)))
    waiter.start()
    ring.close()
    waiter.join(timeout=1)
    assert result == [False]


# --- a whole session ---------------------------------------------------------------------

FPS = 60.0
RADIUS = 60.0
REST, RISE, AFTER = 0.8, 0.6, 1.4       # seconds
LIFT_PX = 240.0


def _bar_y(t):
    s = min(max((t - REST) / RISE, 0.0), 1.0)
    return 480.0 - LIFT_PX * (1 - np.cos(np.pi * s)) / 2


class PacedSource:
    """Frames from memory, released at camera pace."""

    def __init__(self, frames, fps):
        self.frames, self.fps = frames, fps
        self.size = frames[0].shape[1::-1]

    def start(self):
        self._i, self._t0 = 0, None

    def read(self):
        if self._i >= len(self.frames):
            return None
        t = self._i / self.fps
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        wait = self._t0 + t - now
        if wait > 0:
            time.sleep(wait)
        frame = Frame(self.frames[self._i], t, self._i)
        self._i += 1
        return frame

    def stop(self):
        pass


@pytest.fixture(scope="module")
def set_clip():
    frames = []
    for i in range(int((REST + RISE + AFTER) * FPS)):
        img = syn.textured_background((480, 640), seed=3)
        syn.draw_plate(img, (380.0, 150.0), RADIUS, face=syn.RED)        # stays on the stand
        syn.draw_plate(img, (150.0, _bar_y(i / FPS)), RADIUS, face=syn.BLACK)
        frames.append(img)
    return frames


def test_session_keeps_the_lift_from_before_it_started(set_clip, tmp_path):
    config = LiveConfig(check_every=4, search_every_s=0.25, preroll_s=0.4, rest_s=0.5)
    logged = []
    session = Session(PacedSource(set_clip, FPS), str(tmp_path), ring_seconds=1.0,
                      config=config, log=logged.append)
    results = session.run()
    assert len(results) == 1, logged
    result = results[0]
    assert result["ended"] == "plate at rest"
    assert result["trajectory"][0]["t_ms"] < 1000 * REST        # the pre-roll was kept
    rises = [r for mv in result["movements"] for r in mv["rises"]]
    assert len(rises) == 1
    mm_per_px = 450.0 / (2 * RADIUS)
    assert rises[0]["height_mm"] == pytest.approx(LIFT_PX * mm_per_px, rel=0.03)
    peak = LIFT_PX / 2 * np.pi / RISE * mm_per_px / 1000
    assert rises[0]["peak_velocity_mps"] == pytest.approx(peak, rel=0.03)
    assert (tmp_path / result["path"].split("/")[-1] / "result.json").exists()


def test_session_with_the_camera_turned_measures_the_same_lift(set_clip, tmp_path):
    # The camera lies on its side: its picture has to be turned a quarter
    # clockwise to be upright. Only the analysis is told.
    turned = [np.ascontiguousarray(np.rot90(f, 1)) for f in set_clip]     # counter-clockwise
    config = LiveConfig(check_every=4, search_every_s=0.25, preroll_s=0.4, rest_s=0.5)
    logged = []
    session = Session(PacedSource(turned, FPS), str(tmp_path), ring_seconds=1.0,
                      config=config, log=logged.append, quarter_turns=1)
    results = session.run()
    assert len(results) == 1, logged
    rises = [r for mv in results[0]["movements"] for r in mv["rises"]]
    assert len(rises) == 1
    assert rises[0]["height_mm"] == pytest.approx(LIFT_PX * 450.0 / (2 * RADIUS), rel=0.03)
    assert results[0]["frame_size"] == list(set_clip[0].shape[1::-1])      # upright again


@pytest.mark.parametrize("k", [1, 2, 3])
def test_turning_a_recording_moves_crops_with_the_frame(k):
    rng = np.random.default_rng(k)
    frame = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    boxes = [(5, 7, 20, 13), (40, 30, 24, 18)]            # x0, y0, w, h
    rec = SetRecording(crops=[frame[y:y + h, x:x + w].copy() for x, y, w, h in boxes],
                       origins=[(x, y) for x, y, _, _ in boxes], centres=[(9.0, 11.0), None],
                       frame_size=(64, 48))
    turn_recording(rec, k)
    upright = np.ascontiguousarray(np.rot90(frame, -k))    # clockwise
    for crop, (ox, oy) in zip(rec.crops, rec.origins):
        ch, cw = crop.shape[:2]
        assert np.array_equal(crop, upright[oy:oy + ch, ox:ox + cw])
    assert rec.frame_size == upright.shape[1::-1]
    x, y = rec.centres[0]
    assert np.array_equal(upright[int(y), int(x)], frame[11, 9])
    assert turn_point(9, 11, (64, 48), k) == (x, y)


def test_process_analyser_gives_what_the_thread_gives(set_clip, tmp_path):
    config = LiveConfig(check_every=4, search_every_s=0.25, preroll_s=0.4, rest_s=0.5)
    kept = []
    session = Session(PacedSource(set_clip, FPS), str(tmp_path), ring_seconds=1.0, config=config,
                      log=lambda *a: None, analyser=lambda rec, k: kept.append(rec) or None)
    session.run()
    assert len(kept) == 1
    rec = kept[0]
    copy = SetRecording(**{**rec.__dict__, "crops": list(rec.crops)})
    worker = ProcessAnalyser()
    try:
        remote = worker(copy, 0)
    finally:
        worker.close()
    assert all(c is None for c in copy.crops)                  # sent, not kept twice
    from app.capture.session import analyse
    local = analyse(rec, 0)
    assert remote["trajectory"] == local["trajectory"]


def test_video_source_loops_with_a_pause_and_time_keeps_going(tmp_path):
    path = str(tmp_path / "clip.mp4")
    frames, _, _ = syn.rising_clip(n=12)
    syn.write_video(path, frames, 60.0)
    source = VideoFileSource(path, loop=True, pause_s=0.1)
    source.start()
    times = [source.read().t for _ in range(60)]
    steps = np.diff(times)
    assert np.all(steps > 0.01) and np.all(steps < 0.03)     # no jump back, no gap
    source.stop()
