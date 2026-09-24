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
from app.capture.live import LiveConfig
from app.capture.ring import RingBuffer
from app.capture.session import Session
from app.capture.source import Frame


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
