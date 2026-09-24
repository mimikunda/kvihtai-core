#!/usr/bin/env python3
"""Time each stage of the capture pipeline on recorded footage.

Decoding is done first and not timed, since with a camera there is none. Then
every stage runs on frames already in memory, as fast as it can:

    search    the whole-frame look for plates, once a second while idle
    check     the look for each watched plate, every eighth frame while idle
    follow    the per-frame search during a set, which must keep up with the camera
    analysis  the precise stage, after the set

The follow cost per frame is the number that decides the frame rate: above
1000 / fps milliseconds the watcher falls behind, and a set longer than the
ring buffer loses frames.

Usage:
    python3 tools/bench_capture.py FOOTAGE.mp4 [--frames 200]
"""

import argparse
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.capture.live import LiveConfig, LiveTracker  # noqa: E402
from app.capture.ring import RingBuffer  # noqa: E402
from app.capture.session import analyse  # noqa: E402
from app.capture.source import VideoFileSource  # noqa: E402


def _ms(times):
    t = 1000 * np.array(times)
    return f"median {np.median(t):6.1f} ms, 90 % {np.quantile(t, 0.9):6.1f} ms, max {t.max():6.1f} ms"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--frames", type=int, default=None, help="use only the first this many")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    source = VideoFileSource(args.video)
    source.start()
    frames = []
    while args.frames is None or len(frames) < args.frames:
        f = source.read()
        if f is None:
            break
        frames.append(f)
    source.stop()
    h, w = frames[0].image.shape[:2]
    print(f"{args.video}: {w}x{h}, {len(frames)} frames decoded")

    ring = RingBuffer(len(frames), frames[0].image.shape, frames[0].image.dtype)
    for f in frames:
        ring.push(f.image, f.t)
    ring.close()
    config = LiveConfig()
    live = LiveTracker(ring, (w, h), config, log=lambda *_: None)

    searches = []
    for seq in range(0, len(frames), 30):
        t0 = time.perf_counter()
        live.search(seq)
        searches.append(time.perf_counter() - t0)
    print(f"search    {_ms(searches)}")
    live.search(0)
    if not live.watched:
        print("no plate found in the first frame; nothing more to time")
        return 1
    plate = max(live.watched, key=lambda wp: wp.r)
    print(f"          {len(live.watched)} watched, the largest r {plate.r:.0f} px at "
          f"({plate.x:.0f}, {plate.y:.0f})")

    checks = []
    for seq in range(0, len(frames), config.check_every):
        t0 = time.perf_counter()
        live.check(seq)
        checks.append(time.perf_counter() - t0)
    print(f"check     {_ms(checks)} for {len(live.watched)} plates")

    # follow: time each frame by timing the whole run and the frames it kept
    per_frame = []
    original = live._along

    def timed(*a, **k):
        t0 = time.perf_counter()
        out = original(*a, **k)
        per_frame.append(time.perf_counter() - t0)
        return out

    live._along = timed
    plate.moved = 0
    t0 = time.perf_counter()
    rec = live.follow(0, plate, threading.Event())
    total = time.perf_counter() - t0
    print(f"follow    {_ms(per_frame)} searching, {1000 * total / max(1, len(rec.times)):.1f} ms "
          f"per frame in all, over {len(rec.times)} frames ({rec.ended})")
    print(f"          keeps up with at most {len(rec.times) / total:.0f} fps")

    t0 = time.perf_counter()
    result = analyse(rec)
    took = time.perf_counter() - t0
    measured = result["measured"] if result else 0
    print(f"analysis  {took:.2f} s for {len(rec.times)} frames, {1000 * took / max(1, len(rec.times)):.1f} ms "
          f"per frame, {measured} measured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
