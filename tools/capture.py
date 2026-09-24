#!/usr/bin/env python3
"""Run the capture pipeline: watch for a set, keep it, analyse it.

With --video a recorded clip stands in for the camera and is played at its
own pace, so the watcher sees it the way it would see a live camera. On the
Pi, --pi uses the camera module.

Every set found is written to OUT/<start time>/result.json.

Usage:
    .venv/bin/python tools/capture.py --video FOOTAGE.mp4 --out sets/
    python3 tools/capture.py --pi --fps 80 --out sets/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.capture.live import LiveConfig  # noqa: E402
from app.capture.session import Session  # noqa: E402
from app.capture.source import PiCameraSource, VideoFileSource  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="play this file as if it were the camera")
    src.add_argument("--pi", action="store_true", help="use the Raspberry Pi camera")
    p.add_argument("--out", default="sets")
    p.add_argument("--seconds", type=float, default=None, help="stop after this long")
    p.add_argument("--fast", action="store_true", help="with --video: decode as fast as possible")
    p.add_argument("--loop", action="store_true", help="with --video: play it over and over")
    p.add_argument("--size", default="1536x864", help="with --pi: sensor output size")
    p.add_argument("--fps", type=float, default=80.0, help="with --pi")
    p.add_argument("--exposure", type=int, default=2000, help="with --pi: exposure in microseconds")
    p.add_argument("--gain", type=float, default=8.0, help="with --pi: analogue gain")
    p.add_argument("--ring", type=float, default=3.0, help="seconds of frames kept in memory")
    p.add_argument("--keep-frames", action="store_true", help="also save the crops of every set")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.video:
        source = VideoFileSource(args.video, realtime=not args.fast, loop=args.loop)
    else:
        w, h = (int(v) for v in args.size.lower().split("x"))
        source = PiCameraSource((w, h), args.fps, args.exposure, args.gain)
    session = Session(source, args.out, ring_seconds=args.ring, config=LiveConfig(),
                      keep_frames=args.keep_frames)
    results = session.run(args.seconds)
    live = session.live.stats if session.live else {}
    print(f"{session.frames_in} frames in, {live.get('checks', 0)} checks, "
          f"{live.get('searches', 0)} searches, {len(results)} sets analysed")
    for r in results:
        print(f"  {r['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
