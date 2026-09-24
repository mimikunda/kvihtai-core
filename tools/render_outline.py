#!/usr/bin/env python3
"""Draw the detected plate face onto every frame of a clip.

Reads a trajectory CSV written by track_plate.py and renders the video. No
detection happens here, so re-rendering with different styling takes seconds
rather than minutes.

A frame the tracker rejected is drawn in red with the reason; a frame with no
row at all is labelled. Nothing is ever interpolated: an outline that is
guessed rather than measured lags or runs ahead of the bar, which looks like a
tracking error and hides a real one.

Usage:
    .venv/bin/python tools/render_outline.py FOOTAGE.mp4 trajectory.csv outline.mp4
"""

import argparse
import csv
import sys

import cv2

ACCEPTED = (0, 255, 0)
REJECTED = (0, 0, 255)
MISSING = (0, 165, 255)


def load(path):
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows[int(r["frame"])] = {
                "cx": float(r["cx_px"]),
                "cy": float(r["cy_px"]),
                "major": float(r["major_px"]),
                "minor": float(r["minor_px"]),
                "angle": float(r["angle_deg"]),
                "accepted": r["accepted"] == "1",
                "reason": r["reason"],
                "y_mm": r["y_mm"],
                "vy": r["vy_m_s"],
            }
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("csv")
    p.add_argument("out")
    p.add_argument("--trail", action="store_true", help="also draw the path so far")
    p.add_argument("--thickness", type=int, default=1,
                   help="outline width in pixels; keep it thin to judge the edge")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    rows = load(args.csv)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"cannot open {args.video}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    trail = []
    index = 0
    counts = {"accepted": 0, "rejected": 0, "missing": 0}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        e = rows.get(index)
        if e is None:
            counts["missing"] += 1
            colour, status = MISSING, "no detection"
        elif e["accepted"]:
            counts["accepted"] += 1
            colour = ACCEPTED
            trail.append((int(round(e["cx"])), int(round(e["cy"]))))
            speed = f"  v={float(e['vy']):+.2f} m/s" if e["vy"] else ""
            status = f"h={float(e['y_mm']):.0f} mm{speed}"
        else:
            counts["rejected"] += 1
            colour, status = REJECTED, f"rejected: {e['reason']}"

        if args.trail:
            for a, b in zip(trail, trail[1:]):
                cv2.line(frame, a, b, ACCEPTED, 1, cv2.LINE_AA)
        if e is not None:
            centre = (int(round(e["cx"] * 16)), int(round(e["cy"] * 16)))
            axes = (int(round(e["major"] / 2 * 16)), int(round(e["minor"] / 2 * 16)))
            cv2.ellipse(frame, centre, axes, e["angle"], 0, 360, colour, args.thickness, cv2.LINE_AA, 4)
            cv2.drawMarker(frame, (int(round(e["cx"])), int(round(e["cy"]))), colour,
                           cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)

        cv2.rectangle(frame, (0, 0), (w, 64), (0, 0, 0), -1)
        cv2.putText(frame, f"frame {index:3d}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, status, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
        writer.write(frame)
        index += 1

    cap.release()
    writer.release()
    total = max(index, 1)
    print(f"{args.out}: {index} frames")
    for key, count in counts.items():
        print(f"  {key:9s} {count:4d}  ({100.0 * count / total:.1f} %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
