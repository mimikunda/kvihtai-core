#!/usr/bin/env python3
"""Draw the detected plate outline onto every frame of a clip.

Reads a trajectory CSV written by track_plate.py and renders the video. No
detection happens here, so re-rendering with different styling takes seconds
rather than minutes.

A frame with no row in the CSV is labelled instead of drawn. Nothing is ever
interpolated: an outline that is guessed rather than measured lags or runs
ahead of the bar, which looks like a tracking error and hides a real one.

Usage:
    .venv/bin/python tools/render_outline.py FOOTAGE.mp4 trajectory.csv outline.mp4
"""

import argparse
import csv
import sys

import cv2

MEASURED = (0, 255, 0)
INTERPOLATED = (0, 165, 255)
MISSING = (0, 0, 255)


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
                "y_mm": float(r["y_mm"]),
                "ratio": float(r["ratio"]),
                "interpolated": r.get("interpolated") == "1",
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
    measured = interpolated = missing = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        e = rows.get(index)

        if e is not None:
            colour = INTERPOLATED if e["interpolated"] else MEASURED
            trail.append(((int(e["cx"]), int(e["cy"])), colour))
            if e["interpolated"]:
                interpolated += 1
            else:
                measured += 1
        else:
            missing += 1

        if args.trail:
            for (a, _), (b, colour) in zip(trail, trail[1:]):
                cv2.line(frame, a, b, colour, 1, cv2.LINE_AA)

        if e is not None:
            colour = INTERPOLATED if e["interpolated"] else MEASURED
            centre = (int(round(e["cx"])), int(round(e["cy"])))
            axes = (int(round(e["major"] / 2)), int(round(e["minor"] / 2)))
            cv2.ellipse(frame, centre, axes, e["angle"], 0, 360, colour,
                        args.thickness, cv2.LINE_AA)
            cv2.drawMarker(frame, centre, colour, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
            label = "interpolated" if e["interpolated"] else "measured"
            status = f"{label}  h={e['y_mm']:.0f}mm  ratio={e['ratio']:.3f}"
        else:
            status = "no detection"
            colour = MISSING

        cv2.rectangle(frame, (0, 0), (w, 64), (0, 0, 0), -1)
        cv2.putText(frame, f"frame {index:3d}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, status, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
        writer.write(frame)
        index += 1

    cap.release()
    writer.release()
    total = max(index, 1)
    print(f"{args.out}: {index} frames")
    print(f"  measured     {measured:3d}  ({100.0 * measured / total:.1f} %)")
    print(f"  interpolated {interpolated:3d}  ({100.0 * interpolated / total:.1f} %)")
    print(f"  no detection {missing:3d}  ({100.0 * missing / total:.1f} %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
