#!/usr/bin/env python3
"""Track the weight plate through recorded footage and write its bar path.

Any plate colour: nothing in the detector asks what colour the plate is. The
method is described in docs/DESIGN.md under "Tracking"; this script only runs
it on a video file and writes what it found.

Writes one CSV row per frame in which the plate was fitted, accepted or not,
with the reason for every rejection, and prints a summary of the set: the
camera angle, the scale, and every movement with its rises.

Usage:
    .venv/bin/python tools/track_plate.py FOOTAGE.mp4 --csv trajectory.csv
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.analysis import kinematics, reps  # noqa: E402
from app.vision import tracker  # noqa: E402
from app.vision.frames import read_video  # noqa: E402

COLUMNS = ["frame", "t_ms", "accepted", "reason",
           "cx_px", "cy_px", "major_px", "minor_px", "angle_deg",
           "x_mm", "y_mm", "vx_m_s", "vy_m_s",
           "mm_per_px", "outline_x_px", "outline_y_px", "scale_px",
           "residual_px", "sectors", "similarity"]


def write_csv(path, st, t, pos, vel):
    face = st.face
    by_time = {round(tt, 6): k for k, tt in enumerate(t)}
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for m in st.measurements:
            if m is None:
                continue
            k = by_time.get(round(m.t, 6)) if m.accepted else None
            xy = ("", "") if k is None else (round(pos[k, 0], 1), round(pos[k, 1], 1))
            v = ("", "") if k is None or np.isnan(vel[k, 0]) else (round(vel[k, 0] / 1000, 3),
                                                                    round(vel[k, 1] / 1000, 3))
            w.writerow([m.frame, round(m.t * 1000, 2), int(m.accepted), m.reason,
                        round(m.face_x, 2), round(m.face_y, 2),
                        round(face.major * m.scale, 2), round(face.minor * m.scale, 2),
                        round(face.angle_deg, 1), *xy, *v,
                        round(m.mm_per_px, 4), round(m.x, 2), round(m.y, 2), round(m.scale, 2),
                        round(m.residual, 3), m.sectors, round(m.similarity, 3)])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--csv", default="trajectory.csv")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    frames = read_video(args.video)
    try:
        st = tracker.track(frames)
    except ValueError as e:
        raise SystemExit(f"{args.video}: {e}")
    acc = st.accepted
    if not acc:
        raise SystemExit("no frame produced a usable measurement")
    t, pos = kinematics.bar_path(st.measurements, st.face, frames.frame_size)
    vel = kinematics.velocity(t, pos)

    n = len(frames)
    w, h = frames.frame_size
    rate = (n - 1) / (frames.times[-1] - frames.times[0]) if n > 1 else 0.0
    face = st.face
    print(f"{args.video}: {w}x{h} {rate:.2f} fps, {n} frames")
    print(f"measured          {len(acc)} / {n} ({100.0 * len(acc) / n:.1f} %)")
    reasons = {}
    for m in st.measurements:
        key = "not found" if m is None else m.reason
        if m is None or not m.accepted:
            reasons[key] = reasons.get(key, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  rejected        {count:4d}  {reason}")
    print(f"camera angle      {face.camera_angle_deg:.1f} deg  (axis ratio {face.ratio:.3f}, major axis at {face.angle_deg:.0f} deg)"
          + ("" if face.corrected else ", too near square to correct by"))
    k = np.array([m.mm_per_px for m in acc])
    print(f"scale             {np.median(k):.3f} mm/px  (from {k.min():.3f} to {k.max():.3f} over the set)")
    print(f"vertical travel   {pos[:, 1].max() - pos[:, 1].min():.0f} mm")
    print(f"horizontal spread {pos[:, 0].max() - pos[:, 0].min():.0f} mm"
          + ("  (corrected for the camera angle)" if face.corrected else ""))

    for j, mv in enumerate(reps.split(t, pos, vel), 1):
        print(f"movement {j}: {mv.start_t:.2f}-{mv.end_t:.2f} s, bar from {mv.low_mm:.0f} to {mv.high_mm:.0f} mm")
        for r in mv.rises:
            print(f"  rise {r.start_t:.2f}-{r.end_t:.2f} s  {r.height_mm:4.0f} mm  "
                  f"mean {r.mean_velocity:.2f} m/s  peak {r.peak_velocity:.2f} m/s at {r.peak_t:.2f} s")

    write_csv(args.csv, st, t, pos, vel)
    print(f"\nwritten {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
