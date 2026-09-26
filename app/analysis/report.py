"""A set's results as plain data, ready to be saved as JSON or served.

Field names follow docs/API_CONTRACT.md where the contract has one, and add
what the contract has no place for yet: a set holds several rises, and each
gets its own velocities.
"""

import math

import numpy as np

from app.analysis import kinematics, reps


def summarise(set_track, frame_count=None, started_at=None, extra=None):
    """Everything worth keeping about one tracked set, as a dict of plain values."""
    t, pos = kinematics.bar_path(set_track.measurements, set_track.face, set_track.frame_size)
    vel = kinematics.velocity(t, pos)
    movements = reps.split(t, pos, vel)
    measured = set_track.accepted
    n = frame_count if frame_count is not None else len(set_track.measurements)
    rejected = {}
    for m in set_track.measurements:
        if m is not None and not m.accepted:
            rejected[m.reason] = rejected.get(m.reason, 0) + 1
    missing = sum(m is None for m in set_track.measurements)
    if missing:
        rejected["not found"] = missing
    face = set_track.face
    # where the bar end is in the picture, for drawing the path over the video
    px = [(m.face_x, m.face_y) for m in set_track.measurements if m is not None and m.accepted]
    out = {
        "started_at": started_at,
        "frames": n,
        "measured": len(measured),
        "rejected": rejected,
        "frame_size": list(set_track.frame_size),
        "duration_ms": round(1000 * float(t[-1] - t[0]), 1) if len(t) > 1 else 0.0,
        "camera": {
            "angle_deg": round(face.camera_angle_deg, 1),
            "axis_ratio": round(face.ratio, 4),
            "angle_corrected": face.corrected,
            "mm_per_px": round(float(np.median([m.mm_per_px for m in measured])), 4) if measured else None,
        },
        "metrics": {
            "vertical_displacement_mm": round(float(np.ptp(pos[:, 1])), 1) if len(pos) else 0.0,
            "horizontal_deviation_mm": round(float(np.ptp(pos[:, 0])), 1) if len(pos) else 0.0,
        },
        "movements": [
            {
                "start_ms": round(1000 * mv.start_t, 1),
                "end_ms": round(1000 * mv.end_t, 1),
                "low_mm": round(mv.low_mm, 1),
                "high_mm": round(mv.high_mm, 1),
                "rises": [
                    {
                        "start_ms": round(1000 * r.start_t, 1),
                        "end_ms": round(1000 * r.end_t, 1),
                        "height_mm": round(r.height_mm, 1),
                        "mean_concentric_velocity_mps": round(r.mean_velocity, 3),
                        "peak_velocity_mps": round(r.peak_velocity, 3),
                        "peak_ms": round(1000 * r.peak_t, 1),
                    }
                    for r in mv.rises
                ],
            }
            for mv in movements
        ],
        "trajectory": [
            {"t_ms": round(1000 * float(tt), 1), "x_mm": round(float(p[0]), 1), "y_mm": round(float(p[1]), 1),
             "vy_mps": None if math.isnan(v[1]) else round(float(v[1]) / 1000, 3),
             "x_px": round(float(q[0]), 1), "y_px": round(float(q[1]), 1)}
            for tt, p, v, q in zip(t, pos, vel, px)
        ],
    }
    if extra:
        out.update(extra)
    return out
