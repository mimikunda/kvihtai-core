"""Split a recorded set into movements and the upward phases within them.

Live detection only decides that a set is going on; how many lifts it held,
and where each begins, is decided here, from the whole bar path at once.

    rest        the bar barely moves for a while: on the floor, in the rack,
                on the shoulders, overhead before the drop
    movement    everything between two rests
    rise        a stretch of a movement in which the bar goes up, continuously
                enough and far enough to be a lift and not a wobble

A squat set is one movement with a rise per rep. A snatch from the floor is a
movement with two rises, the pull and the stand from the catch. Which is which
depends on the lift, and naming them is left for when the lift is known.
"""

from dataclasses import dataclass, field

import numpy as np

REST_SPEED = 0.08       # m/s; slower than this the bar counts as still
REST_MIN_S = 0.40       # a pause shorter than this is part of the movement
RISE_SPEED = 0.05       # m/s upwards to count as rising
RISE_GAP_S = 0.08       # a dip shorter than this does not end a rise
RISE_MIN_MM = 100.0     # a rise smaller than this is a wobble


@dataclass
class Rise:
    start_t: float
    end_t: float
    height_mm: float            # how far the bar went up
    mean_velocity: float        # m/s over the rise: the mean concentric velocity
    peak_velocity: float        # m/s
    peak_t: float


@dataclass
class Movement:
    start_t: float
    end_t: float
    low_mm: float
    high_mm: float
    rises: list = field(default_factory=list)


def _runs(mask):
    """(start, end) index pairs of the True runs in mask, end inclusive."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def rests(t, vel):
    """Boolean mask: the bar is at rest at these samples."""
    speed = np.hypot(vel[:, 0], vel[:, 1]) / 1000.0
    still = np.nan_to_num(speed, nan=0.0) < REST_SPEED
    mask = np.zeros(len(t), bool)
    for a, b in _runs(still):
        if t[b] - t[a] >= REST_MIN_S:
            mask[a:b + 1] = True
    return mask


def rises(t, up_mm, v_up):
    """Upward phases in one movement."""
    rising = np.nan_to_num(v_up, nan=0.0) / 1000.0 > RISE_SPEED
    runs = _runs(rising)
    merged = []
    for a, b in runs:
        if merged and t[a] - t[merged[-1][1]] <= RISE_GAP_S:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    out = []
    for a, b in merged:
        # extend to the turning points either side, where the bar is lowest and highest
        lo = a
        while lo > 0 and up_mm[lo - 1] < up_mm[lo]:
            lo -= 1
        hi = b
        while hi < len(t) - 1 and up_mm[hi + 1] > up_mm[hi]:
            hi += 1
        height = up_mm[hi] - up_mm[lo]
        if height < RISE_MIN_MM or t[hi] <= t[lo]:
            continue
        seg = np.nan_to_num(v_up[lo:hi + 1], nan=0.0) / 1000.0
        k = int(np.argmax(seg))
        out.append(Rise(float(t[lo]), float(t[hi]), float(height),
                        float(height / 1000.0 / (t[hi] - t[lo])),
                        float(seg[k]), float(t[lo + k])))
    return out


def split(t, pos, vel):
    """Movements of a set, each with its rises. pos and vel in mm and mm/s, (N, 2)."""
    t = np.asarray(t, float)
    if len(t) < 3:
        return []
    still = rests(t, vel)
    out = []
    for a, b in _runs(~still):
        if t[b] - t[a] < 0.2:
            continue
        up = pos[a:b + 1, 1]
        out.append(Movement(float(t[a]), float(t[b]), float(up.min()), float(up.max()),
                            rises(t[a:b + 1], up, vel[a:b + 1, 1])))
    return out
