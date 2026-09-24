"""Which candidate in each frame is the bar's plate.

Choosing the best candidate frame by frame fails exactly where it matters: in
the fast part of a lift the plate blurs, scores less than a static circle in
the background, and a greedy tracker jumps to it and never comes back. With the
whole set recorded, the choice can be made for all frames at once: the path
through the candidates with the highest total score that never moves faster
than a bar can.

The path is anchored to the plate that moves. Circles found by a Hough
transform on a sample of frames are linked into tracks; the bar's plate is the
track that travels furthest, and the path must pass through it wherever it was
seen. A plate on a storage tree or a coil of rope does not move.
"""

import math

from app.vision.candidates import Candidate
from app.vision.plate import max_step_px


def link_circles(detections, times, max_gap_frames, size_tolerance=0.15):
    """Link per-frame circles (frame, [(x, y, r), ...]) into tracks of (frame, (x, y, r))."""
    tracks = []
    for i, circles in detections:
        for x, y, r in circles:
            best = None
            for track in tracks:
                j, (px, py, pr) = track[-1]
                if i - j > max_gap_frames or abs(r - pr) > size_tolerance * pr:
                    continue
                d = math.hypot(x - px, y - py)
                if d <= max_step_px(pr, times[i] - times[j]) + 3 and (best is None or d < best[0]):
                    best = (d, track)
            if best is not None:
                best[1].append((i, (x, y, r)))
            else:
                tracks.append([(i, (x, y, r))])
    return tracks


def moving_track(tracks, min_length=4):
    """The track that travels furthest, weighted by how often it was seen."""
    def travel(track):
        xs = [p[0] for _, p in track]
        ys = [p[1] for _, p in track]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    long_enough = [t for t in tracks if len(t) >= min_length]
    if not long_enough:
        return None
    return max(long_enough, key=lambda t: travel(t) * len(t))


ANCHOR_BONUS = 10.0     # more than any run of skipped frames could save


def anchor(candidates, track, radius, near=0.35):
    """Where the moving track was seen, keep only candidates close to it.

    If none is close the track's own circle is used. Anchored candidates carry
    a bonus no path can afford to skip, so the path goes through every one of
    them and, held by the speed limit in between, cannot wander off to a
    static circle that scores better frame by frame.
    """
    out = [list(c) for c in candidates]
    for i, (x, y, _) in track:
        close = [c for c in out[i] if math.hypot(c.x - x, c.y - y) <= near * radius]
        out[i] = [Candidate(c.x, c.y, c.score + ANCHOR_BONUS) for c in close] or \
            [Candidate(x, y, 1.0 + ANCHOR_BONUS)]
    return out


def best_path(candidates, times, radius, max_gap=6, gap_cost=0.1, smoothness=0.3):
    """Dynamic programming over candidates; returns one Candidate or None per frame.

    A step between two frames may not exceed what a bar can travel in the
    time between them, and costs a little in proportion to its size, so of two
    equally good paths the steadier wins. Frames may be skipped, at gap_cost
    each, which is how a frame where the plate was not among the candidates is
    left empty instead of being filled with the nearest wrong thing.
    """
    n = len(candidates)
    score = [[-math.inf] * len(c) for c in candidates]
    back = [[None] * len(c) for c in candidates]
    for i in range(n):
        for a, ca in enumerate(candidates[i]):
            best, arg = ca.score - gap_cost * i, None
            for g in range(1, max_gap + 1):
                j = i - g
                if j < 0:
                    break
                reach = max_step_px(radius, times[i] - times[j])
                for b, cb in enumerate(candidates[j]):
                    if score[j][b] == -math.inf:
                        continue
                    d = math.hypot(ca.x - cb.x, ca.y - cb.y)
                    if d > reach:
                        continue
                    v = score[j][b] + ca.score - gap_cost * (g - 1) - smoothness * (d / reach) ** 2
                    if v > best:
                        best, arg = v, (j, b)
            score[i][a] = best
            back[i][a] = arg
    end, end_score = None, -math.inf
    for i in range(n):
        for a in range(len(candidates[i])):
            v = score[i][a] - gap_cost * (n - 1 - i)
            if v > end_score:
                end_score, end = v, (i, a)
    path = [None] * n
    node = end
    while node is not None:
        i, a = node
        path[i] = candidates[i][a]
        node = back[i][a]
    return path
