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


def other_end(detections, track, away=2.0, cluster=0.5, member=0.6, min_seen=4, min_share=0.1,
              min_travel=0.5):
    """The plate on the bar's other end, as a track like the followed one.

    A bar is rigid, so the plate on its far end keeps nearly the same place
    relative to the followed one while the bar goes up and down. Circles more
    than away radii from the track are taken as offsets from it, and the
    offset with the most others within cluster radii of it is the other end.
    Each frame contributes the circle nearest the cluster's median offset, if
    it is within member radii of it. Returns [(frame, (x, y, r))], or None if
    the other end was seen in fewer than min_seen frames, or in less than
    min_share of the track's. Seen square from the side, the far plate hides
    behind the near one; what clusters then is a knee or a plate on a tree
    behind, found in 4 to 8 % of the frames, where the far plate of an
    oblique view is found in 15 % or more. And it must travel at least
    min_travel as far as the track does over the same frames.

    The offset is not constant: it turns as the bar tilts and as the view of
    it changes, by up to a radius over a lift. So only the frames in which the
    other end was actually seen are returned, never the followed track moved
    by an offset.
    """
    at = dict(track)
    radius = sorted(p[2] for _, p in track)[len(track) // 2]
    offsets = [(i, x - at[i][0], y - at[i][1], r) for i, circles in detections if i in at
               for x, y, r in circles if math.hypot(x - at[i][0], y - at[i][1]) > away * radius]
    if len(offsets) < min_seen:
        return None

    def around(dx, dy, reach):
        return [o for o in offsets if math.hypot(o[1] - dx, o[2] - dy) <= reach * radius]

    def travel(points):
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    # A plate lying still while the bar rests before the lift also keeps a
    # steady offset, and for as long as the bar rests. Seen where it was, it
    # has not moved; the other end moves as far as the followed one.
    while len(offsets) >= min_seen:
        densest = max(offsets, key=lambda o: len(around(o[1], o[2], cluster)))
        near = around(densest[1], densest[2], member)
        mx = sorted(o[1] for o in near)[len(near) // 2]
        my = sorted(o[2] for o in near)[len(near) // 2]
        seen = {}
        for i, dx, dy, r in around(mx, my, member):
            d = math.hypot(dx - mx, dy - my)
            if i not in seen or d < seen[i][0]:
                seen[i] = (d, (at[i][0] + dx, at[i][1] + dy, r))
        if len(seen) < max(min_seen, min_share * len(track)):
            return None
        if travel([p for _, p in seen.values()]) >= min_travel * travel([at[i] for i in seen]):
            return [(i, seen[i][1]) for i in sorted(seen)]
        offsets = [o for o in offsets if o not in near]
    return None


def nearer(track, other, point):
    """Whether track, not other, is the end of the bar on point's side.

    point is anywhere on the side of the plate nearer the camera: a tap on
    it, or the edge of the picture it is on. In each frame both ends were
    seen, the bar's direction from other to track is compared with the
    direction from the bar's middle to the point; the median decides.
    """
    at = dict(other)
    votes = []
    for i, (x, y, _) in track:
        if i in at:
            ox, oy, _ = at[i]
            mx, my = (x + ox) / 2, (y + oy) / 2
            votes.append((x - ox) * (point[0] - mx) + (y - oy) * (point[1] - my))
    if not votes:
        return True
    return sorted(votes)[len(votes) // 2] >= 0


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
