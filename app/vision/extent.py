"""How far the plate reaches round the centre it was followed by.

The circles the coarse stage follows are often not the rim. A loaded bar
carries change plates, collars and a steel disc inside its 450 mm plates, all
on the same sleeve, and their rings are sharper than the rim: on the new
footage the Hough transform settled on one of them in about one clip in five.
Seen from an angle it is worse. The rim is then an ellipse, a ring two pixels
wide round any one centre catches a sliver of it, and the inner discs stand
out towards the camera along the sleeve, so their centre is off the rim's by
up to half its radius. Looking outwards for the last complete ring finds, in
a busy gym, a ring of background as often as the rim.

What does tell the plate apart is that it moves with the bar and the room
does not. Sampled round the followed centre, in the plate's own frame, the
plate looks the same wherever it has been carried to, and the background
behind it changes. So the spread of each sample over frames in which the
plate stood in different places is small on the plate and large off it, and
where the spread jumps is the silhouette of the plate stack, whatever rings
lie inside it. An ellipse through that boundary gives the rim's size and how
far its centre is from the followed one.

Where the background is plain, a white wall or a dark hall, it does not
change either and the boundary runs off into it. Those directions are left
out: the plate's outline is an ellipse, and a direction whose boundary is far
from the others' is not on it.
"""

import math
import warnings

import cv2
import numpy as np

DIRECTIONS = 90
SPACING = 0.5         # of the followed radius: positions closer than this say nothing new
MIN_POSITIONS = 4
MIN_CONTRAST = 1.5    # the background's spread against the plate's: less, and nothing can be told
BAND = 0.3            # directions further than this from the median boundary are left out
TRIM = 0.8            # share of directions the ellipse is refitted to, twice
MIN_ERROR = 0.08      # of the radius: a direction this close to the ellipse is always kept


def _polar(bgr, x, y, radii, directions=DIRECTIONS):
    """Lab samples round (x, y): (directions, radii, 3), NaN outside the image."""
    th = 2 * math.pi * np.arange(directions) / directions
    xs = (x + np.outer(np.cos(th), radii)).astype(np.float32)
    ys = (y + np.outer(np.sin(th), radii)).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return cv2.remap(lab, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=(float("nan"),) * 3)


def carried_ellipse(images, track, max_radius, directions=DIRECTIONS):
    """The plate's silhouette round a followed track, as an ellipse.

    images(i) returns frame i as the track saw it; track is [(i, (x, y, r))].
    Returns (dx, dy, a, b, angle_deg): the ellipse's centre relative to the
    track's, its semi-axes and orientation, in the track's pixels. None when
    the plate did not move over enough different background to tell.
    """
    r0 = float(np.median([p[2] for _, p in track]))
    radii = np.arange(1.0, float(max_radius), 1.0)
    if len(radii) < 8:
        return None
    taken, samples = [], []
    for i, (x, y, _) in track:
        if any(math.hypot(x - tx, y - ty) < SPACING * r0 for tx, ty in taken):
            continue
        taken.append((x, y))
        samples.append(_polar(images(i), x, y, radii, directions))
    if len(samples) < MIN_POSITIONS:
        return None
    stack = np.stack(samples)
    with warnings.catch_warnings():
        # a direction that leaves the image in every frame is all NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        typical = np.nanmedian(stack, axis=0)
        spread = np.nanmedian(np.linalg.norm(stack - typical[None], axis=3), axis=0)
    spread = np.nan_to_num(spread, nan=float(np.nanmax(spread)) if np.isfinite(spread).any() else 0.0)
    spread = cv2.GaussianBlur(spread.astype(np.float32), (0, 0), 1.0)
    inside = float(np.median(spread[:, :max(2, int(0.5 * r0))]))
    far = min(len(radii) - 8, int(3 * r0))
    outside = float(np.median(spread[:, far:]))
    if outside < MIN_CONTRAST * max(inside, 1e-6):
        return None
    moved = (spread >= 0.5 * (inside + outside)).astype(np.float32)
    # Per direction, the radius that best splits the samples into carried
    # inside and not outside. A thin line of high spread inside, the edge of a
    # change plate where the centres jitter by a pixel, costs a few samples,
    # not the whole direction.
    carried_before = np.concatenate([np.zeros((directions, 1)), np.cumsum(1.0 - moved, axis=1)], axis=1)[:, :-1]
    moved_from = np.cumsum(moved[:, ::-1], axis=1)[:, ::-1]
    edge = radii[np.argmax(carried_before + moved_from, axis=1)]

    th = 2 * math.pi * np.arange(directions) / directions
    pts = np.column_stack([edge * np.cos(th), edge * np.sin(th)]).astype(np.float32)
    typical_r = float(np.median(edge))
    keep = np.abs(edge - typical_r) <= BAND * typical_r
    for _ in range(3):
        if keep.sum() < 6:
            return None
        (cx, cy), (w, h), angle = cv2.fitEllipse(pts[keep])
        a, b = w / 2.0, h / 2.0
        if min(a, b) <= 0:
            return None
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        q = pts - [cx, cy]
        u, v = q[:, 0] * c + q[:, 1] * s, -q[:, 0] * s + q[:, 1] * c
        error = np.abs(np.sqrt((u / a) ** 2 + (v / b) ** 2) - 1.0)
        keep = keep & (error <= max(MIN_ERROR, float(np.quantile(error[keep], TRIM))))
    return float(cx), float(cy), float(max(a, b)), float(min(a, b)), float(angle)

