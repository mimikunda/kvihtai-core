"""Where in a frame a plate of known size could be.

Works on a reduced image and only has to be close enough to aim the rays of
the precise stage. Three pieces of evidence, none of them colour:

  votes           every strong edge votes for the points one radius away
                  along its gradient, both ways, since the plate may be
                  darker or lighter than what is behind it
  coverage        the share of directions round a candidate in which an edge
                  actually sits at that radius; a real rim is there almost
                  all the way round, a coincidence of straight lines is not
  concentricity   the share of the inside's variation that radius alone
                  explains; a plate is rings, hub, face and rim, while a loop
                  of rope or a bicycle wheel has the room behind it inside

Votes alone chose a coil of rope over the plate on the second test clip:
static, round and the right size, it scored steadily while the moving plate
blurred. Concentricity is what separates them.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.vision.edges import colour_gradient, to_lab

VOTE_SHARE = 0.10      # strongest share of pixels that vote
COVERAGE_QUANTILE = 0.85
RIM_SHARE = 0.75       # of the most complete ring's coverage: complete enough to be the rim


@dataclass
class Candidate:
    x: float
    y: float
    score: float


def vote(gx, gy, mag, radius, share=VOTE_SHARE):
    """Accumulator of centre votes at +-radius along the gradient.

    Normalised so that a full, sharp rim scores about 1.
    """
    h, w = mag.shape
    thr = max(float(np.quantile(mag, 1.0 - share)), 1e-3)
    ys, xs = np.nonzero(mag > thr)
    g = mag[ys, xs]
    ux, uy = gx[ys, xs] / g, gy[ys, xs] / g
    acc = np.zeros(h * w, np.float32)
    for sign in (1.0, -1.0):
        vx = np.rint(xs + sign * radius * ux).astype(np.int64)
        vy = np.rint(ys + sign * radius * uy).astype(np.int64)
        ok = (vx >= 0) & (vx < w) & (vy >= 0) & (vy < h)
        acc += np.bincount(vy[ok] * w + vx[ok], minlength=h * w).astype(np.float32)
    acc = acc.reshape(h, w)
    return cv2.boxFilter(acc, -1, (5, 5), normalize=False) / (2.0 * math.pi * radius)


def peaks(acc, count, separation):
    """The count highest local peaks, no two closer than separation, sub-pixel."""
    a = acc.copy()
    out = []
    h, w = a.shape
    for _ in range(count):
        y, x = np.unravel_index(int(np.argmax(a)), a.shape)
        if a[y, x] <= 0:
            break
        dx = dy = 0.0
        if 0 < x < w - 1:
            l, c, r = acc[y, x - 1], acc[y, x], acc[y, x + 1]
            den = l - 2 * c + r
            dx = 0.5 * (l - r) / den if abs(den) > 1e-9 else 0.0
        if 0 < y < h - 1:
            l, c, r = acc[y - 1, x], acc[y, x], acc[y + 1, x]
            den = l - 2 * c + r
            dy = 0.5 * (l - r) / den if abs(den) > 1e-9 else 0.0
        out.append((x + dx, y + dy))
        cv2.circle(a, (int(x), int(y)), int(math.ceil(separation)), 0, -1)
    return out


def coverage(gx, gy, mag, x, y, radius, threshold, directions=48, slack=2):
    """Share of directions with a radial edge within slack pixels of the radius."""
    return float(coverages(gx, gy, mag, x, y, [radius], threshold, directions, slack)[0])


def coverages(gx, gy, mag, x, y, radii, threshold, directions=48, slack=2):
    """coverage() for several radii at once."""
    th = 2 * math.pi * np.arange(directions) / directions
    c, s = np.cos(th)[None, :, None], np.sin(th)[None, :, None]
    r = np.asarray(radii, float)[:, None, None] + np.arange(-slack, slack + 1)[None, None, :]
    px = np.rint(x + r * c).astype(int)
    py = np.rint(y + r * s).astype(int)
    h, w = mag.shape
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    pxc, pyc = np.clip(px, 0, w - 1), np.clip(py, 0, h - 1)
    radial = np.abs(gx[pyc, pxc] * c + gy[pyc, pxc] * s)
    radial[~ok] = 0.0
    return (radial.max(axis=2) > threshold).mean(axis=1)


def concentricity(lab, x, y, radius, rings=16, directions=48):
    """Share of the variation inside the circle that radius alone explains.

    1 for a perfect bullseye, 0 when direction matters as much as radius,
    below 0 when the rings are less alike than the inside as a whole. Medians
    throughout, so the lettering on a plate and a hand across it cost little.
    """
    fr = (np.arange(rings) + 0.5) / rings * 0.9
    a = 2 * math.pi * np.arange(directions) / directions
    xs = (x + np.outer(fr * radius, np.cos(a))).astype(np.float32)
    ys = (y + np.outer(fr * radius, np.sin(a))).astype(np.float32)
    v = cv2.remap(lab, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    per_ring = np.median(v, axis=1, keepdims=True)
    within = np.median(np.linalg.norm(v - per_ring, axis=2))
    overall = np.median(np.linalg.norm(v - np.median(v.reshape(-1, 3), axis=0), axis=2))
    return float(1.0 - within / max(overall, 1e-6))


def find_candidates(bgr, radius, count=12, window=None):
    """Plate candidates of this radius in a (reduced) image, best first.

    window, if given, is (x0, y0, x1, y1): only that part of the image is
    searched, which is how the live tracker keeps to a few milliseconds.
    Coordinates returned are always those of the whole image.
    """
    ox = oy = 0
    if window is not None:
        h, w = bgr.shape[:2]
        x0, y0 = max(0, int(window[0])), max(0, int(window[1]))
        x1, y1 = min(w, int(window[2])), min(h, int(window[3]))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return []
        bgr = bgr[y0:y1, x0:x1]
        ox, oy = x0, y0
    gx, gy, mag = colour_gradient(bgr)
    acc = vote(gx, gy, mag, radius)
    thr = float(np.quantile(mag, COVERAGE_QUANTILE))
    lab = to_lab(bgr)
    out = []
    for x, y in peaks(acc, count, radius * 0.5):
        cov = coverage(gx, gy, mag, x, y, radius, thr)
        con = concentricity(lab, x, y, radius)
        out.append(Candidate(x + ox, y + oy, cov + max(con, -0.5)))
    out.sort(key=lambda c: -c.score)
    return out


def rim_radius(bgr, x, y, radius, max_radius, step=1.0):
    """The radius of the rim round (x, y), looking outwards from 0.6 radius.

    A plate is rings inside rings, and the circle finder often reports the hub
    or the edge of the face rather than the rim. The rim is the outermost ring
    with an edge nearly all the way round. Not the most complete ring: a plate
    standing on the floor loses the bottom of its rim to the floor and its
    shadow, and on the first test clip the face then scored higher. And the
    outermost, since the largest plate on the bar is the one of known size.
    """
    half = int(math.ceil(max_radius)) + 4
    h, w = bgr.shape[:2]
    x0, y0 = max(0, int(x) - half), max(0, int(y) - half)
    patch = bgr[y0:min(h, int(y) + half + 1), x0:min(w, int(x) + half + 1)]
    gx, gy, mag = colour_gradient(patch)
    thr = float(np.quantile(mag, COVERAGE_QUANTILE))
    radii = np.arange(0.6 * radius, max_radius + step / 2, step)
    if len(radii) < 3:
        return float(radius)
    cov = coverages(gx, gy, mag, x - x0, y - y0, radii, thr)
    cov = np.convolve(cov, np.ones(3) / 3, mode="same")
    peak = (cov[1:-1] >= cov[:-2]) & (cov[1:-1] >= cov[2:]) & (cov[1:-1] >= RIM_SHARE * cov.max())
    found = np.nonzero(peak)[0]
    return float(radii[found[-1] + 1]) if len(found) else float(radii[int(np.argmax(cov))])


def hough_circles(gray, min_radius, max_radius):
    """OpenCV's gradient Hough transform: (x, y, r) of every plausible circle."""
    found = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1.5, 20, param1=60, param2=0.75,
                             minRadius=int(min_radius), maxRadius=int(max_radius))
    return [] if found is None else [tuple(map(float, c)) for c in found[0]]
