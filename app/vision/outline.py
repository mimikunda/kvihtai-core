"""The plate's outline, learned once per set and fitted per frame.

A plate seen from the side and a little above is not an ellipse. It is the
front face plus the tread, which shows as a crescent on the side facing away
from the camera, and perspective bends both. Rather than model each effect the
outline is learned: its radius as a function of direction, rho(phi), pooled
from every frame of the set. Per frame only a position and a scale are fitted.

Fitting three numbers to a known shape is what keeps the centre still. A free
ellipse per frame lets shape and position trade against each other, and a
detector that sometimes takes the face edge and sometimes the silhouette moves
the centre sideways by half the tread. The learned outline is the silhouette,
so edges on the face side of the tread fall outside the fit's gate and cannot
pull it.

The face is recovered separately, from the same pooled edges, because the face
is what the geometry needs: its centre is the end of the bar, its major axis is
the true 450 mm, and its axis ratio is the camera angle.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

BINS = 120


@dataclass
class Face:
    """The front face ellipse, in outline units: outline origin at 0, scale 1."""
    dx: float
    dy: float
    major: float        # full axis lengths, 2.0 for a unit circle
    minor: float
    angle_deg: float    # direction of the major axis
    tread_x: float = 0.0  # image of the plate's thickness: far face minus front face
    tread_y: float = 0.0
    corrected: bool = True  # False: the axis is too uncertain to correct the bar path by

    @property
    def ratio(self) -> float:
        return self.minor / self.major

    @property
    def camera_angle_deg(self) -> float:
        return math.degrees(math.acos(min(1.0, self.ratio)))


class Outline:
    def __init__(self, rho=None, face: Face | None = None):
        self.rho = np.ones(BINS) if rho is None else np.asarray(rho, float)
        self.face = face

    def at(self, phi):
        """rho and d rho / d phi at the angles phi; linear between bins, periodic."""
        x = np.mod(phi, 2 * math.pi) / (2 * math.pi) * BINS
        k0 = np.floor(x).astype(int) % BINS
        k1 = (k0 + 1) % BINS
        f = x - np.floor(x)
        r = self.rho[k0] * (1 - f) + self.rho[k1] * f
        dr = (self.rho[k1] - self.rho[k0]) * BINS / (2 * math.pi)
        return r, dr

    def curve(self, n=BINS):
        phi = 2 * math.pi * np.arange(n) / n
        r, _ = self.at(phi)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi)])


@dataclass
class OutlineFit:
    centre: np.ndarray
    scale: float
    residuals: np.ndarray
    inliers: np.ndarray
    sigma: float

    def residual(self) -> float:
        """Median distance of the inlier edges from the outline, pixels."""
        return float(np.median(np.abs(self.residuals[self.inliers]))) if self.inliers.any() else math.inf


def sectors_covered(points, centre, sectors=12) -> int:
    """How many of the directions round the centre hold at least one edge."""
    if len(points) == 0:
        return 0
    a = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    return int(np.unique(((a + math.pi) / (2 * math.pi) * sectors).astype(int) % sectors).size)


def fit_outline(points, outline: Outline, centre, scale, scale_prior=None, prior_weight=0.0,
                gate=0.04, iterations=15) -> OutlineFit | None:
    """Position and scale of the outline on these edges.

    Gauss-Newton on the radial distance of each edge from the outline, with
    Tukey weights on a scale learned from the edges themselves, so a hand or a
    shoulder crossing the rim is ignored rather than averaged in. Edges further
    than gate * scale from the outline take no part at all. scale_prior, when
    given, holds the scale near the value the neighbouring frames agree on;
    where the rim is well covered the edges outvote it, and where only an arc
    is visible it stops the scale and the position trading against each other.
    """
    c = np.array(centre, float)
    s = float(scale)
    gate_px = gate * scale
    sigma = gate_px / 3.0
    if len(points) < 12:
        return None
    for it in range(iterations):
        d = points - c
        dist = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
        phi = np.arctan2(d[:, 1], d[:, 0])
        rho, drho = outline.at(phi)
        res = dist - s * rho
        inside = np.abs(res) <= gate_px
        if inside.sum() < 12:
            return None
        if it > 1:
            mad = 1.4826 * float(np.median(np.abs(res[inside])))
            sigma = max(min(sigma, 3.0 * mad), 0.2)
        u = res / (4.685 * sigma)
        w = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
        ux, uy = d[:, 0] / dist, d[:, 1] / dist
        # phi = atan2(dy, dx), so d phi / d centre = (uy, -ux) / dist
        J = np.column_stack([-ux - s * drho * uy / dist, -uy + s * drho * ux / dist, -rho])
        Jw = J * w[:, None]
        N = Jw.T @ J
        g = -(Jw.T @ res)
        if scale_prior is not None and prior_weight > 0:
            N[2, 2] += prior_weight ** 2
            g[2] -= prior_weight ** 2 * (s - scale_prior)
        try:
            step = np.linalg.solve(N, g)
        except np.linalg.LinAlgError:
            return None
        c += step[:2]
        s += step[2]
        if not np.isfinite(s) or s <= 0:
            return None
        if float(np.linalg.norm(step)) < 1e-4:
            break
    d = points - c
    rho, _ = outline.at(np.arctan2(d[:, 1], d[:, 0]))
    res = np.hypot(d[:, 0], d[:, 1]) - s * rho
    return OutlineFit(c, s, res, np.abs(res) <= max(3.0 * sigma, 0.6), sigma)


def _peak(h, edges, j, width):
    """Sub-bin position of histogram bin j."""
    shift = 0.0
    if 0 < j < len(h) - 1:
        den = h[j - 1] - 2 * h[j] + h[j + 1]
        shift = 0.5 * (h[j - 1] - h[j + 1]) / den if abs(den) > 1e-9 else 0.0
    return edges[j] + (0.5 + shift) * width


def _histogram(values, lo, hi, width):
    edges = np.arange(lo, hi + width, width)
    h, _ = np.histogram(values, bins=edges)
    return np.convolve(h, np.exp(-0.5 * np.arange(-3, 4) ** 2), mode="same"), edges


def _outermost(values, lo, hi, width, share=0.6, reach=0.15):
    """The outermost histogram peak at least share as dense as the densest.

    Only peaks within reach of the densest count, so an edge well outside the
    plate, the far plate's rim for one, cannot take over.
    """
    h, edges = _histogram(values, lo, hi, width)
    top = int(np.argmax(h))
    best = top
    for j in range(top + 1, min(len(h) - 1, top + int(round(reach / width)) + 1)):
        if h[j] >= share * h[top] and h[j] >= h[j - 1] and h[j] >= h[j + 1]:
            best = j
    return _peak(h, edges, best, width), h, edges


def learn_outline(phi, r, lo=0.85, hi=1.20, width=0.004, smooth=2, support=0.35):
    """rho(phi) from pooled edges: in each direction, the silhouette.

    Where the tread shows a direction has two edges, the face meeting the
    tread and the tread meeting the background, and both may be seen in every
    frame. Taking whichever is denser would let the outline switch between
    them from one direction to the next. The silhouette is the outer of the
    two, so the outer one is taken whenever it is seen nearly as often.

    A direction in which the rim is rarely seen, a dark plate against dark
    clothes, gets its radius from its neighbours instead: there the densest
    thing is whatever clutter happens to be there. support is the share of the
    typical direction's density a direction needs to speak for itself.
    """
    k = (np.mod(phi, 2 * math.pi) / (2 * math.pi) * BINS).astype(int) % BINS
    rho = np.full(BINS, np.nan)
    height = np.zeros(BINS)
    for b in range(BINS):
        v = r[(k == b) | (k == (b - 1) % BINS) | (k == (b + 1) % BINS)]
        v = v[(v >= lo) & (v < hi)]
        if v.size >= 8:
            rho[b], h, _ = _outermost(v, lo, hi, width)
            height[b] = h.max()
    known = ~np.isnan(rho)
    if known.sum() < BINS // 3:
        return None
    ok = known & (height >= support * np.median(height[known]))
    if ok.sum() < BINS // 3:
        ok = known
    idx = np.arange(BINS)
    rho = np.interp(idx, idx[ok], rho[ok], period=BINS)
    if smooth:
        kern = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        rho = np.convolve(np.concatenate([rho[-smooth:], rho, rho[:smooth]]), kern, mode="valid")
    return rho


def recentre(rho):
    """Move and rescale an outline so its best-fit circle has centre 0 and radius 1.

    The outline and the per-frame centres can trade a shift between them
    without changing any fit. Pinning the outline's own centre removes that
    freedom, so repeated learning cannot drift. Returns the new rho, the old
    centre of the circle and its radius; a point stays where it was if every
    per-frame centre is moved by shift * scale and every scale multiplied by
    the radius.
    """
    pts = Outline(rho).curve(720)
    x, y = pts[:, 0], pts[:, 1]
    sol, *_ = np.linalg.lstsq(np.column_stack([x, y, np.ones_like(x)]), x * x + y * y, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    radius = math.sqrt(sol[2] + cx * cx + cy * cy)
    q = (pts - [cx, cy]) / radius
    phi = np.mod(np.arctan2(q[:, 1], q[:, 0]), 2 * math.pi)
    rr = np.hypot(q[:, 0], q[:, 1])
    order = np.argsort(phi)
    grid = 2 * math.pi * np.arange(BINS) / BINS
    new = np.interp(grid, phi[order], rr[order], period=2 * math.pi)
    return new, np.array([cx, cy]), radius


def face_from_edges(phi, r_rel, outline: Outline, max_tread=0.35, step=0.01, min_gap=0.015) -> Face:
    """The front face, from pooled edges measured relative to the outline.

    The outline is the silhouette of a cylinder: the front face swept along t,
    the image of the plate's thickness. Where the outline's outward normal has
    a component along t, the silhouette is the far face's rim, and the front
    face's rim is the same curve moved back by t; elsewhere the two coincide.
    So t alone decides where the face edge must be, and t is found as the
    vector that puts the most pooled edges on that predicted edge, above the
    edges' background density in each direction.

    If no t does better than chance, or the tread does not contrast with the
    face, t stays zero and the face is the silhouette itself: the centre is
    then off by half the tread, but by the same amount in every frame.
    """
    # pooled edges as a density over (direction, radius relative to the outline)
    n_dir, lo, hi, width = 180, 0.60, 1.10, 0.004
    k = (np.mod(phi, 2 * math.pi) / (2 * math.pi) * n_dir).astype(int) % n_dir
    j = ((r_rel - lo) / width).astype(int)
    ok = (j >= 0) & (j < int((hi - lo) / width))
    H = np.zeros((n_dir, int((hi - lo) / width)))
    np.add.at(H, (k[ok], j[ok]), 1.0)
    H = H - np.median(H, axis=1, keepdims=True)

    sil = outline.curve(360)
    normal = np.roll(sil, -1, axis=0) - np.roll(sil, 1, axis=0)
    normal = np.column_stack([normal[:, 1], -normal[:, 0]])
    normal /= np.maximum(np.hypot(normal[:, 0], normal[:, 1]), 1e-12)[:, None]
    if np.mean(np.sum(normal * sil, axis=1)) < 0:      # make them point outwards
        normal = -normal

    # Only where the face edge would be clearly inside the silhouette does an
    # edge say anything about t; closer in, it could be the silhouette's own.
    def score(t):
        front = normal @ t > min_gap
        if not front.any():
            return 0.0
        q = sil[front] - t
        a = np.arctan2(q[:, 1], q[:, 0])
        rho, _ = outline.at(a)
        rr = np.hypot(q[:, 0], q[:, 1]) / rho
        kk = (np.mod(a, 2 * math.pi) / (2 * math.pi) * n_dir).astype(int) % n_dir
        jj = ((rr - lo) / width).astype(int)
        inside = (jj >= 0) & (jj < H.shape[1])
        return float(np.clip(H[kk[inside], jj[inside]], 0, None).sum())

    grid = np.arange(-max_tread, max_tread + 1e-9, 2 * step)
    best_t, best = np.zeros(2), 0.0
    for tx in grid:
        for ty in grid:
            if tx * tx + ty * ty > max_tread ** 2:
                continue
            v = score(np.array([tx, ty]))
            if v > best:
                best, best_t = v, np.array([tx, ty])
    for tx in np.arange(best_t[0] - 2 * step, best_t[0] + 2 * step + 1e-9, step / 2):
        for ty in np.arange(best_t[1] - 2 * step, best_t[1] + 2 * step + 1e-9, step / 2):
            v = score(np.array([tx, ty]))
            if v > best:
                best, best_t = v, np.array([tx, ty])
    # Chance level: the same search on the silhouette's own ring alone scores
    # nothing, so demand that the face edge was seen on a fair part of the
    # front arc, as a share of how densely the silhouette itself was seen.
    ring = np.clip(H, 0, None).max(axis=1)
    front = normal @ best_t > min_gap
    if front.sum() < 12 or best < 0.25 * float(np.median(ring)) * front.sum():
        best_t = np.zeros(2)
        front = np.zeros(len(sil), bool)
    shifted = np.clip(normal @ best_t, 0, None) > 0
    boundary = sil - np.where(shifted[:, None], best_t, 0.0)
    (cx, cy), (w, h), angle = cv2.fitEllipseDirect(boundary.astype(np.float32))
    if w >= h:
        major, minor, direction = w, h, angle
    else:
        major, minor, direction = h, w, angle + 90.0
    return Face(float(cx), float(cy), float(major), float(minor), float(direction % 180.0),
                float(best_t[0]), float(best_t[1]))
