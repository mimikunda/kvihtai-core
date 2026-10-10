"""The plate's front face, measured in one frame.

The end of the bar is the centre of the plate's front face, and the face's
rim is what a person marks as the plate. It is not the plate's outline in the
picture. A plate is a cylinder, and seen a little from the side its tread
shows beyond the face on one side, with the plates behind it and the far end
of the bar further out still: lit, above the face, while the bar rests on the
floor below the camera; dark, below it, overhead.

Which side that is, the sleeve says. Its end stands out of the plate towards
the camera, a light colourless disc, and parallax carries it across the face:
below the hub at rest, above it at the top of a lift. The tread lies on the
other side, moved out by the same parallax applied to the plate's thickness.
So on that side edges beyond the face, up to ZONE times the sleeve end's
offset from the centre, cost nothing, and the face is fitted to the rest.

The fit is by soft assignment: every edge on a ray counts, weighed by how near
the face it lies against a level for "none of these is the rim". Its cost is
smooth, so a start a few pixels off does not hold it where it began. A fit that
takes, on each ray, only the edge nearest the model stays wherever it began,
and on the gym footage it began on the outline, tread and all, 5 to 8 px off
the face at rest and at the top.

    sleeve_end    the end of the sleeve near a guess of the centre
    rim_edges     every edge along the rays, at the middle of its transition
    fit_face      the face's ellipse on those edges
    face_colour   the face's colour and how even it is, to know the plate again
    measure       all of it for one frame
"""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

RAYS = 180
STEP = 0.5                  # px between samples along a ray
INNER, OUTER = 0.12, 0.20   # the rim is looked for this far inside the radius and this far outside it
MARGIN = 12                 # px beyond the outermost ray's end that the region read round a guess reaches
SAME_GUESS = 1.0            # px: a guess this near one already fitted is not fitted again
SMOOTH = 1.0                # samples; a Gaussian along each ray before the gradient
MIN_EDGE = 1.0              # Lab units (L 0 to 100) per pixel: weaker steps are texture
KEEP = 0.15                 # and so are steps weaker than this share of the strongest on their ray
REL = 0.3                   # an edge's transition is where its gradient is above this share of its peak
SIGMAS = (3.0, 2.0, 1.5, 1.2, 1.0)     # px; the fit starts loose and ends tight
WIDE_SIGMAS = (8.0, 5.0) + SIGMAS      # for a guess that may be well off
ITERATIONS = 6
CONVERGED = 1e-3            # px: a step smaller than this ends a sigma's iterations
PRUNE_SIGMA, PRUNE = 3.0, 5.0   # from this sigma on, edges further than PRUNE sigmas off are dropped
OUTLIER = 2.5               # sigmas: the level at which a ray's edges are taken for none of them the rim
INLIER = 2.0                # px from the face: an edge on it, for counting
ZONE = 0.3                  # free band beyond the face, on the tread's side, of the sleeve end's offset
RADIUS_WEIGHT = 20.0        # pull of the expected radius, in rays' worth

CAP = 0.12                  # radius of the sleeve's end, of the plate's: 50 mm on 450, a little nearer the camera
CAP_REACH = 0.65            # how far from the centre the end is looked for, of the radius
CAP_NEAR = 0.25             # and, once its offset is known, how far from where it should be

CIRCLE = (1.0, 0.0)         # a face's shape: axis ratio and direction of the major axis, radians


@dataclass
class Face:
    """The front face's shape, in units of its semi-major axis: an ellipse of
    full axes major (2.0) and minor, its major axis at angle_deg. Its centre is
    the end of the bar, its major axis the plate's true 450 mm, its axis ratio
    the camera angle."""
    major: float = 2.0
    minor: float = 2.0
    angle_deg: float = 90.0
    corrected: bool = False   # True: the bar path is redrawn as if the camera stood square

    @property
    def ratio(self) -> float:
        return self.minor / self.major

    @property
    def camera_angle_deg(self) -> float:
        return math.degrees(math.acos(min(1.0, self.ratio)))


@dataclass
class FaceFit:
    x: float                # centre of the face, image pixels
    y: float
    r: float                # semi-major axis, pixels
    rays: int               # rays with an edge on the face (or in the free band)
    sectors: int            # of 12 directions round the face with such a ray
    cost: float             # how much of the rim the face leaves unexplained, in rays
    rms: float              # of the edges on it, pixels
    points: np.ndarray = field(repr=False, default=None)   # nearest edge per ray and its distance from the face
    sleeve: tuple = None    # the sleeve's end, (x, y), if found


def _lab(bgr):
    """float BGR (0 to 255) to Lab, L 0 to 100; NaN where any channel is NaN."""
    bad = np.isnan(bgr).any(axis=-1)
    lab = cv2.cvtColor(np.where(bad[..., None], 0.0, bgr * (1.0 / 255.0)).astype(np.float32), cv2.COLOR_BGR2LAB)
    lab[bad] = np.nan
    return lab


def sample(roi, xs, ys):
    """Lab of a float BGR region at the points xs, ys (float32); NaN outside it."""
    v = cv2.remap(roi, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(np.nan,) * 3)
    return _lab(v)


def silver(lab):
    """Light and without colour: the steel of the sleeve, where plates are coloured or dark."""
    return lab[..., 0] - np.hypot(lab[..., 1], lab[..., 2])


SECTORS = 6                 # the ring round the sleeve's end, in this many parts, to check it is darker all round
PEAKS = 4                   # blobs checked, strongest first


def _blob_kernels(rc):
    """Kernels: the disc's mean less the ring's, then the mean of each part of
    the ring less the disc's; and the cover of disc and ring together."""
    k = int(math.ceil(1.6 * rc))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    d = np.hypot(xx, yy)
    disc = (d <= rc).astype(np.float32)
    ring = (d > 1.15 * rc) & (d <= 1.6 * rc)
    part = (np.floor((np.arctan2(yy, xx) + math.pi) / (2 * math.pi) * SECTORS).astype(int)) % SECTORS
    kernels = [disc / disc.sum() - ring / ring.sum()]
    for j in range(SECTORS):
        m = (ring & (part == j)).astype(np.float32)
        kernels.append(m / m.sum() - disc / disc.sum())
    cover = disc + ring
    return np.stack(kernels), cover / cover.sum(), k


def sleeve_end(roi, c, R, reach=CAP_REACH):
    """The sleeve's end near c in a float BGR region: (x, y, score), or None.

    The strongest blob of the end's size in silver within reach of c: the
    mean over a disc less the mean over the ring round it. A blob counts only
    if the disc is lighter than nearly all of the ring: the tube of the sleeve
    may run into one part of it, but the edge of a steel hub, which also has
    light inside and dark outside, has light on half of it. Found at half
    resolution, then the response is taken at full resolution round it and
    refined to a fraction of a pixel by a parabola either way.
    """
    rc = CAP * R
    h = int(math.ceil((reach + 2 * CAP) * R))
    x0, y0 = int(round(c[0])) - h, int(round(c[1])) - h
    H, W = roi.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + 2 * h + 1), min(H, y0 + 2 * h + 1)
    if xb - xa < 4 or yb - ya < 4:
        return None
    n = 2 * h + 1
    s = np.zeros((n, n), np.float32)
    s[ya - y0:yb - y0, xa - x0:xb - x0] = silver(_lab(roi[ya:yb, xa:xb]))
    partial = xa > x0 or ya > y0 or xb < x0 + n or yb < y0 + n
    valid = np.zeros_like(s)
    valid[ya - y0:yb - y0, xa - x0:xb - x0] = 1

    # half resolution: where the end is, to a pixel or two
    s2 = cv2.resize(s, (n // 2, n // 2), interpolation=cv2.INTER_AREA)
    kerns2, cover2, _ = _blob_kernels(rc / 2)
    resp = cv2.filter2D(s2, -1, kerns2[0], borderType=cv2.BORDER_CONSTANT)
    if partial:
        v2 = cv2.resize(valid, (n // 2, n // 2), interpolation=cv2.INTER_AREA)
        resp[cv2.filter2D(v2, -1, cover2, borderType=cv2.BORDER_CONSTANT) < 0.97] = -1e9
    gy, gx = np.mgrid[0:resp.shape[0], 0:resp.shape[1]]
    centre2 = ((c[0] - x0 - 0.5) / 2, (c[1] - y0 - 0.5) / 2)
    resp[np.hypot(gx - centre2[0], gy - centre2[1]) > reach * R / 2] = -1e9
    peak = (resp > 0) & (resp >= cv2.dilate(resp, np.ones((5, 5), np.uint8)))
    ys, xs = np.nonzero(peak)
    order = np.argsort(-resp[ys, xs])[:PEAKS]
    sec = [cv2.filter2D(s2, -1, kk, borderType=cv2.BORDER_CONSTANT) for kk in kerns2[1:]]
    found = None
    for j in order:
        darker = sum(float(m[ys[j], xs[j]]) < 0 for m in sec)
        if darker >= SECTORS - 1:
            found = (ys[j], xs[j])
            break
    if found is None:
        return None
    px, py = 2 * found[1] + 1, 2 * found[0] + 1    # the middle of the half-resolution pixel, roughly

    # full resolution, round it
    kerns, cover, k = _blob_kernels(rc)
    m = 3
    lo_x, lo_y = max(k, px - m), max(k, py - m)
    hi_x, hi_y = min(n - k - 1, px + m), min(n - k - 1, py + m)
    if hi_x < lo_x or hi_y < lo_y:
        return None
    views = np.lib.stride_tricks.sliding_window_view(s[lo_y - k:hi_y + k + 1, lo_x - k:hi_x + k + 1], cover.shape)
    full = np.einsum("ijkl,kl->ij", views, kerns[0])
    if partial:
        vv = np.lib.stride_tricks.sliding_window_view(valid[lo_y - k:hi_y + k + 1, lo_x - k:hi_x + k + 1], cover.shape)
        full[np.einsum("ijkl,kl->ij", vv, cover) < 0.99] = -1e9
    wy, wx = np.mgrid[lo_y:hi_y + 1, lo_x:hi_x + 1]
    full[np.hypot(wx - (c[0] - x0), wy - (c[1] - y0)) > reach * R] = -1e9
    q = int(np.argmax(full))
    qy, qx = divmod(q, full.shape[1])
    if full[qy, qx] <= 0:
        return None

    def sub(a, b, cc):
        dd = a - 2 * b + cc
        return 0.5 * (a - cc) / dd if abs(dd) > 1e-9 else 0.0

    fx, fy = float(qx), float(qy)
    if 0 < qx < full.shape[1] - 1:
        fx += sub(full[qy, qx - 1], full[qy, qx], full[qy, qx + 1])
    if 0 < qy < full.shape[0] - 1:
        fy += sub(full[qy - 1, qx], full[qy, qx], full[qy + 1, qx])
    return fx + lo_x + x0, fy + lo_y + y0, float(full[qy, qx])


def rim_edges(v, radii):
    """Every edge along each ray of v (rays, samples, 3 Lab): radii (rays, K), NaN-padded, and strengths.

    The gradient is the length of the change in Lab, so a red plate against
    skin counts as much as a black one against a white wall. An edge is a
    peak of it, and it is placed at the middle of its transition, the
    centroid of the gradient above REL of the peak between the minima either
    side: for an edge smeared by motion, where it was at mid-exposure.
    """
    rows, n = v.shape[:2]
    h = int(math.ceil(3 * SMOOTH))
    k = np.exp(-0.5 * (np.arange(-h, h + 1) / SMOOTH) ** 2)
    k /= k.sum()
    vv = cv2.filter2D(np.nan_to_num(v, nan=-1e3), -1, k[None, :], borderType=cv2.BORDER_REPLICATE)
    g = np.zeros((rows, n), np.float32)
    g[:, 1:-1] = np.linalg.norm(vv[:, 2:] - vv[:, :-2], axis=2) / (2 * STEP)
    bad = np.isnan(v).any(axis=2)
    if bad.any():
        g[cv2.dilate(bad.astype(np.uint8), np.ones((1, 2 * h + 3), np.uint8)) > 0] = 0
    floor = np.maximum(MIN_EDGE, KEEP * g.max(axis=1, keepdims=True))
    peak = np.zeros((rows, n), bool)
    peak[:, 1:-1] = (g[:, 1:-1] > g[:, :-2]) & (g[:, 1:-1] >= g[:, 2:]) & (g[:, 1:-1] > floor)
    low = np.zeros((rows, n), bool)
    low[:, 1:-1] = (g[:, 1:-1] <= g[:, :-2]) & (g[:, 1:-1] < g[:, 2:])

    # Each minimum starts a segment of its ray; an edge's transition is its
    # segment and the minimum that ends it, the first sample of the next.
    seg = (np.cumsum(low, axis=1) + (np.arange(rows) * (n + 1))[:, None]).ravel()
    fg = g.ravel()
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    ends = np.r_[starts[1:], fg.size]
    nxt = np.minimum(ends, fg.size - 1)
    right = (ends < fg.size) & (nxt // n == starts // n)
    g_right = np.where(right, fg[nxt], 0.0)
    top = np.maximum(np.maximum.reduceat(fg, starts), np.where(right, g_right, -np.inf))
    has_peak = np.add.reduceat(peak.ravel().astype(np.int32), starts) > 0
    fr = np.tile(radii.astype(np.float32), rows)
    w = np.maximum(fg - np.repeat(REL * top, ends - starts), 0.0)
    w_right = np.where(right, np.maximum(g_right - REL * top, 0.0), 0.0)
    sw = np.add.reduceat(w, starts) + w_right
    swr = np.add.reduceat(w * fr, starts) + w_right * np.where(right, fr[nxt], 0.0)
    keep = has_peak & (sw > 0)
    rc, st, ray = swr[keep] / sw[keep], top[keep], starts[keep] // n
    counts = np.bincount(ray, minlength=rows)
    K = max(1, int(counts.max()) if counts.size else 1)
    rank = np.arange(ray.size) - np.repeat(np.cumsum(counts) - counts, counts)
    rad = np.full((rows, K), np.nan)
    strength = np.full((rows, K), np.nan)
    rad[ray, rank] = rc
    strength[ray, rank] = st
    return rad, strength


def _shape(phi, shape):
    """The face's radius at the angles phi, in units of its semi-major axis, and its derivative."""
    q, th = shape
    if q >= 1.0:
        return np.ones_like(phi), np.zeros_like(phi)
    c, s = np.cos(phi - th), np.sin(phi - th)
    den = np.sqrt(q * q * c * c + s * s)
    return q / den, -q * (1 - q * q) * c * s / den ** 3


def fit_face(c0, phi, cands, start, sleeve=None, zone=0.0, r_prior=None, shape=CIRCLE, sigmas=SIGMAS):
    """The face on the edges of rays cast from c0 at the angles phi: a FaceFit (local pixels), or None.

    Every candidate on a ray counts, weighed by a Gaussian of its distance from
    the face against an outlier level, and the weights are refound as the face
    moves, with sigma shrinking from loose to tight. With sleeve, on the side
    away from it edges up to zone times its offset beyond the face count as on
    it. r_prior (R, w) holds the semi-major axis near R with the weight of w rays.
    """
    rad, _ = cands
    has = ~np.isnan(rad).all(axis=1)
    rad, ph = rad[has], phi[has]
    if len(rad) < 10:
        return None
    ray, col = np.nonzero(~np.isnan(rad))          # every candidate, ray by ray
    r_ = rad[ray, col]
    PX = c0[0] + r_ * np.cos(ph)[ray]
    PY = c0[1] + r_ * np.sin(ph)[ray]
    cx, cy, R = (float(v) for v in start)
    floor = math.exp(-0.5 * OUTLIER ** 2)
    circle = shape[0] >= 1.0

    def residuals(px, py, cx, cy, R):
        dx, dy = px - cx, py - cy
        d = np.hypot(dx, dy)
        d[d < 1e-9] = 1e-9
        ux, uy = dx / d, dy / d
        if circle:
            rho, drho = 1.0, 0.0
        else:
            rho, drho = _shape(np.arctan2(dy, dx), shape)
        raw = d - R * rho
        res = raw
        if sleeve is not None and zone > 0:
            ox, oy = cx - sleeve[0], cy - sleeve[1]
            res = np.where(raw < 0, raw, np.maximum(0.0, raw - np.maximum(0.0, zone * (ux * ox + uy * oy))))
        return res, raw, d, ux, uy, rho, drho

    px, py, rr = PX, PY, ray
    pruned = False
    prior = r_prior is not None
    for sg in sigmas:
        if not pruned and sg <= PRUNE_SIGMA:
            # edges this far off weigh nothing from here on
            near = np.abs(residuals(px, py, cx, cy, R)[0]) < PRUNE * sg
            px, py, rr = px[near], py[near], rr[near]
            pruned = True
            if len(px) < 10:
                return None
        k = -0.5 / (sg * sg)
        for _ in range(ITERATIONS):
            res, _, d, ux, uy, rho, drho = residuals(px, py, cx, cy, R)
            g = np.exp(k * res * res)
            w = g / (np.bincount(rr, weights=g, minlength=len(ph)) + floor)[rr]
            if circle:
                J = np.stack([-ux, -uy, -np.ones_like(ux)])
            else:
                J = np.stack([-ux - R * drho * uy / d, -uy + R * drho * ux / d, -rho])
            Jw = J * w
            N = Jw @ J.T
            gv = -(Jw @ res)
            if prior:
                N[2, 2] += r_prior[1]
                gv[2] -= r_prior[1] * (R - r_prior[0])
            try:
                step = np.linalg.solve(N, gv)
            except np.linalg.LinAlgError:
                return None
            cx, cy, R = cx + step[0], cy + step[1], R + step[2]
            if not (np.isfinite(R) and R > 0):
                return None
            if np.abs(step).max() < CONVERGED:
                break
    res, raw, *_ = residuals(PX, PY, cx, cy, R)
    starts = np.flatnonzero(np.r_[True, ray[1:] != ray[:-1]])
    best = np.minimum.reduceat(np.abs(res), starts)
    inl = best < INLIER
    sectors = len(np.unique((np.mod(ph[inl], 2 * math.pi) / (2 * math.pi) * 12).astype(int) % 12))
    u = np.clip(best / INLIER, 0, 1)
    cost = float((1 - (1 - u * u) ** 2).sum())
    order = np.lexsort((np.abs(raw), ray))
    first = order[starts]
    points = np.column_stack([PX[first], PY[first], raw[first]])
    return FaceFit(cx, cy, R, int(inl.sum()), sectors, cost,
                   float(np.sqrt((best[inl] ** 2).mean())) if inl.any() else 99.0, points)


def face_colour(roi, c, R, inner=0.6, outer=0.88, directions=180):
    """Median Lab over a ring of the face, where neither the hub nor the rim is,
    and the robust spread (1.4826 MAD) of each channel there: (6,), or None."""
    phi = 2 * math.pi * np.arange(directions) / directions
    f = np.linspace(inner, outer, 4)[:, None]
    xs = (c[0] + f * R * np.cos(phi)).astype(np.float32)
    ys = (c[1] + f * R * np.sin(phi)).astype(np.float32)
    lab = sample(roi, xs, ys).reshape(-1, 3)
    lab = lab[~np.isnan(lab).any(axis=1)]
    if len(lab) < 50:
        return None
    med = np.median(lab, axis=0)
    return np.concatenate([med, 1.4826 * np.median(np.abs(lab - med), axis=0)])


def measure(image, origin, guides, R, sleeve_offset=None, shape=CIRCLE, sigmas=SIGMAS, r_weight=None,
            colour=False):
    """The face in one frame, from guesses of its centre: (FaceFit or None, colour or None).

    image is the whole frame or a crop of it whose top-left corner is at
    origin; guides and everything returned are in full-frame pixels. The
    sleeve's end is looked for round the first guess, near where
    sleeve_offset puts it if that is known, and the centre it gives is
    tried first. Of the fits from the guesses the one that leaves least of
    the rim unexplained is kept. colour: also the face's colour round it,
    see face_colour.
    """
    ox, oy = origin
    gx, gy = guides[0][0] - ox, guides[0][1] - oy
    h = int((1 + OUTER) * R + MARGIN)
    H, W = image.shape[:2]
    x0, y0 = max(0, int(gx) - h), max(0, int(gy) - h)
    x1, y1 = min(W, int(gx) + h + 1), min(H, int(gy) + h + 1)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None, None
    roi = image[y0:y1, x0:x1].astype(np.float32)
    if sleeve_offset is not None:
        cap = sleeve_end(roi, (gx + sleeve_offset[0] - x0, gy + sleeve_offset[1] - y0), R, CAP_NEAR)
    else:
        cap = sleeve_end(roi, (gx - x0, gy - y0), R)
    tried = [(g[0] - ox - x0, g[1] - oy - y0) for g in guides]
    if cap is not None and sleeve_offset is not None:
        tried.insert(0, (cap[0] - sleeve_offset[0], cap[1] - sleeve_offset[1]))
    phi = 2 * math.pi * np.arange(RAYS) / RAYS
    radii = np.arange((1 - INNER) * R, (1 + OUTER) * R, STEP)
    edges = {}
    best = None
    done = []
    for gl in tried:
        if any(math.hypot(gl[0] - p[0], gl[1] - p[1]) < SAME_GUESS for p in done):
            continue
        done.append(gl)
        key = (round(gl[0]), round(gl[1]))
        if key not in edges:
            xs = (key[0] + np.outer(np.cos(phi), radii)).astype(np.float32)
            ys = (key[1] + np.outer(np.sin(phi), radii)).astype(np.float32)
            edges[key] = rim_edges(sample(roi, xs, ys), radii)
        f = fit_face(key, phi, edges[key], (gl[0], gl[1], R), sleeve=None if cap is None else cap[:2],
                     zone=ZONE if cap is not None else 0.0, r_prior=(R, r_weight or RADIUS_WEIGHT), shape=shape,
                     sigmas=sigmas)
        if f is not None and (best is None or f.cost < best.cost):
            best = f
    if best is None:
        return None, None
    col = face_colour(roi, (best.x, best.y), best.r) if colour else None
    dx, dy = x0 + ox, y0 + oy
    best.x += dx
    best.y += dy
    best.points = best.points + [dx, dy, 0.0]
    if cap is not None:
        best.sleeve = (cap[0] + dx, cap[1] + dy)
    return best, col
