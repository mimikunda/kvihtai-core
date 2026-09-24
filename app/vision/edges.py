"""Edges without a reference colour.

Plates come red, blue, yellow, green, black and grey, and the same plate
changes colour as it moves through the light: the black plate in the second
test clip turns bright blue at the bottom of the squat. So nothing here asks
what colour the plate is. An edge is wherever the colour changes, measured as
the length of the change in Lab space, which weighs a red plate against skin as
much as a black plate against a white wall.
"""

import math

import cv2
import numpy as np

RAYS = 180
STEP = 0.25        # sampling step along a ray, pixels
MIN_GRAD = 6.0     # Lab units per pixel; below this a change is texture, not a rim
MAX_EDGES = 3      # per ray: the silhouette, the face edge where the tread shows, and one spare


def to_lab(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def colour_gradient(bgr: np.ndarray):
    """Sobel gradient of whichever Lab channel changes most at each pixel.

    Returns gx, gy and the magnitude, float32. The sign of the gradient is
    kept, but callers should not rely on it: whether the plate is darker or
    lighter than what is behind it changes round the rim.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gx = [cv2.Sobel(lab[:, :, c], cv2.CV_32F, 1, 0, ksize=3) for c in range(3)]
    gy = [cv2.Sobel(lab[:, :, c], cv2.CV_32F, 0, 1, ksize=3) for c in range(3)]
    m = [cv2.magnitude(gx[c], gy[c]) for c in range(3)]
    use1 = m[1] > m[0]
    bx = np.where(use1, gx[1], gx[0])
    by = np.where(use1, gy[1], gy[0])
    bm = np.maximum(m[0], m[1])
    use2 = m[2] > bm
    return np.where(use2, gx[2], bx), np.where(use2, gy[2], by), np.maximum(bm, m[2])


def rim_patch(image: np.ndarray, centre, reach: float, blur: float = 1.0):
    """The square of half-size reach round centre, lightly blurred, and its origin.

    The patch stays BGR, scaled to [0, 1]: ray_edges samples it and converts
    only the samples to Lab, which is a fraction of the pixels. The blur suppresses
    sensor noise without moving an edge, since a symmetric blur leaves the
    middle of a step where it was.
    """
    h, w = image.shape[:2]
    half = int(math.ceil(reach))
    x0, y0 = max(0, int(centre[0]) - half), max(0, int(centre[1]) - half)
    x1, y1 = min(w, int(centre[0]) + half + 1), min(h, int(centre[1]) + half + 1)
    if x1 <= x0 or y1 <= y0:
        return None, np.zeros(2)
    patch = image[y0:y1, x0:x1].astype(np.float32) * (1.0 / 255.0)
    if blur:
        # in floating point: blurred and rounded back to 8 bits, the steps of
        # the rounding show in the sub-pixel peak and double the jitter
        patch = cv2.GaussianBlur(patch, (0, 0), blur)
    return patch, np.array([x0, y0], np.float64)


def ray_edges(patch, origin, centre, scale, inner, outer, rays=RAYS, step=STEP,
              min_grad=MIN_GRAD, max_edges=MAX_EDGES):
    """Edges crossed by rays cast outwards from centre.

    Rays run from inner * scale to outer * scale. Along each one the colour
    gradient is the Lab distance between samples half a pixel either side, and
    every local maximum above min_grad is an edge, refined to sub-pixel by a
    parabola through the peak. A ray may cross several edges: where the tread
    of the plate is in view it crosses the face edge and then the silhouette,
    and choosing between them is left to the outline fit, which knows the
    shape.

    Returns points (N, 2) in image coordinates, their gradient strength (N,),
    and the index of the ray each came from (N,).
    """
    empty = (np.zeros((0, 2)), np.zeros(0), np.zeros(0, int))
    if patch is None:
        return empty
    ang = 2.0 * math.pi * np.arange(rays) / rays
    cos, sin = np.cos(ang), np.sin(ang)
    radii = np.arange(inner * scale, outer * scale, step)
    k = max(1, int(round(0.5 / step)))
    if radii.size < 2 * k + 3:
        return empty
    lx, ly = centre[0] - origin[0], centre[1] - origin[1]
    xs = (lx + cos[:, None] * radii[None, :]).astype(np.float32)
    ys = (ly + sin[:, None] * radii[None, :]).astype(np.float32)
    prof = cv2.remap(patch, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    prof = cv2.cvtColor(prof, cv2.COLOR_BGR2LAB)
    prof[:, :, 0] *= 255.0 / 100.0          # same units as 8-bit Lab, for min_grad
    g = np.linalg.norm(prof[:, 2 * k:, :] - prof[:, :-2 * k, :], axis=2) / (2 * k * step)

    peak = np.zeros(g.shape, bool)
    peak[:, 1:-1] = (g[:, 1:-1] > g[:, :-2]) & (g[:, 1:-1] >= g[:, 2:]) & (g[:, 1:-1] > min_grad)
    score = np.where(peak, g, -np.inf)
    m = min(max_edges, score.shape[1])
    order = np.argpartition(-score, m - 1, axis=1)[:, :m]
    ray, slot = np.nonzero(np.isfinite(np.take_along_axis(score, order, 1)))
    if ray.size == 0:
        return empty
    j = order[ray, slot]
    y0, y1, y2 = g[ray, j - 1], g[ray, j], g[ray, j + 1]
    den = y0 - 2.0 * y1 + y2
    shift = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
    r = radii[j + k] + shift * step
    pts = np.column_stack([centre[0] + r * cos[ray], centre[1] + r * sin[ray]])
    return pts, y1, ray
