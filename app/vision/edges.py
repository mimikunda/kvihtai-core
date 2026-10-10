"""Edges without a reference colour.

Plates come red, blue, yellow, green, black and grey, and the same plate
changes colour as it moves through the light: the black plate in the second
test clip turns bright blue at the bottom of the squat. So nothing here asks
what colour the plate is. An edge is wherever the colour changes, measured as
the length of the change in Lab space, which weighs a red plate against skin as
much as a black plate against a white wall.
"""

import cv2
import numpy as np


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
