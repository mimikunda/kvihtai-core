"""Is what was followed a plate at all?

A plate carries its face with it; a ring, a wheel or a loop of rope shows the
room behind it. See inside() and travels_with().
"""

import math

import cv2
import numpy as np

RINGS = 32
DIRECTIONS = 64
INSIDE = (0.15, 0.85)     # the band of the face sampled by inside(), of the radius


def inside(image, centre, scale, rings=RINGS, directions=DIRECTIONS, band=INSIDE):
    """The face sampled in the plate's own frame: lightness, (rings, directions)."""
    fr = np.linspace(band[0], band[1], rings)
    phi = 2 * math.pi * np.arange(directions) / directions
    xs = (centre[0] + np.outer(fr * scale, np.cos(phi))).astype(np.float32)
    ys = (centre[1] + np.outer(fr * scale, np.sin(phi))).astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    v = cv2.remap(lab, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return v[:, :, 0].astype(np.float32)


def travels_with(a, b):
    """Correlation of two inside() samples, in [-1, 1].

    A plate carries its face: sampled in the circle's own frame it is the same
    picture however far the plate has moved. A ring, a wheel or a loop of rope
    shows the room behind it, and the room slides through it as it moves. So
    two frames the object has moved a radius between separate the two, which
    concentricity does not: on real footage a plate with a printed label and a
    lighting gradient across it scores as low as a ring does.
    """
    a = a - a.mean()
    b = b - b.mean()
    return float(a.ravel() @ b.ravel() / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9))
