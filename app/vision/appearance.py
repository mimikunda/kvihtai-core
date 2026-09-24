"""Does the thing inside the fitted outline look like this set's plate?

The outline fit only asks whether edges line up with the outline, and in the
last frames of the first test clip, with the plate dropped out of the picture,
a monitor, a head and an arm lined up well enough. What they do not do is look
like the plate: hub, lettering band and face in rings of known size.

So every fitted frame is summarised by its lightness as a function of radius,
taken as a median round the circle so that lettering, rotation and a hand
across the face barely move it. The set's own median profile is the
reference. Correlation, not difference, because the same plate changes
brightness as it moves through the light. On the test clips the real plate
scores 0.29 and above and the wrong fits score -0.57 and below.
"""

import math

import cv2
import numpy as np

RINGS = 32
DIRECTIONS = 64


def radial_profile(image, centre, scale, outline, rings=RINGS, directions=DIRECTIONS):
    """Median Lab value at each fraction of the radius, (rings, 3)."""
    phi = 2 * math.pi * np.arange(directions) / directions
    rho, _ = outline.at(phi)
    fr = (np.arange(rings) + 0.5) / rings * 0.92
    xs = centre[0] + np.outer(fr, scale * rho * np.cos(phi))
    ys = centre[1] + np.outer(fr, scale * rho * np.sin(phi))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    v = cv2.remap(lab, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    return np.median(v, axis=1)


def similarity(profile, reference):
    """Correlation of the lightness profiles, in [-1, 1]."""
    a = profile[:, 0] - profile[:, 0].mean()
    b = reference[:, 0] - reference[:, 0].mean()
    return float(a @ b / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9))
