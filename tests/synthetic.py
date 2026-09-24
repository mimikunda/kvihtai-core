"""Synthetic plates for the tests: drawn at sub-pixel precision, any colour.

Shapes are drawn at SUPER times the resolution and averaged down, so an edge
lands between pixels the way a real one does and a centre can be placed at a
fraction of a pixel. That is what lets the tests ask for sub-pixel accuracy.
"""

import math

import cv2
import numpy as np

SUPER = 8
SHIFT = 4                      # fixed-point bits for cv2 drawing

RED = (40, 40, 200)
BLUE = (170, 70, 30)
BLACK = (32, 30, 30)
YELLOW = (40, 200, 220)
GREEN = (60, 160, 40)
HUB = (150, 150, 150)
SLEEVE = (210, 210, 210)


def _fx(v):
    """A length in fixed point at the drawing resolution."""
    return int(round(v * SUPER * (1 << SHIFT)))


def _at(v):
    """A position in fixed point at the drawing resolution.

    Pixel i of the final image averages drawing pixels 8i to 8i + 7, whose
    middle is 8i + 3.5, so a position is not simply scaled up.
    """
    return int(round(((v + 0.5) * SUPER - 0.5) * (1 << SHIFT)))


def textured_background(size, seed=0, base=(90, 110, 130)):
    """A mottled, not flat, background: flat backgrounds make edge finding too easy."""
    rng = np.random.default_rng(seed)
    w, h = size
    noise = rng.normal(0, 1, (h // 8 + 1, w // 8 + 1, 3)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    img = np.clip(np.array(base, np.float32) + 18 * noise, 0, 255)
    return img.astype(np.uint8)


def _region(image, cx, cy, reach):
    """Integer box round a shape, clipped to the image, and the shape centre inside it."""
    h, w = image.shape[:2]
    x0, y0 = max(0, int(cx - reach)), max(0, int(cy - reach))
    x1, y1 = min(w, int(cx + reach) + 2), min(h, int(cy + reach) + 2)
    return x0, y0, x1, y1


def draw_plate(image, centre, radius, face=BLACK, tread=None, tread_colour=None,
               ratio=1.0, angle_deg=90.0, hub=HUB):
    """A plate: tread (if any), face, hub and sleeve end, on image, in place.

    tread is the image offset of the far face from the near one, in pixels.
    ratio and angle_deg shape the face as an ellipse with the major axis at
    angle_deg.
    """
    reach = radius + 4 + (0 if tread is None else math.hypot(*tread))
    x0, y0, x1, y1 = _region(image, centre[0], centre[1], reach)
    region = image[y0:y1, x0:x1]
    centre = (centre[0] - x0, centre[1] - y0)
    big = cv2.resize(region, ((x1 - x0) * SUPER, (y1 - y0) * SUPER), interpolation=cv2.INTER_NEAREST)
    axes = (_fx(radius), _fx(radius * ratio))
    c = (_at(centre[0]), _at(centre[1]))
    if tread is not None:
        tc = tread_colour or tuple(int(v * 0.8) for v in face)
        steps = 12
        for k in range(steps, 0, -1):
            f = k / steps
            cc = (_at(centre[0] + f * tread[0]), _at(centre[1] + f * tread[1]))
            cv2.ellipse(big, cc, axes, angle_deg, 0, 360, tc, -1, cv2.LINE_8, SHIFT)
    cv2.ellipse(big, c, axes, angle_deg, 0, 360, face, -1, cv2.LINE_8, SHIFT)
    cv2.ellipse(big, c, (_fx(radius * 0.38), _fx(radius * 0.38 * ratio)), angle_deg, 0, 360,
                hub, -1, cv2.LINE_8, SHIFT)
    cv2.ellipse(big, c, (_fx(radius * 0.11), _fx(radius * 0.11 * ratio)), angle_deg, 0, 360,
                SLEEVE, -1, cv2.LINE_8, SHIFT)
    region[:] = cv2.resize(big, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    return image


def draw_ring(image, centre, radius, colour=(200, 200, 200), thickness=3.0):
    """A thin ring with the background inside: round, but not a plate."""
    x0, y0, x1, y1 = _region(image, centre[0], centre[1], radius + thickness + 2)
    region = image[y0:y1, x0:x1]
    big = cv2.resize(region, ((x1 - x0) * SUPER, (y1 - y0) * SUPER), interpolation=cv2.INTER_NEAREST)
    cv2.circle(big, (_at(centre[0] - x0), _at(centre[1] - y0)), _fx(radius), colour,
               int(thickness * SUPER), cv2.LINE_8, SHIFT)
    region[:] = cv2.resize(big, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    return image


def rising_clip(n=40, size=(320, 480), radius=50.0, face=BLACK, fps=60.0, tread=None,
                distractor=True, seed=1, ratio=1.0):
    """Frames of a plate rising at constant speed, and its true centres."""
    frames, centres = [], []
    for i in range(n):
        img = textured_background(size, seed)
        if distractor:
            draw_ring(img, (250.0, 90.0), radius)
        c = (140.0 + 0.3 * i, 380.0 - 5.5 * i)
        draw_plate(img, c, radius, face=face, tread=tread, ratio=ratio)
        frames.append(img)
        centres.append(c)
    return frames, np.arange(n) / fps, centres


def write_video(path, frames, fps):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise OSError("no mp4 writer available")
    for f in frames:
        writer.write(f)
    writer.release()


def circle_points(cx, cy, r, count=180, arc=2 * math.pi, start=0.0, ratio=1.0, angle_deg=0.0):
    t = math.radians(angle_deg)
    out = []
    for k in range(count):
        phi = start + arc * k / count
        x, y = r * math.cos(phi), r * ratio * math.sin(phi)
        out.append((cx + x * math.cos(t) - y * math.sin(t), cy + x * math.sin(t) + y * math.cos(t)))
    return np.array(out)
