"""Bar path in millimetres, and its velocity.

The plate is its own ruler in every frame. Its face is a circle 450 mm across;
seen at an angle it is an ellipse, and the map that turns that ellipse back
into a circle of the right size turns the plane the bar moves in into
millimetres:

    stretch across the minor axis by 1 / ratio   undoes the camera angle
    rotate so the image's vertical stays vertical  gravity is the reference
    multiply by 450 / major axis                   pixels to millimetres

The scale is taken per frame, not once per set. The plate's size in pixels
changes when the lifter moves along the bar towards or away from the camera,
or when a phone is held by hand; on the second test clip it varied by 7 %.
Positions are measured from the image centre, where a change of distance
moves nothing, so a plate that comes closer grows round that point rather than
appearing to slide across the frame. The image centre stands in for the lens
axis, which is close enough for phone and Pi cameras.

Only positions in the plane the bar moves in are right. The sleeve end and the
far plate sit at other distances and have other scales.
"""

import math

import numpy as np


def plane_matrix(face) -> np.ndarray:
    """2x2 map from image offsets, in pixels of the face's own scale, to the bar's plane.

    Rows are (forward, up); the result is in units of the face scale, so multiply
    by the millimetres per pixel of the frame to get millimetres. y in the
    image grows downwards and 'up' is returned positive.
    """
    t = math.radians(face.angle_deg)
    major = np.array([math.cos(t), math.sin(t)])
    minor = np.array([-math.sin(t), math.cos(t)])
    stretch = np.outer(major, major) + np.outer(minor, minor) / face.ratio
    down = stretch @ np.array([0.0, 1.0])
    a = math.atan2(down[0], down[1])          # angle of the image's down direction after stretching
    c, s = math.cos(a), math.sin(a)
    rotate = np.array([[c, -s], [s, c]])        # puts it back to straight down
    to_plane = rotate @ stretch
    flip = np.array([[1.0, 0.0], [0.0, -1.0]])  # image down -> plane up
    return flip @ to_plane


def bar_path(measurements, face, frame_size):
    """Times (s) and positions (mm, forward and up) of every accepted frame.

    Origin: the image centre, at the median distance of the set. A change of
    scale during the set is applied round the centre, as a change of distance
    would be.
    """
    acc = [m for m in measurements if m is not None and m.accepted]
    if not acc:
        return np.zeros(0), np.zeros((0, 2))
    w, h = frame_size
    centre = np.array([w / 2.0, h / 2.0])
    M = plane_matrix(face)
    t = np.array([m.t for m in acc])
    pos = np.array([[m.face_x, m.face_y] for m in acc]) - centre
    k = np.array([m.mm_per_px for m in acc])
    return t, (pos @ M.T) * k[:, None]


def velocity(t, pos, half_window_s=0.04):
    """Velocity by a local quadratic fit over neighbours within the window.

    The window is in seconds, so missing frames do not change what it means,
    but it never holds fewer than two frames either side: at 30 fps 40 ms
    would be one. Frames with too few neighbours get NaN.
    """
    t = np.asarray(t, float)
    pos = np.asarray(pos, float)
    v = np.full(pos.shape, np.nan)
    if len(t) > 1:
        half_window_s = max(half_window_s, 2.2 * float(np.median(np.diff(t))))
    for i in range(len(t)):
        sel = np.abs(t - t[i]) <= half_window_s
        if sel.sum() < 4:
            continue
        dt = t[sel] - t[i]
        A = np.column_stack([np.ones_like(dt), dt, dt * dt])
        coef, *_ = np.linalg.lstsq(A, pos[sel], rcond=None)
        v[i] = coef[1]
    return v

