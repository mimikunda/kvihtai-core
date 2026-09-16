#!/usr/bin/env python3
"""Track the weight plate through recorded footage and write its trajectory.

The plate is a circle of known diameter. Seen from an angle it projects to an
ellipse whose major axis stays the true diameter and whose minor axis is
shortened by the cosine of that angle. Both facts are used here: the major axis
gives the scale in millimetres per pixel, and the ratio gives the angle, which
is what lets the bar path be redrawn as if the camera had stood square to the
lifter.

Three properties are treated as constants of a set rather than measurements of
a frame, because the camera does not move and the plate does not change size:

    the direction of the major axis
    the ratio between the axes
    the plate diameter in pixels

Estimating them once, from every frame at once, and then fitting only the
position per frame is what makes the result stable. A free per-frame ellipse
has five parameters, and since the plate is nearly circular in a side view its
orientation is decided by noise, which then leaks into the ratio.

The rim itself is found by casting rays outwards from the centre and locating
the colour step along each one, refined to sub-pixel by a parabola. A colour
threshold cannot do this job: it puts the boundary wherever the lighting
crosses the chosen level, and bare skin is close enough to plate red that any
threshold loose enough to catch the whole plate also catches the lifter's hands.

Usage:
    .venv/bin/python tools/track_plate.py FOOTAGE.mp4 --csv trajectory.csv
"""

import argparse
import csv
import math
import os
import statistics
import sys

import cv2
import numpy as np

# --- rim search -------------------------------------------------------------

RAYS = 180
STEP = 0.25            # ray sampling step, in pixels
WIDE = (0.55, 1.45)    # search band before the plate size is known
NARROW = (0.85, 1.15)  # once it is: the far plate's rim is then out of reach
MIN_POINTS = 24        # rays that must find a step before a frame is usable

# --- acceptance -------------------------------------------------------------

MIN_RAY_SHARE = 0.12       # share of rays surviving the outlier trim
# Rim scatter, as a fraction of the plate radius. Measured against a fit whose
# radius is held fixed, which does not minimise the scatter the way a free
# radius does, so this is looser than it would need to be for a free fit.
MAX_RESIDUAL = 0.025
# The diameter is read off the cleanest frames only. Accepting a frame and
# trusting it to define the plate size are different jobs: a frame can be good
# enough to report a position while still being too scattered to set the scale
# that every other frame then inherits.
STRICT_RESIDUAL = 0.020
MAX_SIZE_DRIFT = 0.10      # disagreement with the set diameter
MIN_HUB_CONTRAST = 50.0    # plate hub against plate face, see hub_contrast
MAX_STEP_PX = 45.0         # a plate cannot move further between frames
MAX_STEP_SPAN = 10         # frames the step allowance may accumulate over

# Shape limits for the coarse stage. They only have to rule out things that
# cannot be a plate at all, so they are deliberately loose.
MIN_MAJOR_PX = 90.0
MAX_MAJOR_PX = 480.0
MIN_AXIS_RATIO = 0.30      # cos(72 deg); beyond that a plate is edge-on
MIN_COARSE_FILL = 0.70     # contour area over the area of its fitted ellipse

# Peak bar speed in a snatch is about 2 m/s. At roughly 2.8 mm/px and 60 fps
# that is 12 px between frames, so MAX_STEP_PX leaves a wide margin and still
# rules out a jump to a plate lying on the floor.

RED_RANGES = (
    ((0, 90, 60), (10, 255, 255)),
    ((170, 90, 60), (180, 255, 255)),
)


def redness(bgr):
    """High on the plate, lower on skin, wood and floor. Never thresholded."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    return lab[:, :, 1] * (hsv[:, :, 1] / 255.0)


def _sample(img, xs, ys):
    return cv2.remap(img, xs.astype(np.float32), ys.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()


def rim_points(field, cx, cy, radius, inner, outer):
    """One sub-pixel rim crossing per ray."""
    radii = np.arange(radius * inner, radius * outer, STEP)
    points = []
    for k in range(RAYS):
        theta = 2 * math.pi * k / RAYS
        xs = (cx + radii * math.cos(theta)).reshape(-1, 1)
        ys = (cy + radii * math.sin(theta)).reshape(-1, 1)
        profile = _sample(field, xs, ys)
        if profile.size < 5:
            continue
        gradient = np.diff(profile)
        i = int(np.argmin(gradient))
        if i <= 0 or i >= gradient.size - 1 or -gradient[i] < 1.0:
            continue
        y0, y1, y2 = gradient[i - 1], gradient[i], gradient[i + 1]
        denom = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-6 else 0.0
        r = radii[i] + (shift + 0.5) * STEP
        points.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    return np.array(points, dtype=np.float32)


def hub_contrast(field, cx, cy, radius):
    """How much the middle of a fit looks like a hub rather than plate face.

    A plate has a metal hub, so redness is low at the centre and high across
    the disc. A fit that has settled between the two plates on the bar sits on
    plate material instead and the contrast collapses. Measured on this
    footage, good fits score about 90 and straddling fits score below 20, with
    nothing in between, which is a far sharper separation than the share of
    rays that agreed.
    """
    def band(lo, hi):
        vals = []
        for frac in np.arange(lo, hi, 0.05):
            for k in range(48):
                th = 2 * math.pi * k / 48
                x, y = cx + frac * radius * math.cos(th), cy + frac * radius * math.sin(th)
                if 0 <= x < field.shape[1] and 0 <= y < field.shape[0]:
                    vals.append(field[int(y), int(x)])
        return float(np.median(vals)) if vals else 0.0
    return band(0.45, 0.75) - band(0.05, 0.25)


# --- fitting ----------------------------------------------------------------

def _unsqueeze(points, theta_deg, ratio):
    """Map an ellipse of this shape onto a circle."""
    t = math.radians(theta_deg)
    cos_t, sin_t = math.cos(t), math.sin(t)
    x, y = points[:, 0], points[:, 1]
    along = x * cos_t + y * sin_t
    across = (-x * sin_t + y * cos_t) / ratio
    return np.column_stack([along * cos_t - across * sin_t,
                            along * sin_t + across * cos_t]).astype(np.float64)


def _resqueeze(point, theta_deg, ratio):
    t = math.radians(theta_deg)
    cos_t, sin_t = math.cos(t), math.sin(t)
    x, y = point
    along = x * cos_t + y * sin_t
    across = (-x * sin_t + y * cos_t) * ratio
    return along * cos_t - across * sin_t, along * sin_t + across * cos_t


def _fit_circle(points):
    """Algebraic least squares. Linear, so it has no local minima."""
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    A = np.column_stack([x, y, np.ones_like(x)])
    try:
        sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    return (cx, cy), r, float(np.median(np.abs(np.hypot(x - cx, y - cy) - r)))


def _fit_centre_only(points, radius, iterations=12):
    """Fit the centre when the radius is already known.

    Where the rim is partly covered, a free radius and the centre trade against
    each other and the fitted edge rides off towards the side with more rays.
    Holding the diameter fixed removes that trade, and a partial arc still pins
    the centre.
    """
    c = points.mean(axis=0).astype(np.float64)
    for _ in range(iterations):
        d = points - c
        dist = np.hypot(d[:, 0], d[:, 1])
        dist[dist < 1e-6] = 1e-6
        new = (points - (d.T / dist * radius).T).mean(axis=0)
        if np.hypot(*(new - c)) < 1e-4:
            c = new
            break
        c = new
    d = points - c
    return c, float(np.median(np.abs(np.hypot(d[:, 0], d[:, 1]) - radius)))


def fit_shaped(points, theta_deg, ratio, radius=None):
    """Fit position, and size unless the size is given."""
    circle_pts = _unsqueeze(points, theta_deg, ratio)
    if radius is not None:
        centre, residual = _fit_centre_only(circle_pts, radius)
        r = radius
    else:
        out = _fit_circle(circle_pts)
        if out is None:
            return None
        centre, r, residual = out
    cx, cy = _resqueeze(centre, theta_deg, ratio)
    return {"cx": cx, "cy": cy, "major": 2 * r, "minor": 2 * r * ratio,
            "ratio": ratio, "angle": theta_deg, "residual": residual}


def trim(points, theta_deg, ratio, sigma=2.0, rounds=3):
    """Drop rays that disagree with the common shape, such as those on a hand."""
    kept = points
    for _ in range(rounds):
        if len(kept) < MIN_POINTS:
            return kept
        circle_pts = _unsqueeze(kept, theta_deg, ratio)
        out = _fit_circle(circle_pts)
        if out is None:
            return kept
        (cx, cy), r, _ = out
        d = np.abs(np.hypot(circle_pts[:, 0] - cx, circle_pts[:, 1] - cy) - r)
        keep = d <= sigma * max(float(d.std()), 1e-3)
        if keep.all() or keep.sum() < MIN_POINTS:
            break
        kept = kept[keep]
    return kept


def _score_shape(all_points, theta_deg, ratio):
    scores = []
    for pts in all_points:
        kept = trim(pts, theta_deg, ratio)
        if len(kept) < MIN_POINTS:
            continue
        got = fit_shaped(kept, theta_deg, ratio)
        if got and got["major"] > 0:
            scores.append(got["residual"] / (got["major"] / 2))
    if len(scores) < 5:
        return float("inf")
    return float(np.median(scores))


def search_shape(all_points):
    """Find the one ellipse shape that explains every frame best."""
    best = (float("inf"), 90.0, 1.0)
    for theta in range(0, 180, 5):
        for ratio in np.arange(0.50, 1.001, 0.02):
            s = _score_shape(all_points, float(theta), float(ratio))
            if s < best[0]:
                best = (s, float(theta), float(ratio))
    _, theta0, ratio0 = best
    for theta in np.arange(theta0 - 6, theta0 + 6.1, 1.5):
        for ratio in np.arange(max(0.40, ratio0 - 0.04), min(1.0, ratio0 + 0.04) + 1e-9, 0.005):
            s = _score_shape(all_points, float(theta), float(ratio))
            if s < best[0]:
                best = (s, float(theta) % 180, float(ratio))
    return best


# --- coarse stage -----------------------------------------------------------

def _red_mask(frame):
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = None
    for low, high in RED_RANGES:
        part = cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _fill_holes(mask):
    filled = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, -1)
    return filled


def coarse_locate(frame, state):
    """Roughly where the plate is. Only has to be close enough to aim rays."""
    filled = _fill_holes(_red_mask(frame))
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return None
    _, sure = cv2.threshold(dist, 0.55 * dist.max(), 255, cv2.THRESH_BINARY)
    sure = sure.astype(np.uint8)
    count, markers = cv2.connectedComponents(sure)
    if count <= 1:
        return None
    markers = markers + 1
    markers[cv2.subtract(filled, sure) == 255] = 0
    markers = cv2.watershed(frame, markers)

    regions = []
    for label in range(2, count + 1):
        piece = np.uint8(markers == label) * 255
        contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 5:
                continue
            (cx, cy), (aw, ah), _ = cv2.fitEllipse(contour)
            major, minor = max(aw, ah), min(aw, ah)
            if major <= 0:
                continue
            ellipse_area = math.pi * major * minor / 4.0
            if ellipse_area <= 0:
                continue
            fill = cv2.contourArea(contour) / ellipse_area
            if not (MIN_MAJOR_PX <= major <= MAX_MAJOR_PX
                    and minor / major >= MIN_AXIS_RATIO
                    and fill >= MIN_COARSE_FILL):
                continue
            regions.append({"cx": cx, "cy": cy, "major": major})
    if not regions:
        return None

    previous = state.get("previous")
    if previous is None:
        pick = max(regions, key=lambda e: e["major"])
    else:
        reachable = [e for e in regions
                     if math.hypot(e["cx"] - previous["cx"], e["cy"] - previous["cy"]) <= MAX_STEP_PX]
        if not reachable:
            state["lost"] = state.get("lost", 0) + 1
            if state["lost"] >= 8:
                state["previous"] = None
                state["lost"] = 0
            return None
        pick = min(reachable,
                   key=lambda e: math.hypot(e["cx"] - previous["cx"], e["cy"] - previous["cy"]))
    state["lost"] = 0
    state["previous"] = pick
    return pick


def _predict(history, index):
    """Constant-velocity guess from the last two accepted centres."""
    if len(history) < 2:
        return history[-1][1] if history else None
    (i1, p1), (i2, p2) = history[-2], history[-1]
    if i2 == i1:
        return p2
    step = (index - i2) / (i2 - i1)
    return p2[0] + (p2[0] - p1[0]) * step, p2[1] + (p2[1] - p1[1]) * step


# --- pipeline ---------------------------------------------------------------

def collect_rays(video, band, known_radius=None):
    """Cast rays for every frame and return the rim points found."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    meta = {"fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    state, history, per_frame, sizes = {}, [], [], []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        field = cv2.GaussianBlur(redness(frame), (0, 0), 1.5)
        coarse = coarse_locate(frame, state)
        if coarse is not None:
            seed = (coarse["cx"], coarse["cy"])
            radius = known_radius or coarse["major"] / 2
        else:
            seed = _predict(history, index)
            radius = known_radius or (statistics.median(sizes) / 2 if sizes else None)

        pts = None
        if seed is not None and radius:
            found = rim_points(field, seed[0], seed[1], radius, *band)
            if len(found) >= MIN_POINTS:
                pts = found
                rough = cv2.fitEllipse(found)
                history.append((index, (rough[0][0], rough[0][1])))
                sizes.append(max(rough[1]))
                if len(history) > 40:
                    history.pop(0)
        per_frame.append(pts)
        index += 1
    cap.release()
    return meta, per_frame


def fit_all(video, per_frame, theta, ratio, rounds=3):
    """Fit every frame, reject what cannot be the plate, then fill the holes."""
    results = []
    for pts in per_frame:
        if pts is None:
            results.append(None)
            continue
        kept = trim(pts, theta, ratio)
        if len(kept) < MIN_POINTS:
            results.append(None)
            continue
        got = fit_shaped(kept, theta, ratio)
        if got is not None:
            got["rays"] = len(kept) / RAYS
        results.append(got)

    strong = [r for r in results if r is not None
              and r["rays"] >= MIN_RAY_SHARE
              and r["residual"] / (r["major"] / 2) <= STRICT_RESIDUAL]
    if len(strong) < 5:
        raise SystemExit("too few usable frames to establish a plate diameter")
    radius = statistics.median(r["major"] for r in strong) / 2

    # Refit with the radius taken from its neighbours rather than from this one
    # frame, before anything is judged. Where the rim is partly covered a free
    # radius and the centre trade against each other and the fitted edge rides
    # off towards the side with more rays.
    for i, r in enumerate(results):
        if r is None or per_frame[i] is None:
            continue
        kept = trim(per_frame[i], theta, ratio)
        if len(kept) < MIN_POINTS:
            continue
        got = fit_shaped(kept, theta, ratio, radius=radius)
        if got is not None:
            got["rays"] = len(kept) / RAYS
            results[i] = got

    for i, r in enumerate(results):
        if r is None:
            continue
        if (r["rays"] < MIN_RAY_SHARE
                or r["residual"] / (r["major"] / 2) > MAX_RESIDUAL
                or abs(r["major"] - 2 * radius) / (2 * radius) > MAX_SIZE_DRIFT):
            results[i] = None

    last_index = last_centre = None
    for i, r in enumerate(results):
        if r is None:
            continue
        if last_centre is not None:
            # The allowance grows with the gap, but not without limit: over a
            # long gap an unbounded allowance covers the whole frame and lets a
            # plate lying on the floor be accepted. A floor plate has a hub of
            # its own, so the hub check does not catch it either.
            span = min(max(1, i - last_index), MAX_STEP_SPAN)
            if math.hypot(r["cx"] - last_centre[0], r["cy"] - last_centre[1]) > MAX_STEP_PX * span:
                results[i] = None
                continue
        last_index, last_centre = i, (r["cx"], r["cy"])

    _verify_hub(video, results)

    # A frame that failed usually failed because the ray search started in the
    # wrong place, not because the plate was hidden: a constant-velocity guess
    # is wrong exactly where the bar changes speed. Now that the frames either
    # side are measured, a gap can be seeded by interpolating between them.
    for _ in range(rounds):
        if not _reseed(video, results, per_frame, theta, ratio, radius):
            break
    return results, radius


def _verify_hub(video, results):
    cap = cv2.VideoCapture(video)
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = results[index] if index < len(results) else None
        if r is not None:
            field = cv2.GaussianBlur(redness(frame), (0, 0), 1.5)
            if hub_contrast(field, r["cx"], r["cy"], r["major"] / 2) < MIN_HUB_CONTRAST:
                results[index] = None
        index += 1
    cap.release()


def _reseed(video, results, per_frame, theta, ratio, radius, radii=None):
    measured = [i for i, r in enumerate(results) if r is not None]
    if len(measured) < 2:
        return 0
    holes = {}
    for a, b in zip(measured, measured[1:]):
        for i in range(a + 1, b):
            holes[i] = (a, b)
    if not holes:
        return 0

    cap = cv2.VideoCapture(video)
    index = recovered = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index in holes:
            a, b = holes[index]
            t = (index - a) / (b - a)
            sx = results[a]["cx"] + t * (results[b]["cx"] - results[a]["cx"])
            sy = results[a]["cy"] + t * (results[b]["cy"] - results[a]["cy"])
            field = cv2.GaussianBlur(redness(frame), (0, 0), 1.5)
            pts = rim_points(field, sx, sy, radius, *NARROW)
            if len(pts) >= MIN_POINTS:
                kept = trim(pts, theta, ratio)
                if len(kept) >= MIN_POINTS:
                    got = fit_shaped(kept, theta, ratio, radius=radius)
                    if got is not None:
                        got["rays"] = len(kept) / RAYS
                        if (got["rays"] >= MIN_RAY_SHARE
                                and got["residual"] / (got["major"] / 2) <= MAX_RESIDUAL
                                and math.hypot(got["cx"] - sx, got["cy"] - sy) <= MAX_STEP_PX
                                and hub_contrast(field, got["cx"], got["cy"],
                                                 got["major"] / 2) >= MIN_HUB_CONTRAST):
                            results[index] = got
                            recovered += 1
        index += 1
    cap.release()
    return recovered


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--csv", default="trajectory.csv")
    p.add_argument("--max-residual", type=float, default=MAX_RESIDUAL,
                   help="how far the rim points may scatter, as a fraction of the radius")
    p.add_argument("--theta", type=float, default=None,
                   help="override the major axis direction instead of searching for it")
    p.add_argument("--ratio", type=float, default=None,
                   help="override the axis ratio instead of searching for it")
    p.add_argument("--rounds", type=int, default=8,
                   help="how many times to retry the gaps with better seeds")
    p.add_argument("--plate-mm", type=float, default=450.0,
                   help="true plate diameter; 450 for a competition plate")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    # First pass, wide band, only to learn how big the plate is in pixels.
    meta, wide_points = collect_rays(args.video, WIDE)
    usable = [pts for pts in wide_points if pts is not None]
    if len(usable) < 5:
        raise SystemExit("the plate was not found often enough to continue")
    score, theta, ratio = search_shape(usable)
    rough, radius = fit_all(args.video, wide_points, theta, ratio, rounds=0)

    # Second pass, narrow band. The plate on the far end of the bar also
    # produces a strong step, and a wide band lets rays lock onto it instead.
    meta, points = collect_rays(args.video, NARROW, known_radius=radius)
    usable = [pts for pts in points if pts is not None]
    score, theta, ratio = search_shape(usable)
    if args.theta is not None:
        theta = args.theta
    if args.ratio is not None:
        ratio = args.ratio
    results, radius = fit_all(args.video, points, theta, ratio, rounds=args.rounds)

    good = [r for r in results if r is not None]
    if not good:
        raise SystemExit("no frame produced a usable measurement")
    diameter = statistics.median(r["major"] for r in good)
    mm_per_px = args.plate_mm / diameter
    tilt = math.degrees(math.acos(min(1.0, ratio)))

    print(f"{args.video}: {meta['width']}x{meta['height']} {meta['fps']:.2f} fps, {len(results)} frames")
    print(f"measured          {len(good)} / {len(results)} ({100.0 * len(good) / len(results):.1f} %)")
    print(f"major axis        {theta:.1f} deg")
    print(f"axis ratio        {ratio:.3f}  -> camera {tilt:.1f} deg off the plate normal")
    print(f"rim residual      {100 * score:.2f} % of the plate radius")
    print(f"plate diameter    {diameter:.1f} px  ->  {mm_per_px:.3f} mm/px")

    ys = [(meta["height"] - r["cy"]) * mm_per_px for r in good]
    xs = [r["cx"] * mm_per_px / ratio for r in good]
    print(f"vertical travel   {max(ys) - min(ys):.0f} mm")
    print(f"horizontal spread {max(xs) - min(xs):.0f} mm  (corrected for the camera angle)")

    directory = os.path.dirname(os.path.abspath(args.csv))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_ms", "cx_px", "cy_px", "x_mm", "y_mm",
                    "major_px", "minor_px", "ratio", "angle_deg", "rays", "residual"])
        for i, r in enumerate(results):
            if r is None:
                continue
            w.writerow([i,
                        round(i * 1000.0 / meta["fps"], 2) if meta["fps"] else "",
                        round(r["cx"], 2), round(r["cy"], 2),
                        round(r["cx"] * mm_per_px / ratio, 1),
                        round((meta["height"] - r["cy"]) * mm_per_px, 1),
                        round(r["major"], 1), round(r["minor"], 1),
                        round(r["ratio"], 4), round(r["angle"], 1),
                        round(r["rays"], 3),
                        round(r["residual"] / (r["major"] / 2), 4)])
    print(f"\nwritten {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
