"""Measure the plate in every frame of a set.

    coarse    where the plate roughly is, frame by frame (candidates, path)
    outline   the shape of this set's plate, learned from a sample of its frames
    scale     the plate's size in pixels over time
    fit       position and scale of the outline in each frame
    face      the front face inside the outline: bar end, true diameter, angle
    judge     keep a frame only if its fit is believable

Frames that fail are reported as rejected, with the reason, and are never
filled in by interpolation: an interpolated position lags or runs ahead of the
bar, which looks like a tracking error and hides a real one.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.vision import appearance
from app.vision.candidates import concentricity, find_candidates, hough_circles
from app.vision.edges import MAX_EDGES, ray_edges, rim_patch
from app.vision.frames import Frames
from app.vision.outline import (Face, Outline, face_from_edges, fit_outline, learn_outline,
                                recentre, sectors_covered)
from app.vision.path import anchor, best_path, link_circles, moving_track
from app.vision.plate import PLATE_DIAMETER_MM

# --- coarse ---------------------------------------------------------------------
WORK_WIDTH = 360          # the coarse stage runs on frames reduced to this width
HOUGH_EVERY = 4           # frames between Hough samples when looking for the moving plate
HOUGH_RADIUS = (0.03, 0.20)   # of the reduced frame height

# --- outline and fit --------------------------------------------------------------
LEARN_ROUNDS = 3
LEARN_FRAMES = 120        # the outline is a constant of the set; this many frames teach it
FIRST_BAND = 0.18         # search band round the outline before its shape is known
BAND = 0.10               # and after, as a fraction of the scale
FIRST_GATE = 0.08
GATE = 0.04               # edges further than this from the outline take no part
PRIOR_WEIGHT = 3.0        # pull of the smoothed scale, in rim points' worth
SCALE_WINDOW_S = 0.25     # half-width of the running median over the scale
MAX_RESEED_GAP = 12       # frames; longer gaps are left empty

# The scale is read off frames with the rim in view almost all round. Accepting
# a frame and trusting it to set the size are different jobs.
GOOD_SECTORS = 10
GOOD_RESIDUAL = 0.012     # of the scale

# --- judging ------------------------------------------------------------------------
MIN_SECTORS = 5           # of 12 directions round the rim that must hold edges
MAX_RESIDUAL = 0.03       # median edge distance from the outline, of the scale
MAX_SCALE_DRIFT = 0.05    # disagreement with the smoothed scale
MIN_SIMILARITY = 0.0      # see appearance.py
# The appearance check compares each frame with the set's own median, so a set
# that followed the wrong thing from start to finish would agree with itself.
# Whatever was followed must also be built like a plate, in rings.
MIN_CONCENTRICITY = 0.15


@dataclass
class Measurement:
    frame: int
    t: float
    x: float                # outline centre, pixels in the full frame
    y: float
    scale: float            # outline scale, pixels
    face_x: float           # centre of the front face: the end of the bar
    face_y: float
    mm_per_px: float
    residual: float         # median edge distance from the outline, pixels
    sectors: int
    similarity: float
    accepted: bool
    reason: str = ""


@dataclass
class SetTrack:
    outline: Outline
    measurements: list      # one Measurement or None per frame
    frame_size: tuple
    radius_px: float        # scale the coarse stage started from
    edges: dict = field(default_factory=dict, repr=False)

    @property
    def accepted(self):
        return [m for m in self.measurements if m is not None and m.accepted]

    @property
    def face(self) -> Face:
        return self.outline.face


# --- coarse ---------------------------------------------------------------------

def _reduce(image, scale):
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def coarse_path(frames: Frames, work_width=WORK_WIDTH):
    """Rough plate centre per frame (full-frame pixels, or None) and its radius.

    Needs whole frames; on the Pi the live tracker does this job instead.
    """
    scale = work_width / frames.frame_size[0]
    detections = []
    for i in range(0, len(frames), HOUGH_EVERY):
        gray = cv2.cvtColor(_reduce(frames.image(i)[0], scale), cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        detections.append((i, hough_circles(gray, HOUGH_RADIUS[0] * h, HOUGH_RADIUS[1] * h)))
    times = frames.times
    tracks = link_circles(detections, times, 6 * HOUGH_EVERY)
    bar = moving_track(tracks)
    if bar is None:
        return [None] * len(frames), None
    radius = float(np.median([p[2] for _, p in bar]))
    candidates = [find_candidates(_reduce(frames.image(i)[0], scale), radius) for i in range(len(frames))]
    candidates = anchor(candidates, bar, radius)
    path = best_path(candidates, times, radius)
    centres = [None if c is None else (c.x / scale, c.y / scale) for c in path]
    return centres, radius / scale


# --- fine -----------------------------------------------------------------------

def _edges(frames, i, centre, scale, outline, band, inner=None, max_edges=MAX_EDGES):
    """Edges within band of the outline placed at centre with this scale.

    inner, if given, widens the band on the inside only, to that fraction of
    the outline: the face edge can sit a quarter of the radius inside the
    silhouette when the camera is well off to the side.
    """
    img, org = frames.image(i)
    local = np.asarray(centre, float) - org
    reach = scale * (outline.rho.max() + band) + 4
    patch, off = rim_patch(img, local, reach)
    lo = max(0.3, outline.rho.min() * (inner if inner is not None else 1.0) - band)
    hi = outline.rho.max() + band
    pts, _, _ = ray_edges(patch, off, local, scale, lo, hi, max_edges=max_edges)
    if len(pts) == 0:
        return pts
    pts = pts + org
    d = pts - centre
    rho, _ = outline.at(np.arctan2(d[:, 1], d[:, 0]))
    rel = np.hypot(d[:, 0], d[:, 1]) / (scale * rho)
    floor = 1.0 - band / rho if inner is None else inner
    return pts[(rel >= floor) & (rel <= 1.0 + band / rho)]


def _mean_scale(learned, radius):
    """Learned scales relative to the coarse radius: the outline is recentred and rescaled."""
    if not learned:
        return 1.0
    return float(np.median([s for _, s in learned.values()])) / radius


def _smooth(times, values: dict, keys, half_window):
    """Running median of values at keys, interpolated to every frame."""
    keys = sorted(keys)
    if not keys:
        return None
    t = np.asarray([times[k] for k in keys])
    v = np.asarray([values[k] for k in keys])
    med = np.array([np.median(v[np.abs(t - t[j]) <= half_window]) for j in range(len(keys))])
    return np.interp(times, t, med)


def learn(frames, centres, radius, rounds=LEARN_ROUNDS, sample=LEARN_FRAMES):
    """Learn the outline from an even sample of frames.

    Returns it with the refined (centre, scale) of the sampled frames.
    """
    outline = Outline()
    known = [i for i, c in enumerate(centres) if c is not None]
    known = known[::max(1, len(known) // sample)]
    state = {i: (np.asarray(centres[i], float), float(radius)) for i in known}
    for rnd in range(rounds + 1):
        band, gate = (FIRST_BAND, FIRST_GATE) if rnd == 0 else (BAND, GATE)
        phis, rs = [], []
        for i, (c, s) in list(state.items()):
            pts = _edges(frames, i, c, s, outline, band)
            fit = fit_outline(pts, outline, c, s, gate=gate)
            if fit is None:
                continue
            state[i] = (fit.centre, fit.scale)
            d = pts - fit.centre
            phis.append(np.arctan2(d[:, 1], d[:, 0]))
            rs.append(np.hypot(d[:, 0], d[:, 1]) / fit.scale)
        if not phis:
            break
        rho = learn_outline(np.concatenate(phis), np.concatenate(rs))
        if rho is None:
            break
        rho, shift, factor = recentre(rho)
        outline = Outline(rho)
        state = {i: (c + shift * s, s * factor) for i, (c, s) in state.items()}
    return outline, state


def measure(frames: Frames, centres, radius) -> SetTrack:
    """Everything after the coarse stage."""
    n = len(frames)
    times = frames.times
    outline, learned = learn(frames, centres, radius)

    # One set of edges per frame, found round the coarse centre with the learned
    # outline, serves both fits below. A frame that was in the learning sample
    # starts from its refined centre and scale.
    start = {}
    for i, c in enumerate(centres):
        if c is not None:
            start[i] = learned.get(i, (np.asarray(c, float), float(radius) * _mean_scale(learned, radius)))
    free, good, edges = {}, [], {}
    for i, (c, s) in start.items():
        pts = _edges(frames, i, c, s, outline, FIRST_BAND if i not in learned else BAND)
        fit = fit_outline(pts, outline, c, s, gate=GATE)
        if fit is None:
            continue
        free[i], edges[i] = fit, pts
        if (sectors_covered(pts[fit.inliers], fit.centre) >= GOOD_SECTORS
                and fit.residual() <= GOOD_RESIDUAL * fit.scale):
            good.append(i)
    scales = {i: f.scale for i, f in free.items()}
    scale_t = _smooth(times, scales, good or list(free), SCALE_WINDOW_S)
    if scale_t is None:
        return SetTrack(outline, [None] * n, frames.frame_size, radius)

    fits = {}

    def fit_at(i, centre, pts=None):
        if pts is None:
            pts = _edges(frames, i, centre, scale_t[i], outline, BAND)
        f = fit_outline(pts, outline, centre, scale_t[i], scale_prior=scale_t[i],
                        prior_weight=PRIOR_WEIGHT, gate=GATE)
        if f is not None:
            fits[i], edges[i] = f, pts
        return f

    for i, f in free.items():
        # Edges found far from where the fit ended up are refound round it.
        moved = np.hypot(*(f.centre - start[i][0])) > 0.03 * f.scale
        fit_at(i, f.centre, None if moved else edges[i])
    # A frame lost in the coarse stage was usually lost because the plate
    # moved faster than the candidates could follow, not because it was hidden.
    # With the frames either side measured, the gap can be seeded between them.
    for _ in range(2):
        done = sorted(fits)
        for a, b in zip(done, done[1:]):
            if b - a - 1 > MAX_RESEED_GAP:
                continue
            for i in range(a + 1, b):
                if i in fits:
                    continue
                w = (times[i] - times[a]) / max(times[b] - times[a], 1e-9)
                fit_at(i, fits[a].centre + w * (fits[b].centre - fits[a].centre))

    if not fits:
        return SetTrack(outline, [None] * n, frames.frame_size, radius)
    outline.face = _face(frames, fits, edges, outline)
    profiles = {}
    for i, f in fits.items():
        img, org = frames.image(i)
        profiles[i] = appearance.radial_profile(img, f.centre - org, f.scale, outline)
    trusted = [i for i in fits if sectors_covered(edges[i][fits[i].inliers], fits[i].centre) >= GOOD_SECTORS]
    reference = np.median(np.array([profiles[i] for i in (trusted or list(fits))]), axis=0)
    rings = _concentricity(frames, fits, trusted or list(fits))
    not_a_plate = rings < MIN_CONCENTRICITY

    face = outline.face
    measurements = [None] * n
    for i, f in fits.items():
        sectors = sectors_covered(edges[i][f.inliers], f.centre)
        sim = appearance.similarity(profiles[i], reference)
        reason = ""
        if not_a_plate:
            reason = "not built like a plate"
        elif sectors < MIN_SECTORS:
            reason = "rim not in view"
        elif f.residual() > MAX_RESIDUAL * f.scale:
            reason = "edges scattered"
        elif abs(f.scale - scale_t[i]) > MAX_SCALE_DRIFT * scale_t[i]:
            reason = "wrong size"
        elif sim < MIN_SIMILARITY:
            reason = "does not look like the plate"
        fx = f.centre[0] + face.dx * f.scale
        fy = f.centre[1] + face.dy * f.scale
        measurements[i] = Measurement(
            frame=i, t=float(times[i]), x=float(f.centre[0]), y=float(f.centre[1]), scale=float(f.scale),
            face_x=float(fx), face_y=float(fy),
            mm_per_px=PLATE_DIAMETER_MM / (face.major * float(scale_t[i])),
            residual=f.residual(), sectors=sectors, similarity=sim,
            accepted=not reason, reason=reason)
    return SetTrack(outline, measurements, frames.frame_size, radius, edges)


def _concentricity(frames, fits, keys, sample=30):
    """Median concentricity of what was fitted, over a sample of frames."""
    values = []
    for i in keys[::max(1, len(keys) // sample)]:
        f = fits[i]
        img, org = frames.image(i)
        c = f.centre - org
        half = int(f.scale) + 2
        x0, y0 = max(0, int(c[0]) - half), max(0, int(c[1]) - half)
        crop = img[y0:int(c[1]) + half + 1, x0:int(c[0]) + half + 1]
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        values.append(concentricity(lab, c[0] - x0, c[1] - y0, f.scale))
    return float(np.median(values)) if values else -1.0


def _face(frames, fits, edges, outline, sample=LEARN_FRAMES):
    phis, rel = [], []
    keys = sorted(fits)
    for i in keys[::max(1, len(keys) // sample)]:
        f = fits[i]
        # lettering and the hub sit inside too, so allow a ray more edges
        pts = _edges(frames, i, f.centre, f.scale, outline, BAND, inner=0.6, max_edges=8)
        d = pts - f.centre
        phi = np.arctan2(d[:, 1], d[:, 0])
        rho, _ = outline.at(phi)
        phis.append(phi)
        rel.append(np.hypot(d[:, 0], d[:, 1]) / (f.scale * rho))
    return face_from_edges(np.concatenate(phis), np.concatenate(rel), outline)


def track(frames: Frames) -> SetTrack:
    centres, radius = coarse_path(frames)
    if radius is None:
        raise ValueError("no moving plate found")
    return measure(frames, centres, radius)
