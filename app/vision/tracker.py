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

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from app.vision import appearance
from app.vision.candidates import concentricity, find_candidates, hough_circles
from app.vision.edges import MAX_EDGES, RAYS, ray_edges, rim_patch
from app.vision.extent import carried_ellipse
from app.vision.frames import Frames
from app.vision.outline import (Face, Outline, face_from_edges, fit_outline, learn_outline,
                                recentre, sectors_covered)
from app.vision.path import anchor, best_path, link_circles, moving_track, nearer, other_end
from app.vision.plate import PLATE_DIAMETER_MM

# --- coarse ---------------------------------------------------------------------
WORK_WIDTH = 360          # the coarse stage runs on frames reduced to this width
HOUGH_EVERY = 4           # frames between Hough samples when looking for the moving plate
HOUGH_RADIUS = (0.03, 0.20)   # of the reduced frame height
# The two ends of a bar are about 1.7 m apart, so the near plate is larger
# in the picture by a factor that depends on how far away the camera stands.
# On ten phone clips it was 1.22 to 1.35 times the far one.
NEAR_SIZE = (0.9, 1.6)    # of the far plate's radius
OTHER_REACH = 1.2         # of its radius: window round where the rigid bar puts the other end
OTHER_SCORE = 0.8         # coverage + concentricity a candidate there needs

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
# Where the tread or the plates behind show, the outline has two ways to sit
# on a frame's edges: its silhouette on the stack's silhouette, or on the front
# face's edge a tread's width in. On the first test clip, in the catch, the
# fit flipped between the two from one frame to the next, 12 px apart, and
# both had a residual under 1.5 px and edges in 12 of 12 sectors. What tells
# them apart is how much of the rim they explain: 155 to 170 of 180 rays with
# an edge on the outline for the right one, 120 to 126 for the other. So a fit
# that disagrees with what its neighbours predict is tried again from their
# prediction, and whichever explains more of the rim is kept.
AGREE = 0.02              # of the scale; closer to the prediction than this, no second try
SUPPORT_MARGIN = 3        # rays; a second try must beat the first by more than this

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
# What was followed must also carry its face: sampled in its own frame, the
# inside of a plate is the same picture wherever the plate has moved to, while
# a ring or a wheel shows the room behind it and the room slides through it.
# Two frames a radius apart settle it. Concentricity, which used to, cannot:
# on this footage a plate with a printed label and a lighting gradient across
# it scores 0.08 where a ring scores 0.11. It is still the test for a set that
# never moved a radius, which is the case it was measured on.
MIN_INTERIOR = 0.25       # correlation of the inside between two such frames
MIN_CONCENTRICITY = 0.15

# --- the camera angle --------------------------------------------------------------
# Near square to the plate, the face is so nearly round that the direction of
# its axis is decided by noise: on the first test clip four quarters of the
# set put it anywhere from 38 to 159 degrees. Correcting along a wrong axis
# moves vertical distances, by 1.3 % on that clip, while leaving a 15 degree
# view uncorrected costs horizontal distances 3.4 % and vertical ones nothing.
# So the bar path is corrected only when the angle is large and the quarters agree.
MIN_CORRECTED_ANGLE = 15.0     # degrees
FACE_SUBSETS = 4
MIN_AXIS_AGREEMENT = 0.9       # length of the mean of the quarters' doubled axis angles


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
    support: float = 0.0        # share of the rays with an edge on the outline
    interior: float = 0.0       # the inside against the set's, see appearance.inside()


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


def coarse_path(frames: Frames, work_width=WORK_WIDTH, near=None):
    """Rough plate centre per frame (full-frame pixels, or None) and its radius.

    near, if given, is a point (full-frame pixels) on the side of the plate
    nearer the camera. With the camera off to the side both ends of the bar
    are in view, and the far plate is often the easier one to see: sharper
    against a plain wall, while the near one is a dark plate against dark
    tiles. But it is the one the lifter covers with arms and head, so the
    near one is followed.

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
    small = lambda i: _reduce(frames.image(i)[0], scale)
    far = None
    if near is not None:
        other = other_end(detections, bar)
        if other is not None and not nearer(bar, other, (near[0] * scale, near[1] * scale)):
            bar, far = _along_the_bar(small, bar, other), bar
    # The Hough transform reports whichever ring round the centre is sharpest,
    # often a change plate or the steel disc inside the rim, and seen from an
    # angle with its centre off the rim's. Everything below is built on this
    # radius and these centres: the votes that find the plate in the other
    # frames, the scale the fine stage starts from, and the millimetres per
    # pixel. So the plate's extent is measured from what the track carried
    # with it; see app.vision.extent.
    radius = float(np.median([p[2] for _, p in bar]))
    extent = carried_ellipse(small, bar, HOUGH_RADIUS[1] * h)
    if extent is not None and far is not None:
        # The extent runs off into a plain wall more easily round the near
        # plate, which is larger and nearer the edge of the picture: on one
        # clip it came out four times the plate. The far plate bounds it.
        far_extent = carried_ellipse(small, far, HOUGH_RADIUS[1] * h)
        far_radius = 0.5 * (far_extent[2] + far_extent[3]) if far_extent else float(np.median([p[2] for _, p in far]))
        if not NEAR_SIZE[0] * far_radius <= 0.5 * (extent[2] + extent[3]) <= NEAR_SIZE[1] * far_radius:
            extent = None
    if extent is not None:
        dx, dy, a, b, _ = extent
        radius = 0.5 * (a + b)
        bar = [(i, (x + dx, y + dy, radius)) for i, (x, y, _) in bar]
    candidates = [find_candidates(small(i), radius) for i in range(len(frames))]
    candidates = anchor(candidates, bar, radius)
    path = best_path(candidates, times, radius)
    centres = [None if c is None else (c.x / scale, c.y / scale) for c in path]
    return centres, radius / scale


def _along_the_bar(small, track, other):
    """The other end in every frame of track, where it can be found.

    The Hough transform finds the near plate in fewer frames than the far
    one when the near one is the harder to see, which is why the far one was
    followed: on one clip in 13 of 33, all of them overhead. Anchored that
    sparsely, the path went to a knee and a board in between. So where the
    other end was not seen, it is looked for where the rigid bar puts it:
    the followed end's position plus the offset in the nearest frame both
    were seen in, among candidates of its own size.
    """
    seen = dict(other)
    at = dict(track)
    keys = np.array(sorted(seen))
    r = float(np.median([p[2] for _, p in other]))
    out = []
    for i, (x, y, _) in track:
        if i in seen:
            out.append((i, seen[i]))
            continue
        j = int(keys[np.argmin(np.abs(keys - i))])
        px, py = x + seen[j][0] - at[j][0], y + seen[j][1] - at[j][1]
        reach = OTHER_REACH * r
        found = [c for c in find_candidates(small(i), r, count=3, window=(px - reach, py - reach, px + reach, py + reach))
                 if c.score >= OTHER_SCORE]
        if found:
            out.append((i, (found[0].x, found[0].y, r)))
    return out


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
    _settle(frames, free, edges, outline)
    for i, fit in free.items():
        if (sectors_covered(edges[i][fit.inliers], fit.centre) >= GOOD_SECTORS
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
    insides = {}
    for i, f in fits.items():
        img, org = frames.image(i)
        insides[i] = appearance.inside(img, f.centre - org, f.scale)
    inside_ref = np.median(np.array([insides[i] for i in (trusted or list(fits))]), axis=0)
    not_a_plate = not _carries_its_face(frames, fits, trusted or list(fits))

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
            accepted=not reason, reason=reason,
            support=_support(f, edges[i]) / RAYS,
            interior=appearance.travels_with(insides[i], inside_ref))
    return SetTrack(outline, measurements, frames.frame_size, radius, edges)


def _support(fit, pts):
    """How many of the rays saw an edge on the outline where this fit put it."""
    near = np.abs(fit.residuals) <= GATE * fit.scale
    return sectors_covered(pts[near], fit.centre, sectors=RAYS)


def _settle(frames, fits, edges, outline, passes=2):
    """Try each fit again from where its neighbours put the plate; keep the better.

    Forward, the prediction is the last two fits carried on at their speed;
    backward, the next two. A fit already where the prediction is costs
    nothing. Changes fits and edges in place.
    """
    keys = sorted(fits)
    for p in range(passes):
        order = keys if p % 2 == 0 else keys[::-1]
        step = 1 if p % 2 == 0 else -1
        changed = 0
        for i in order:
            a, b = fits.get(i - step), fits.get(i - 2 * step)
            if a is None:
                continue
            pred = a.centre if b is None else 2 * a.centre - b.centre
            f = fits[i]
            if np.hypot(*(f.centre - pred)) <= AGREE * f.scale:
                continue
            pts = _edges(frames, i, pred, a.scale, outline, BAND)
            g = fit_outline(pts, outline, pred, a.scale, gate=GATE)
            if g is not None and _support(g, pts) > _support(f, edges[i]) + SUPPORT_MARGIN:
                fits[i], edges[i] = g, pts
                changed += 1
        if not changed and p > 0:
            break


def _carries_its_face(frames, fits, keys, sample=15):
    """Is what was fitted a plate, or something the background shows through?

    Pairs of frames the fit moved a radius between are compared inside the
    circle, in the circle's own frame. Where the set never moved that far the
    question cannot be put that way, and the old test is used instead: the
    inside of a plate is built in rings.
    """
    keys = sorted(keys)
    centres = np.array([fits[i].centre for i in keys])
    scales = np.array([fits[i].scale for i in keys])
    pairs = []
    for a in range(0, len(keys), max(1, len(keys) // sample)):
        far = np.nonzero(np.hypot(*(centres - centres[a]).T) >= scales[a])[0]
        if far.size:
            pairs.append((keys[a], keys[int(far[np.argmin(np.abs(far - a))])]))
    if not pairs:
        return _concentricity(frames, fits, keys) >= MIN_CONCENTRICITY
    values = []
    for a, b in pairs:
        samples = []
        for i in (a, b):
            img, org = frames.image(i)
            samples.append(appearance.inside(img, fits[i].centre - org, fits[i].scale))
        values.append(appearance.travels_with(*samples))
    return float(np.median(values)) >= MIN_INTERIOR


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
    face = face_from_edges(np.concatenate(phis), np.concatenate(rel), outline)
    if face.camera_angle_deg < MIN_CORRECTED_ANGLE or len(phis) < 2 * FACE_SUBSETS:
        return replace(face, corrected=False)
    axes = [face_from_edges(np.concatenate(phis[k::FACE_SUBSETS]), np.concatenate(rel[k::FACE_SUBSETS]),
                            outline).angle_deg for k in range(FACE_SUBSETS)]
    agreement = abs(np.mean(np.exp(2j * np.radians(axes))))
    return face if agreement >= MIN_AXIS_AGREEMENT else replace(face, corrected=False)


def track(frames: Frames, near=None) -> SetTrack:
    """near: a point on the side of the plate nearer the camera, see coarse_path."""
    centres, radius = coarse_path(frames, near=near)
    if radius is None:
        raise ValueError("no moving plate found")
    return measure(frames, centres, radius)
