"""Measure the plate in every frame of a set.

    coarse    where the plate roughly is, frame by frame (candidates, path)
    start     the front face in the first frame it can be found in, its radius
              searched round the coarse one
    follow    the face frame by frame, both ways from there; the sleeve's end
              and the last frames say where to look, the rim says exactly where
    judge     keep a frame only if what was measured is this set's plate

Frames that fail are reported as rejected, with the reason, and are never
filled in by interpolation: an interpolated position lags or runs ahead of the
bar, which looks like a tracking error and hides a real one.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.vision import appearance
from app.vision.candidates import concentricity, find_candidates, hough_circles
from app.vision.extent import carried_ellipse
from app.vision.face import CIRCLE, SIGMAS, WIDE_SIGMAS, Face, measure as measure_face
from app.vision.frames import Frames
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

# --- following the face ------------------------------------------------------------
# Where the live stage loses the plate, it keeps crops wide enough for the
# plate to be found again for this many frames (see app.capture.live); the
# face is followed through them from its last speed.
MAX_RESEED_GAP = 12

START_TRIES = 30          # frames from the start of the set the face is looked for in
# The coarse radius is the plate's outline, or the outermost ring with an edge
# nearly all round it; the face is inside that by the tread.
START_RADII = np.arange(0.80, 1.151, 0.025)
START_RAYS = 140          # of 180: the rim seen nearly all round, to start from
START_SECTORS = 11
REF_FRAMES = 20           # frames from the start that say what the plate looks like

MIN_RAYS = 60             # of 180, with an edge on the face
MIN_SECTORS = 8           # of 12 directions round it
# The radius is compared with the frames before, not with the start: a plate
# carried towards the camera grows, by 17 % on one clip from the floor to
# overhead, and held to the start's it was rejected, then fitted inside its rim.
MAX_DR = 0.03             # change of the radius from the frames before
R_FOLLOW = 0.2            # share of each accepted frame's radius in the one the next is measured with
DROP_SPEED = 6.0          # m/s: a bar dropped from overhead
# What the face looks like, against the frames just before: its colour, and
# how even it is. After the bar was dropped on the gym footage, a ring and a
# rack were followed; a face is nearly even, 1 to 3.5 in L, they were not, 15
# to 21. Against the frames just before rather than the start, because the
# light changes as the bar moves through it, and a phone's exposure follows.
MAX_DAB = 7.0             # a and b of Lab; the right plate stayed within 5
MAX_DL = 18.0             # lightness
MAX_SPREAD = 2.5          # spread of each, against theirs
COLOUR_FRAMES = 10        # accepted frames the colour is compared with
MAX_SLEEVE_JUMP = 0.12    # change of the sleeve end's offset between frames, of the radius
# Once lost, the face is looked for in the whole image (a whole frame, or the
# wide crop the live stage keeps while it has lost the plate too): circles of
# its size from a Hough transform at half resolution, each measured, and of
# those this set's face could be, the nearest to where it was going. Every
# frame at first, then every few.
REACQUIRE = 12            # circles tried
REACQUIRE_SHARE = 0.8     # of the most rays among them, for the nearest to count
REACQUIRE_EVERY = 4       # frames, once the face has been lost for longer than MAX_RESEED_GAP
SCALE_WINDOW_S = 0.25     # half-width of the running median that gives each frame its millimetres

# The face's shape. Seen off square it is an ellipse, and a circle fitted to
# it leaves the rays near the ends of its axes off the rim. The shape is
# learned from the frames at the start, then fitted with them.
SHAPE_ROUNDS = 3
SHAPE_POOL = 0.08         # edges this far from the face, of the radius, are pooled to learn the shape
MIN_ELLIPTICITY = 0.005   # 1 - axis ratio below which the face is taken for a circle: 6 degrees

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

# --- the whole set ---------------------------------------------------------------
# The checks above compare each frame with the start of the set, so a set that
# followed the wrong thing from start to finish would agree with itself. What
# was followed must also carry its face: sampled in its own frame, the inside
# of a plate is the same picture wherever the plate has moved to, while a ring
# or a wheel shows the room behind it and the room slides through it. Two
# frames a radius apart settle it. A set that never moved a radius is asked
# whether the inside is built in rings instead.
MIN_INTERIOR = 0.25       # correlation of the inside between two such frames
MIN_CONCENTRICITY = 0.15


@dataclass
class Measurement:
    frame: int
    t: float
    x: float                # centre of the front face, pixels in the full frame: the end of the bar
    y: float
    scale: float            # the face's semi-major axis, pixels
    face_x: float           # the same centre, by the name the analysis uses
    face_y: float
    mm_per_px: float
    residual: float         # rms distance of the rim's edges from the face, pixels
    sectors: int            # of 12 directions round the face with rim in them
    rays: int               # of 180 rays with an edge on the face
    accepted: bool
    reason: str = ""
    sleeve_x: float = math.nan   # the sleeve's end, where it was found
    sleeve_y: float = math.nan


@dataclass
class SetTrack:
    face: Face
    measurements: list      # one Measurement or None per frame
    frame_size: tuple
    radius_px: float        # radius the coarse stage started from

    @property
    def accepted(self):
        return [m for m in self.measurements if m is not None and m.accepted]


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

def _max_step(radius_px, dt):
    """How far the face can move in dt: a dropped bar, in pixels of a plate this size."""
    return DROP_SPEED * 1000.0 * abs(dt) * radius_px / (PLATE_DIAMETER_MM / 2.0)


def _verdict(f, colour, R, ref, last, dt):
    """Why a frame's fit is not this set's face, or ''."""
    if f is None:
        return "rim not found"
    if f.rays < MIN_RAYS or f.sectors < MIN_SECTORS:
        return "rim not in view"
    if abs(f.r - R) > MAX_DR * R:
        return "wrong size"
    if last is not None and math.hypot(f.x - last[0], f.y - last[1]) > _max_step(R, dt):
        return "faster than a bar"
    if ref is not None and ref.get("colour") is not None and colour is not None:
        rc = ref["colour"]
        if any(colour[3 + k] >= max(3.0, MAX_SPREAD * rc[3 + k]) for k in range(3)):
            return "face not even"
        if math.hypot(colour[1] - rc[1], colour[2] - rc[2]) >= MAX_DAB or abs(colour[0] - rc[0]) >= MAX_DL:
            return "not the plate's colour"
    return ""


def _start(frames, centres, radius):
    """The first frame the face can be found in round the coarse centre, and its fit.

    The radius is searched: the coarse one is the outline, and the face is
    inside it by the tread. Of the fits the one with most of the rim on it
    is kept, and the first frame where that is nearly all of it is the start;
    failing that, the best frame tried.
    """
    tried, best_any = 0, None
    for i, c in enumerate(centres):
        if c is None:
            continue
        img, org = frames.image(i)
        best = None
        for k in START_RADII:
            f, _ = measure_face(img, org, [c], k * radius, sigmas=WIDE_SIGMAS, r_weight=2.0)
            if f is not None and (best is None or (f.rays, -f.cost) > (best.rays, -best.cost)):
                best = f
        if best is not None:
            if best.rays >= START_RAYS and best.sectors >= START_SECTORS:
                return i, best
            if best_any is None or best.rays > best_any[1].rays:
                best_any = (i, best)
        tried += 1
        if tried >= START_TRIES:
            break
    if best_any is not None and best_any[1].rays >= MIN_RAYS and best_any[1].sectors >= MIN_SECTORS:
        return best_any
    return None


def _reacquire(img, org, R, near, shape, ref, last, dt):
    """The face looked for in the whole image: a fit, its colour and '' as _verdict says, or None."""
    g = cv2.cvtColor(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    found = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1.5, R / 2, param1=80, param2=20,
                             minRadius=int(0.45 * R), maxRadius=int(0.55 * R) + 1)
    if found is None:
        return None
    good = []
    for x, y, _ in found[0][:REACQUIRE]:
        f, col = measure_face(img, org, [(2 * x + org[0], 2 * y + org[1])], R, None, shape, colour=ref is not None)
        if not _verdict(f, col, R, ref, last, dt):
            good.append((f, col))
    if not good:
        return None
    # the nearest to where the face was going, of those with nearly the most rim on them
    most = max(f.rays for f, _ in good)
    return min(((f, col) for f, col in good if f.rays >= REACQUIRE_SHARE * most),
               key=lambda fc: math.hypot(fc[0].x - near[0], fc[0].y - near[1]))


def _follow(frames, centres, order, origin, R, shape, ref, out, stop_after=None):
    """Follow the face through the frames in order (either way in time) from
    origin, the fit in the frame before them: {frame: (fit, colour, reason)} into out.

    Each frame is looked for where the last two accepted ones say it is, and
    round where the sleeve's end, found near where its last offset puts it,
    says. Where that fails, the coarse stage's centre for the frame is tried,
    moved by how far the face has been from it, and once the face has been
    lost for a few frames, the whole image. Returns the number accepted.
    """
    times = frames.times
    i0, f0 = origin
    hist = [(i0, f0.x, f0.y)]
    sleeve = None if f0.sleeve is None else (f0.sleeve[0] - f0.x, f0.sleeve[1] - f0.y)
    offsets = []                # face centre less the coarse one, recent frames
    if centres[i0] is not None:
        offsets.append((f0.x - centres[i0][0], f0.y - centres[i0][1]))
    accepted = 0
    want_colour = ref is not None and ref["colour"] is not None
    if want_colour:
        ref = dict(ref)
        recent = []
    for i in order:
        img, org = frames.image(i)
        a = hist[-1]
        if len(hist) >= 2 and abs(a[0] - hist[-2][0]) <= 3 and abs(i - a[0]) <= MAX_RESEED_GAP:
            # carried on at the speed of the last two, over any frames lost since
            b = hist[-2]
            k = (times[i] - times[a[0]]) / (times[a[0]] - times[b[0]])
            pred = (a[1] + k * (a[1] - b[1]), a[2] + k * (a[2] - b[2]))
        else:
            pred = (a[1], a[2])
        resumed = abs(i - a[0]) > 2
        f, col = measure_face(img, org, [pred], R, None if resumed else sleeve, shape,
                              WIDE_SIGMAS if resumed else SIGMAS, colour=want_colour)
        why = _verdict(f, col, R, ref, a[1:], times[i] - times[a[0]])
        if why and centres[i] is not None:
            off = np.median(np.array(offsets), axis=0) if offsets else (0.0, 0.0)
            g = (centres[i][0] + off[0], centres[i][1] + off[1])
            if math.hypot(g[0] - pred[0], g[1] - pred[1]) > 2.0:
                f2, col2 = measure_face(img, org, [g], R, None, shape, WIDE_SIGMAS, colour=want_colour)
                why2 = _verdict(f2, col2, R, ref, a[1:], times[i] - times[a[0]])
                if not why2:
                    f, col, why = f2, col2, why2
        lost = abs(i - a[0]) - 1
        if why and resumed and (lost <= MAX_RESEED_GAP or lost % REACQUIRE_EVERY == 0):
            found = _reacquire(img, org, R, pred, shape, ref, a[1:], times[i] - times[a[0]])
            if found is not None:
                f, col = found
                why = ""
        out[i] = (f, col, why)
        if why:
            continue
        accepted += 1
        hist = hist[-1:] + [(i, f.x, f.y)]
        R = (1 - R_FOLLOW) * R + R_FOLLOW * f.r
        if f.sleeve is not None:
            off = (f.sleeve[0] - f.x, f.sleeve[1] - f.y)
            if sleeve is None or math.hypot(off[0] - sleeve[0], off[1] - sleeve[1]) < MAX_SLEEVE_JUMP * R:
                sleeve = off
        if centres[i] is not None:
            offsets = offsets[-29:] + [(f.x - centres[i][0], f.y - centres[i][1])]
        if want_colour and col is not None:
            # the light changes as the bar moves through it, and a phone's exposure follows
            recent = recent[-(COLOUR_FRAMES - 1):] + [col]
            if len(recent) >= COLOUR_FRAMES // 2:
                ref["colour"] = np.median(np.array(recent), axis=0)
        if stop_after is not None and accepted >= stop_after:
            break
    return accepted


def _ellipse(points):
    """Semi-axes and direction (radians) of the ellipse round 0 through points (N, 2), robustly."""
    x, y = points[:, 0], points[:, 1]
    A = np.column_stack([x * x, 2 * x * y, y * y])
    w = np.ones(len(x))
    for _ in range(10):
        sol = np.linalg.lstsq(A * w[:, None], w, rcond=None)[0]
        r = np.sqrt(np.maximum(A @ sol, 1e-9)) - 1
        s = max(1.4826 * float(np.median(np.abs(r))), 0.002)
        u = r / (4.685 * s)
        w = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
    lam, vec = np.linalg.eigh(np.array([[sol[0], sol[1]], [sol[1], sol[2]]]))
    if lam[0] <= 0:
        return None
    return 1 / math.sqrt(lam[0]), 1 / math.sqrt(lam[1]), math.atan2(vec[1, 0], vec[0, 0]) % math.pi


def _pooled(fits, shape):
    """Edges near the face in these fits, in each face's own frame and size."""
    pts = []
    for f in fits:
        p = f.points
        near = np.abs(p[:, 2]) < SHAPE_POOL * f.r
        if f.sleeve is not None:
            # on the tread's side the edges beyond the face are the tread's
            ox, oy = f.x - f.sleeve[0], f.y - f.sleeve[1]
            n = math.hypot(ox, oy)
            if n > 0.05 * f.r:
                beyond = ((p[:, 0] - f.x) * ox + (p[:, 1] - f.y) * oy) / n > 0.3 * f.r
                near &= ~beyond
        pts.append((p[near, :2] - [f.x, f.y]) / f.r)
    return pts


def _learn_shape(frames, centres, i0, f0):
    """The face's shape and the fits that taught it, from the frames after i0: (shape, fits)."""
    shape = CIRCLE
    for _ in range(SHAPE_ROUNDS):
        img, org = frames.image(i0)
        f, _ = measure_face(img, org, [(f0.x, f0.y)], f0.r, shape=shape, sigmas=WIDE_SIGMAS)
        if f is None:
            f = f0
        out = {}
        _follow(frames, centres, range(i0 + 1, len(frames)), (i0, f), f.r, shape, None, out, stop_after=REF_FRAMES)
        fits = [f] + [v[0] for v in out.values() if not v[2]]
        pts = np.concatenate(_pooled(fits, shape))
        e = _ellipse(pts) if len(pts) >= 50 else None
        if e is None:
            break
        new = (e[1] / e[0], e[2]) if 1 - e[1] / e[0] >= MIN_ELLIPTICITY else CIRCLE
        same = abs(new[0] - shape[0]) < 0.002 and (new[0] == 1.0 or abs(math.sin(new[1] - shape[1])) < 0.02)
        shape = new
        if same:
            break
    return shape, fits


def _face(shape, fits):
    """The face for the analysis, and whether its axis is known well enough to correct by."""
    q, th = shape
    face = Face(major=2.0, minor=2.0 * q, angle_deg=math.degrees(th) % 180.0)
    if face.camera_angle_deg < MIN_CORRECTED_ANGLE or len(fits) < 2 * FACE_SUBSETS:
        return face
    axes = []
    for k in range(FACE_SUBSETS):
        pts = np.concatenate(_pooled(fits[k::FACE_SUBSETS], shape))
        e = _ellipse(pts) if len(pts) >= 50 else None
        if e is None:
            return face
        axes.append(e[2])
    agreement = abs(np.mean(np.exp(2j * np.array(axes))))
    face.corrected = bool(agreement >= MIN_AXIS_AGREEMENT)
    return face


def _reference(frames, fits, i_list):
    """What the plate looked like at the start: its colour and evenness."""
    cols = []
    for i, f in zip(i_list, fits):
        img, org = frames.image(i)
        _, col = measure_face(img, org, [(f.x, f.y)], f.r, colour=True)
        if col is not None:
            cols.append(col)
    return {"colour": np.median(np.array(cols), axis=0) if cols else None}


def measure(frames: Frames, centres, radius) -> SetTrack:
    """Everything after the coarse stage: centres (full-frame pixels or None per frame) and its radius."""
    n = len(frames)
    times = frames.times
    found = _start(frames, centres, radius)
    if found is None:
        return SetTrack(Face(), [None] * n, frames.frame_size, radius)
    i0, f0 = found
    shape, start_fits = _learn_shape(frames, centres, i0, f0)
    img, org = frames.image(i0)
    f, _ = measure_face(img, org, [(f0.x, f0.y)], f0.r, shape=shape, sigmas=WIDE_SIGMAS)
    f0 = f or f0
    out = {}
    _follow(frames, centres, range(i0 + 1, min(n, i0 + 1 + 2 * REF_FRAMES)), (i0, f0), f0.r, shape, None, out,
            stop_after=REF_FRAMES)
    ref_frames = [i0] + [i for i in sorted(out) if not out[i][2]]
    ref = _reference(frames, [f0] + [out[i][0] for i in ref_frames[1:]], ref_frames)
    results = {i0: (f0, None, "")}
    _follow(frames, centres, range(i0 + 1, n), (i0, f0), f0.r, shape, ref, results)
    _follow(frames, centres, range(i0 - 1, -1, -1), (i0, f0), f0.r, shape, ref, results)

    fits = {i: v[0] for i, v in results.items() if not v[2]}
    if not fits:
        return SetTrack(Face(), [None] * n, frames.frame_size, radius)
    face = _face(shape, start_fits)
    not_a_plate = not _carries_its_face(frames, fits)
    keys = sorted(fits)
    scale_t = _smooth(times, {i: fits[i].r for i in keys}, keys, SCALE_WINDOW_S)
    measurements = [None] * n
    for i, (f, _, why) in results.items():
        if f is None:
            continue
        reason = "not built like a plate" if not_a_plate and not why else why
        measurements[i] = Measurement(
            frame=i, t=float(times[i]), x=float(f.x), y=float(f.y), scale=float(f.r),
            face_x=float(f.x), face_y=float(f.y),
            mm_per_px=PLATE_DIAMETER_MM / (face.major * float(scale_t[i])),
            residual=float(f.rms), sectors=int(f.sectors), rays=int(f.rays),
            accepted=not reason, reason=reason,
            sleeve_x=math.nan if f.sleeve is None else float(f.sleeve[0]),
            sleeve_y=math.nan if f.sleeve is None else float(f.sleeve[1]))
    return SetTrack(face, measurements, frames.frame_size, radius)


def _smooth(times, values: dict, keys, half_window):
    """Running median of values at keys, interpolated to every frame."""
    keys = sorted(keys)
    t = np.asarray([times[k] for k in keys])
    v = np.asarray([values[k] for k in keys])
    med = np.array([np.median(v[np.abs(t - t[j]) <= half_window]) for j in range(len(keys))])
    return np.interp(times, t, med)


def _carries_its_face(frames, fits, sample=15):
    """Is what was followed a plate, or something the background shows through?

    Pairs of frames the face moved a radius between are compared inside it,
    in its own frame. Where the set never moved that far the question cannot
    be put that way, and the inside must be built in rings instead.
    """
    keys = sorted(fits)
    centres = np.array([(fits[i].x, fits[i].y) for i in keys])
    radii = np.array([fits[i].r for i in keys])
    pairs = []
    for a in range(0, len(keys), max(1, len(keys) // sample)):
        far = np.nonzero(np.hypot(*(centres - centres[a]).T) >= radii[a])[0]
        if far.size:
            pairs.append((keys[a], keys[int(far[np.argmin(np.abs(far - a))])]))
    if not pairs:
        return _concentricity(frames, fits, keys) >= MIN_CONCENTRICITY
    values = []
    for a, b in pairs:
        samples = []
        for i in (a, b):
            img, org = frames.image(i)
            samples.append(appearance.inside(img, np.array([fits[i].x, fits[i].y]) - org, fits[i].r))
        values.append(appearance.travels_with(*samples))
    return float(np.median(values)) >= MIN_INTERIOR


def _concentricity(frames, fits, keys, sample=30):
    """Median concentricity of what was followed, over a sample of frames."""
    values = []
    for i in keys[::max(1, len(keys) // sample)]:
        f = fits[i]
        img, org = frames.image(i)
        c = np.array([f.x, f.y]) - org
        half = int(f.r) + 2
        x0, y0 = max(0, int(c[0]) - half), max(0, int(c[1]) - half)
        crop = img[y0:int(c[1]) + half + 1, x0:int(c[0]) + half + 1]
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        values.append(concentricity(lab, c[0] - x0, c[1] - y0, f.r))
    return float(np.median(values)) if values else -1.0


def track(frames: Frames, near=None) -> SetTrack:
    """near: a point on the side of the plate nearer the camera, see coarse_path."""
    centres, radius = coarse_path(frames, near=near)
    if radius is None:
        raise ValueError("no moving plate found")
    return measure(frames, centres, radius)
