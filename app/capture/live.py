"""Deciding, while the camera runs, whether a set is going on.

    idle      the plates in view are known; every eighth frame each is looked
              for where it was, and a plate that has moved starts a set
    active    the moving plate is followed in every frame, from before it
              moved, and a crop round it is kept
    done      the plate has been still for a while, or is gone: the crops go
              to the analysis and the watcher goes back to idle

Only cheap work happens here, on reduced images and in small windows. The
precise rim, the outline and every number that is reported come afterwards,
from the crops, in app.vision.

The plates to watch are found by looking, not told: every second while idle
the whole frame is searched for plate-shaped circles, and each is watched.
Plates on a storage tree are watched too and never move; the one on the bar
is the one that does.
"""

import math
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.vision.candidates import find_candidates, hough_circles, rim_radius
from app.vision.plate import max_step_px
from app.vision.tracker import MAX_RESEED_GAP


@dataclass
class LiveConfig:
    work_side: int = 360           # all live searching runs on frames reduced to this short side
    plate_px: float = 40.0         # and a plate is looked for no larger than this radius in them
    check_every: int = 8           # frames between checks while idle
    search_every_s: float = 1.0    # full-frame search for plates while idle
    min_score: float = 1.0         # coverage + concentricity a watched plate needs
    max_watched: int = 4
    move_fraction: float = 0.08    # of the radius: moved this far, the plate is moving
    confirm: int = 2               # consecutive checks that must agree
    preroll_s: float = 1.0         # kept from before the movement was noticed
    rest_fraction: float = 0.04    # of the radius per rest window: still
    rest_s: float = 3.0            # still for this long ends the set
    lost_s: float = 2.0            # not found for this long ends the set
    crop_radii: float = 1.6        # half-size of the kept crop, in plate radii
    max_windows: int = 8           # searched along the predicted path while the plate is missing
    wide_crops: int = MAX_RESEED_GAP  # missing frames kept whole round the predicted path;
                                      # a longer gap is not bridged by the analysis anyway
    max_set_s: float = 180.0
    max_bytes: int = 1_500_000_000  # crops kept per set; a set that would need more is cut


@dataclass
class Watched:
    x: float                       # full-frame pixels
    y: float
    r: float
    moved: int = 0


@dataclass
class SetRecording:
    """What the live stage keeps of a set, for the analysis."""
    crops: list = field(default_factory=list)
    origins: list = field(default_factory=list)
    times: list = field(default_factory=list)
    centres: list = field(default_factory=list)     # live estimate, full-frame pixels
    radius: float = 0.0
    frame_size: tuple = (0, 0)
    lost_frames: int = 0
    started_wall: float = 0.0
    ended: str = ""
    quarter_turns: int = 0         # how the camera stood, see app.capture.session


class LiveTracker:
    def __init__(self, ring, frame_size, config: LiveConfig | None = None, on_set=None, log=print,
                 clock=time.time):
        self.ring = ring
        self.w, self.h = frame_size
        self.cfg = config or LiveConfig()
        # The short side, not the width: the Pi camera is landscape unless it
        # is turned, and at 360 across 1536 a plate would be searched for at
        # half the resolution it gets in a portrait frame.
        self.scale = self.cfg.work_side / min(self.w, self.h)
        self.on_set = on_set
        self.log = log
        self.clock = clock                 # wall time, for naming sets
        self.watched: list[Watched] = []
        self.state = "idle"                # or "active" while a set is followed
        self.sighting = None               # (t, x, y, r) of the plate followed, while active
        self._fresh = None                 # a watch list the searcher found, not yet taken up
        self._search_due = -math.inf       # frame time of the next background search
        self._lock = threading.Lock()
        self.stats = {"checks": 0, "searches": 0, "sets": 0, "lost": 0}

    # --- helpers ----------------------------------------------------------------

    def _window(self, seq, cx, cy, half, half_y=None):
        """Read a box round (cx, cy) from frame seq; returns (image, origin, t) or None."""
        half_y = half if half_y is None else half_y
        x0, y0 = max(0, int(cx - half)), max(0, int(cy - half_y))
        x1, y1 = min(self.w, int(cx + half) + 1), min(self.h, int(cy + half_y) + 1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        got = self.ring.read(seq, (x0, y0, x1, y1))
        if got is None:
            return None
        return got[0], np.array([x0, y0], float), got[1]

    def _scale_for(self, r):
        """The reduction for looking for a plate of radius r.

        No finer than puts the plate at plate_px: both test clips were verified
        with the plate at about 40 px, and on the Pi's landscape frame a near
        plate would otherwise be twice the area, and following it too slow to
        keep up with 60 fps on a Pi 4.
        """
        return min(self.scale, self.cfg.plate_px / r)

    def _look(self, image, origin, r, near=None, reach=None):
        """Best candidate of radius r in a full-resolution window, in full-frame pixels."""
        scale = self._scale_for(r)
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cands = find_candidates(small, r * scale, count=4)
        best = None
        for c in cands:
            x, y = origin[0] + c.x / scale, origin[1] + c.y / scale
            d = 0.0 if near is None else math.hypot(x - near[0], y - near[1])
            if reach is not None and d > reach:
                continue
            value = c.score - (0.0 if reach is None else 0.3 * (d / reach) ** 2)
            if best is None or value > best[3]:
                best = (x, y, c.score, value)
        return best

    # --- idle ---------------------------------------------------------------------

    def search(self, seq):
        """Find plate-shaped circles in frame seq; they become the watch list."""
        found = self._find(seq)
        if found is not None:
            self.watched = found

    def _find(self, seq):
        """Plate-shaped circles in the whole of frame seq, best first; None if it is gone."""
        got = self.ring.read(seq)
        if got is None:
            return None
        image, _ = got
        small = cv2.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        found = []
        for x, y, r in hough_circles(gray, 0.03 * h, 0.20 * h):
            r = rim_radius(small, x, y, r, 0.20 * h)
            cands = find_candidates(small, r, count=2,
                                    window=(x - 1.5 * r, y - 1.5 * r, x + 1.5 * r, y + 1.5 * r))
            if not cands or cands[0].score < self.cfg.min_score:
                continue
            c = cands[0]
            # A circle cut by the edge of the frame moves whenever its visible
            # part is found a little differently, and on the Pi one did, twice
            # in a row, and started a set. A plate waiting to be lifted is in view.
            if min(c.x - r, c.y - r, small.shape[1] - c.x - r, small.shape[0] - c.y - r) < 0:
                continue
            found.append((c.score, Watched(c.x / self.scale, c.y / self.scale, r / self.scale)))
        found.sort(key=lambda sw: -sw[0])
        kept = []
        for _, w in found:
            if all(math.hypot(w.x - k.x, w.y - k.y) > k.r for k in kept):
                kept.append(w)
        self.stats["searches"] += 1
        return kept[: self.cfg.max_watched]

    def check(self, seq):
        """Look for each watched plate near where it was. Returns the one that moved, if any."""
        self.stats["checks"] += 1
        for w in self.watched:
            dt = self.cfg.check_every / 30.0            # generous: any camera is at least this fast
            reach = max_step_px(w.r, dt) + w.r
            win = self._window(seq, w.x, w.y, w.r * 1.3 + reach)
            if win is None:
                continue
            image, origin, _ = win
            best = self._look(image, origin, w.r, near=(w.x, w.y), reach=reach)
            if best is None or best[2] < 0.8 * self.cfg.min_score:
                w.moved = 0                             # hidden, or gone: not evidence of movement
                continue
            d = math.hypot(best[0] - w.x, best[1] - w.y)
            if d > self.cfg.move_fraction * w.r:
                w.moved += 1
                if w.moved >= self.cfg.confirm:
                    return w
            else:
                w.moved = 0
                w.x += 0.2 * (best[0] - w.x)            # follow slow drift, not a lift
                w.y += 0.2 * (best[1] - w.y)
        return None

    # --- active -------------------------------------------------------------------

    def follow(self, start_seq, plate: Watched, stop):
        """Track the plate from start_seq until the set ends; returns the recording.

        Each frame is searched in small windows along the line from where the
        plate was last seen to where it would be had it kept its speed, and
        only candidates a bar could have reached since are considered, the
        one nearest the prediction preferred. The windows stay small because
        the vote that finds a plate is shared among the strongest edges in
        view: in a whole frame of gym, a motion-blurred plate is not among the
        peaks at all, and a coil of rope in the corner was followed instead.
        """
        cfg = self.cfg
        r = plate.r
        rec = SetRecording(radius=r, frame_size=(self.w, self.h), started_wall=self.clock())
        seen = [(None, np.array([plate.x, plate.y]))]       # (t, centre) of sightings
        last_seen_t = None
        misses = 0
        kept_bytes = 0
        seq = start_seq
        crop_half = cfg.crop_radii * r
        while not stop.is_set():
            if not self.ring.wait(seq, timeout=0.5):
                if self.ring.closed:
                    rec.ended = "camera stopped"
                    break
                continue
            if seq < self.ring.oldest:
                # fell behind by more than the buffer holds
                rec.lost_frames += self.ring.oldest - seq
                seq = self.ring.oldest
            stamp = self.ring.read(seq, (0, 0, 1, 1))
            if stamp is None:
                rec.lost_frames += 1
                seq += 1
                continue
            t = stamp[1]
            last_t, last_p = seen[-1]
            since = (t - last_t) if last_t is not None else 0.1
            reach = max_step_px(r, since) + 2
            pred = last_p.copy()
            if len(seen) > 1 and seen[-2][0] is not None and last_t > seen[-2][0]:
                step = (last_p - seen[-2][1]) * since / (last_t - seen[-2][0])
                pred = last_p + step * min(1.0, reach / max(np.hypot(*step), 1e-9))
            pred = np.clip(pred, 0, [self.w - 1, self.h - 1])
            n = min(cfg.max_windows - 1, int(np.hypot(*(pred - last_p)) // r))
            aims = [last_p + (pred - last_p) * k / max(n, 1) for k in range(n + 1)]
            lo, hi = np.min(aims, axis=0) - crop_half, np.max(aims, axis=0) + crop_half
            win = self._window(seq, *(lo + hi) / 2, *(hi - lo) / 2)
            if win is None:
                rec.lost_frames += 1
                seq += 1
                continue
            image, origin, _ = win
            best = self._along(image, origin, r, aims, last_p, reach, pred)
            if best is not None and best[2] >= 0.6 * cfg.min_score:
                centre = np.array([best[0], best[1]])
                self.sighting = (t, best[0], best[1], r)
                seen.append((t, centre))
                seen = seen[-400:]
                last_seen_t = t
                misses = 0
                lo, hi = centre - crop_half, centre + crop_half
            else:
                centre = None
                misses += 1
                if misses > cfg.wide_crops:
                    # too long a gap for the analysis to bridge; keep the usual size
                    lo, hi = pred - crop_half, pred + crop_half
            # keep a crop from the window already read: round the plate, or,
            # while it is missing, round everywhere it could be
            x0, y0 = np.maximum(0, (lo - origin).astype(int))
            x1, y1 = np.minimum(image.shape[1::-1], (hi - origin).astype(int) + 1)
            crop = image[y0:y1, x0:x1].copy()
            kept_bytes += crop.nbytes
            rec.crops.append(crop)
            rec.origins.append((origin[0] + x0, origin[1] + y0))
            rec.times.append(t)
            rec.centres.append(None if centre is None else tuple(centre))

            t0 = rec.times[0]
            if last_seen_t is not None and t - last_seen_t > cfg.lost_s:
                rec.ended = "plate lost"
                break
            if t - t0 > cfg.max_set_s:
                rec.ended = "too long"
                break
            if kept_bytes > cfg.max_bytes:
                rec.ended = "memory full"
                break
            if t - t0 > cfg.rest_s + cfg.preroll_s and self._resting(seen, t, r):
                rec.ended = "plate at rest"
                break
            seq += 1
        # back to watching this plate where it now lies
        plate.x, plate.y = seen[-1][1]
        plate.moved = 0
        return rec

    def _along(self, image, origin, r, aims, last_p, reach, pred):
        """Best candidate near any of the aim points and within reach of the last sighting."""
        scale = self._scale_for(r)
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rs = r * scale
        spread = r + np.hypot(*(pred - last_p))
        best = None
        for aim in aims:
            ax, ay = (aim - origin) * scale
            half = (self.cfg.crop_radii + 0.5) * rs
            for c in find_candidates(small, rs, count=4, window=(ax - half, ay - half, ax + half, ay + half)):
                x, y = origin[0] + c.x / scale, origin[1] + c.y / scale
                if math.hypot(x - last_p[0], y - last_p[1]) > reach:
                    continue
                d = math.hypot(x - pred[0], y - pred[1])
                value = c.score - 0.3 * (d / spread) ** 2
                if best is None or value > best[3]:
                    best = (x, y, c.score, value)
        return best

    def _resting(self, history, now, radius):
        pts = [p for (t, p) in history if t is not None and t >= now - self.cfg.rest_s]
        if len(pts) < 5:
            return False
        pts = np.array(pts)
        spread = np.hypot(*(pts.max(axis=0) - pts.min(axis=0)))
        return spread < self.cfg.rest_fraction * radius

    # --- loop ---------------------------------------------------------------------

    def run(self, stop):
        """Consume the ring until stop is set or the camera stops.

        The whole-frame search runs in a thread of its own. On a Pi 4 it takes
        a third of a second, and done in line it held up the checks for that
        long, which is time a starting lift is not noticed in.
        """
        idle = threading.Event()
        idle.set()
        searcher = threading.Thread(target=self._searcher, args=(stop, idle), name="search", daemon=True)
        searcher.start()
        seq = max(0, self.ring.newest)
        while not stop.is_set():
            if not self.ring.wait(seq, timeout=0.5):
                if self.ring.closed:
                    break
                continue
            seq = max(seq, self.ring.newest)            # idle: only the present matters
            # A new watch list replaces the old one, so not while a plate is
            # part way through being confirmed as moving.
            with self._lock:
                if self._fresh is not None and not any(w.moved for w in self.watched):
                    self.watched, self._fresh = self._fresh, None
            moved = self.check(seq) if self.watched else None
            if moved is not None:
                idle.clear()
                self.state = "active"
                fps = self._rate()
                start = max(self.ring.oldest, seq - int(self.cfg.preroll_s * fps))
                self.log(f"set: plate at ({moved.x:.0f}, {moved.y:.0f}) r {moved.r:.0f} moved, "
                         f"keeping {seq - start} frames from before")
                rec = self.follow(start, moved, stop)
                self.stats["sets"] += 1
                self.stats["lost"] += rec.lost_frames
                self.log(f"set ended ({rec.ended}): {len(rec.times)} frames, {rec.lost_frames} lost")
                self.state, self.sighting = "idle", None
                if self.on_set is not None:
                    self.on_set(rec)
                seq = self.ring.newest
                with self._lock:
                    self._fresh = None
                    self._search_due = -math.inf        # the room may have changed during the set
                idle.set()
            seq += self.cfg.check_every
        searcher.join(timeout=5)

    def _searcher(self, stop, idle):
        """Refresh the watch list from the newest frame, every search_every_s while idle."""
        while not stop.is_set() and not self.ring.closed:
            if not idle.wait(0.2):
                continue
            seq = self.ring.newest
            stamp = self.ring.read(seq, (0, 0, 1, 1))
            if stamp is None or stamp[1] < self._search_due:
                time.sleep(0.02)
                continue
            found = self._find(seq)
            with self._lock:
                self._search_due = stamp[1] + self.cfg.search_every_s
                if found is not None and idle.is_set():     # a set started meanwhile: stale
                    self._fresh = found

    def _rate(self):
        """Frames per second, from the timestamps in the ring."""
        a, b = self.ring.oldest, self.ring.newest
        ga, gb = self.ring.read(a, (0, 0, 1, 1)), self.ring.read(b, (0, 0, 1, 1))
        if ga is None or gb is None or gb[1] <= ga[1]:
            return 30.0
        return (b - a) / (gb[1] - ga[1])
