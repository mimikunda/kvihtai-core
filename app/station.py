"""The station: camera, watcher, analysis and recording, running inside the API.

One process holds everything, so that the API can show what the camera sees
and what the watcher is doing, and change the camera's settings, without a
second process to talk to. The capture session runs in threads of its own
(app.capture.session); the station starts it, restarts it if the camera fails,
and stores every analysed set in the database.

Between sets it also keeps the picture bright enough. The exposure stays short
and fixed, for the reason given in app.capture.source, and the gain follows the
light. It is never changed during a set: a plate whose brightness changes
half way through looks less like itself to the analysis.

It also keeps track of itself: what went wrong is kept in incidents.json, its
log in logs/station.log, and run.json says while it runs that it has not been
stopped, so that the next start can tell a power cut or a crash from a
shutdown.
"""

import logging
import math
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime

import cv2
import numpy as np

from app import logbook, system
from app.capture.recorder import list_segments
from app.capture.session import Session, turn_point
from app.capture.worker import ProcessAnalyser
from app.config import Settings
from app.files import read_json, write_json
from app.vision.plate import PLATE_DIAMETER_MM

log = logging.getLogger("kvihtai.station")

TARGET_BRIGHTNESS = 110      # median of the picture, 0-255, that the gain aims for
MIN_GAIN = 1.0
MAX_LENS = 15.0              # dioptres, the Camera Module 3's nearest focus
STALL_S = 5.0                # no frame for this long, and the camera has stopped
FIRST_FRAME_S = 20.0         # the same, while a camera that was just opened starts
KEEP_INCIDENTS = 50
AF_SCANNING, AF_FOCUSED, AF_FAILED = 1, 2, 3     # libcamera's AfState
HEARTBEAT_S = 60             # how often run.json says the station is still running
WATCHDOG_S = 5               # how often systemd hears that it is, see deploy/kvihtai.service
LOOP_SILENT_S = 30           # the API's event loop silent this long, and systemd is not told


@dataclass
class CameraSettings:
    quarter_turns: int = 0          # clockwise turns that make the picture upright
    exposure_us: int = 2000
    gain: float = 8.0
    auto_gain: bool = True
    lens_position: float | None = None   # dioptres; None until focused once
    # A point on the plate nearer the camera, tapped in the app: (x, y) from 0
    # to 1 across the camera's own frame, so that turning the picture later
    # does not move it. None until tapped. See app.capture.live.
    near_plate: list | None = None


def turn_unit(x, y, quarter_turns):
    """A point given from 0 to 1 across a frame, after turning the frame clockwise."""
    k = quarter_turns % 4
    return [(x, y), (1 - y, x), (1 - x, 1 - y), (y, 1 - x)][k]


def rise_list(result):
    return [r for mv in result.get("movements", []) for r in mv["rises"]]


def contract_summary(result):
    """A result in the shape of app.api.schemas.SetSummary: one rep per rise."""
    traj = result.get("trajectory", [])
    t0 = traj[0]["t_ms"] if traj else 0.0
    reps = []
    for i, r in enumerate(rise_list(result)):
        pts = [p for p in traj if r["start_ms"] <= p["t_ms"] <= r["end_ms"]]
        xs = [p["x_mm"] for p in pts]
        reps.append({
            "rep_number": i + 1,
            "metrics": {
                "mean_concentric_velocity_mps": r["mean_concentric_velocity_mps"],
                "peak_velocity_mps": r["peak_velocity_mps"],
                "horizontal_loop_deviation_mm": round(max(xs) - min(xs), 1) if xs else 0.0,
                "vertical_displacement_mm": r["height_mm"],
            },
            "trajectory": [{"x_mm": p["x_mm"], "y_mm": p["y_mm"], "t_ms": round(p["t_ms"] - t0, 1),
                            "vy_mps": p.get("vy_mps")} for p in pts],
        })
    return {"set_id": result["set_id"], "timestamp": result["started_at"],
            "processed_video_url": (result.get("video") or {}).get("url"), "reps": reps}


class Station:
    def __init__(self, cfg: Settings, store=None):
        self.cfg = cfg
        self.data_dir = str(cfg.data_dir)
        self.sets_dir = os.path.join(self.data_dir, "sets")
        self.recordings_dir = os.path.join(self.data_dir, "recordings")
        self.store = store                      # app.db.repository, or a stand-in for tests
        self.camera = self._load_camera()
        self.session = None
        self.source = None
        self.recorder = None
        self.error = None
        self.clock_offset = 0.0                 # seconds added to the system clock, see set_clock
        self.last_set_id = None
        self.last_event = None                  # ("completed" | "failed", wall time)
        self.lines = deque(maxlen=200)          # recent log, shown in the app
        self.brightness = None
        self.fps = 0.0
        self.dropped = 0.0                      # share of the camera's frames lost, over the last 5 s
        self.system = {}
        self._stop = threading.Event()
        self._threads = []
        self._path = deque(maxlen=12)           # recent (t, x_mm, y_mm) of the followed plate
        self._path_t = None
        self.analyser = ProcessAnalyser()
        self.started = time.monotonic()
        self.focus = None                       # "focusing", "focused" or "failed", after the last autofocus
        self.incidents = self._load_incidents()
        self.exit = os._exit                    # how the station ends itself, see _watch_camera
        self._watched = (None, 0, 0.0)          # (session, frames in, when that number last changed)
        self.log_dir = os.path.join(self.data_dir, "logs")
        self.loop_beat = time.monotonic()       # the API's event loop sets this while it runs, see app.main
        self._run_state = None                  # what run.json holds while running
        for inc in self.incidents[-3:]:
            self.lines.append(f"earlier, {inc['at']}: {inc['what']}")

    # --- lifecycle ------------------------------------------------------------------

    def start(self):
        os.makedirs(self.sets_dir, exist_ok=True)
        logbook.install(self.log_dir)
        self._check_last_run()
        self._check_database()
        if self.cfg.camera != "none":
            self._threads.append(threading.Thread(target=self._run, name="station", daemon=True))
        self._threads.append(threading.Thread(target=self._housekeeping, name="housekeeping", daemon=True))
        for th in self._threads:
            th.start()

    def stop(self):
        # first: told to stop is a clean stop, even if stopping then takes too long
        self._clear_run()
        self._stop.set()
        if self.session is not None:
            self.session.stop()
        for th in self._threads:
            th.join(timeout=10)
        self.analyser.close()
        logbook.uninstall()

    def now(self):
        return time.time() + self.clock_offset

    def note(self, message, level=logging.INFO):
        stamp = datetime.fromtimestamp(self.now()).strftime("%H:%M:%S")
        for line in str(message).splitlines():
            self.lines.append(f"{stamp} {line}")
        log.log(level, message)

    def warn(self, message):
        self.note(message, logging.WARNING)

    def _make_source(self):
        kind = self.cfg.camera
        if kind.startswith("video:"):
            from app.capture.source import VideoFileSource
            return VideoFileSource(kind[len("video:"):], realtime=True, loop=True, pause_s=4.0)
        if kind == "pi":
            from app.capture.recorder import Recorder
            from app.capture.source import PiCameraSource
            if self.cfg.record:
                self.recorder = Recorder(self.recordings_dir, segment_s=self.cfg.record_segment_s,
                                         min_free_bytes=self.cfg.record_min_free_gb * 1e9,
                                         wall_clock=self.now, log=self.note)
                self.recorder.extra = {"quarter_turns": self.camera.quarter_turns}
            w, h = (int(v) for v in self.cfg.camera_size.lower().split("x"))
            src = PiCameraSource((w, h), self.cfg.target_fps, self.camera.exposure_us, self.camera.gain,
                                 recorder=self.recorder)
            return src
        raise ValueError(f"unknown camera {kind!r}")

    def _run(self):
        """Run capture sessions until stopped; a camera that fails is opened again."""
        while not self._stop.is_set():
            try:
                self.source = self._make_source()
                # 4 s of frames, about 1 GB at 60 fps: the slack the watcher has
                # when following a near plate on a Pi 4 cannot quite keep up
                self.session = Session(self.source, self.sets_dir, ring_seconds=4.0,
                                       log=self.note, on_result=self._on_result,
                                       clock=self.now, quarter_turns=self.camera.quarter_turns,
                                       analyser=self._analyse, near=self.camera.near_plate)
                self.error = None
                self.note(f"camera {self.cfg.camera} starting")
                if self.camera.lens_position is not None and hasattr(self.source, "set_controls"):
                    self._after_start(lambda: self.source.set_controls(AfMode=0,
                                                                      LensPosition=self.camera.lens_position))
                self.session.run()
                if not self._stop.is_set():
                    self.warn("camera stopped; starting it again")
            except Exception as e:
                self.error = repr(e)
                self.warn(f"camera failed: {e!r}")
            finally:
                self.recorder = None
            self._stop.wait(3.0)

    def _after_start(self, action):
        def later():
            for _ in range(100):
                if self._stop.is_set():
                    return
                if self.session is not None and self.session.ring is not None:
                    try:
                        action()
                    except Exception as e:
                        self.warn(f"camera setting failed: {e!r}")
                    return
                time.sleep(0.1)
        threading.Thread(target=later, daemon=True).start()

    # --- sets -------------------------------------------------------------------------

    def _analyse(self, rec, quarter_turns):
        """The analyser, with a failure kept as an incident: it is a fault to report."""
        try:
            return self.analyser(rec, quarter_turns)
        except Exception as e:
            self.add_incident(f"a set could not be analysed: {e}", "analysis")
            raise

    def _on_result(self, rec, result):
        """Store an analysed set, and cut its clip from the recording."""
        set_id = result["set_id"]
        if not rise_list(result):
            # the watcher saw a plate move, but nothing was lifted: a plate
            # knocked or carried past. It stays on disk, not in the app.
            self.note(f"set {set_id}: no lift in it, not kept")
            self.last_event = ("failed", time.monotonic())
            return
        self.last_set_id = set_id
        self.last_event = ("completed", time.monotonic())
        if self.store is not None:
            self.store.save_result(set_id, result["started_at"], result)
            self.store.save_set_summary(contract_summary(result))
        if self.recorder is not None and hasattr(self.source, "sensor_us"):
            span = (self.source.sensor_us(rec.times[0]), self.source.sensor_us(rec.times[-1]))
            threading.Thread(target=self._clip, args=(result, span), daemon=True).start()

    def _clip(self, result, span):
        """Cut the set out of the recording, once the segment holding it is closed."""
        recorder = self.recorder
        recorder.split()
        seg = None
        for _ in range(40):
            seg = recorder.find(*span)
            if seg is not None:
                break
            time.sleep(0.25)
        if seg is None:
            self.warn(f"set {result['set_id']}: no finished recording holds it, no clip")
            return
        seg_start = seg["sensor_origin_us"] + seg["encoder_start_us"]
        want = (span[0] - seg_start) / 1e6
        keys = [k / 1e6 for k in seg.get("keyframes_us", [0]) if k / 1e6 <= want] or [0.0]
        start = keys[-1]
        length = (span[1] - seg_start) / 1e6 - start + 0.2
        out = os.path.join(result["path"], "clip.mp4")
        turn = (result.get("quarter_turns") or 0) % 4
        cmd = ["ffmpeg", "-y", "-v", "error", "-display_rotation", str((360 - 90 * turn) % 360),
               "-ss", f"{start + 0.0005:.4f}", "-i", seg["path"], "-t", f"{length:.3f}",
               "-c", "copy", "-movflags", "+faststart", out]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            self.warn(f"set {result['set_id']}: cutting the clip failed: {done.stderr.strip()[:200]}")
            return
        # the clip's time zero on the set's own clock, which the trajectory uses
        t0_ms = result["source_span_s"][0] * 1000 - (want - start) * 1000
        result["video"] = {"url": f"/api/v1/sets/{result['set_id']}/clip.mp4", "t0_ms": round(t0_ms, 1),
                           "recording": seg["name"]}
        self._write_result(result)
        if self.store is not None:
            self.store.save_result(result["set_id"], result["started_at"], result)
            self.store.save_set_summary(contract_summary(result))

    def _set_dir(self, set_id):
        return os.path.join(self.sets_dir, os.path.basename(set_id))

    def _write_result(self, result):
        directory = self._set_dir(result["set_id"])
        if os.path.isdir(directory):
            write_json(os.path.join(directory, "result.json"), result, indent=1)

    def clip_path(self, set_id):
        path = os.path.join(self._set_dir(set_id), "clip.mp4")
        return path if os.path.exists(path) else None

    def delete_set(self, set_id):
        """Forget a set, and delete its files, so that it cannot come back from them."""
        gone = self.store.delete_result(set_id) if self.store is not None else False
        directory = self._set_dir(set_id)
        if os.path.basename(set_id) and os.path.isdir(directory):
            shutil.rmtree(directory, ignore_errors=True)
            gone = True
        if gone:
            self.note(f"set {set_id} deleted")
        return gone

    def _check_database(self):
        """Fill a database that was made new from the sets on disk."""
        if self.store is None or not hasattr(self.store, "health"):
            return
        health = self.store.health()
        if health["broken"]:
            self.add_incident(f"the database could not be read and was started again; the broken one is "
                              f"{health['broken']}", "storage")
        if not health["created"]:
            return
        added = 0
        for name in sorted(os.listdir(self.sets_dir)):
            result = read_json(os.path.join(self.sets_dir, name, "result.json"))
            if not isinstance(result, dict) or "set_id" not in result or not rise_list(result):
                continue
            try:
                self.store.save_result(result["set_id"], result["started_at"], result)
                self.store.save_set_summary(contract_summary(result))
                added += 1
            except Exception as e:              # one bad file must not stop the rest
                log.warning("set %s could not be added again: %r", name, e)
        if added:
            self.note(f"the database was new: {added} sets added again from the card")

    # --- the camera -------------------------------------------------------------------

    def _load_camera(self):
        data = read_json(os.path.join(self.data_dir, "camera.json"))
        try:
            return CameraSettings(**{k: v for k, v in data.items() if k in CameraSettings.__dataclass_fields__})
        except (AttributeError, TypeError):
            return CameraSettings()

    def _save_camera(self):
        os.makedirs(self.data_dir, exist_ok=True)
        write_json(os.path.join(self.data_dir, "camera.json"), asdict(self.camera))

    def update_camera(self, **changes):
        cam = self.camera
        if changes.get("quarter_turns") is not None:
            cam.quarter_turns = int(changes["quarter_turns"]) % 4
            if self.session is not None:
                self.session.quarter_turns = cam.quarter_turns
            if self.recorder is not None:
                self.recorder.extra["quarter_turns"] = cam.quarter_turns
        if changes.get("near_plate") is not None:
            # given as seen in the app, upright; kept as the camera sees it
            point = changes["near_plate"]
            cam.near_plate = None if not point else \
                [round(min(max(float(v), 0.0), 1.0), 4) for v in turn_unit(*point[:2], -cam.quarter_turns)]
            if self.session is not None:
                self.session.set_near(cam.near_plate)
        if changes.get("auto_gain") is not None:
            cam.auto_gain = bool(changes["auto_gain"])
        controls = {}
        if changes.get("exposure_us") is not None:
            cam.exposure_us = int(min(max(changes["exposure_us"], 100), int(1e6 / self.cfg.target_fps) - 500))
            controls["ExposureTime"] = cam.exposure_us
        if changes.get("gain") is not None:
            cam.gain = float(min(max(changes["gain"], MIN_GAIN), 16.0))
            cam.auto_gain = bool(changes.get("auto_gain", False))
            controls["AnalogueGain"] = cam.gain
        if changes.get("lens_position") is not None:
            cam.lens_position = round(float(min(max(changes["lens_position"], 0.0), MAX_LENS)), 3)
            controls.update(AfMode=0, LensPosition=cam.lens_position)     # manual, held there
            self.focus = None
        if controls and hasattr(self.source, "set_controls"):
            self.source.set_controls(**controls)
        self._save_camera()
        return self.camera_state()

    def autofocus(self):
        if not hasattr(self.source, "autofocus"):
            return False
        self.source.autofocus()
        self.focus = "focusing"

        def remember():
            # The lens moves for a second or two. Read before the camera says it
            # is done, the position is wherever the scan had got to: on the Pi
            # that once kept 0.1 m for a wall 0.2 m away.
            scanned = False
            state = None
            for _ in range(60):                 # metadata comes every 16th frame; 6 s in all
                time.sleep(0.1)
                state = (getattr(self.source, "metadata", {}) or {}).get("AfState")
                scanned = scanned or state == AF_SCANNING
                if scanned and state in (AF_FOCUSED, AF_FAILED):
                    break
            lens = (getattr(self.source, "metadata", {}) or {}).get("LensPosition")
            self.focus = "focused" if state == AF_FOCUSED else "failed"
            if lens is None:
                return
            self.camera.lens_position = round(float(lens), 3)
            self._save_camera()
            where = "far away" if lens < 0.05 else f"{1 / lens:.1f} m"
            self.note(f"focused at {where}" if self.focus == "focused" else
                      f"autofocus found nothing sharp; the lens stays at {where}")
        threading.Thread(target=remember, daemon=True).start()
        return True

    def camera_state(self):
        meta = getattr(self.source, "metadata", {}) or {}
        near = self.camera.near_plate
        return {
            **asdict(self.camera),
            # upright, as the app shows the picture
            "near_plate": None if near is None else [round(v, 4) for v in turn_unit(*near, self.camera.quarter_turns)],
            "kind": self.cfg.camera,
            "exposure_actual_us": meta.get("ExposureTime"),
            "gain_actual": round(meta["AnalogueGain"], 2) if "AnalogueGain" in meta else None,
            "lens_actual": round(meta["LensPosition"], 3) if "LensPosition" in meta else None,
            "lux": round(meta["Lux"], 1) if "Lux" in meta else None,
            "brightness": self.brightness,
            "focus": self.focus,
            "too_dark":bool(self.brightness is not None and self.brightness < 60 and self.camera.gain >= 15.9),
        }

    def _upright_size(self):
        if self.session is None or self.session.ring is None:
            return None
        h, w = self.session.ring._images.shape[1:3]
        return (h, w) if self.camera.quarter_turns % 2 else (w, h)

    def preview_jpeg(self, width=480, quality=70):
        """The newest frame, upright and reduced, as JPEG bytes; None without a camera."""
        ring = self.session.ring if self.session is not None else None
        if ring is None or ring.newest < 0:
            return None
        k = self.camera.quarter_turns % 4
        h, w = ring._images.shape[1:3]
        upright_w = h if k % 2 else w
        step = max(1, int(upright_w / width))           # a strided copy, not the whole 4 MB frame
        got = ring.read(ring.newest, step=step) or ring.read(ring.newest - 1, step=step)
        if got is None:
            return None
        image = got[0]
        scale = min(1.0, width / (upright_w / step))
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if k:
            small = cv2.rotate(small, {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180,
                                       3: cv2.ROTATE_90_COUNTERCLOCKWISE}[k])
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    # --- what is going on ---------------------------------------------------------------

    def _upright(self, x, y):
        ring = self.session.ring
        h, w = ring._images.shape[1:3]
        return turn_point(x, y, (w, h), self.camera.quarter_turns)

    def status(self):
        s = self.session
        live = s.live if s is not None else None
        size = self._upright_size()
        watched, plate = [], None
        if live is not None and size is not None:
            for p in list(live.watched):
                x, y = self._upright(p.x, p.y)
                watched.append({"x": round(x, 1), "y": round(y, 1), "r": round(p.r, 1)})
            if live.state == "active" and live.sighting is not None:
                _, x, y, r = live.sighting
                x, y = self._upright(x, y)
                plate = {"x": round(x, 1), "y": round(y, 1), "r": round(r, 1)}
        if self.cfg.camera == "none":
            state = "no camera"
        elif self.error is not None or live is None:
            state = "starting" if self.error is None else "camera error"
        elif live.state == "active":
            state = "set"
        elif s.analysing > 0:
            state = "analysing"
        else:
            state = "watching"
        rec = self.recorder
        return {
            "state": state,
            "error": self.error,
            "camera": self.cfg.camera,
            "fps": round(self.fps, 1),
            "dropped": round(self.dropped, 3),
            "frames_in": s.frames_in if s is not None else 0,
            "frame": None if size is None else {"w": size[0], "h": size[1]},
            "watched": watched,
            "plate": plate,
            "analysing": s.analysing if s is not None else 0,
            "last_set_id": self.last_set_id,
            "recording": None if rec is None else {
                "file": os.path.basename(rec.current() or "") or None,
                "error": rec.error,
                "free_gb": round(self.system.get("free_bytes", 0) / 1e9, 1),
            },
            "system": {k: v for k, v in self.system.items() if k != "free_bytes"},
            "clock": {"offset_s": round(self.clock_offset, 1), "now": datetime.fromtimestamp(self.now()).isoformat(
                timespec="seconds")},
            "brightness": self.brightness,
            "uptime_s": round(time.monotonic() - self.started),
            "incidents": self.incidents[-10:],
        }

    def live_payload(self):
        """The newest state in the shape of the live stream in docs/API_CONTRACT.md."""
        s = self.session
        live = s.live if s is not None else None
        state = "idle"
        if live is not None and live.state == "active":
            state = "active"
        elif self.last_event is not None and time.monotonic() - self.last_event[1] < 10:
            state = self.last_event[0]
        x_mm = y_mm = v = a = 0.0
        phase = "idle"
        if state == "active" and live.sighting is not None:
            t, x, y, r = live.sighting
            size = self._upright_size()
            ux, uy = self._upright(x, y)
            k = PLATE_DIAMETER_MM / 2 / r
            x_mm, y_mm = (ux - size[0] / 2) * k, (size[1] / 2 - uy) * k
            if self._path_t != t:
                self._path.append((t, x_mm, y_mm))
                self._path_t = t
            v, a = self._kinematics()
            phase = "rising" if v > 0.08 else "lowering" if v < -0.08 else "still"
        else:
            self._path.clear()
        return {
            "timestamp_ms": int(self.now() * 1000),
            "frame_id": s.frames_in if s is not None else 0,
            "lift_state": state,
            "bar_center": {"x_mm": round(x_mm, 1), "y_mm": round(y_mm, 1)},
            "kinematics": {"current_velocity_mps": round(v, 3), "acceleration_mps2": round(a, 2)},
            "phase": phase,
        }

    def _kinematics(self):
        """Vertical velocity and acceleration from the last few sightings; rough, for display only."""
        pts = list(self._path)
        if len(pts) < 3:
            return 0.0, 0.0
        t = np.array([p[0] for p in pts[-6:]])
        y = np.array([p[2] for p in pts[-6:]]) / 1000
        if t[-1] - t[0] <= 0:
            return 0.0, 0.0
        coef = np.polyfit(t - t[-1], y, 2 if len(t) >= 4 else 1)
        if len(coef) == 3:
            return float(coef[1]), float(2 * coef[0])
        return float(coef[0]), 0.0

    # --- clock ------------------------------------------------------------------------

    def set_clock(self, epoch_ms):
        """Take the time from a phone, if this computer has no better source.

        At the gym the Pi runs its own hotspot, has no internet and no clock
        of its own, and wakes up at whatever time it was switched off. The
        phone knows the time. The system clock is set if that is allowed,
        otherwise the station keeps the difference itself.
        """
        diff = epoch_ms / 1000 - time.time()
        synced = _ntp_synchronized()
        if synced or abs(diff) < 2:
            return {"adjusted": False, "by_s": round(diff, 1), "ntp": synced}
        done = subprocess.run(["sudo", "-n", "date", "-s", f"@{epoch_ms / 1000:.3f}"], capture_output=True)
        if done.returncode == 0:
            self.clock_offset = 0.0
            how = "system clock"
        else:
            self.clock_offset = diff
            how = "offset"
        self.note(f"clock set from the phone, {diff:+.0f} s ({how})")
        return {"adjusted": True, "by_s": round(diff, 1), "ntp": False, "how": how}

    # --- housekeeping -------------------------------------------------------------------

    def _housekeeping(self):
        counts = deque(maxlen=10)               # (time, frames in, frames the camera made), 5 s of them
        tick = 0
        system.notify("READY=1")
        while not self._stop.wait(0.5):
            tick += 1
            try:
                self._keep_alive(tick)
            except Exception as e:              # never let this take the station down
                log.warning("keeping alive: %r", e)
            s = self.session
            now = time.monotonic()
            if s is not None:
                made = getattr(self.source, "_index", None)
                counts.append((now, s.frames_in, made))
                t0, in0, made0 = counts[0]
                if now > t0 and s.frames_in >= in0:
                    self.fps = (s.frames_in - in0) / (now - t0)
                    if made is not None and made0 is not None and made > made0:
                        # the camera numbers its frames; a frame it made that
                        # never reached the ring buffer was dropped for want of time
                        self.dropped = max(0.0, 1 - (s.frames_in - in0) / (made - made0))
                try:
                    self._measure_brightness()
                except Exception as e:          # never let this take the station down
                    log.debug("brightness: %r", e)
                self._watch_camera(s, now)
            if tick % 10 == 1:
                self.system = system.vitals(self.data_dir)
                self._note_power_trouble()

    def _watch_camera(self, s, now):
        """End the station when the camera has stopped sending frames.

        On the Pi 4B the Camera Module 3 has stopped for good, with no error,
        20 s after it was opened. picamera2 then waits for the next
        frame forever, and the app went on showing the last picture at 0 fps
        until the Pi was switched off. Stopping a camera whose driver has hung
        can hang as well, so the process ends, systemd starts it again, and the
        camera is opened afresh. What happened is kept in incidents.json and
        shown in the app.
        """
        if self.cfg.camera != "pi" or self.error is not None or self._stop.is_set():
            return
        session, frames, since = self._watched
        if session is not s or frames != s.frames_in:
            self._watched = (s, s.frames_in, now)
            return
        limit = FIRST_FRAME_S if frames == 0 else STALL_S
        if now - since < limit:
            return
        what = ("the camera sent no picture after it was opened" if frames == 0 else
                f"the camera stopped sending pictures after {frames} frames")
        self.add_incident(f"{what}; the station restarted", "camera")
        self.warn(f"{what}: restarting the station")
        # the incident says why; run.json would have the next start report a crash
        self._clear_run()
        logging.shutdown()
        self.exit(75)

    def _load_incidents(self):
        got = read_json(os.path.join(self.data_dir, "incidents.json"), [])
        return list(got)[-KEEP_INCIDENTS:] if isinstance(got, list) else []

    def add_incident(self, what, kind=None):
        """Note something that went wrong, so that it is still known after a restart."""
        incident = {"at": datetime.fromtimestamp(self.now()).isoformat(timespec="seconds"), "what": what}
        if kind:
            incident["kind"] = kind
        self.incidents.append(incident)
        del self.incidents[:-KEEP_INCIDENTS]
        log.warning("incident: %s", what)
        self._save_incidents()

    def clear_incidents(self):
        self.incidents = []
        self._save_incidents()
        self.note("incidents cleared from the app")

    def _save_incidents(self):
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            write_json(os.path.join(self.data_dir, "incidents.json"), self.incidents)
        except OSError as e:
            log.warning("incidents: %r", e)

    # --- how the last run ended -----------------------------------------------------------

    def _run_path(self):
        return os.path.join(self.data_dir, "run.json")

    def _check_last_run(self):
        """Tell from run.json whether the station was stopped last time, and if not, why not.

        A run.json left behind means the process ended without being stopped.
        In the same boot it crashed, ran out of memory or was killed; in an
        earlier boot the computer went off under it, which is nearly always
        the plug pulled without shutting down.
        """
        last = read_json(self._run_path())
        boot = system.boot_id()
        now = datetime.fromtimestamp(self.now()).isoformat(timespec="seconds")
        same_boot = isinstance(last, dict) and last.get("boot_id") == boot
        self._run_state = {"boot_id": boot, "pid": os.getpid(), "started": now, "seen": now,
                           "noted": list(last.get("noted", [])) if same_boot else []}
        if isinstance(last, dict):
            seen = str(last.get("seen") or last.get("started") or "")
            at = seen[11:16] if len(seen) >= 16 else "an unknown time"
            day = "" if seen[:10] == now[:10] else f" on {seen[:10]}"
            if boot is not None and last.get("boot_id") not in (None, boot):
                self.add_incident(f"the Pi went off at about {at}{day} without being shut down", "power")
            else:
                why = system.last_exit_reason()
                self.add_incident(f"the station stopped unexpectedly at about {at}{day}"
                                  f"{f' ({why})' if why else ''} and was started again", "crash")
        self._write_run()

    def _write_run(self):
        if self._run_state is None:
            return
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            write_json(self._run_path(), self._run_state)
        except OSError as e:
            log.warning("run.json: %r", e)

    def _clear_run(self):
        self._run_state = None
        try:
            os.remove(self._run_path())
        except OSError:
            pass

    def _keep_alive(self, tick):
        """Say that the station still runs: to run.json now and then, to systemd often."""
        if self._run_state is not None and tick % int(HEARTBEAT_S / 0.5) == 0:
            self._run_state["seen"] = datetime.fromtimestamp(self.now()).isoformat(timespec="seconds")
            self._write_run()
        # only while the API answers too: a station that shows nothing is no use
        if tick % int(WATCHDOG_S / 0.5) == 0 and time.monotonic() - self.loop_beat < LOOP_SILENT_S:
            system.notify("WATCHDOG=1")

    def _note_power_trouble(self):
        """Keep undervoltage and overheating as incidents, once a boot: they come and go too fast to see."""
        if self._run_state is None:
            return
        for flag, kind, what in (
                ("undervoltage_since_boot", "undervoltage",
                 "the power supply's voltage dropped too low (undervoltage); use the official supply "
                 "and a short cable"),
                ("throttled_since_boot", "heat", "the Pi got too hot and slowed itself down")):
            if self.system.get(flag) and kind not in self._run_state["noted"]:
                self._run_state["noted"].append(kind)
                self.add_incident(what, kind)
                self._write_run()

    # --- power and recordings, from the app ----------------------------------------------

    def power_abilities(self):
        return system.power_abilities(self.cfg.power_control)

    def power(self, action):
        """Restart the station, or reboot or shut down the computer. (accepted, why not)."""
        can = self.power_abilities()
        if not can.get(action):
            return False, can["why"].get(action, "not possible")
        words = {"restart": "restarting the station", "reboot": "rebooting the Pi",
                 "shutdown": "shutting the Pi down"}[action]
        self.note(f"{words}, as asked in the app")

        def later():
            time.sleep(1.0)                     # let the answer reach the phone first
            if action == "restart":
                system.restart_self()
                return
            failed = system.power(action)
            if failed:
                self.add_incident(f"{words} failed: {failed}", "power")
        threading.Thread(target=later, name="power", daemon=True).start()
        return True, None

    def recordings(self):
        rec = self.recorder
        return list_segments(self.recordings_dir, rec.current() if rec is not None else None)



def _ntp_synchronized():
    try:
        out = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
                             capture_output=True, text=True, timeout=2)
        return out.stdout.strip() == "yes"
    except (OSError, subprocess.SubprocessError):
        return False
