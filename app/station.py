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
"""

import json
import logging
import math
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime

import cv2
import numpy as np

from app.capture.session import Session, turn_point
from app.capture.worker import ProcessAnalyser
from app.config import Settings
from app.vision.plate import PLATE_DIAMETER_MM

log = logging.getLogger("kvihtai.station")

TARGET_BRIGHTNESS = 110      # median of the picture, 0-255, that the gain aims for
MIN_GAIN = 1.0


@dataclass
class CameraSettings:
    quarter_turns: int = 0          # clockwise turns that make the picture upright
    exposure_us: int = 2000
    gain: float = 8.0
    auto_gain: bool = True
    lens_position: float | None = None   # dioptres; None until focused once


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

    # --- lifecycle ------------------------------------------------------------------

    def start(self):
        os.makedirs(self.sets_dir, exist_ok=True)
        if self.cfg.camera != "none":
            self._threads.append(threading.Thread(target=self._run, name="station", daemon=True))
        self._threads.append(threading.Thread(target=self._housekeeping, name="housekeeping", daemon=True))
        for th in self._threads:
            th.start()

    def stop(self):
        self._stop.set()
        if self.session is not None:
            self.session.stop()
        for th in self._threads:
            th.join(timeout=10)
        self.analyser.close()

    def now(self):
        return time.time() + self.clock_offset

    def note(self, message):
        stamp = datetime.fromtimestamp(self.now()).strftime("%H:%M:%S")
        for line in str(message).splitlines():
            self.lines.append(f"{stamp} {line}")
        log.info(message)

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
                                       analyser=self.analyser)
                self.error = None
                self.note(f"camera {self.cfg.camera} starting")
                if self.camera.lens_position is not None and hasattr(self.source, "set_controls"):
                    self._after_start(lambda: self.source.set_controls(AfMode=0,
                                                                      LensPosition=self.camera.lens_position))
                self.session.run()
                if not self._stop.is_set():
                    self.note("camera stopped; starting it again")
            except Exception as e:
                self.error = repr(e)
                self.note(f"camera failed: {e!r}")
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
                        self.note(f"camera setting failed: {e!r}")
                    return
                time.sleep(0.1)
        threading.Thread(target=later, daemon=True).start()

    # --- sets -------------------------------------------------------------------------

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
            self.note(f"set {result['set_id']}: no finished recording holds it, no clip")
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
            self.note(f"set {result['set_id']}: cutting the clip failed: {done.stderr.strip()[:200]}")
            return
        # the clip's time zero on the set's own clock, which the trajectory uses
        t0_ms = result["source_span_s"][0] * 1000 - (want - start) * 1000
        result["video"] = {"url": f"/api/v1/sets/{result['set_id']}/clip.mp4", "t0_ms": round(t0_ms, 1),
                           "recording": seg["name"]}
        with open(os.path.join(result["path"], "result.json"), "w") as fh:
            json.dump(result, fh, indent=1)
        if self.store is not None:
            self.store.save_result(result["set_id"], result["started_at"], result)
            self.store.save_set_summary(contract_summary(result))

    def clip_path(self, set_id):
        path = os.path.join(self.sets_dir, os.path.basename(set_id), "clip.mp4")
        return path if os.path.exists(path) else None

    # --- the camera -------------------------------------------------------------------

    def _load_camera(self):
        try:
            with open(os.path.join(self.data_dir, "camera.json")) as fh:
                data = json.load(fh)
            return CameraSettings(**{k: v for k, v in data.items() if k in CameraSettings.__dataclass_fields__})
        except (OSError, ValueError, TypeError):
            return CameraSettings()

    def _save_camera(self):
        os.makedirs(self.data_dir, exist_ok=True)
        with open(os.path.join(self.data_dir, "camera.json"), "w") as fh:
            json.dump(asdict(self.camera), fh)

    def update_camera(self, **changes):
        cam = self.camera
        if changes.get("quarter_turns") is not None:
            cam.quarter_turns = int(changes["quarter_turns"]) % 4
            if self.session is not None:
                self.session.quarter_turns = cam.quarter_turns
            if self.recorder is not None:
                self.recorder.extra["quarter_turns"] = cam.quarter_turns
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
        if controls and hasattr(self.source, "set_controls"):
            self.source.set_controls(**controls)
        self._save_camera()
        return self.camera_state()

    def autofocus(self):
        if not hasattr(self.source, "autofocus"):
            return False
        self.source.autofocus()

        def remember():
            time.sleep(2.5)
            lens = (getattr(self.source, "metadata", {}) or {}).get("LensPosition")
            if lens is not None:
                self.camera.lens_position = round(float(lens), 3)
                self._save_camera()
                self.note(f"focused at {1 / max(lens, 1e-3):.1f} m")
        threading.Thread(target=remember, daemon=True).start()
        return True

    def camera_state(self):
        meta = getattr(self.source, "metadata", {}) or {}
        return {
            **asdict(self.camera),
            "kind": self.cfg.camera,
            "exposure_actual_us": meta.get("ExposureTime"),
            "gain_actual": round(meta["AnalogueGain"], 2) if "AnalogueGain" in meta else None,
            "lens_actual": round(meta["LensPosition"], 3) if "LensPosition" in meta else None,
            "lux": round(meta["Lux"], 1) if "Lux" in meta else None,
            "brightness": self.brightness,
            "too_dark": bool(self.brightness is not None and self.brightness < 60 and self.camera.gain >= 15.9),
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
        while not self._stop.wait(0.5):
            tick += 1
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
            if tick % 10 == 1:
                self.system = _system_state(self.data_dir)

    def _measure_brightness(self):
        s = self.session
        ring = s.ring if s is not None else None
        if ring is None or ring.newest < 0:
            return
        got = ring.read(ring.newest, step=8)
        if got is None:
            return
        gray = cv2.cvtColor(got[0], cv2.COLOR_BGR2GRAY)
        self.brightness = int(np.median(gray))
        cam = self.camera
        if (not cam.auto_gain or s.live is None or s.live.state != "idle" or s.analysing
                or not hasattr(self.source, "set_controls")):
            return
        # a half step towards the target each time, in the log of the gain
        want = cam.gain * math.sqrt(TARGET_BRIGHTNESS / max(self.brightness, 4))
        want = float(min(max(want, MIN_GAIN), 16.0))
        if abs(math.log(want / cam.gain)) > 0.05:
            cam.gain = round(want, 2)
            self.source.set_controls(AnalogueGain=cam.gain)


def _ntp_synchronized():
    try:
        out = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
                             capture_output=True, text=True, timeout=2)
        return out.stdout.strip() == "yes"
    except (OSError, subprocess.SubprocessError):
        return False


def _system_state(path):
    state = {}
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as fh:
            state["temp_c"] = round(int(fh.read()) / 1000, 1)
    except OSError:
        pass
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2)
        value = int(out.stdout.strip().split("=")[1], 16)
        state["throttled_now"] = bool(value & 0x4)
        state["throttled_since_boot"] = bool(value & 0x40000)
        state["undervoltage"] = bool(value & 0x1)
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        pass
    try:
        st = os.statvfs(path)
        state["free_bytes"] = st.f_bavail * st.f_frsize
    except OSError:
        pass
    state["load"] = round(os.getloadavg()[0], 2)
    return state
