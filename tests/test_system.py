"""The station's upkeep: power, how the last run ended, logs, and a database
that cannot be read."""

import io
import json
import logging
import os
import threading
import zipfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import logbook, system
from app.config import Settings
from app.db import database
from app.main import create_app
from app.station import Station
from tests.test_api import _result


@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c


class Store:
    """A database in memory, in place of app.db.repository."""

    def __init__(self, created=False, broken=None):
        self.results = {}
        self.state = {"created": created, "broken": broken}

    def health(self):
        return self.state

    def save_result(self, set_id, started_at, result):
        self.results[set_id] = json.loads(json.dumps(result))

    def save_set_summary(self, data):
        pass

    def get_result(self, set_id):
        r = self.results.get(set_id)
        return None if r is None else json.loads(json.dumps(r))

    def delete_result(self, set_id):
        return self.results.pop(set_id, None) is not None


def _station(tmp_path, store=None, camera="pi"):
    station = Station(Settings(camera=camera, data_dir=tmp_path), store=store)
    station.exit = lambda code: None
    return station


# --- how the last run ended -------------------------------------------------------------

def test_a_power_cut_is_told_from_a_crash_and_from_a_shutdown(tmp_path, monkeypatch):
    monkeypatch.setattr(system, "boot_id", lambda: "boot-2")
    run = tmp_path / "run.json"

    run.write_text(json.dumps({"boot_id": "boot-1", "seen": "2026-10-03T08:12:30"}))
    station = _station(tmp_path)
    station._check_last_run()
    assert station.incidents[-1]["kind"] == "power"
    assert "08:12" in station.incidents[-1]["what"] and "without being shut down" in station.incidents[-1]["what"]
    assert json.loads(run.read_text())["boot_id"] == "boot-2"      # this run, now

    monkeypatch.setattr(system, "last_exit_reason", lambda: "out of memory")
    station = _station(tmp_path)
    station._check_last_run()                                        # the run.json the last one left
    assert station.incidents[-1]["kind"] == "crash"
    assert "(out of memory)" in station.incidents[-1]["what"]

    station._clear_run()                                             # as stop() does
    before = len(station.incidents)
    station = _station(tmp_path)
    station._check_last_run()
    assert len(station.incidents) == before


def test_a_camera_restart_is_not_taken_for_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(system, "boot_id", lambda: "boot-1")
    station = _station(tmp_path)
    station._check_last_run()
    s = SimpleNamespace(frames_in=10)
    station._watch_camera(s, 0.0)
    station._watch_camera(s, 60.0)
    assert not (tmp_path / "run.json").exists()
    station = _station(tmp_path)
    station._check_last_run()
    assert [i["kind"] for i in station.incidents] == ["camera"]


def test_undervoltage_is_kept_once_a_boot(tmp_path, monkeypatch):
    monkeypatch.setattr(system, "boot_id", lambda: "boot-1")
    station = _station(tmp_path)
    station._check_last_run()
    station.system = {"undervoltage_since_boot": True, "throttled_since_boot": False}
    station._note_power_trouble()
    station._note_power_trouble()
    assert [i["kind"] for i in station.incidents] == ["undervoltage"]
    # a station restarted in the same boot does not say it again
    station._clear_run()
    station._run_state = None
    again = _station(tmp_path)
    (tmp_path / "run.json").write_text(json.dumps({"boot_id": "boot-1", "noted": ["undervoltage"]}))
    again._check_last_run()
    again.system = station.system
    again._note_power_trouble()
    assert [i["kind"] for i in again.incidents] == ["undervoltage", "crash"]


# --- power -------------------------------------------------------------------------------

def test_the_computer_is_not_switched_off_unless_allowed(client, monkeypatch):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    for action in ("shutdown", "reboot", "restart"):
        r = client.post("/api/v1/system/power", json={"action": action})
        assert r.status_code == 409 and r.json()["detail"]
    assert client.post("/api/v1/system/power", json={"action": "explode"}).status_code == 422


def test_a_shutdown_is_handed_to_systemd(tmp_path, monkeypatch):
    monkeypatch.setattr(system, "power_abilities",
                        lambda enabled: {"shutdown": True, "reboot": True, "restart": True, "why": {}})
    asked = []
    monkeypatch.setattr(system, "power", lambda action: asked.append(action) or None)
    station = _station(tmp_path)
    assert station.power("shutdown") == (True, None)
    for th in threading.enumerate():
        if th.name == "power":
            th.join(timeout=5)
    assert asked == ["shutdown"]
    assert any("shutting the Pi down" in line for line in station.lines)


# --- the log ------------------------------------------------------------------------------

def test_the_log_keeps_warnings_from_anywhere_and_whole_tracebacks(tmp_path):
    logbook.install(str(tmp_path))
    try:
        logging.getLogger("kvihtai.station").info("camera pi starting")
        logging.getLogger("uvicorn.access").info("GET /api/v1/status 200")       # noise, not kept
        logging.getLogger("somewhere").warning("disk is slow")

        def die():
            raise ValueError("broken frame")
        th = threading.Thread(target=die, name="watcher")
        th.start()
        th.join()
    finally:
        logbook.uninstall()
    records = logbook.read(str(tmp_path))
    assert any("camera pi starting" in r for r in records)
    assert not any("GET /api" in r for r in records)
    problems = logbook.read(str(tmp_path), problems=True)
    assert not any("camera pi starting" in r for r in problems)
    assert any("disk is slow" in r for r in problems)
    crash = [r for r in problems if "thread watcher died" in r]
    assert crash and "ValueError: broken frame" in crash[0] and "Traceback" in crash[0]


def test_log_and_journal_over_the_api(client):
    logging.getLogger("kvihtai.station").warning("something to see")
    records = client.get("/api/v1/system/log", params={"problems": True}).json()["records"]
    assert any("something to see" in r for r in records)
    assert client.get("/api/v1/system/journal", params={"source": "nonsense"}).status_code == 422
    got = client.get("/api/v1/system/journal", params={"lines": 5}).json()
    assert set(got) == {"available", "lines", "error"}


def test_incidents_are_listed_and_cleared(client):
    client.app.state.station.add_incident("the Pi went off at about 08:12 without being shut down", "power")
    listed = client.get("/api/v1/incidents").json()
    assert listed[-1]["kind"] == "power"
    assert client.delete("/api/v1/incidents").status_code == 200
    assert client.get("/api/v1/incidents").json() == []


def test_system_info_and_the_diagnostics_bundle(client):
    info = client.get("/api/v1/system").json()
    for key in ("hostname", "model", "os", "version", "memory", "disk", "network", "power", "storage"):
        assert key in info
    assert info["power"]["shutdown"] is False and info["power"]["why"]["shutdown"]

    r = client.get("/api/v1/system/diagnostics.zip")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    for name in ("status.json", "system.json", "camera.json", "incidents.json", "sets.json",
                 "journal/station-this-boot.txt"):
        assert name in names


def test_a_deleted_set_takes_its_files_with_it(tmp_path):
    store = Store()
    station = _station(tmp_path, store)
    result = _result("20261003-100000", "2026-10-03T08:00:00+00:00")
    store.save_result(result["set_id"], result["started_at"], result)
    os.makedirs(tmp_path / "sets" / "20261003-100000")
    assert station.delete_set("20261003-100000")
    assert not (tmp_path / "sets" / "20261003-100000").exists()
    assert not station.delete_set("20261003-100000")
    assert not station.delete_set("../../etc")


# --- the database ------------------------------------------------------------------------------

def test_a_broken_database_is_moved_aside(tmp_path, monkeypatch):
    path = tmp_path / "kvihtai.db"
    path.write_bytes(b"this is not a database" * 100)
    monkeypatch.setattr(database, "DB_PATH", path)
    monkeypatch.setattr(database, "CREATED", False)
    monkeypatch.setattr(database, "BROKEN", None)
    database.initialize_db()
    assert database.CREATED and database.BROKEN.startswith("kvihtai.db.broken-")
    assert (tmp_path / database.BROKEN).exists()
    with database.get_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 0


def test_a_new_database_is_filled_from_the_sets_on_disk(tmp_path):
    store = Store(created=True, broken="kvihtai.db.broken-20261003-080000")
    station = _station(tmp_path, store)
    os.makedirs(station.sets_dir)
    for set_id, rises in (("20261003-070000", True), ("20261003-071000", False)):
        result = _result(set_id, "2026-10-03T05:00:00+00:00")
        if not rises:
            result["movements"] = []
        os.makedirs(os.path.join(station.sets_dir, set_id))
        with open(os.path.join(station.sets_dir, set_id, "result.json"), "w") as fh:
            json.dump(result, fh)
    os.makedirs(os.path.join(station.sets_dir, "20261003-072000"))        # cut off before its result
    station._check_database()
    assert list(store.results) == ["20261003-070000"]
    assert station.incidents[-1]["kind"] == "storage"
