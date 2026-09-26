"""The API, with the station running without a camera."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from app.api.schemas import SetSummary
from app.config import settings
from app.db import repository
from app.main import create_app
from app.station import contract_summary


def _result(set_id, started_at):
    traj = [{"t_ms": 1000.0 + 10 * i, "x_mm": 0.5 * i, "y_mm": 5.0 * i, "vy_mps": 0.5,
             "x_px": 100.0, "y_px": 200.0 - i} for i in range(100)]
    rise = {"start_ms": 1100.0, "end_ms": 1800.0, "height_mm": 350.0, "mean_concentric_velocity_mps": 0.5,
            "peak_velocity_mps": 0.9, "peak_ms": 1500.0}
    return {"set_id": set_id, "started_at": started_at, "frames": 100, "measured": 100, "rejected": {},
            "frame_size": [720, 1280], "duration_ms": 990.0,
            "camera": {"angle_deg": 10.0, "axis_ratio": 0.98, "angle_corrected": False, "mm_per_px": 2.8},
            "metrics": {"vertical_displacement_mm": 495.0, "horizontal_deviation_mm": 49.5},
            "movements": [{"start_ms": 1000.0, "end_ms": 1990.0, "low_mm": 0.0, "high_mm": 495.0, "rises": [rise]}],
            "trajectory": traj, "path": "/somewhere/on/disk"}


@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c


def test_status_without_a_camera(client):
    status = client.get("/api/v1/status").json()
    assert status["state"] == "no camera"
    assert status["frame"] is None and status["watched"] == []


def test_sets_are_listed_served_and_deleted(client):
    for set_id, when in [("20260926-100000", "2026-09-26T08:00:00+00:00"),
                         ("20260926-110000", "2026-09-26T09:00:00+00:00")]:
        result = _result(set_id, when)
        repository.save_result(set_id, when, result)
        repository.save_set_summary(contract_summary(result))

    listed = client.get("/api/v1/sets").json()
    assert [s["set_id"] for s in listed][:2] == ["20260926-110000", "20260926-100000"]
    assert "trajectory" not in listed[0] and listed[0]["rises"][0]["height_mm"] == 350.0

    full = client.get("/api/v1/sets/20260926-110000").json()
    assert len(full["trajectory"]) == 100 and "path" not in full

    latest = client.get("/api/v1/sets/latest").json()
    SetSummary.model_validate(latest)
    assert latest["set_id"] == "20260926-110000"
    rep = latest["reps"][0]
    assert rep["metrics"]["vertical_displacement_mm"] == 350.0
    assert rep["trajectory"][0]["t_ms"] == pytest.approx(100.0)       # from the start of the set
    assert all(1100.0 <= p["t_ms"] + 1000.0 <= 1800.0 for p in rep["trajectory"])

    assert client.delete("/api/v1/sets/20260926-110000").status_code == 200
    assert client.get("/api/v1/sets/20260926-110000").status_code == 404
    assert client.get("/api/v1/sets/latest").json()["set_id"] == "20260926-100000"
    assert client.get("/api/v1/sets/20260926-110000/clip.mp4").status_code == 404


def test_camera_settings_are_kept(client):
    changed = client.put("/api/v1/camera", json={"quarter_turns": 5}).json()
    assert changed["quarter_turns"] == 1
    with open(os.path.join(settings.data_dir, "camera.json")) as fh:
        assert json.load(fh)["quarter_turns"] == 1
    assert client.get("/api/v1/camera").json()["quarter_turns"] == 1
    client.put("/api/v1/camera", json={"quarter_turns": 0})


def test_no_picture_and_no_focus_without_a_camera(client):
    assert client.get("/api/v1/camera/preview.jpg").status_code == 503
    assert client.post("/api/v1/camera/focus").status_code == 409


def test_live_stream_follows_the_contract(client):
    with client.websocket_connect("/ws/live") as ws:
        live = ws.receive_json()
    assert set(live) == {"timestamp_ms", "frame_id", "lift_state", "bar_center", "kinematics", "phase"}
    assert live["lift_state"] == "idle"
    assert set(live["bar_center"]) == {"x_mm", "y_mm"}
    assert set(live["kinematics"]) == {"current_velocity_mps", "acceleration_mps2"}
