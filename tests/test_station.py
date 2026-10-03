"""The station's own upkeep, without a camera."""

from types import SimpleNamespace

from app.config import Settings
from app.station import FIRST_FRAME_S, STALL_S, Station


def _station(tmp_path, camera="pi"):
    station = Station(Settings(camera=camera, data_dir=tmp_path))
    exits = []
    station.exit = exits.append
    return station, exits


def test_a_camera_that_stops_sending_frames_restarts_the_station(tmp_path):
    station, exits = _station(tmp_path)
    s = SimpleNamespace(frames_in=0)
    station._watch_camera(s, 0.0)
    s.frames_in = 1200
    station._watch_camera(s, 1.0)                   # frames coming: nothing to do
    station._watch_camera(s, 1.0 + STALL_S - 0.5)
    assert exits == []
    station._watch_camera(s, 1.0 + STALL_S + 0.5)
    assert exits == [75]
    assert "after 1200 frames" in station.incidents[-1]["what"]
    # kept for the station that starts next
    assert Station(Settings(camera="pi", data_dir=tmp_path)).incidents == station.incidents


def test_a_camera_just_opened_gets_longer_for_its_first_frame(tmp_path):
    station, exits = _station(tmp_path)
    s = SimpleNamespace(frames_in=0)
    station._watch_camera(s, 0.0)
    station._watch_camera(s, STALL_S + 1)
    assert exits == []
    station._watch_camera(s, FIRST_FRAME_S + 1)
    assert exits == [75]


def test_a_new_session_and_other_sources_are_not_taken_for_a_stall(tmp_path):
    station, exits = _station(tmp_path)
    station._watch_camera(SimpleNamespace(frames_in=50), 0.0)
    station._watch_camera(SimpleNamespace(frames_in=50), 30.0)      # opened again since
    assert exits == []
    video, exits = _station(tmp_path, camera="video:clip.mp4")      # pauses between plays
    s = SimpleNamespace(frames_in=50)
    video._watch_camera(s, 0.0)
    video._watch_camera(s, 30.0)
    assert exits == []
