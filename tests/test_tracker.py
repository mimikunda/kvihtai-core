"""The whole tracker on synthetic clips held in memory.

Each clip is a plate rising past a ring of exactly its size, the thing that
fooled the first version of the coarse stage on real footage. The clips are
easy on purpose: they check that the stages fit together and that nothing in
them depends on the plate's colour, not how the tracker copes with a gym.
"""

import numpy as np
import pytest

import synthetic as syn
from app.analysis import kinematics
from app.vision import tracker
from app.vision.frames import Frames


def run(frames, times):
    return tracker.track(Frames(frames, times))


@pytest.mark.parametrize("colour", ["BLACK", "RED", "YELLOW", "BLUE"])
def test_follows_a_plate_of_any_colour_past_a_ring(colour):
    frames, times, truth = syn.rising_clip(n=30, face=getattr(syn, colour))
    st = run(frames, times)
    acc = st.accepted
    assert len(acc) >= 27, [(m.frame, m.reason) for m in st.measurements if m and not m.accepted]
    err = [np.hypot(m.face_x - truth[m.frame][0], m.face_y - truth[m.frame][1]) for m in acc]
    assert max(err) < 0.3, max(err)


def test_finds_the_face_inside_a_visible_tread():
    """The silhouette's centre is half the tread off; the face's is on the bar."""
    tread = (-6.0, -3.0)
    frames, times, truth = syn.rising_clip(n=30, face=syn.RED, tread=tread)
    st = run(frames, times)
    acc = st.accepted
    assert len(acc) >= 27
    face_err = np.median([np.hypot(m.face_x - truth[m.frame][0], m.face_y - truth[m.frame][1]) for m in acc])
    outline_err = np.median([np.hypot(m.x - truth[m.frame][0], m.y - truth[m.frame][1]) for m in acc])
    assert outline_err > 2.0          # the outline is the silhouette, tread and all
    assert face_err < 0.5, face_err


def test_scale_and_bar_path_in_millimetres():
    frames, times, truth = syn.rising_clip(n=30, radius=50.0)
    st = run(frames, times)
    t, pos = kinematics.bar_path(st.measurements, st.face, (320, 480))
    mm_per_px = 450.0 / 100.0
    assert np.median([m.mm_per_px for m in st.accepted]) == pytest.approx(mm_per_px, rel=0.01)
    rise = pos[-1, 1] - pos[0, 1]
    true_rise = (truth[0][1] - truth[len(pos) - 1][1]) * mm_per_px if len(pos) == 30 else None
    if true_rise is not None:
        assert rise == pytest.approx(true_rise, rel=0.01)
    v = kinematics.velocity(t, pos)
    # 5.5 px a frame at 60 fps, 4.5 mm a pixel: 1.485 m/s upwards
    assert np.nanmedian(v[:, 1]) / 1000 == pytest.approx(1.485, rel=0.02)


def test_a_ring_alone_is_not_a_plate():
    frames, times = [], np.arange(20) / 60.0
    for i in range(20):
        img = syn.textured_background((320, 480), seed=2)
        syn.draw_ring(img, (160.0, 300.0 - 5.0 * i), 50.0)
        frames.append(img)
    try:
        st = run(frames, times)
    except ValueError:
        return                                   # no plate found at all: fine
    assert len(st.accepted) <= 2


def test_a_square_view_is_left_uncorrected():
    frames, times, _ = syn.rising_clip(n=30)
    st = run(frames, times)
    assert st.face.camera_angle_deg < tracker.MIN_CORRECTED_ANGLE
    assert not st.face.corrected


def test_an_oblique_view_is_measured_and_corrected():
    """25 degrees off square: the face is an ellipse with its long axis upright.

    Not more: from about 30 degrees the coarse stage, which looks for circles,
    hands the fine stage a centre off by more than it can recover from. See
    docs/DESIGN.md.
    """
    ratio = float(np.cos(np.radians(25.0)))
    frames, times, _ = syn.rising_clip(n=30, ratio=ratio)
    st = run(frames, times)
    assert len(st.accepted) >= 27
    assert st.face.corrected
    assert st.face.camera_angle_deg == pytest.approx(25.0, abs=1.5)
    t, pos = kinematics.bar_path(st.measurements, st.face, (320, 480))
    v = kinematics.velocity(t, pos)
    # 0.3 px a frame sideways at 60 fps and 4.5 mm a pixel, foreshortened by the ratio
    assert np.nanmedian(v[:, 0]) == pytest.approx(0.3 * 60 * 4.5 / ratio, rel=0.05)
    assert np.nanmedian(v[:, 1]) / 1000 == pytest.approx(1.485, rel=0.02)
