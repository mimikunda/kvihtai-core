"""End-to-end test of the tracker on a synthetic clip.

The geometry tests exercise the maths one function at a time, which does not
catch a function called with the wrong arguments, a stale signature, or a flag
wired to nothing. Those failures only show up when the whole thing runs. This
test therefore draws a plate moving across a few frames, writes it as a video,
and asks the tracker to follow it, so a break anywhere along the chain fails
here rather than in front of real footage.

The clip is deliberately easy. It is a regression test that the pipeline runs
and reports roughly the right thing, not a measure of how well detection works;
that can only be judged against real video.
"""

import csv
import math
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

TOOLS = os.path.join(os.path.dirname(__file__), "..", "tools")
sys.path.insert(0, TOOLS)

PLATE_RADIUS = 60
FRAMES = 40


def draw_plate(size, centre, radius, ratio):
    """A red disc with a dull metal hub, on a plain background."""
    frame = np.full((size[1], size[0], 3), 40, np.uint8)
    axes = (int(radius), int(radius * ratio))
    cv2.ellipse(frame, (int(centre[0]), int(centre[1])), axes, 90, 0, 360, (40, 40, 200), -1)
    cv2.ellipse(frame, (int(centre[0]), int(centre[1])),
                (int(radius * 0.28), int(radius * 0.28 * ratio)), 90, 0, 360, (130, 130, 130), -1)
    return frame


@pytest.fixture(scope="module")
def synthetic_clip(tmp_path_factory):
    """A plate rising at a constant speed, seen slightly off square."""
    path = str(tmp_path_factory.mktemp("clip") / "plate.mp4")
    size = (320, 480)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 60.0, size)
    assert writer.isOpened(), "no mp4 writer available"
    positions = []
    for i in range(FRAMES):
        centre = (160.0, 380.0 - 6.0 * i)
        positions.append(centre)
        writer.write(draw_plate(size, centre, PLATE_RADIUS, 0.9))
    writer.release()
    return path, positions


def test_tracker_follows_the_plate(synthetic_clip, tmp_path):
    path, positions = synthetic_clip
    out = str(tmp_path / "trajectory.csv")

    result = subprocess.run(
        [sys.executable, os.path.join(TOOLS, "track_plate.py"), path, "--csv", out],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    with open(out) as fh:
        rows = {int(r["frame"]): r for r in csv.DictReader(fh)}

    assert len(rows) >= FRAMES * 0.8, f"only {len(rows)} of {FRAMES} frames measured"

    for frame, row in rows.items():
        want = positions[frame]
        assert math.hypot(float(row["cx_px"]) - want[0],
                          float(row["cy_px"]) - want[1]) < 4.0, f"frame {frame} is off"

    diameters = [float(r["major_px"]) for r in rows.values()]
    assert max(diameters) - min(diameters) < 1e-6, "the diameter should be one value per set"
    assert abs(diameters[0] - 2 * PLATE_RADIUS) < 6.0


def test_the_residual_flag_is_actually_wired(synthetic_clip, tmp_path):
    """A flag that parses but reaches nothing is the failure this catches."""
    path, _ = synthetic_clip
    out = str(tmp_path / "strict.csv")
    result = subprocess.run(
        [sys.executable, os.path.join(TOOLS, "track_plate.py"), path,
         "--csv", out, "--max-residual", "0.0001"],
        capture_output=True, text=True,
    )
    # Either it refuses outright or it keeps far fewer frames; what it must not
    # do is quietly behave as though the flag were absent.
    if result.returncode == 0:
        with open(out) as fh:
            kept = sum(1 for _ in csv.DictReader(fh))
        assert kept < FRAMES, "an impossible residual limit changed nothing"
