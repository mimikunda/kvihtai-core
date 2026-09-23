"""End-to-end test of the tracking tool on a synthetic video file.

The unit tests exercise one piece at a time, which does not catch a function
called with the wrong arguments, a stale signature, or a flag wired to
nothing. Those failures only show up when the whole thing runs, from reading
the file to writing the CSV, so this test writes a clip to disk and runs the
tool on it as a user would.

The clip is deliberately easy. It is a regression test that the tool runs and
reports roughly the right thing, not a measure of how well detection works;
that can only be judged against real video.
"""

import csv
import os
import subprocess
import sys

import pytest

import synthetic as syn

TOOL = os.path.join(os.path.dirname(__file__), "..", "tools", "track_plate.py")
FRAMES = 40


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("clip") / "plate.mp4")
    frames, times, truth = syn.rising_clip(n=FRAMES, face=syn.BLACK)
    syn.write_video(path, frames, 60.0)
    return path, truth


def test_tool_follows_the_plate(clip, tmp_path):
    path, truth = clip
    out = str(tmp_path / "trajectory.csv")
    result = subprocess.run([sys.executable, TOOL, path, "--csv", out], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    with open(out) as fh:
        rows = [r for r in csv.DictReader(fh)]
    accepted = [r for r in rows if r["accepted"] == "1"]
    assert len(accepted) >= FRAMES * 0.9, f"only {len(accepted)} of {FRAMES} frames measured"
    for r in accepted:
        want = truth[int(r["frame"])]
        # the video codec blurs the rim a little: a pixel, not a tenth
        assert abs(float(r["cx_px"]) - want[0]) < 1.0 and abs(float(r["cy_px"]) - want[1]) < 1.0, r["frame"]
    assert float(accepted[0]["mm_per_px"]) == pytest.approx(4.5, rel=0.02)
    assert "rise" in result.stdout


def test_tool_refuses_a_clip_without_a_plate(tmp_path):
    path = str(tmp_path / "empty.mp4")
    syn.write_video(path, [syn.textured_background((320, 480), seed=9) for _ in range(20)], 60.0)
    result = subprocess.run([sys.executable, TOOL, path, "--csv", str(tmp_path / "x.csv")],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "plate" in (result.stdout + result.stderr)
