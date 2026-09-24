"""Splitting a bar path into movements and rises, on paths made up here."""

import numpy as np
import pytest

from app.analysis import kinematics, reps

FPS = 60.0
REST_S = 0.6
DOWN_S, UP_S = 1.0, 2.0
DEPTH_MM = 500.0


def squat_set(n_reps):
    """Bar height of a squat set, each rep slowed almost to a stop half way up."""
    parts = [np.zeros(int(REST_S * FPS))]
    for _ in range(n_reps):
        s = np.arange(int(DOWN_S * FPS)) / (DOWN_S * FPS)
        parts.append(-DEPTH_MM * (1 - np.cos(np.pi * s)) / 2)
        s = np.arange(int(UP_S * FPS)) / (UP_S * FPS)
        # speed 1 - 0.97 cos(4 pi s): 3 % of the mean at the sticking point
        parts.append(-DEPTH_MM + DEPTH_MM * (s - 0.97 * np.sin(4 * np.pi * s) / (4 * np.pi)))
    parts.append(np.zeros(int(REST_S * FPS)))
    up = np.concatenate(parts)
    t = np.arange(len(up)) / FPS
    return t, np.column_stack([np.zeros_like(up), up])


def test_velocity_of_a_parabola():
    t = np.arange(60) / FPS
    pos = np.column_stack([100 * t, 500 * t * t])
    v = kinematics.velocity(t, pos)
    inner = ~np.isnan(v[:, 1])
    assert np.allclose(v[inner, 0], 100.0, atol=1e-6)
    assert np.allclose(v[inner, 1], 1000 * t[inner], atol=1e-6)


def test_a_squat_set_is_one_movement_with_a_rise_per_rep():
    t, pos = squat_set(3)
    movements = reps.split(t, pos, kinematics.velocity(t, pos))
    assert len(movements) == 1
    rises = movements[0].rises
    assert len(rises) == 3                   # the sticking point does not split a rep
    for r in rises:
        assert r.height_mm == pytest.approx(DEPTH_MM, rel=0.01)
        assert r.mean_velocity == pytest.approx(DEPTH_MM / 1000 / UP_S, rel=0.03)
        assert r.peak_velocity == pytest.approx(1.97 * DEPTH_MM / 1000 / UP_S, rel=0.03)


def test_a_still_bar_has_no_movement():
    t = np.arange(120) / FPS
    pos = np.column_stack([np.zeros_like(t), 0.2 * np.sin(7 * t)])     # sub-millimetre jitter
    assert reps.split(t, pos, kinematics.velocity(t, pos)) == []
