"""Tests for the geometry the plate tracker is built on.

Everything here runs on synthetic points and synthetic images, so the suite
needs no footage and says nothing about whether the tracker finds a real plate.
What it pins down is the maths underneath: that unsqueezing an ellipse really
does make it round, that a circle can be recovered from part of its rim when
the radius is known, and that the two checks which decide whether a frame is
believed respond to the thing they claim to measure.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from track_plate import (  # noqa: E402
    _fit_centre_only,
    _fit_circle,
    _resqueeze,
    _unsqueeze,
    fit_shaped,
    hub_contrast,
    sector_coverage,
    trim,
)


def ellipse_points(cx, cy, major, ratio, theta_deg, count=180, arc=2 * math.pi, start=0.0):
    """Points on an ellipse, optionally only along part of it."""
    t = math.radians(theta_deg)
    a, b = major / 2, major * ratio / 2
    out = []
    for k in range(count):
        phi = start + arc * k / count
        x, y = a * math.cos(phi), b * math.sin(phi)
        out.append((cx + x * math.cos(t) - y * math.sin(t),
                    cy + x * math.sin(t) + y * math.cos(t)))
    return np.array(out, dtype=np.float32)


class TestSqueeze:
    def test_round_trip_returns_the_original_point(self):
        for theta in (0.0, 37.5, 91.5, 175.0):
            for ratio in (0.4, 0.75, 0.945, 1.0):
                # float32 in, so the tolerance is single precision, not double
                there = _unsqueeze(np.array([[123.0, -45.0]], np.float32), theta, ratio)[0]
                back = _resqueeze(there, theta, ratio)
                assert back[0] == pytest.approx(123.0, abs=1e-4)
                assert back[1] == pytest.approx(-45.0, abs=1e-4)

    def test_unsqueezing_an_ellipse_makes_it_round(self):
        pts = ellipse_points(400, 300, major=160, ratio=0.6, theta_deg=91.5)
        circle = _unsqueeze(pts, 91.5, 0.6)
        (cx, cy), r, residual = _fit_circle(circle)
        assert residual == pytest.approx(0.0, abs=1e-3)
        assert r == pytest.approx(80.0, abs=1e-3)

    def test_a_circle_is_unchanged_by_a_ratio_of_one(self):
        pts = ellipse_points(10, 20, major=50, ratio=1.0, theta_deg=0.0)
        assert _unsqueeze(pts, 45.0, 1.0) == pytest.approx(pts, abs=1e-4)


class TestCircleFit:
    def test_recovers_a_known_circle(self):
        pts = ellipse_points(250.0, 133.0, major=158.0, ratio=1.0, theta_deg=0.0)
        (cx, cy), r, residual = _fit_circle(pts)
        assert cx == pytest.approx(250.0, abs=1e-3)
        assert cy == pytest.approx(133.0, abs=1e-3)
        assert r == pytest.approx(79.0, abs=1e-3)
        assert residual == pytest.approx(0.0, abs=1e-3)

    def test_centre_only_fit_recovers_the_centre_from_a_third_of_the_rim(self):
        """The reason the radius is held fixed: a partial arc still pins the centre."""
        pts = ellipse_points(300.0, 400.0, major=158.0, ratio=1.0,
                             theta_deg=0.0, count=60, arc=2 * math.pi / 3)
        centre, residual = _fit_centre_only(pts, 79.0)
        assert centre[0] == pytest.approx(300.0, abs=0.05)
        assert centre[1] == pytest.approx(400.0, abs=0.05)
        assert residual == pytest.approx(0.0, abs=1e-3)

    def test_a_free_radius_drifts_on_a_partial_arc_where_a_fixed_one_does_not(self):
        noisy = ellipse_points(300.0, 400.0, major=158.0, ratio=1.0,
                               theta_deg=0.0, count=60, arc=2 * math.pi / 3)
        noisy = noisy + np.random.default_rng(0).normal(0, 0.8, noisy.shape).astype(np.float32)
        (fx, fy), free_r, _ = _fit_circle(noisy)
        fixed, _ = _fit_centre_only(noisy, 79.0)
        free_error = math.hypot(fx - 300.0, fy - 400.0)
        fixed_error = math.hypot(fixed[0] - 300.0, fixed[1] - 400.0)
        assert fixed_error < free_error


class TestFitShaped:
    def test_recovers_a_known_ellipse(self):
        pts = ellipse_points(360.0, 640.0, major=159.2, ratio=0.945, theta_deg=91.5)
        got = fit_shaped(pts, 91.5, 0.945)
        assert got["cx"] == pytest.approx(360.0, abs=0.01)
        assert got["cy"] == pytest.approx(640.0, abs=0.01)
        assert got["major"] == pytest.approx(159.2, abs=0.01)
        assert got["ratio"] == pytest.approx(0.945)

    def test_a_given_radius_is_kept_exactly(self):
        pts = ellipse_points(100.0, 100.0, major=200.0, ratio=0.9, theta_deg=0.0)
        got = fit_shaped(pts, 0.0, 0.9, radius=79.6)
        assert got["major"] == pytest.approx(159.2)


class TestTrim:
    def test_drops_points_that_are_not_on_the_rim(self):
        pts = ellipse_points(300.0, 300.0, major=160.0, ratio=0.95, theta_deg=90.0, count=100)
        strays = np.array([[300.0, 300.0], [500.0, 120.0], [120.0, 500.0]], np.float32)
        kept = trim(np.vstack([pts, strays]), 90.0, 0.95)
        assert len(kept) <= len(pts)
        for stray in strays:
            assert not any(np.allclose(k, stray) for k in kept)


class TestSectorCoverage:
    """The check that separates a blurred plate from a fit on the wrong thing."""

    def test_a_full_rim_covers_every_sector(self):
        pts = ellipse_points(300.0, 300.0, major=160.0, ratio=0.95, theta_deg=90.0)
        fit = fit_shaped(pts, 90.0, 0.95)
        assert sector_coverage(pts, fit, 90.0, 0.95) == 12

    def test_a_quarter_of_the_rim_covers_about_a_quarter_of_the_sectors(self):
        pts = ellipse_points(300.0, 300.0, major=160.0, ratio=0.95, theta_deg=90.0,
                             count=60, arc=math.pi / 2)
        fit = fit_shaped(pts, 90.0, 0.95)
        assert sector_coverage(pts, fit, 90.0, 0.95) <= 4


class TestHubContrast:
    """The check that rejects a fit sitting between the two plates on the bar."""

    @staticmethod
    def plate_field(size=300, centre=(150, 150), radius=80, hub=0.3):
        """A synthetic plate: a red disc with a dull metal hub in the middle."""
        yy, xx = np.mgrid[0:size, 0:size]
        d = np.hypot(xx - centre[0], yy - centre[1])
        field = np.zeros((size, size), np.float32)
        field[d <= radius] = 100.0
        field[d <= radius * hub] = 5.0
        return field

    def test_high_on_a_plate_centred_on_its_hub(self):
        field = self.plate_field()
        assert hub_contrast(field, 150, 150, 80) > 50

    def test_collapses_when_the_fit_sits_on_the_face_instead(self):
        """Offset by half a radius: the middle now lands on plate, not hub."""
        field = self.plate_field()
        assert hub_contrast(field, 150 + 40, 150, 80) < 50
