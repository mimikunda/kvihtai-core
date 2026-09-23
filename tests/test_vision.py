"""Tests for the pieces the plate tracker is built from.

Everything here runs on synthetic points and synthetic images, so the suite
needs no footage and says nothing about how well the tracker does on a real
plate in a real gym; that is judged against footage, by eye. What it pins down
is that each piece does what it claims: that an edge is found to a fraction of
a pixel whatever the plate's colour, that a partial rim still gives the
centre, that a ring the size of a plate does not pass for one, and that the
path through the candidates follows the plate that moves.
"""

import math

import numpy as np
import pytest

from app.vision import appearance
from app.vision.candidates import Candidate, concentricity, find_candidates
from app.vision.edges import ray_edges, rim_patch, to_lab
from app.vision.outline import Outline, fit_outline, learn_outline, recentre, sectors_covered
from app.vision.path import anchor, best_path
from app.vision.plate import max_step_px
import synthetic as syn

COLOURS = {"black": syn.BLACK, "red": syn.RED, "blue": syn.BLUE, "yellow": syn.YELLOW, "green": syn.GREEN}


class TestPlate:
    def test_step_limit_grows_with_size_and_time(self):
        assert max_step_px(100, 1 / 60) == pytest.approx(2 * max_step_px(50, 1 / 60))
        assert max_step_px(100, 1 / 30) == pytest.approx(2 * max_step_px(100, 1 / 60))

    def test_step_limit_is_in_millimetres_of_the_plate(self):
        # a 450 mm plate 450 px across is 1 px per mm; 3 m/s for 10 ms is 30 mm
        assert max_step_px(225, 0.01) == pytest.approx(30.0)


class TestEdges:
    @pytest.mark.parametrize("name", sorted(COLOURS))
    def test_centre_to_a_few_hundredths_of_a_pixel_whatever_the_colour(self, name):
        img = syn.textured_background((240, 240), seed=3)
        true = np.array([118.37, 121.81])
        syn.draw_plate(img, true, 60.0, face=COLOURS[name])
        # rays from a centre that is off by a couple of pixels, as a coarse one is
        start = true + [2.0, -1.5]
        patch, origin = rim_patch(img, start, 80)
        pts, strength, ray = ray_edges(patch, origin, start, 60.0, 0.85, 1.15)
        dist = np.hypot(pts[:, 0] - true[0], pts[:, 1] - true[1]) - 60.0
        on_rim = np.abs(dist) < 0.5
        assert len(np.unique(ray[on_rim])) >= 0.9 * 180
        # a single edge is good to a couple of tenths; the rim as a whole far better
        assert np.median(np.abs(dist[on_rim])) < 0.2
        fit = fit_outline(pts, Outline(), start, 60.0)
        assert np.hypot(*(fit.centre - true)) < 0.05
        # Lab is not linear in the camera's values, so the steepest point of a
        # blurred step leans a little towards the darker side. The lean is the
        # same all round the rim: it moves the size by 0.2 %, never the centre.
        assert fit.scale == pytest.approx(60.0, abs=0.2)

    def test_a_ray_reports_both_edges_where_the_tread_shows(self):
        img = syn.textured_background((240, 240), seed=4)
        syn.draw_plate(img, (130.0, 130.0), 60.0, face=syn.RED, tread=(-8.0, 0.0),
                       tread_colour=(120, 150, 230))
        patch, origin = rim_patch(img, (130.0, 130.0), 80)
        pts, _, ray = ray_edges(patch, origin, (130.0, 130.0), 60.0, 0.85, 1.25)
        left = pts[ray == 90]      # the ray pointing left, into the tread
        xs = sorted(left[:, 0])
        assert any(abs(x - 62.0) < 0.6 for x in xs), xs     # tread meets background
        assert any(abs(x - 70.0) < 0.6 for x in xs), xs     # face meets tread


class TestOutlineFit:
    def test_recovers_centre_and_scale(self):
        pts = syn.circle_points(200.3, 150.7, 80.0)
        fit = fit_outline(pts, Outline(), (197.0, 153.0), 76.0)
        assert fit.centre == pytest.approx([200.3, 150.7], abs=1e-3)
        assert fit.scale == pytest.approx(80.0, abs=1e-3)

    def test_a_third_of_the_rim_and_many_stray_edges(self):
        """A hand across the plate: a third of the rim left, and clutter everywhere."""
        rng = np.random.default_rng(0)
        rim = syn.circle_points(300.0, 400.0, 79.0, count=60, arc=2 * math.pi / 3)
        rim = rim + rng.normal(0, 0.3, rim.shape)
        clutter = np.array([300.0, 400.0]) + rng.uniform(-90, 90, (50, 2))
        pts = np.vstack([rim, clutter])
        fit = fit_outline(pts, Outline(), (302.0, 398.0), 79.0, scale_prior=79.0, prior_weight=3.0)
        assert np.hypot(*(fit.centre - [300.0, 400.0])) < 0.5
        # a plain least-squares circle through the same points is nowhere near
        x, y = pts[:, 0], pts[:, 1]
        sol, *_ = np.linalg.lstsq(np.column_stack([x, y, np.ones_like(x)]), x * x + y * y, rcond=None)
        assert np.hypot(sol[0] / 2 - 300.0, sol[1] / 2 - 400.0) > 2

    def test_the_prior_stops_scale_and_position_trading_on_a_short_arc(self):
        rim = syn.circle_points(0.0, 0.0, 100.0, count=40, arc=math.pi / 3, start=-math.pi / 6)
        rim[:, 0] += 0.5                     # a small bias, as blur or a shadow gives
        loose = fit_outline(rim, Outline(), (2.0, 0.0), 100.0)
        held = fit_outline(rim, Outline(), (2.0, 0.0), 100.0, scale_prior=100.0, prior_weight=30.0)
        assert abs(held.centre[0] - 0.5) < abs(loose.centre[0] - 0.5) + 1e-9
        assert held.centre[0] == pytest.approx(0.5, abs=0.05)

    def test_sectors_count_directions_not_points(self):
        pts = syn.circle_points(0, 0, 50, count=100, arc=math.pi)
        assert sectors_covered(pts, (0, 0)) == 6


class TestOutlineLearning:
    def test_learns_an_ellipse_from_many_frames(self):
        rng = np.random.default_rng(1)
        phis, rs = [], []
        for _ in range(30):
            scale = rng.uniform(70, 90)
            pts = syn.circle_points(0, 0, scale, ratio=0.8, angle_deg=90) + rng.normal(0, 0.2, (180, 2))
            phis.append(np.arctan2(pts[:, 1], pts[:, 0]))
            rs.append(np.hypot(pts[:, 0], pts[:, 1]) / scale)
        rho = learn_outline(np.concatenate(phis), np.concatenate(rs), lo=0.6, hi=1.2)
        rho, shift, factor = recentre(rho)
        outline = Outline(rho)
        # major axis vertical, 0.8 across: after recentring the extremes keep that ratio
        up, _ = outline.at(np.array([math.pi / 2]))
        side, _ = outline.at(np.array([0.0]))
        assert side[0] / up[0] == pytest.approx(0.8, abs=0.01)
        assert np.hypot(*shift) < 0.01


class TestCandidates:
    def test_a_plate_is_concentric_and_a_ring_is_not(self):
        img = syn.textured_background((300, 200), seed=5)
        syn.draw_plate(img, (80.0, 100.0), 45.0)
        syn.draw_ring(img, (220.0, 100.0), 45.0)
        lab = to_lab(img)
        assert concentricity(lab, 80, 100, 45) > 0.4
        assert concentricity(lab, 220, 100, 45) < 0.2

    @pytest.mark.parametrize("name", sorted(COLOURS))
    def test_the_plate_outscores_a_ring_of_its_size(self, name):
        img = syn.textured_background((300, 200), seed=6)
        syn.draw_plate(img, (80.0, 100.0), 45.0, face=COLOURS[name])
        syn.draw_ring(img, (220.0, 100.0), 45.0)
        best = find_candidates(img, 45.0)[0]
        assert math.hypot(best.x - 80, best.y - 100) < 3


class TestPath:
    def test_follows_the_plate_that_moves_not_the_brighter_one_that_does_not(self):
        times = np.arange(20) / 60.0
        cands = []
        for i in range(20):
            frame = [Candidate(200.0, 50.0, 1.5)]              # static, round, strong
            if i % 5 != 3:                                       # the plate, missed now and then
                frame.append(Candidate(100.0, 300.0 - 4.0 * i, 1.0))
            cands.append(frame)
        track = [(i, (100.0, 300.0 - 4.0 * i, 40.0)) for i in range(0, 20, 4)]
        path = best_path(anchor(cands, track, 40.0), times, 40.0)
        for i, c in enumerate(path):
            if c is not None:
                assert c.x == 100.0, i
        assert sum(c is None for c in path) <= 4

    def test_leaves_a_frame_empty_rather_than_step_faster_than_a_bar(self):
        times = np.arange(3) / 60.0
        radius = 50.0
        far = max_step_px(radius, 1 / 60) * 1.5
        cands = [[Candidate(0.0, 0.0, 1.0)], [Candidate(far, 0.0, 1.0)], [Candidate(0.0, 0.0, 1.0)]]
        path = best_path(cands, times, radius)
        assert path[0] is not None and path[2] is not None
        assert path[1] is None


class TestAppearance:
    def test_similarity_is_correlation_of_the_profile(self):
        p = np.column_stack([np.linspace(0, 100, 32), np.zeros(32), np.zeros(32)])
        assert appearance.similarity(p, p) == pytest.approx(1.0)
        assert appearance.similarity(p, p * 0.5 + 40) == pytest.approx(1.0)
        assert appearance.similarity(p[::-1], p) == pytest.approx(-1.0)
