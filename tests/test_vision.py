"""Tests for the pieces the plate tracker is built from.

Everything here runs on synthetic points and synthetic images, so the suite
needs no footage and says nothing about how well the tracker does on a real
plate in a real gym; that is judged against footage, by eye. What it pins down
is that each piece does what it claims: that an edge is found to a fraction of
a pixel whatever the plate's colour, that the tread beside the face does not
move it, that a partial rim still gives the centre, that a ring the size of a
plate does not pass for one, and that the path through the candidates follows
the plate that moves.
"""

import math

import numpy as np
import pytest

from app.vision import face
from app.vision.candidates import Candidate, concentricity, find_candidates, rim_radius
from app.vision.edges import to_lab
from app.vision.path import anchor, best_path, nearer, other_end
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


def _rays(img, centre, R):
    """Edges along the rays the face fit casts from centre."""
    roi = img.astype(np.float32)
    phi = 2 * math.pi * np.arange(face.RAYS) / face.RAYS
    radii = np.arange((1 - face.INNER) * R, (1 + face.OUTER) * R, face.STEP)
    xs = (centre[0] + np.outer(np.cos(phi), radii)).astype(np.float32)
    ys = (centre[1] + np.outer(np.sin(phi), radii)).astype(np.float32)
    return phi, face.rim_edges(face.sample(roi, xs, ys), radii)


class TestFace:
    @pytest.mark.parametrize("name", sorted(COLOURS))
    def test_centre_to_a_few_hundredths_of_a_pixel_whatever_the_colour(self, name):
        img = syn.textured_background((240, 240), seed=3)
        true = np.array([118.37, 121.81])
        syn.draw_plate(img, true, 60.0, face=COLOURS[name])
        # from a guess off by a couple of pixels, as the frame before gives
        f, _ = face.measure(img, (0, 0), [true + [2.0, -1.5]], 60.0)
        assert np.hypot(f.x - true[0], f.y - true[1]) < 0.05
        assert f.r == pytest.approx(60.0, abs=0.2)
        assert f.rays >= 0.95 * face.RAYS and f.sectors == 12

    def test_a_crop_is_measured_in_full_frame_pixels(self):
        img = syn.textured_background((240, 240), seed=3)
        syn.draw_plate(img, (118.37, 121.81), 60.0)
        whole, _ = face.measure(img, (0, 0), [(120.0, 120.0)], 60.0)
        crop, _ = face.measure(img[30:, 20:], (20, 30), [(120.0, 120.0)], 60.0)
        assert (crop.x, crop.y, crop.r) == pytest.approx((whole.x, whole.y, whole.r), abs=1e-6)

    def test_an_edge_is_the_middle_of_its_transition(self):
        # a step blurred over several pixels, lighter on the outside and with
        # a lean of colour that a lightness-only edge would not see
        radii = np.arange(30.0, 70.0, face.STEP)
        s = 0.5 * (1 + np.tanh((radii - 50.3) / 2.0))
        v = np.zeros((1, len(radii), 3), np.float32)
        v[0, :, 0] = 20.0 + 40.0 * s
        v[0, :, 1] = 30.0 * s
        rad, _ = face.rim_edges(v, radii)
        assert rad[0, 0] == pytest.approx(50.3, abs=0.05)

    def test_every_edge_on_a_ray_is_a_candidate(self):
        img = syn.textured_background((240, 240), seed=4)
        syn.draw_plate(img, (130.0, 130.0), 60.0, face=syn.RED, tread=(-8.0, 0.0),
                       tread_colour=(120, 150, 230))
        phi, (rad, _) = _rays(img, (130.0, 130.0), 60.0)
        left = rad[face.RAYS // 2]          # the ray pointing left, into the tread
        assert np.any(np.abs(left - 68.0) < 0.6), left       # tread meets background
        assert np.any(np.abs(left - 60.0) < 0.6), left       # face meets tread

    def test_the_tread_beyond_the_face_does_not_pull_it(self):
        """The tread shows on the side away from the sleeve's end, which says which side that is."""
        img = syn.textured_background((260, 260), seed=7)
        true = (131.3, 128.6)
        syn.draw_plate(img, true, 60.0, face=syn.RED, tread=(-6.0, -3.0), sleeve=(24.0, 12.0))
        f, _ = face.measure(img, (0, 0), [(130.0, 130.0)], 60.0)
        assert f.sleeve == pytest.approx((true[0] + 24.0, true[1] + 12.0), abs=1.0)
        assert np.hypot(f.x - true[0], f.y - true[1]) < 0.3

    def test_the_fit_ends_where_it_ends_whatever_the_start(self):
        img = syn.textured_background((240, 240), seed=8)
        true = (121.2, 119.4)
        syn.draw_plate(img, true, 60.0, face=syn.BLUE, tread=(5.0, -4.0), sleeve=(-20.0, 16.0))
        ends = []
        for a in np.arange(8) * math.pi / 4:
            f, _ = face.measure(img, (0, 0), [(true[0] + 5 * math.cos(a), true[1] + 5 * math.sin(a))], 60.0)
            ends.append((f.x, f.y))
        assert np.ptp(np.array(ends), axis=0).max() < 0.05

    def test_a_third_of_the_rim_and_many_stray_edges(self):
        """A hand across the plate: a third of the rim left, and clutter all round."""
        rng = np.random.default_rng(0)
        c0, R = np.array([300.0, 400.0]), 79.0
        phi = 2 * math.pi * np.arange(180) / 180
        rad = np.full((180, 2), np.nan)
        arc = phi < 2 * math.pi / 3
        rad[arc, 0] = R + rng.normal(0, 0.3, arc.sum())
        stray = rng.choice(180, 50, replace=False)
        rad[stray, 1] = rng.uniform(0.85 * R, 1.2 * R, 50)
        f = face.fit_face(c0, phi, (rad, None), (302.0, 398.0, R), r_prior=(R, 20.0))
        assert np.hypot(f.x - c0[0], f.y - c0[1]) < 0.3
        assert f.rays >= 58          # the arc, and the odd stray edge that happens to lie on the rim

    def test_an_ellipse_is_fitted_with_its_shape(self):
        img = syn.textured_background((260, 260), seed=9)
        true = (129.6, 131.2)
        syn.draw_plate(img, true, 70.0, face=syn.GREEN, ratio=0.85, angle_deg=80.0, hub=(60, 60, 60))
        shape = (0.85, math.radians(80.0))
        f, _ = face.measure(img, (0, 0), [(128.0, 132.0)], 70.0, shape=shape)
        assert np.hypot(f.x - true[0], f.y - true[1]) < 0.1
        assert f.r == pytest.approx(70.0, abs=0.3)
        circle, _ = face.measure(img, (0, 0), [(128.0, 132.0)], 70.0)
        assert f.rays > circle.rays

    def test_the_sleeve_end_is_found_where_it_stands(self):
        img = syn.textured_background((240, 240), seed=10)
        syn.draw_plate(img, (120.0, 120.0), 60.0, face=syn.BLACK, sleeve=(-14.3, 27.6))
        e = face.sleeve_end(img.astype(np.float32), (120.0, 120.0), 60.0)
        # the tube running into it from one side leans it by a fraction of a pixel; it is only a guide
        assert (e[0], e[1]) == pytest.approx((120.0 - 14.3, 120.0 + 27.6), abs=1.0)

    def test_a_face_is_even_and_a_ring_with_the_room_inside_is_not(self):
        img = syn.textured_background((300, 200), seed=5)
        syn.draw_plate(img, (80.0, 100.0), 45.0, face=syn.RED)
        syn.draw_ring(img, (220.0, 100.0), 45.0)
        roi = img.astype(np.float32)
        plate = face.face_colour(roi, (80.0, 100.0), 45.0)
        ring = face.face_colour(roi, (220.0, 100.0), 45.0)
        assert plate[3] < 1.0 and ring[3] > 3.0


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

    def test_the_rim_is_found_from_the_hub(self):
        img = syn.textured_background((240, 240))
        syn.draw_plate(img, (120.0, 120.0), 60.0, face=syn.BLUE)
        # starting from the hub's radius, as the circle finder often reports it
        assert rim_radius(img, 120.0, 120.0, 0.38 * 60.0, 90.0) == pytest.approx(60.0, abs=2.0)

    def test_the_rim_is_found_on_a_plate_standing_on_the_floor(self):
        img = syn.textured_background((240, 240))
        syn.draw_plate(img, (120.0, 120.0), 60.0, face=syn.BLACK)
        img[int(120 + 0.85 * 60):] = (70, 75, 80)         # the floor hides the bottom of the rim
        assert rim_radius(img, 120.0, 120.0, 0.38 * 60.0, 90.0) == pytest.approx(60.0, abs=2.0)


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

    def test_finds_the_other_end_of_the_bar_and_which_end_is_near(self):
        # followed: the far plate, at x 100; the near one, larger, at x 300,
        # seen in half the frames and a little lower as the bar tilts; a
        # static ring on a storage tree, and a knee seen once
        track = [(i, (100.0, 300.0 - 4.0 * i, 40.0)) for i in range(0, 80, 4)]
        detections = []
        for k, (i, (x, y, r)) in enumerate(track):
            circles = [(x, y, r), (500.0, 60.0, 40.0)]
            if k % 2 == 0:
                circles.append((x + 200.0 + 0.2 * k, y + 10.0 - 0.3 * k, 50.0))
            if k == 7:
                circles.append((x - 90.0, y + 150.0, 30.0))
            detections.append((i, circles))
        other = other_end(detections, track)
        assert other is not None and len(other) == 10
        assert all(abs(x - 300.0) < 5.0 and r == 50.0 for _, (x, y, r) in other)
        assert not nearer(track, other, (720.0, 400.0))      # a tap on the right: the other end
        assert nearer(track, other, (0.0, 400.0))
        assert nearer(other, track, (720.0, 400.0))

    def test_no_other_end_when_the_far_plate_is_hidden(self):
        track = [(i, (100.0, 300.0 - 4.0 * i, 40.0)) for i in range(0, 80, 4)]
        detections = [(i, [(x, y, r)] + ([(x + 150.0, y + 20.0, 30.0)] if k == 3 else []))
                      for k, (i, (x, y, r)) in enumerate(track)]
        assert other_end(detections, track) is None
