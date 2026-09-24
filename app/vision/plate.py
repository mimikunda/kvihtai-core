"""What is known about a weight plate before anything is measured."""

# Every bumper plate and every competition plate is 450 mm across, whatever
# its weight, colour or maker. That is what lets the plate serve as the scale.
PLATE_DIAMETER_MM = 450.0

# Peak bar speed in a snatch is about 2 m/s. The limit leaves a margin, and a
# plate that seems to move faster than this between two frames is something
# else, most often another plate lying on the floor.
MAX_BAR_SPEED_M_S = 3.0


def max_step_px(radius_px: float, dt_s: float) -> float:
    """How far a plate of this pixel radius can travel in dt seconds, in pixels.

    Derived from the plate's own size, so it needs no calibration: the radius
    in pixels says how many pixels a millimetre is at the plate's distance.
    """
    px_per_mm = radius_px / (PLATE_DIAMETER_MM / 2.0)
    return MAX_BAR_SPEED_M_S * 1000.0 * abs(dt_s) * px_per_mm
