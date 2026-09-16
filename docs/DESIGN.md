# KvihtAI Core Design Notes

Working design decisions for `kvihtai-core`. This document records what has been agreed so far and what is still open. It changes as the design evolves.

## Stack and Hardware

- **Language and API:** Python with FastAPI.
- **Computer vision:** OpenCV is the likely choice. Other tools are not ruled out.
- **Hardware:** Raspberry Pi 5 with Raspberry Pi Camera Module 3.
- **Development:** starts on a laptop with recorded footage. Tuning for the Pi comes later.

## Tracking

- Tracking is markerless. The camera views the lifter from the side, and the tracker follows the round weight plates.
- A trained detection model for the plates is under consideration.

### Geometry

The plate is a circle of known diameter, 450 mm for a competition plate. Seen
from an angle it projects to an ellipse. The major axis stays the true
diameter; the minor axis is shortened by the cosine of the angle between the
camera and the plate. So one measurement gives both the scale and the viewing
angle, and no calibration target has to be placed in the shot:

- major axis against 450 mm gives millimetres per pixel, and it is unaffected
  by the angle, so vertical distances need no correction;
- minor over major gives the angle, and horizontal distances are divided by
  that ratio to redraw the bar path as if the camera had stood square.

The camera may therefore be placed off to the side. Precision improves as the
angle grows, because the ratio changes with the sine of the angle: it is least
sensitive near a square-on view, where no correction is needed anyway.

The correction only holds in the plane the bar moves in. Anything nearer to or
further from the camera has a different scale.

### How the rim is measured

Implemented in `tools/track_plate.py`.

Three properties are constants of a set, not measurements of a frame, because
the camera does not move and the plate does not change size: the direction of
the major axis, the ratio between the axes, and the diameter in pixels. They
are estimated once from all frames together, and only the position is fitted
per frame. A free per-frame ellipse has five parameters, and a nearly circular
plate leaves its orientation to be decided by noise, which then leaks into the
ratio.

The rim is found by casting 180 rays from the centre and locating the colour
step along each, refined to sub-pixel with a parabola. Colour thresholds were
tried first and abandoned: a threshold puts the boundary wherever the lighting
crosses the chosen level, and bare skin is close enough to plate red that any
threshold loose enough to catch the whole plate also catches the hands.

Two checks decide whether a frame is believed:

- **hub contrast.** A plate has a metal hub, so redness is low at the centre
  and high across the disc. A fit that has settled between the two plates on
  the bar sits on plate material and the contrast collapses. Good fits score
  about 90, straddling fits below 20, with nothing between.
- **reachability.** Peak bar speed in a snatch is about 2 m/s, which is roughly
  12 px between frames at 60 fps. A fit further than 45 px from the previous
  one is measuring something else, usually a plate lying on the floor.

Frames that fail are left empty rather than interpolated. An interpolated
outline lags or runs ahead of the bar, which looks like a tracking error and
hides real ones.

### What has been verified, on one clip

A 5.3 s snatch, 720x1280 at 60 fps, shot on a phone at about 19 degrees off
square. 98 % of frames measured and none interpolated, rim scatter under 0.5 %
of the plate radius, and the major axis direction recovered as vertical, which
is what the geometry above predicts and which was not imposed on the search.
The frames with no measurement are one to three frames long, either where the
hands cover too much of the rim or at the very end where the dropped bar is
too motion-blurred to have an edge.

Every reported frame was compared against the footage by eye. That check is not
optional: the scatter of the rim points around the fitted ellipse measures how
well those points agree with each other, not whether the ellipse is on the
plate, so a fit that has settled on the wrong thing can score well. Reading a
low scatter as a good fit was wrong twice during development, and only a
contact sheet of every frame caught it.

Not verified: any other plate colour, gym, lighting or angle. The detector
finds red, so blue, yellow and green plates will not be found at all. This is
the strongest argument for a learned segmentation model, which would replace
only the step that says which pixels are plate and leave the geometry alone.

## Capture and Lift Detection

- The camera records continuously at about 80 fps into a RAM ring buffer. The oldest frames are overwritten.
- Every 8th frame is checked for plate movement.
- When the plate moves, real recording starts. The frames already in the ring buffer are kept as pre-roll, so the start of the lift is not lost.
- The pre-roll must be longer than 8 frames. The exact length will be set from real footage.
- A lift ends when the plate is back at its starting height and stationary.
- A set can contain several lifts.

## Open Questions

- **Squats:** start and end detection differs from floor lifts. One option is to treat the whole set as one lift.
- **Splitting a set into lifts:** proposed direction is that live detection only decides whether a set is active, and analysis after the set splits it into individual lifts. Details are not decided.
- **Frame cropping and RAM sizing:** to be decided on real hardware.
- **Plate colour:** detection is currently tied to red. Needed for every other
  plate weight before this is usable in a gym.
- **Absolute scale:** the chain is self-consistent but has never been checked
  against a length that is not part of the calculation. The bar is 2200 mm and
  is in shot, which would settle it without any new footage.
