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
