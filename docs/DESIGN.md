# KvihtAI Core Design Notes

Working design decisions for `kvihtai-core`. This document records what has been agreed so far and what is still open. It changes as the design evolves.

## Stack and Hardware

- **Language and API:** Python with FastAPI.
- **Computer vision:** OpenCV is the likely choice. Other tools are not ruled out.
- **Hardware:** Raspberry Pi 5 with Raspberry Pi Camera Module 3.
- **Development:** starts on a laptop with recorded footage. Tuning for the Pi comes later.

## Tracking

- Tracking is markerless. The camera views the lifter from the side, and the
  tracker follows the round weight plates.
- Nothing in it asks what colour the plate is. Black iron plates, coloured
  bumpers and competition plates go through the same code.
- A trained detection model was considered because the first tracker found
  only red plates. That reason is gone.

Code: `app/vision/` (finding and measuring the plate), `app/analysis/` (bar
path, reps, the result), `tools/track_plate.py` (runs it on a video file).

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

The correction only holds in the plane the bar moves in. Anything nearer to or
further from the camera has a different scale.

Near square, the angle cannot be measured well enough to correct by. The face
is then so nearly round that the direction of its axis is decided by noise: on
the first test clip, four quarters of the set put it anywhere from 38 to 159
degrees. Correcting along a wrong axis moves vertical distances, by 1.3 % on
that clip, while leaving a 15 degree view uncorrected costs horizontal
distances 3.4 % and vertical ones nothing. So the bar path is corrected only
when the angle is at least 15 degrees and four quarters of the set agree on the
axis. Both test clips are below that and are reported uncorrected.

A plate is not flat. Its thickness shows as a crescent of tread on the side
facing away from the camera, and the outline in the image is the silhouette of
a cylinder: the front face swept along the image of the thickness. The centre
of that silhouette is off the bar by half the tread. What is on the bar is the
centre of the front face.

### How the plate is found and measured

Two stages. The coarse one only has to say roughly where the plate is; the fine
one measures it.

**Coarse.** Every strong colour edge votes for the points one plate radius
away along its gradient, both ways, since a plate can be darker or lighter than
what is behind it. Each peak is scored on two things:

- **coverage**, the share of directions round it with an edge at that radius.
  A real rim is there almost all the way round, a coincidence of lines is not.
- **concentricity**, the share of the variation inside the circle that radius
  alone explains. A plate is rings: hub, face, rim. A loop of rope or a wheel
  has the room behind it inside.

Votes and coverage alone chose a coil of rope over the plate on the second test
clip: static, round, the right size, and scoring steadily while the moving
plate blurred. Concentricity is what separates them.

On a video file the path through the candidates is chosen for the whole clip at
once: a path that no bar could travel, faster than 3 m/s scaled by the plate's
size in pixels, is not allowed, and circles found by OpenCV's Hough transform
on the plate that moves the furthest are anchors the path must pass through. A
greedy frame-by-frame tracker lost the plate to static circles. On the Pi the
live watcher, below, does this job instead.

**Fine.** Edges are found along 180 rays from the centre, as the largest steps
in Lab colour, refined to sub-pixel with a parabola. Lab, because a plate can
differ from its background in lightness alone (black on dark clothes) or in
hue alone (red on skin). The image is blurred in floating point before this:
blurring in 8 bits rounds every edge to the same few levels and doubled the
jitter.

The outline is learned once per set rather than assumed. Its radius as a
function of direction is pooled from a sample of frames, and in each direction
the outermost edge seen nearly as often as the most common one is taken, so it
is the silhouette and never the face edge inside the tread. Taking whichever
is more common let the outline switch between the two from one direction to
the next. Per frame, only a position and a scale are fitted to that fixed
shape, robustly, so a hand across the rim is ignored rather than averaged in.
A free ellipse per frame lets shape and position trade against each other, and
that trade is exactly the sideways jumping of the bar centre.

The scale is held to a running median of the frames where the rim is seen all
the way round, over a quarter of a second. It still follows the plate towards
and away from the camera, but a frame that sees only an arc cannot trade its
scale against its position.

The front face is then recovered from the same pooled edges: it is the tread
vector that puts the most edges where the face edge would have to be, inside
the silhouette on the tread side. The face's centre is the bar end, its major
axis is the 450 mm, and its axis ratio is the camera angle.

Frames are accepted on four checks: edges in at least 5 of 12 directions round
the rim, a median edge distance from the outline under 3 % of the radius, a
scale within 5 % of the running median, and a radial brightness profile that
correlates with the set's median profile. The whole set must also be built like
a plate, in rings. Rejected frames are reported with the reason and never
interpolated. A frame the coarse stage lost between two measured ones is
looked for again between them, up to 12 frames.

### What has been verified

Two phone clips, both handheld, and synthetic plates.

- **A 5.3 s snatch**, 720x1280 at 60 fps, red competition plates, about 13
  degrees off square. 311 of 319 frames measured. The 8 rejected are the
  dropped bar at the end, too blurred to recognise. Pull 1015 mm, peak 2.05 m/s.
  The centre's frame-to-frame noise, measured as the scatter about a local
  quadratic over five frames, is 0.15 px, against 0.19 px for the earlier
  red-only tracker.
- **A 6.2 s clip of three lifts**, 1440x1920 at 30 fps, about 11 degrees off
  square. 185 of 185 frames measured. At 30 fps the plate moves up to 50 px
  between frames and is visibly smeared, and the noise is 0.6 px.
- **Synthetic plates** of every colour, with and without visible tread, and a
  ring of the plate's exact size in the background. Face centre within 0.3 px,
  scale within 1 %, velocity within 2 %.

The earlier red-only tracker put the first clip at 19 degrees, this one at 13.
Neither can be trusted at that angle, for the reason given under Geometry.

Both clips were shot handheld. The camera moves by up to 38 px during them,
zooms by 1 % and turns by 1 degree. The per-frame scale absorbs the zoom, but
the camera's movement is in the bar path as if the bar had made it. On a tripod
this goes away; on these clips it cannot be separated from the bar.

**Oblique views have a limit.** On synthetic plates the tracker is exact to 25
degrees off square. From 30 degrees the coarse stage, which looks for circles,
hands the fine stage a centre off by 5 to 11 px, and at 40 degrees the camera
angle comes out as 25. At 50 degrees nothing is found. The fine stage's
outline learning also assumes the silhouette lies within 15 % of a circle. Both
need work before a camera well off to the side can be supported.

Verification habit: check every reported frame against the footage, as a
contact sheet of all of them. Rim scatter says how well the edges agree with
each other, not whether they are on the plate, and a fit on the wrong object
scores well. Sampling a few frames hid real errors more than once.

## Capture and Lift Detection

Code: `app/capture/`, run with `tools/capture.py`, timed with
`tools/bench_capture.py`.

Three threads, and a fourth for searching:

    camera      source -> ring buffer, never waits for anyone
    watcher     ring buffer -> live tracker -> a recording per set
    search      whole-frame search for plates, once a second while idle
    analysis    recording -> precise stage -> result.json

- **Ring buffer.** Memory is allocated once, for a fixed number of frames, and
  the camera copies each frame into the next slot. A reader that falls behind
  by more than the buffer loses frames, and is told so; it never gets a frame
  that was overwritten while it was being copied.
- **Idle.** Every second the newest frame is searched for plate-shaped circles,
  reduced to 360 px on its short side, and each becomes a watched plate. Plates on a storage tree
  are watched too and never move. Every 8th frame each watched plate is looked
  for where it was. A plate found 8 % of its radius away in two checks in a row
  has started to move. A circle cut by the edge of the frame is not watched: its
  visible part is found slightly differently each time, and on the Pi one
  appeared to move and started a set.
- **The rim among the rings.** The Hough transform often reports the hub or the
  face edge. The radius used is the outermost ring round the centre with an
  edge nearly all the way round. Not the most complete ring: a plate standing on
  the floor loses the bottom of its rim to the floor, and the face then scores
  higher.
- **Pre-roll.** When a plate starts to move, the set starts one second back in
  the ring buffer, so the start of the lift is kept.
- **Active.** Every frame the plate is looked for in small windows along the
  line from where it was last seen to where it would be at its last speed.
  Only candidates a bar could have reached since are considered. A crop of
  1.6 plate radii round it is kept; while the plate is missing, the crop covers
  everywhere it could be, for as many frames as the analysis can bridge.
  Searching the whole frame instead lost the plate: the votes that find a plate
  go to the strongest edges in view, and in a whole frame of gym a blurred plate
  is not among the peaks.
- **Detail.** Checks and following look for the plate at no more than 40 px
  radius in the reduced image, which is where both test clips had it. On the
  Pi's landscape frame a near plate would otherwise be twice that area: on a
  synthetic 1536x864 clip with a plate 220 px across, following took 17 ms a
  frame on the Pi 4 without the cap and 15.5 ms with it, against 16.7 ms
  between frames at 60 fps.
- **End.** The set ends when the plate has been still for 3 s, lost for 2 s,
  after 180 s, or when its crops pass 1.5 GB.
- **Analysis.** After the set, on the crops. Live detection only decides that a
  set is going on; how many lifts it held is decided here, from the whole bar
  path. Rests are stretches of 0.4 s under 0.08 m/s, movements are what lies
  between them, and a rise is a stretch of a movement going up by at least
  100 mm. A sticking point that slows the bar without stopping it does not
  split a rise. Each rise gets its height, mean and peak velocity.
- **Camera settings.** Short exposure, set by hand: 2 ms. The phone footage
  exposes for up to a thirtieth of a second, and the blur, not the detector, is
  what limits it. Noise from the gain averages out in the rim fit; blur does
  not. The frame rate is 60 fps, not 80, because the station records every
  frame, see Station below.

Live and offline agree on both clips: rises of 1016 and 715 mm live against
1015 and 721 mm offline on the first, and about 1 % lower live on the second.

Measured on the test Pi 4B, with the clip decoded into memory first:

| | first clip, 720x1280, plate 158 px across | second clip, 1440x1920, 316 px |
|---|---|---|
| search, in its own thread | 164 ms | 85 ms |
| check, every 8th frame | 20 ms | 51 ms |
| follow, every frame | 13 ms, 75 fps at most | 23 ms, 44 fps at most |
| analysis, after the set | 66 ms a frame | 134 ms a frame |

Both clips played at their own rate through the whole pipeline on the Pi 4,
decoding included, and lost no frames. The Pi 5 is expected to be two to three
times faster.

## Station

Code: `app/station.py`, `app/capture/recorder.py`, `app/capture/worker.py`,
`deploy/`. The web app is the separate `kvihtai-web` repository, built and
served by the API at `/`.

The station is the API process with the capture session running inside it.
One process, because the camera belongs to one process, and the app needs to
see what the camera sees and change its settings while it runs. It starts at
boot as a systemd service, and a camera that fails is opened again.

- **Recording.** Everything the camera sees is recorded, not only the sets.
  The sets the watcher misses are the ones most worth having for improving it.
  The Pi's hardware H.264 encoder takes the same frames as the tracker. It is
  asked for 12 Mbit/s, about 5 GB an hour, but in a dark room at full gain the
  noise took it to 24 Mbit/s, 11 GB an hour. Segments are 5 minutes of fragmented MP4, so
  that pulling the plug loses a fragment, not the file, and a JSON file beside
  each holds the sensor time of its first frame, its keyframes and how the
  camera was turned. The oldest segments are deleted only when less than 5 GB
  is free.
- **60 fps.** On the Pi 4B the encoder and the copy of each frame into the ring
  buffer compete for memory. At 80 fps with the encoder running, 45 of the 80
  frames a second reached the ring; copying straight from the camera's buffer
  instead of through an extra array raised that to 67. At 60 fps all of them
  do, with 1 to 4 % lost while the app's camera preview is open.
- **A clip per set.** When a set ends, the segment is closed at the next
  keyframe, and the set is cut out of it without re-encoding, from the keyframe
  before it. Keyframes are every half second. The clip carries the camera's
  rotation as metadata, and the app draws the measured path over it.
- **Analysis in a process of its own.** In a thread, the analysis held the
  interpreter's lock long enough that a set played on the Pi while the one
  before it was being analysed lost more than half of its frames. The crops go
  to the worker one at a time and are dropped as they go, so a set is never in
  memory twice. The worker runs at a lower priority.
- **Turned cameras.** The enclosure stands the camera on its side. The live
  stage does not care which way is up, so frames are kept as the camera gives
  them, and only a set's crops, their positions and the live centres are turned
  upright before the analysis, which needs to know where gravity points.
- **Light.** The exposure stays at what was set; the gain follows the light
  between sets, towards a median brightness of 110, and is never changed during
  a set.
- **Sets with no rise.** A plate that moved without being lifted, knocked or
  carried past, is analysed, kept on disk, and left out of the app.
- **Time.** At the gym the Pi has no internet and no clock of its own, and wakes
  at whatever time it was switched off. The app sends the phone's time when it
  connects, and the station sets the system clock from it when it has no NTP.
- **Network.** At boot the Pi waits 40 s for a Wi-Fi it knows. If none comes,
  it opens its own hotspot, `KvihtAI`, and the station is at
  `http://10.42.0.1:8080`. Waiting first matters: an access point is always
  available to NetworkManager, and would win over the home Wi-Fi.

## Open Questions

- **Oblique views.** Beyond about 25 degrees off square the coarse stage and the
  outline learning need work, see above. A clip at 45 to 60 degrees is the test.
- **Analysis time on the Pi.** At 66 to 134 ms a frame on the Pi 4, a 30 s set
  at 60 fps takes minutes; a 4.5 s set took 28 s in the station. About half of
  a short set's time is learning the outline, which is the same for any
  length. Options: fewer rays, fewer frames for the outline, measuring every
  frame only where the bar moves fast, or spreading the frames over the Pi's
  idle cores.
- **Following at 60 fps on a Pi 4.** With a plate near the camera the watcher
  needs nearly all of the 16.7 ms between frames, and falls behind when
  anything else runs. The 4 s ring buffer absorbs that for a while; a long set
  may still lose frames, and each set reports how many. The Pi 5 should not
  have this problem.
- **Absolute scale:** the chain is self-consistent but has never been checked
  against a length that is not part of the calculation. The bar is 2200 mm and
  is in shot, which would settle it without any new footage.
- **Tripod footage.** Both clips are handheld, so the camera's own movement is
  in every bar path measured so far.
- **Naming the rises.** A snatch from the floor is one movement with two rises,
  the pull and the stand from the catch; a squat set is one movement with a rise
  per rep. Which rise is the lift depends on the lift, and is not decided.
