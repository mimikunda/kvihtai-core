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

The Hough circle is often not the rim: on the new footage it was a change
plate or the steel disc inside the rim in about one clip in five, and seen from
an angle its centre is off the rim's. So the plate's size is measured from what
the followed track carries with it. Sampled round the followed centre, in the
plate's own frame, the plate looks the same wherever it has been carried to and
the background changes; where the spread of the samples over frames jumps is
the silhouette of the plate stack. An ellipse through that boundary gives the
radius and the centre's offset. Where the background is a plain wall it does
not change either, and those directions are left out of the ellipse.

**The near plate.** With the camera off to the side both ends of the bar are
in view, and the far plate is often the easier to see: sharp against a plain
wall, while the near one is a dark plate on dark tiles. On 14 of 37 new clips
the far one was followed. It is the one the lifter's arms and head cover, and
telling the two apart by size from the picture alone did not work: the hub of
the near plate often has more edges round it than its rim. So the person
setting up the station taps the near plate once in the app (see Station). On a
video file, `tools/track_plate.py --near left|right` does the same.

Given that side, the other end of the bar is found as the Hough circles that
keep a steady offset from the followed track, since the bar is rigid, and that
travel as far as it does, since a plate lying still while the bar rests before
the lift also keeps a steady offset. The end on the tapped side is followed.
Where Hough missed it, it is looked for where the rigid bar puts it. Its extent
is bounded by the far plate's: on ten clips the near plate was 1.22 to 1.35
times the far one in the picture, and round the near plate the extent can run
off into a plain wall; on one clip it came out four times the plate.

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

Where the tread or the plates behind show, the outline can sit on the stack's
silhouette or on the front face's edge a tread's width in. In the catch of the
first test clip the fit flipped between the two from frame to frame, 12 px
apart, and both had a residual under 1.5 px and edges in 12 of 12 sectors. What
tells them apart is the share of rays with an edge on the outline: 155 to 170
of 180 for the right one, 120 to 126 for the other. So a fit that disagrees
with what its neighbours predict is tried again from their prediction, and
whichever explains more of the rim is kept.

The front face is then recovered from the same pooled edges: it is the tread
vector that puts the most edges where the face edge would have to be, inside
the silhouette on the tread side. The face's centre is the bar end, its major
axis is the 450 mm, and its axis ratio is the camera angle.

Frames are accepted on four checks: edges in at least 5 of 12 directions round
the rim, a median edge distance from the outline under 3 % of the radius, a
scale within 5 % of the running median, and a radial brightness profile that
correlates with the set's median profile. A set that followed the wrong thing
from start to finish agrees with its own median, so the whole set must also
carry its face: sampled in the circle's own frame, the inside of a plate is the
same picture wherever it has moved to, while a ring or a wheel shows the room
behind it sliding through. Frames a radius apart are compared. Concentricity,
which used to decide this, cannot on the new footage: a plate with a printed
label and a lighting gradient scores 0.08 where a ring scores 0.11. A set that
never moved a radius still has to be built in rings. Rejected frames are
reported with the reason and never interpolated. A frame the coarse stage lost between two measured ones is
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

**37 more phone clips**, from six sessions in four gyms, handheld and upright:
31 at 720x1280 and 60 fps, 6 at 480x848 and 24 or 30 fps, 526 s in all. With
the near side given, 24,273 of 27,845 frames were measured (87 %), on the near
plate. The median noise is 0.42 px, from 0.12 to 1.3 px. Before the near
plate was followed the share was 92 %, but on 14 clips of the far plate, which
is easier to measure: its noise was 0.15 to 0.3 px on the clips where it has
since gone to 0.3 to 1.3. Not yet right:

- two clips measure nothing: on one the path leaves the near plate for a knee
  and a board where Hough never saw the plate, on the other the plate comes out
  1.8 times its size;
- on one clip a plate lying still on the floor is followed instead of the bar,
  the camera's shake giving it the most travel;
- the appearance check rejects frames that are right: 225 of 755 on one clip.

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
- **The near plate.** Both ends of the bar start to move in the same check.
  The one followed is on the side of the near plate tapped in the app, not
  whichever was confirmed first. Round the tap the search also looks for a
  plate at every size, since the Hough transform can miss the near plate
  altogether: on one clip only the far one was ever watched.
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
- **A camera that stops.** On the Pi 4B the Camera Module 3 has stopped
  sending frames for good, with no error, 20 s after it was opened. picamera2
  then waits for the next frame forever, and the app showed the last picture
  at 0 fps until the Pi was switched off. Now the station ends itself when no
  frame has come for 5 s, or for 20 s after the camera was opened, and systemd
  starts it again, which opens the camera afresh. Ending the process rather
  than closing the camera, because closing a camera whose driver has hung can
  hang too. What happened is kept in `incidents.json` and shown in the app.
  Why the camera stops is not known. At the moment it stopped, libcamera
  logged `PDAF data in unsupported format`, which it does when a frame's
  focus data cannot be read. The same message came 9 times in the first
  minute after the camera was opened, then about once a minute, and not at
  all in 4 minutes with the fan off, at full speed, or switched on and off.
- **Turned cameras.** The enclosure stands the camera on its side. The live
  stage does not care which way is up, so frames are kept as the camera gives
  them, and only a set's crops, their positions and the live centres are turned
  upright before the analysis, which needs to know where gravity points.
- **The near plate.** Tapped once in the app after the camera is set up, not
  before every lift: it stays on the same side of the picture until the camera
  is moved. It is kept in `camera.json` as the camera sees it, so turning the
  picture later does not move it.
- **Light.** The exposure stays at what was set; the gain follows the light
  between sets, towards a median brightness of 110, and is never changed during
  a set.
- **Focus.** Set once, like the near plate, and then held: autofocus in the
  app, or a distance by hand. The autofocus result is read when the camera
  says it has finished. Read after a fixed 2.5 s it was wherever the scan had
  got to, and the station kept 0.1 m for a wall 0.2 m away.
- **Sets with no rise.** A plate that moved without being lifted, knocked or
  carried past, is analysed, kept on disk, and left out of the app.
- **Time.** At the gym the Pi has no internet and no clock of its own, and wakes
  at whatever time it was switched off. The app sends the phone's time when it
  connects, and the station sets the system clock from it when it has no NTP.
- **Network.** At boot the Pi waits 40 s for a Wi-Fi it knows. If none comes,
  it opens its own hotspot, `KvihtAI`, and the station is at
  `http://10.42.0.1:8080`. Waiting first matters: an access point is always
  available to NetworkManager, and would win over the home Wi-Fi.
- **Shutting down.** The Pi was switched off by pulling the plug, which can
  corrupt the card and loses the open fragment of the recording. The app has
  Shut down, Reboot and Restart station. Shut down and reboot go through
  `sudo systemctl`, allowed by `deploy/setup_pi.sh` for exactly those two
  commands, and only where `KVIHTAI_POWER_CONTROL` is set, so that a laptop
  running the station is never switched off from a phone. The app follows the
  station until it stops answering and then counts down the few seconds the Pi
  needs before it says to unplug. Stopping gives up a set still being
  analysed: waiting for it held a restart up past systemd's 20 s, and the
  station was killed before it had closed the recording. A stop during a set
  now takes about a second on the Pi 4B.
- **How the last run ended.** While it runs, the station keeps `run.json` with
  the boot's id and, once a minute, the time. A clean stop deletes it. Found at
  start, it says the station was not stopped: from the same boot it crashed or
  was killed, and the journal is asked whether for memory or by the watchdog;
  from an earlier boot the Pi went off under it. Either is an incident. The
  camera restart above deletes it first, since its incident already says why.
- **Undervoltage and heat** come and go faster than the app polls, so the
  firmware's since-boot flags are kept as an incident, once a boot.
- **Logs.** `logs/station.log` on the card, 5 MB at most, holds the station's
  lines, the camera library's, and every warning and error from the process,
  a thread that died included. The journal is kept on the card too, 200 MB at
  most; Raspberry Pi OS keeps it in memory by default, and what led up to a
  power cut went with it. The app shows both, and the kernel's messages, and
  can filter them to the problems by their words: the journal gives every line
  of the service the same priority. uvicorn's access log is off; the status
  poll filled the journal with it. The app's diagnostics bundle is one zip of
  the logs, the journal of this boot and the one before, the settings, the
  incidents and the lists of sets and recordings.
- **Watchdog.** The station tells systemd every 5 s that it runs, but only
  while the API's event loop has run in the last 30 s. Silent for 60 s, it is
  killed and started again. A station that runs but does not answer is no use
  at the gym.
- **Files that survive a power cut.** `camera.json`, `result.json`,
  `incidents.json` and the rest are written to a temporary file, flushed and
  renamed. A database that cannot be read is moved aside and a new one filled
  from the sets' `result.json` files; deleting a set deletes its folder, so
  that it does not come back from there.
- **What was lifted.** The app says what the next set will be, a lift and a
  weight, and every set that ends from then on carries it; it is taken when
  the set ends, as the rotation is. A set's lift, weight and a note can be
  changed afterwards. With weights given, the history plots each set's best
  mean velocity against its weight. Sets export as CSV, a row per rise.
- **Missed sets.** A lifter who lifted and saw no set can say so in the app.
  The recording of the last five minutes is then kept: deleted for space only
  when nothing else is left, since it is the footage the watcher most needs.

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
- **The watcher on the new footage.** Played through the watcher frame by
  frame, as if the Pi kept up with every frame, with the near plate tapped
  where it lay, the live centre was within half a radius of the offline one in
  58 % of the offline run's frames, 47.5 % without the tap. On 8 of 37 clips
  the set still starts on the other plate: at 360 px the near plate's rim
  scores 0.5 to 0.9 where a watched plate needs 1.0, so it is not watched, or
  its rim is taken to be a ring well outside it. On many the plate is lost
  during the lift. Played in real time through a whole station session, six
  clips at once, the same clip varied by tens of percent between runs.
- **Absolute scale:** the chain is self-consistent but has never been checked
  against a length that is not part of the calculation. The bar is 2200 mm and
  is in shot, which would settle it without any new footage.
- **Tripod footage.** Both clips are handheld, so the camera's own movement is
  in every bar path measured so far.
- **Naming the rises.** A snatch from the floor is one movement with two rises,
  the pull and the stand from the catch; a squat set is one movement with a rise
  per rep. Which rise is the lift depends on the lift, and is not decided.
