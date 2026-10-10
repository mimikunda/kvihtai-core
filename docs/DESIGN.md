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

**Fine.** What is measured is the front face's rim. Its centre is the bar end,
and it is what a person marks as the plate. The outline in the picture is not
the face: seen a little from the side, the tread shows beyond the face on one
side, with the plates behind it and the far end of the bar further out still.
On the station's gym footage that band was 13 to 19 px wide at the top of a
lift, on a plate 300 px across.

Which side it shows on, the sleeve says. Its end stands out of the plate
towards the camera, and parallax carries it across the face: below the hub
while the bar rests on the floor below the camera, above it overhead. The tread
lies on the other side, moved out by the same parallax applied to the plate's
thickness. So the sleeve's end is found in every frame, as the strongest blob
of its size, 0.12 of the plate's radius, in lightness less chroma: steel is
light and without colour where plates are coloured or dark. A blob counts only
if it is lighter than nearly all of the ring round it; the edge of a steel hub
is light inside too, but light on half its ring. On the side away from the
sleeve's end, edges beyond the face, up to 0.3 times the end's offset from the
centre, cost nothing.

Edges are found along 180 rays, every peak of the colour gradient in Lab, each
placed at the middle of its transition: the centroid of the gradient above 30 %
of its peak between the minima either side. For a rim smeared by motion that
is where it was at mid-exposure; on synthetic blur it stayed within half a
pixel up to an 8 px smear. Lab, because a plate can differ from its background
in lightness alone (black on dark clothes) or in hue alone (red on skin).

The face is fitted by soft assignment: every edge on a ray counts, weighed by
a Gaussian of its distance from the face against a level for "none of these is
the rim", with sigma shrinking from 3 px to 1 px. The cost is smooth, so a
start a few pixels off does not hold the fit where it began. A fit that takes on
each ray only the edge nearest the model stays where it started, and on the
gym footage it started on the outline, tread and all, and stayed 5 to 8 px off
the face at rest and at the top of the lift. The radius is held near the frames
before. Off square the face is an ellipse; its axis ratio and direction are
learned from the first 20 frames of the set, and only position and size are
fitted per frame.

It is not unique where the rim is a band. The rounded edge of a plate shows 5
to 6 px wide, lit where it faces the light, and the fit can take the outer edge
of it at the top and the inner at the bottom, or the other way round: the same
radius, the centre 3 px apart either way. Which one it settles on depends on
the centre the rays are cast from and the start, both taken from the frames
before, so it carries over from frame to frame. On a gym
set, this tracker and its prototype, which fit a frame alike to 0.3 px given
the same start, put the centre more than 2 px apart in a tenth of the frames,
in runs of up to a dozen.

The face is followed frame by frame, both ways from the first frame in which it
is found round the coarse centre with the rim nearly all round; its radius is
searched there from 0.8 to 1.15 of the coarse one, which is the outline. Each
frame is looked for where the two before put it, and where the sleeve's end,
looked for near where its last offset puts it, says. That offset changes only
slowly, so the sleeve places the face within a pixel or two even where the bar
moves 30 px between frames. Where both fail, the coarse centre is tried, and
once the face has been lost for a few frames, circles of its size from a Hough
transform over the whole image: the frame, or the wide crop the live stage
keeps while it has lost the plate too. Of those this set's face could be, the
nearest to where it was going is taken; taking the one with most rim on it
jumped to a railing on one clip.

A frame is accepted when:

- the rim is on at least 60 of the 180 rays, in at least 8 of 12 directions;
- the radius is within 3 % of the frames before. Not of the start: a plate
  carried towards the camera grows, by 17 % from the floor to overhead on one
  clip, and held to the start's size it was rejected, then fitted inside its
  rim;
- the face has not moved faster than a dropped bar, 6 m/s;
- the face has the colour of the 10 accepted frames before, within 7 in a and
  b of Lab and 18 in L, and is as even: the spread of each at most 2.5 times
  theirs. After the bar was dropped on the gym footage, a ring and a rack were
  followed until this check: a face spreads 1 to 3.5 in L, they spread 15 to
  21. Against the frames before rather than the start, because the light
  changes as the bar moves through it and a phone's exposure follows; against
  the start it rejected 193 frames of one clip that were right, and against
  the 30 frames before it still rejected the top of the third lift on the
  second test clip.

A set that followed the wrong thing from start to finish agrees with itself, so
the whole set must also carry its face: sampled in the circle's own frame, the
inside of a plate is the same picture wherever it has moved to, while a ring or
a wheel shows the room behind it sliding through. Frames a radius apart are
compared. Concentricity, which used to decide this, cannot on the new footage:
a plate with a printed label and a lighting gradient scores 0.08 where a ring
scores 0.11. A set that never moved a radius still has to be built in rings.
Rejected frames are reported with the reason and never interpolated.

Each frame's millimetres per pixel come from the face's major axis, smoothed
by a running median over a quarter of a second: positions are measured from
the image centre and scaled per frame, and a scale that jittered by 0.3 % would
move a plate 500 px from the centre by 1.5 px.

The tracker before this one learned the outline once per set and fitted only
its position and scale per frame, then put the face a fixed tread's width
inside it. One outline cannot follow a tread that swings from above the face
to below it during a lift: against hand-marked rims it was off by a median of
4.8 px and up to 19 px.

### What has been verified

Two phone clips, both handheld, synthetic plates, 37 more phone clips, and six
sets from the station at the gym checked against rims marked by hand.

Noise below is the scatter of the centre about a local quadratic over five
frames, with the same script for the earlier tracker, the one that learned the
outline once per set.

- **A 5.3 s snatch**, 720x1280 at 60 fps, red competition plates, about 10
  degrees off square. 319 of 319 frames measured, against 311. Pull 1032 mm,
  peak 1.86 m/s. The earlier tracker gave 2.05 m/s, from a step of 18 px in
  one frame after two frames 6 px behind; this one moves 10 to 12 px a frame
  there. Noise 0.26 px, against 0.31.
- **A 6.2 s clip of three lifts**, 1440x1920 at 30 fps, near square. 185 of
  185 frames measured, against 182. At 30 fps the plate moves up to 50 px
  between frames and is visibly smeared; noise 0.82 px, against 2.0.
- **Synthetic plates** of every colour, with and without visible tread, and a
  ring of the plate's exact size in the background. Face centre within 0.5 px,
  scale within 1 %, velocity within 2 %.
- **Gym sets from the station**, 1536x864 at 60 fps, against the front face's
  rim marked by hand in 10 frames each of two sets: the centre is a median of
  1.8 and 1.4 px off, at most 5.1 and 2.5 px. The earlier tracker was 3.9 and
  6.2 px off, at most 5.4 and 13.6, and rejected 3 of the 20 frames. The marks
  themselves scatter by about 1.2 px. The largest miss is the band described
  above, at the top of a fast pull. Of six usable sets 96 % of frames were
  measured; the rest is mostly the plate leaving the picture.

The earlier red-only tracker put the first clip at 19 degrees, the outline
tracker at 13, this one at 10. None can be trusted at that angle, for the
reason given under Geometry.

Both clips were shot handheld. The camera moves by up to 38 px during them,
zooms by 1 % and turns by 1 degree. The per-frame scale absorbs the zoom, but
the camera's movement is in the bar path as if the bar had made it. On a tripod
this goes away; on these clips it cannot be separated from the bar.

**Oblique views have a limit.** On synthetic plates the tracker is exact to 25
degrees off square. From 30 degrees the coarse stage, which looks for circles,
hands the fine stage a centre off by 5 to 11 px, and at 40 degrees the camera
angle comes out as 25. At 50 degrees nothing is found. The fine stage looks
for the rim from 0.88 to 1.2 of the face's semi-major axis round its centre,
so beyond 28 degrees, where the minor axis is shorter than 0.88 of the major,
the rim near the ends of the minor axis falls outside what it looks at. Both
need work before a camera well off to the side can be supported.

**37 more phone clips**, from six sessions in four gyms, handheld and upright:
31 at 720x1280 and 60 fps, 6 at 480x848 and 24 or 30 fps, 526 s in all. With
the near side given, 26,359 of 27,845 frames were measured (95 %), against
24,273 (87 %) with the outline tracker. The median noise is 0.26 px, from 0.13
to 0.78, against 0.42, from 0.09 to 1.24. Not yet right, all of it in the
coarse stage or the start:

- two clips measure nothing, as before. Started from another frame, one of
  them was measured whole;
- on two clips the face is followed from the first frame of a coarse path that
  begins on the wrong thing and reaches the near plate only after a gap: on
  one it stays on the far plate (608 frames), on the other on the lifter's
  legs and then the far plate (496 frames). The outline tracker measured each
  frame round the coarse centre, and on the first of them went over to the
  near plate with the path;
- on one clip a plate lying still on the floor is followed instead of the bar
  (747 frames), the camera's shake giving it the most travel;
- on one clip the start took the outline of the plate and the one behind it
  for the face, about 7 % too large.

Starting from the longest stretch the coarse stage followed without a break
put four of these right and broke three clips that were right: on one the path
held a circle in the empty background after the plate had left, and the face
was followed there and accepted; on another the start took a smaller circle
off the face's centre. So the start is chosen as before, from the first frame.

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
- **Glare.** A circle more than a quarter clipped white is not watched either.
  With the sun in a window at home, the search found circles in the blown-out
  glass and the frame bars across it. Their fit wandered by more than 8 % of
  a radius, and 25 of the 31 sets started that morning were on them. No plate
  on the 37 phone clips had more than 6 % of its circle clipped. Played
  through the watcher frame by frame, 20 minutes of that morning's recordings
  started 8 sets before and 2 after, and the 37 clips came out the same. The
  2 are on the corner of a chair, which this does not catch.
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

Live and offline: rises of 1018 and 717 mm live against 1032 and 711 mm
offline on the first clip; 535, 801 and 856 mm against 560, 823 and 853 on the
second. With the outline tracker the second clip agreed to 1 %; the first rise
there is now 5 % apart, not yet looked into.

Measured on the test Pi 4B, with the clip decoded into memory first:

| | first clip, 720x1280, plate 158 px across | second clip, 1440x1920, 316 px |
|---|---|---|
| search, in its own thread | 164 ms | 85 ms |
| check, every 8th frame | 20 ms | 51 ms |
| follow, every frame | 13 ms, 75 fps at most | 23 ms, 44 fps at most |
| analysis, after the set | 66 ms a frame | 134 ms a frame |

The analysis row is the outline tracker's. Measured later on the same Pi with
the station recording in the background, it took 163 and 337 ms a frame, and
the face tracker 137 and 272.

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
- **One set waits, not more.** A plate that starts to move while one set is
  being analysed and another waits is not followed; the log says so, and the
  recording still has it. One morning the watcher started 180 s sets on a
  sunlit window every three minutes, each took 10 to 12 minutes to analyse
  on the Pi 4B, and each waited in memory for its turn.
  Holding back every set until the one before is analysed would lose real
  ones: a 25 s set takes two to three minutes there, about the rest before
  the next.
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
  fine stage's search band need work, see above. A clip at 45 to 60 degrees is the test.
- **Analysis time on the Pi.** At 137 to 272 ms a frame on the Pi 4 with the
  station recording, a 30 s set at 60 fps takes minutes. On the laptop 60 %
  of it is the fit, iterated at each of five sigmas, a fifth finding the edges
  and an eighth the sleeve's end. Options: fewer sigmas or rays, measuring
  every frame only where the bar moves fast, or spreading the frames over the
  Pi's idle cores.
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
