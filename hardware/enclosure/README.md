# Camera enclosure

A 3D-printed enclosure that holds a Raspberry Pi 5 with the official Active
Cooler and a Raspberry Pi Camera Module 3 in one box. It is built in FreeCAD from
a Python script, checked against Raspberry Pi's own STEP models, and printed
without supports on a Prusa Core One with a 0.4 mm nozzle.

![Standing, as used](out/render/standing_3q.png)

## How it is used

The box stands on its GPIO side, on the table or on a tripod. The lens looks out
of the large front face. In this position the camera's long image axis is
vertical, so the image is portrait, which suits a lifter filmed from the side.

`rpicam` can only rotate by 0 or 180 degrees, so the 90 degree turn to portrait
is done in software, for example `cv2.rotate`. The mount fixes the rotation, so
it is a constant. Check its direction once with a test frame.

Outside size: 91 x 69.5 x 29.7 mm (wide, tall, deep, when standing).

| Side, when standing | What is there |
| --- | --- |
| Front | Lens, fan intake grille, name |
| Top | USB-C power, both micro-HDMI, vent slots |
| Bottom | 1/4"-20 tripod nut |
| Left, seen from the front | USB 3.0, USB 2.0, Ethernet |
| Right, seen from the front | microSD, power button, status LED, blower exhaust slots |
| Back | Four screw heads, vent slots under the board |

## Parts

| Qty | Part | Notes |
| --- | --- | --- |
| 1 | Raspberry Pi 5 | |
| 1 | Raspberry Pi Active Cooler | Expected. The lid height and the vents are designed round it. |
| 1 | Raspberry Pi Camera Module 3, standard lens | For the wide lens, build with `KVIHTAI_CAMERA=wide`. |
| 1 | Pi Zero camera cable, 38 mm | 22-way 0.5 mm pitch to 15-way 1 mm pitch. See the note below. |
| 4 | M2.5 x 12 socket head screw, DIN 912 | Hold the Pi and close the box. 10 to 16 mm also fits. |
| 4 | M2 x 5 or M2 x 6 screw | Hold the camera. Not longer than 6 mm, or it comes out of the front face. |
| 1 | 1/4"-20 UNC hex nut | 11.1 mm across flats, the standard tripod thread. |

The M2.5 and M2 screws cut their own thread in the plastic. No inserts are
needed.

**Camera cable.** The camera sits right above the Pi's camera connector, so the
cable is short and does not fold. The design is checked for a 38 mm cable,
which lies in a small loop up towards the lid. The 200 mm cable that comes with
the Camera Module 3 for the Pi 5 does not fit neatly. Other short lengths are
not checked.

## Printing

Open `out/print/enclosure.3mf` in PrusaSlicer. It holds all three parts on one
plate, already turned the way they print, with the settings they need stored
per part. Select these profiles, slice, and print:

- Printer: Prusa CORE One HF0.4 nozzle.
- Print: `0.20mm STRUCTURAL @COREONE 0.4`.
- Filament: `Prusament PETG @COREONE HF0.4`, or another PETG. PLA is not
  recommended next to a warm Pi 5.

The whole plate takes about 2 h 30 min and 49 g of PETG (PrusaSlicer 2.9
estimate).

| Part | Lies on | Time alone | Filament |
| --- | --- | --- | --- |
| Base | its back | 1 h 42 min | 30.8 g |
| Lid | its front face | 52 min | 18.2 g |
| Button pin | its collar | 1 min | 0.1 g |

The settings stored in the project, if you slice the single STLs in
`out/print/` instead:

- Perimeters: 3.
- Bottom solid layers: 5. The back and the front face are 2 mm thick. With the
  profile's 4 layers, the slicer leaves one layer of sparse infill in them.
- Supports: off. Nothing needs them.
- Lid only, bridging angle 90 degrees. The first layers above the engraved name
  are bridges. At 90 degrees they cross the letters in short spans; at the
  default angle PrusaSlicer warns about long bridges there.

The lid prints on its front face, so the face takes the texture of the sheet.
A satin or textured sheet gives the best-looking front.

Details that let it print without supports:

- Every horizontal hole (LED window, power button, tripod thread) is a
  teardrop with a 45 degree roof.
- The counterbores for the screw heads have two sacrificial bridge layers.
  After printing, push a screw through to break them if they did not open.
- The vent slots in the side walls are vertical when printing.
- The outer edges on the bed are chamfered, which stops elephant's foot from
  showing.

## Assembly

1. Fit the Active Cooler to the Pi 5 and plug in its fan. Route the fan cable
   round the lid post at the mounting hole between the GPIO header and the USB
   ports, not over it.
2. Drop the 1/4"-20 nut into the slot in the block on the GPIO-side wall.
3. Put the button pin into its hole in the short wall at the microSD end, from
   the inside, collar inwards. Hold it with a finger or a piece of tape.
4. Plug the 22-way end of the cable into CAM/DISP 0, contacts facing the
   Ethernet jack.
5. Lower the Pi into the base, USB and Ethernet end first and tilted, so the
   ports go into their openings. Then lower the other end flat onto the four
   standoffs. The Pi now holds the button pin in place.
6. Screw the camera onto the four standoffs inside the lid with the M2 screws,
   lens through the round opening.
7. Plug the 15-way end of the cable into the camera, contacts facing the camera
   board. The lid opens about 70 degrees on the cable, hinged along the port
   side, which is enough to reach the connector latch.
8. Close the lid. Its lip goes inside the walls, and a tab on the lid rests on
   the tripod nut.
9. Turn the box over and fit the four M2.5 screws from the back. They pass
   through the base and the Pi and cut into the posts of the lid. Tighten
   gently; the lid clamps the Pi against the standoffs.

## Checks

`check.py` loads Raspberry Pi's STEP models of the Pi 5 and the Camera Module 3
and tests the printed parts against them. With the current parameters:

- No overlap between any printed part and the Pi, the camera, the Active
  Cooler or the camera cable. The cable is also checked against the cooler, the
  Pi and the camera.
- Nothing in the base reaches over the outline of the Pi, so the Pi can be
  lowered in.

Smallest gaps, from `out/check/report.json`:

| Between | Gap | Where |
| --- | --- | --- |
| Lid post and cooler push-pin cap | 0.45 mm | mounting hole by the USB-C connector |
| Lid and Pi components | 0.83 mm | same post |
| Camera connector and cooler | 0.41 mm | over the blower |
| Camera cable and lid | 0.36 mm | top of the cable loop |
| Camera cable and cooler | 0.95 mm | over the blower |
| Button pin and Pi power button | 0.25 mm | free play before the pin presses |
| Standoffs and Pi board | 0.03 mm | the board rests on the standoffs |

The modelled cable path is 35 mm, counting the ends inside both connectors,
against 38 mm of cable, so the cable is not pulled tight.

The wide-lens build is only checked for valid geometry. Raspberry Pi's STEP
model is of the standard lens, so the wide lens is not tested against the lid.

Renders and cross-sections are in `out/render/`.

## Limits

- The Active Cooler is not in Raspberry Pi's STEP model. It is modelled as an
  envelope from its mechanical drawing, with every height taken as an upper
  bound.
- The camera cable is modelled as a path, not simulated. Its real shape
  depends on how it is folded in.
- Hole sizes assume PETG on a Core One. Other printers or materials may need
  the pilot holes in `enclosure.py` opened up or closed down by 0.1 mm.
- Nothing here has been printed yet.

## Rebuilding

The design is `enclosure.py`; all dimensions are in `Params` at the top.

```sh
freecad.cmd build.py      # writes out/*.step, out/enclosure.FCStd, out/print/
freecad.cmd check.py      # downloads the vendor models into ref/ on first run
python render.py          # needs numpy and opencv, writes out/render/*.png
```

`KVIHTAI_CAMERA=wide freecad.cmd build.py` builds the lid for the wide-angle
camera, 1.2 mm deeper. The STEP files in `out/` hold the parts in their
assembled position for use in other CAD programs.
