"""Parametric enclosure for a Raspberry Pi and a Camera Module 3.

Two boards: a Raspberry Pi 5 with the Active Cooler (``board="pi5"``, the
default), and a Raspberry Pi 4 Model B with the fan of the Raspberry Pi 4 Case
Fan (``board="pi4b"``).

Run through FreeCAD, for example ``freecad.cmd build.py``. This module only builds
geometry; build.py exports it and check.py tests it against the vendor models.

All coordinates are in the frame of the board, the same frame as Raspberry Pi's
STEP model of the Pi 5: origin at the board corner next to the USB-C
connector, x along the 85 mm edge towards the USB and Ethernet ports, y along
the 56 mm edge towards the GPIO header, z up from the underside of the PCB.
The two boards share the outline and the mounting holes.

The lid is the front face and carries the camera, looking out along +z. The
enclosure stands on its +y face (the GPIO side), where the tripod nut is. In
that position the image is portrait, which suits a lifter seen from the side.
"""

import math

import FreeCAD as App
import Part

V = App.Vector

# --- Parts the enclosure is built around (from the vendor drawings and STEP models)

PI_HOLES = [(3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5)]
BOARDS = ("pi5", "pi4b")

# Active Cooler, read off Raspberry Pi's mechanical drawing. The drawing gives
# 13.7 mm from the push-pin tips to the top; the tips sit below the PCB, so
# measuring from the PCB underside gives an upper bound for every height here.
COOLER_TOP = 13.7                 # push-pin caps and fan screw heads
COOLER_BODY_TOP = 12.5            # fins and fan housing
COOLER_FAN = (27.5, 57.5, 18.0, 47.9)            # x0, x1, y0, y1 of the blower
COOLER_FAN_SCREWS = [(31.5, 46.2), (54.5, 45.5), (54.5, 20.4)]
COOLER_PINS = [(3.5, 9.5), (61.5, 46.5)]

# Camera Module 3, vendor model frame: PCB back at z = 0, lens towards -z,
# FPC connector on the back along the x = 23.86 edge.
CAM_PCB_X = 23.862
CAM_PCB_Y = 25.0
CAM_PCB_T = 0.74
CAM_HOLES = [(2.0, 2.0), (14.5, 2.0), (2.0, 23.0), (14.5, 23.0)]
CAM_LENS = (14.4, 12.5)
CAM_CONN_H = 2.60                 # connector height behind the PCB
CAM_LENS_TIP = {"standard": 7.65, "wide": 8.81}   # PCB back to lens tip
CAM_LENS_D = {"standard": 5.77, "wide": 7.04}     # widest part that enters the lid

# Hardware
M25_PILOT = 2.2                   # self-tapping M2.5 into PETG
M25_CLEAR = 2.9
M25_HEAD_D, M25_HEAD_H = 5.2, 2.7  # counterbore for DIN 912 M2.5
M2_PILOT = 1.7
NUT_AF, NUT_T = 11.11, 5.56       # 1/4"-20 UNC hex nut

# Pi 4 only. The fan of the Raspberry Pi 4 Case Fan, taken out of its clip-in
# housing, is an ADDA AD0205MX-K50, 25 x 25 x 6 mm. Its corner holes are taken
# to be on the usual 20 mm square; not measured. Its leads end in female Dupont
# housings on GPIO pins 4, 6 and 8.
FAN_SIZE, FAN_T, FAN_HOLES = 25.0, 6.0, 20.0
GPIO_BASE_H = 2.5                 # plastic base of the GPIO header, above the PCB
DUPONT_H = 14.0                   # female Dupont housing, standing on that base
PI4_LEDS = (7.9, 11.5)            # y of the power and activity LEDs, which light out of the x = 0 edge

# Per board: the parts of Params that differ.
BOARD_PARAMS = {
    "pi5": {
        "pcb_top": 1.31,          # top face of the PCB in the vendor model
        "lens_x": 41.0,           # 1.5 mm left of centre, which gives the camera cable room for its loop
        "cam_flip": False,
        "grille": (42.3, 35.0, 11.0, 24.9),   # centre, radius, lowest y: over the blower, clear of the camera
    },
    "pi4b": {
        # The Pi 4 model has 6 mm pads round the mounting holes, 0.1 mm proud of
        # both faces of the 1.6 mm board. The board rests on them, so z = 0 is
        # the underside of the pads and the lid posts clamp their top.
        "pcb_top": 1.8,
        # The Pi 4 wants the cable's contacts facing the micro-HDMI sockets. With
        # a cable that has its contacts on the same side at both ends, that is
        # only possible with the camera turned so its connector faces -x.
        "cam_flip": True,
        "lens_x": 43.5,           # the camera ends 1.2 mm short of the lid post by the audio jack
        "fan_xy": (33.0, 37.0),   # over the SoC, clear of the camera
        "fan_gap": 3.0,           # between the fan and the lid: room for the air it draws in
        "lead_bend": 2.0,         # above the Dupont housings, for the fan leads to bend over
        "grille": (33.0, 37.0, 11.0, 0.0),    # the fan's intake
    },
}


class Params:
    board = "pi5"
    camera = "standard"

    wall = 2.0
    floor = 2.0
    lid = 2.0
    corner_r = 4.0
    edge_chamfer = 0.8

    # inner faces of the walls
    x_in = (-1.0, 86.0)           # right wall: the USB and Ethernet shells sit in its cut-outs
    y_in = (-2.0, 63.5)           # port side clears the micro-HDMI shells; GPIO side makes room for the tripod nut
    standoff = 4.0                # floor to PCB underside

    lens_y = 11.3                 # as close to the Pi's camera connectors as the port wall allows
    lens_recess = 0.5
    cam_clear = 0.4               # camera connector above the cooler

    lid_lip_h = 2.5
    lid_lip_t = 1.2
    lid_lip_gap = 0.25
    seam_gap = 0.3                # lid clamps the PCB through its posts, not the walls

    post_r = 2.4
    tripod_z = 10.0

    def __init__(self, **kw):
        board = kw.get("board", self.board)
        if board not in BOARDS:
            raise ValueError(f"unknown board {board!r}, expected one of {BOARDS}")
        for k, v in BOARD_PARAMS[board].items():
            setattr(self, k, v)
        for k, v in kw.items():
            setattr(self, k, v)
        self.x_out = (self.x_in[0] - self.wall, self.x_in[1] + self.wall)
        self.y_out = (self.y_in[0] - self.wall, self.y_in[1] + self.wall)
        self.z_floor = -self.standoff
        self.z_bot = self.z_floor - self.floor
        self.cam_x0 = self.lens_x - CAM_LENS[0]
        if self.board == "pi5":
            # The camera hangs from the lid over the blower. Its connector must
            # clear the cooler, and its lens tip sits just under the front face;
            # that fixes the height of the lid.
            self.cam_zb = COOLER_BODY_TOP + self.cam_clear + CAM_CONN_H
        else:
            # The fan leads on the GPIO header are the tallest thing in the box,
            # and they fix the height of the lid. The camera hangs from it.
            z_li = self.pcb_top + GPIO_BASE_H + DUPONT_H + self.lead_bend
            self.cam_zb = z_li + self.lid - CAM_LENS_TIP["standard"] - self.lens_recess
        self.z_top = self.cam_zb + CAM_LENS_TIP[self.camera] + self.lens_recess
        self.z_li = self.z_top - self.lid
        self.z_wall = self.z_li - self.seam_gap
        self.x_mid = (self.x_out[0] + self.x_out[1]) / 2

    def cam_to_box(self, xm, ym, zm=0.0):
        """Camera model coordinates to box coordinates.

        The model looks along -z and the box along +z, so the camera is turned
        180 degrees about x; with cam_flip, 180 degrees about y instead, which
        puts its cable connector on the -x side."""
        if self.cam_flip:
            return V(self.lens_x - (xm - CAM_LENS[0]), self.lens_y + (ym - CAM_LENS[1]), self.cam_zb - zm)
        return V(self.cam_x0 + xm, self.lens_y - (ym - CAM_LENS[1]), self.cam_zb - zm)

    def cam_placement(self):
        if self.cam_flip:
            rot = App.Rotation(V(0, 1, 0), 180)
            return App.Placement(V(self.lens_x + CAM_LENS[0], self.lens_y - CAM_LENS[1], self.cam_zb), rot)
        rot = App.Rotation(V(1, 0, 0), 180)
        return App.Placement(V(self.cam_x0, self.lens_y + CAM_LENS[1], self.cam_zb), rot)


# --- geometry helpers

def box(x0, x1, y0, y1, z0, z1):
    return Part.makeBox(x1 - x0, y1 - y0, z1 - z0, V(x0, y0, z0))


def rbox(x0, x1, y0, y1, z0, z1, r):
    """Box with its vertical edges rounded."""
    b = box(x0, x1, y0, y1, z0, z1)
    if r <= 0:
        return b
    vert = [e for e in b.Edges if abs(e.Vertexes[0].Point.z - e.Vertexes[1].Point.z) > 1e-6]
    return b.makeFillet(r, vert)


def zcyl(x, y, r, z0, z1):
    return Part.makeCylinder(r, z1 - z0, V(x, y, z0))


def rslot_y(x, z, w, h, r, y0, y1):
    """Rounded rectangle in the xz plane, centred on (x, z), extruded along y."""
    b = box(x - w / 2, x + w / 2, y0, y1, z - h / 2, z + h / 2)
    edges = [e for e in b.Edges if abs(e.Vertexes[0].Point.y - e.Vertexes[1].Point.y) > 1e-6]
    return b.makeFillet(r, edges) if r > 0 else b


def rslot_x(y, z, w, h, r, x0, x1):
    """Rounded rectangle in the yz plane, centred on (y, z), extruded along x."""
    b = box(x0, x1, y - w / 2, y + w / 2, z - h / 2, z + h / 2)
    edges = [e for e in b.Edges if abs(e.Vertexes[0].Point.x - e.Vertexes[1].Point.x) > 1e-6]
    return b.makeFillet(r, edges) if r > 0 else b


def teardrop(axis, c1, c2, r, a0, a1):
    """Horizontal hole that prints without support: a circle with a 45 degree roof.

    axis is "x" or "y"; (c1, z = c2) is the centre in the other two coordinates.
    """
    k = r / math.sqrt(2)
    if axis == "x":
        cyl = Part.makeCylinder(r, a1 - a0, V(a0, c1, c2), V(1, 0, 0))
        pts = [V(a0, c1 - k, c2 + k), V(a0, c1 + k, c2 + k), V(a0, c1, c2 + r * math.sqrt(2)), V(a0, c1 - k, c2 + k)]
        roof = Part.Face(Part.makePolygon(pts)).extrude(V(a1 - a0, 0, 0))
    else:
        cyl = Part.makeCylinder(r, a1 - a0, V(c1, a0, c2), V(0, 1, 0))
        pts = [V(c1 - k, a0, c2 + k), V(c1 + k, a0, c2 + k), V(c1, a0, c2 + r * math.sqrt(2)), V(c1 - k, a0, c2 + k)]
        roof = Part.Face(Part.makePolygon(pts)).extrude(V(0, a1 - a0, 0))
    return cyl.fuse(roof)


def fuse_all(shapes):
    out = shapes[0]
    if len(shapes) > 1:
        out = out.fuse(shapes[1:])
    return out


def shell_outline(p, z0, z1):
    """The outside of the enclosure between two heights."""
    return rbox(p.x_out[0], p.x_out[1], p.y_out[0], p.y_out[1], z0, z1, p.corner_r)


def chamfer_face_edges(shape, z, d):
    """Chamfer every edge that lies in the plane z (used on the outer faces)."""
    edges = [e for e in shape.Edges
             if abs(e.BoundBox.ZMin - z) < 1e-6 and abs(e.BoundBox.ZMax - z) < 1e-6]
    return shape.makeChamfer(d, edges)


# --- base

# Openings in the port wall: x, z, width, height, corner radius. They are sized
# for plug overmoulds rather than the sockets, because the sockets sit about
# 2.4 mm behind the outer face.
PORT_WALL = {
    "pi5": [(11.2, 2.47, 13.0, 7.4, 2.2),    # USB-C power
            (25.8, 2.46, 11.4, 7.6, 2.0),    # micro-HDMI 0
            (39.2, 2.46, 11.4, 7.6, 2.0)],   # micro-HDMI 1
    "pi4b": [(11.2, 2.88, 13.0, 7.4, 2.2),
             (26.0, 2.95, 11.4, 7.6, 2.0),
             (39.5, 2.95, 11.4, 7.6, 2.0),
             (54.0, 4.43, 7.8, 7.9, 1.5)],   # audio jack, which stands 2.5 mm proud of the board
}

# Shells that reach into the right wall, from the board models: y0, y1, z0, z1.
USB_ETH_SHELLS = {
    "pi5": [(2.28, 18.22, -0.66, 14.64),     # Ethernet
            (13.28, 19.80, 1.38, 5.49),      # Ethernet shield finger
            (21.75, 36.25, -0.16, 17.53),    # USB 3.0 stack
            (39.75, 54.25, -0.17, 17.36)],   # USB 2.0 stack
    # The Pi 4 model is lower than Raspberry Pi's drawing on the USB 2.0 stack
    # (16.0 mm above the PCB); the taller of the two is used.
    "pi4b": [(1.28, 16.72, -1.85, 17.70),    # USB 2.0 stack
             (19.15, 34.85, -1.48, 18.04),   # USB 3.0 stack
             (37.65, 53.85, -1.85, 15.35)],  # Ethernet
}


def usb_hdmi_cutouts(p):
    """Openings in the port wall."""
    y0, y1 = p.y_out[0] - 0.1, p.y_in[0] + 0.1
    return [rslot_y(x, z, w, h, r, y0, y1) for x, z, w, h, r in PORT_WALL[p.board]]


def usb_eth_cutouts(p):
    """Openings round the shells that reach into the right wall."""
    c = 0.45
    x0, x1 = p.x_in[1] - 0.1, p.x_out[1] + 0.1
    out = []
    for y0, y1, z0, z1 in USB_ETH_SHELLS[p.board]:
        out.append(rslot_x((y0 + y1) / 2, (z0 + z1) / 2, y1 - y0 + 2 * c, z1 - z0 + 2 * c, 0.8, x0, x1))
    return out


def left_wall_features(p):
    """Status LED windows, captive power-button pin (Pi 5), microSD slot."""
    x0, x1 = p.x_out[0] - 0.1, p.x_in[0] + 0.1
    if p.board == "pi4b":
        leds = [teardrop("x", y, 2.2, 1.1, x0, x1) for y in PI4_LEDS]
        return leds + [box(x0, x1, 21.9, 34.3, -2.2, 0.6)]
    led = teardrop("x", 13.3, 1.84, 1.25, x0, x1)
    pin_hole = teardrop("x", 18.4, 3.04, 1.75, x0, x1)
    pin_pocket = teardrop("x", 18.4, 3.04, 2.75, BUTTON["pocket_x"], x1)
    sd = box(x0, x1, 21.9, 34.3, -2.2, 0.6)
    return [led, pin_hole, pin_pocket, sd]


def vents(p):
    out = []
    # left end; on the Pi 5, beside the heatsink fins, where the blower pushes its exhaust
    y = 1.5
    while y + 1.6 <= p.y_in[1] - 2.0:
        out.append(box(p.x_out[0] - 0.1, p.x_in[0] + 0.1, y, y + 1.6, 7.5, 18.5))
        y += 3.4
    # port side, above the connectors: the top face when the enclosure stands up
    x = 1.0
    while x + 1.6 <= 45.0:
        out.append(box(x, x + 1.6, p.y_out[0] - 0.1, p.y_in[0] + 0.1, 8.0, 14.5))
        x += 3.4
    # back, under the board
    x = 12.0
    while x + 2.0 <= 58.5:
        out.append(box(x, x + 2.0, 13.0, 43.0, p.z_bot - 0.1, p.z_floor + 0.1))
        x += 4.0
    return out


TRIPOD_BACK = 1.2                 # wall behind the nut


def tripod_boss(p):
    """Block on the inside of the GPIO-side wall holding a 1/4"-20 nut.

    The nut drops into a slot from the open top and is held down by a tab on the
    lid. The block must not reach over the PCB edge (y = 56), or the Pi could
    not be lowered past it."""
    y0 = p.y_in[1] - (NUT_T + 0.34) - TRIPOD_BACK
    if y0 < 56.3:
        raise ValueError(f"tripod boss reaches over the PCB edge (y0 = {y0:.2f})")
    return box(p.x_mid - 7.7, p.x_mid + 7.7, y0, p.y_in[1] + 0.01, p.z_floor - 0.01, p.z_wall)


def tripod_cuts(p):
    af = NUT_AF + 0.35
    rr = af / math.sqrt(3)
    xc, zc = p.x_mid, p.tripod_z
    y0, y1 = p.y_in[1] - (NUT_T + 0.34), p.y_in[1]
    hexpts = [V(xc + rr * math.cos(math.radians(90 + 60 * k)), y0, zc + rr * math.sin(math.radians(90 + 60 * k)))
              for k in range(6)]
    hexpts.append(hexpts[0])
    nut = Part.Face(Part.makePolygon(hexpts)).extrude(V(0, y1 - y0, 0))
    slot = box(xc - af / 2, xc + af / 2, y0, y1, zc, p.z_wall + 1)
    hole = teardrop("y", xc, zc, 3.35, y0 - TRIPOD_BACK - 0.1, p.y_out[1] + 0.1)
    cone = Part.makeCone(3.35, 4.15, 0.8, V(xc, p.y_out[1] - 0.8, zc), V(0, 1, 0))
    return [nut, slot, hole, cone]


def pi_standoffs(p):
    posts = [zcyl(x, y, 3.0, p.z_floor - 0.01, 0.0) for x, y in PI_HOLES]
    cuts = []
    for x, y in PI_HOLES:
        zc = p.z_bot + M25_HEAD_H
        cuts.append(zcyl(x, y, M25_CLEAR / 2, p.z_bot - 0.1, 0.1))
        cuts.append(zcyl(x, y, M25_HEAD_D / 2, p.z_bot - 0.1, zc))
        # two sacrificial bridge layers so the counterbore ceiling prints cleanly
        cuts.append(box(x - M25_CLEAR / 2, x + M25_CLEAR / 2, y - M25_HEAD_D / 2, y + M25_HEAD_D / 2, zc - 0.01, zc + 0.2))
        cuts.append(box(x - M25_CLEAR / 2, x + M25_CLEAR / 2, y - M25_CLEAR / 2, y + M25_CLEAR / 2, zc + 0.19, zc + 0.4))
    return posts, cuts


def build_base(p):
    outer = shell_outline(p, p.z_bot, p.z_wall)
    outer = chamfer_face_edges(outer, p.z_bot, p.edge_chamfer)
    cavity = rbox(p.x_in[0], p.x_in[1], p.y_in[0], p.y_in[1], p.z_floor, p.z_wall + 1,
                  p.corner_r - p.wall)
    base = outer.cut(cavity)
    posts, post_cuts = pi_standoffs(p)
    base = base.fuse(posts + [tripod_boss(p)])
    cuts = (usb_hdmi_cutouts(p) + usb_eth_cutouts(p) + left_wall_features(p) + vents(p)
            + tripod_cuts(p) + post_cuts)
    base = base.cut(cuts)
    return base.removeSplitter()


# --- lid

def lid_lip(p):
    g, t = p.lid_lip_gap, p.lid_lip_t
    x0, x1 = p.x_in[0] + g, p.x_in[1] - g
    y0, y1 = p.y_in[0] + g, p.y_in[1] - g
    r = p.corner_r - p.wall - g
    z0 = p.z_li - p.lid_lip_h
    ring = rbox(x0, x1, y0, y1, z0, p.z_li + 0.01, r).cut(
        rbox(x0 + t, x1 - t, y0 + t, y1 - t, z0 - 1, p.z_li + 1, max(r - t, 0.3)))
    # leave room for the tripod boss
    ring = ring.cut(box(p.x_mid - 8.0, p.x_mid + 8.0, 50.0, 70.0, z0 - 1, p.z_li + 1))
    # and for the USB and Ethernet shells where they reach up to the lip (Pi 4)
    c = 0.45
    for y0s, y1s, _z0s, z1s in USB_ETH_SHELLS[p.board]:
        if z1s + c > z0:
            ring = ring.cut(box(x1 - t - 1, x1 + 1, y0s - c, y1s + c, z0 - 1, z1s + c))
    return ring


def grille(p):
    cx, cy, r, y_min = p.grille
    disc = zcyl(cx, cy, r, p.z_li - 0.1, p.z_top + 0.1)
    keep = box(cx - r, cx + r, max(y_min, cy - r), cy + r, p.z_li - 0.2, p.z_top + 0.2)
    slots = []
    y = cy - r + 0.3
    while y < cy + r:
        slots.append(box(cx - r, cx + r, y, y + 1.8, p.z_li - 0.1, p.z_top + 0.1))
        y += 3.3
    return fuse_all(slots).common(disc).common(keep)


def engraving(p, text="KvihtAI", height=5.5, depth=0.6,
              font="/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"):
    """Name on the front face, upright when the enclosure stands on its GPIO side.

    The depth is three 0.2 mm layers, so no layer is sliced exactly on the floor
    of the letters.
    """
    try:
        chars = Part.makeWireString(text, font, height, 0.0)
    except Exception:
        print(f"engraving skipped: font {font} not found")
        return None
    faces = []
    for wires in chars:
        if wires:
            faces.append(Part.makeFace(wires, "Part::FaceMakerBullseye"))
    if not faces:
        return None
    solid = Part.makeCompound(faces).extrude(V(0, 0, depth + 0.1))
    bb = solid.BoundBox
    solid.translate(V(-(bb.XMin + bb.XMax) / 2, -(bb.YMin + bb.YMax) / 2, 0))
    # readable from the front with +y pointing down means reading towards -x
    solid.rotate(V(0, 0, 0), V(0, 0, 1), 180)
    solid.translate(V(p.x_mid, 56.0, p.z_top - depth))
    return solid


def build_lid(p):
    plate = shell_outline(p, p.z_li, p.z_top)
    plate = chamfer_face_edges(plate, p.z_top, p.edge_chamfer)
    adds = [lid_lip(p)]
    cuts = []

    # posts that clamp the Pi onto the base standoffs; M2.5 screws come in from the back
    for x, y in PI_HOLES:
        adds.append(zcyl(x, y, p.post_r, p.pcb_top, p.z_li + 0.01))
        cuts.append(zcyl(x, y, M25_PILOT / 2, p.pcb_top - 0.1, p.pcb_top + 14.0))

    # camera standoffs, M2 screws through the camera PCB from behind
    z_front = p.cam_zb + CAM_PCB_T
    for hx, hy in CAM_HOLES:
        c = p.cam_to_box(hx, hy)
        adds.append(zcyl(c.x, c.y, 2.1, z_front, p.z_li + 0.01))
        cuts.append(zcyl(c.x, c.y, M2_PILOT / 2, z_front - 0.1, z_front + 6.0))

    # lens opening with a small chamfer on the face
    lr = CAM_LENS_D[p.camera] / 2 + 0.36
    cuts.append(zcyl(p.lens_x, p.lens_y, lr, p.z_li - 0.2, p.z_top + 0.1))
    cuts.append(Part.makeCone(lr, lr + 0.6, 0.6, V(p.lens_x, p.lens_y, p.z_top - 0.6)))

    cuts.append(grille(p))

    if p.board == "pi4b":
        # posts the fan is screwed to through its corner holes, M2 from the Pi side
        cx, cy = p.fan_xy
        z0 = p.z_li - p.fan_gap
        for dx in (-1, 1):
            for dy in (-1, 1):
                x, y = cx + dx * FAN_HOLES / 2, cy + dy * FAN_HOLES / 2
                adds.append(zcyl(x, y, 2.0, z0, p.z_li + 0.01))
                cuts.append(zcyl(x, y, M2_PILOT / 2, z0 - 0.1, p.z_top - 0.6))

    # tab that keeps the tripod nut seated
    af = NUT_AF + 0.35
    nut_top = p.tripod_z + af / math.sqrt(3)
    adds.append(box(p.x_mid - af / 2 + 0.3, p.x_mid + af / 2 - 0.3,
                    p.y_in[1] - (NUT_T + 0.34) + 0.25, p.y_in[1] - 0.25,
                    nut_top + 0.3, p.z_li + 0.01))

    lid = plate.fuse(adds)
    text = engraving(p)
    if text is not None:
        cuts.append(text)
    lid = lid.cut(cuts)
    return lid.removeSplitter()


# --- captive power-button pin

BUTTON = {
    "axis": (18.4, 3.04),     # y, z of the Pi 5 power button
    "plunger_x": -0.45,       # front of the button plunger
    "gap": 0.25,
    "collar_d": 5.0, "collar_t": 1.1,
    "shaft_d": 3.0, "shaft_out": 0.6,   # how far the shaft stands proud of the wall
    "pocket_x": -1.8,         # bottom of the pocket the collar sits in
}


def build_button(p):
    y, z = BUTTON["axis"]
    x_in = BUTTON["plunger_x"] - BUTTON["gap"]
    x_collar = x_in - BUTTON["collar_t"]
    x_out = p.x_out[0] - BUTTON["shaft_out"]
    collar = Part.makeCylinder(BUTTON["collar_d"] / 2, BUTTON["collar_t"], V(x_collar, y, z), V(1, 0, 0))
    shaft = Part.makeCylinder(BUTTON["shaft_d"] / 2, x_collar - x_out + 0.01, V(x_out, y, z), V(1, 0, 0))
    return collar.fuse(shaft).removeSplitter()


def fan_envelope(p):
    """The fan as it sits on its posts in the lid (Pi 4)."""
    cx, cy = p.fan_xy
    h = FAN_SIZE / 2
    z1 = p.z_li - p.fan_gap
    return box(cx - h, cx + h, cy - h, cy + h, z1 - FAN_T, z1)


def build(**kw):
    p = Params(**kw)
    parts = {"base": build_base(p), "lid": build_lid(p)}
    if p.board == "pi5":
        parts["button"] = build_button(p)
    return p, parts
