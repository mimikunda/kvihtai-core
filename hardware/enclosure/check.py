"""Check the enclosure against the board and camera models. Run with ``freecad.cmd check.py``.

Set KVIHTAI_BOARD=pi4b for the Raspberry Pi 4 Model B; the default is the Pi 5.

Needs STEP models of the board and of the Camera Module 3. They are not
redistributed here; the script downloads them into ref/ on first run. For the
Pi 5 and the camera they are Raspberry Pi's own. Raspberry Pi publishes no STEP
model of the Pi 4, so for it the script uses the one on step.parts, which
agrees with Raspberry Pi's mechanical drawing on every part the enclosure is
built round.

Reports every overlap between a printed part and the Pi, the camera, the
cooler or fan and the camera ribbon, and the smallest gaps at the places where
the design is tight. Writes meshes and cross-sections to out/<board>/check/ for
render.py.
"""

import io
import json
import time
import math
import os
import sys
import urllib.request
import zipfile

import FreeCAD as App
import MeshPart
import Part

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import enclosure as E  # noqa: E402

V = App.Vector
BOARD = os.environ.get("KVIHTAI_BOARD", "pi5")
REF = os.environ.get("KVIHTAI_REF", os.path.join(HERE, "ref"))
OUT = os.path.join(HERE, "out", BOARD, "check")

MODELS = {
    "pi5": ("https://datasheets.raspberrypi.com/rpi5/RaspberryPi5-step.zip", "rpi-5b_no_graphics.step"),
    "pi4b": ("https://media.githubusercontent.com/media/earthtojake/step.parts/"
             "c6113328a5695b976a010a203a90fe86191769bf/catalog/step/raspberry_pi_4_model_b.step",
             "raspberry_pi_4_model_b.step"),
    "cm3": ("https://datasheets.raspberrypi.com/camera/camera-module-3-step.zip",
            "Camera_module_3_std_model_simple.stp"),
}

# Where the ribbon plugs into the Pi: x of the ribbon in the connector, top of
# the connector, and the box round the connector's own pieces.
PI_CONNECTOR = {
    "pi5": {"x": 55.0, "top": 5.29, "pieces": (53.0, 56.5, 17.0)},
    "pi4b": {"x": 46.1, "top": 7.38, "pieces": (43.8, 49.0, 23.0)},
}


def model_path(key):
    url, name = MODELS[key]
    for root, _dirs, files in os.walk(REF):
        if name in files and "__MACOSX" not in root:
            return os.path.join(root, name)
    os.makedirs(os.path.join(REF, key), exist_ok=True)
    print("downloading", url)
    data = urllib.request.urlopen(url).read()
    if url.endswith(".zip"):
        zipfile.ZipFile(io.BytesIO(data)).extractall(os.path.join(REF, key))
    else:
        with open(os.path.join(REF, key, name), "wb") as f:
            f.write(data)
    return model_path(key)


def load(path):
    s = Part.Shape()
    s.read(path)
    return s


def pieces(shape):
    """Solids, plus the shells that the Pi 5 model uses for its micro-HDMI sockets."""
    out = list(shape.Solids)
    in_solids = {x.hashCode() for s in shape.Solids for x in s.Shells}
    for sh in shape.Shells:
        if sh.hashCode() not in in_solids:
            bb = sh.BoundBox
            if bb.XLength > 3 and bb.YLength > 3:
                try:
                    out.append(Part.Solid(sh))
                except Exception:
                    pass
    return out


def shell_boxes(shells):
    """Each connected group of open shells as its bounding box, a solid."""
    boxes = [grown(sh.BoundBox, 0.02) for sh in shells]
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    active = []
    for i in sorted(range(len(boxes)), key=lambda k: boxes[k].XMin):
        active = [j for j in active if boxes[j].XMax >= boxes[i].XMin]
        for j in active:
            if boxes[i].intersect(boxes[j]):
                parent[find(i)] = find(j)
        active.append(i)
    groups = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        bb = App.BoundBox()
        for i in members:
            bb.add(shells[i].BoundBox)
        if min(bb.XLength, bb.YLength, bb.ZLength) > 0.05:
            out.append(E.box(bb.XMin, bb.XMax, bb.YMin, bb.YMax, bb.ZMin, bb.ZMax))
    return out


def pi4b_pieces(shape):
    """The Pi 4 model, moved into the board frame.

    The model lies in its xz plane with the components towards +y and the top
    of the PCB at y = 0. The board is 1.6 mm thick, and the pads round its
    mounting holes stand 0.1 mm proud of both faces; it rests on those, so
    their underside goes to z = 0. The micro-HDMI sockets and one USB stack
    are open shells, which cannot be tested for overlap; each connected group
    of them is replaced by its bounding box, which is at least as large as the
    part."""
    shape.Placement = App.Placement(V(0, 0, 1.7), App.Rotation(V(1, 0, 0), 90))
    shape = shape.copy()
    in_solids = {x.hashCode() for s in shape.Solids for x in s.Shells}
    free = [sh for sh in shape.Shells if sh.hashCode() not in in_solids]
    return list(shape.Solids) + shell_boxes(free)


def cooler_envelope(p):
    """Active Cooler as a conservative envelope, from the mechanical drawing."""
    z0 = p.pcb_top
    outline = [(6.95, 48.95), (63.95, 48.95), (63.95, 18.0), (45.6, 18.0), (35.5, 8.9),
               (3.5, 6.4), (0.4, 9.5), (3.5, 12.6), (6.95, 18.6), (6.95, 48.95)]
    plate = Part.Face(Part.makePolygon([V(x, y, z0) for x, y in outline])).extrude(V(0, 0, 4.6 - z0))
    parts = [plate,
             E.box(6.95, 27.5, 8.9, 48.95, z0, E.COOLER_BODY_TOP),       # pin fins
             E.box(27.5, 63.95, 18.0, 48.95, z0, E.COOLER_BODY_TOP)]     # blower
    for x, y in E.COOLER_FAN_SCREWS:
        parts.append(E.zcyl(x, y, 1.75, z0, E.COOLER_TOP))
    for x, y in E.COOLER_PINS:
        parts.append(E.zcyl(x, y, 3.15, z0, E.COOLER_TOP))   # spring and hex cap
        parts.append(E.zcyl(x, y, 1.6, -2.5, z0))            # push-pin through the board
    return E.fuse_all(parts).removeSplitter()


def fan_leads(p):
    """Female Dupont housings of the fan leads on GPIO pins 4, 6 and 8, the outer row."""
    x1 = 32.5 - 9.5 * 2.54          # pins 1 and 2
    y = 52.5 + 1.27
    z0 = p.pcb_top + E.GPIO_BASE_H
    h = 2.54 / 2
    return E.fuse_all([E.box(x1 + k * 2.54 - h, x1 + k * 2.54 + h, y - h, y + h, z0, z0 + E.DUPONT_H)
                       for k in (1, 2, 3)]).removeSplitter()


def extras(p):
    """What is in the box besides the Pi, the camera and the ribbon."""
    if p.board == "pi5":
        return {"cooler": cooler_envelope(p)}
    return {"fan": E.fan_envelope(p), "leads": fan_leads(p)}


def ribbon_path(p):
    """Control points of the camera FPC in the xz plane, and its width and centre y along the way."""
    zr = p.cam_zb - 1.3
    pc = PI_CONNECTOR[p.board]
    if p.board == "pi5":
        x_cam = p.cam_x0 + 23.49 - 3.5          # end of the ribbon inside the camera connector
        x_pi, z_pi = pc["x"], pc["top"] - 3.0   # end inside the Pi connector
        # a loop up towards the lid takes up the slack of a 38 mm cable
        x_e = p.cam_x0 + 23.86
        ctrl = [(x_cam, zr), (x_e, zr), (x_e + 1.6, 16.4), (x_e + 2.0, 19.4), (x_e + 3.6, 20.9),
                (56.6, 20.9), (58.3, 19.0), (58.4, 15.0), (57.8, 11.5), (56.6, 9.4),
                (55.3, 7.4), (55.0, 5.2), (x_pi, z_pi)]

        def width(t):   # 15-way end 16 mm wide and centred on the camera, 22-way end 11.5 mm on the Pi connector
            return 16.0 + (11.5 - 16.0) * max(0.0, (t - 0.55) / 0.45)

        def centre(t):
            return p.lens_y + (8.5 - p.lens_y) * t
        return ctrl, width, centre

    # Pi 4: the camera is flipped, so the ribbon leaves it towards -x. It runs
    # along under the lid to the left end, turns down and comes back above the
    # board to the Pi's camera connector, which it enters from above.
    x_cam = p.cam_to_box(23.49 - 3.5, 0).x
    x_e = p.cam_to_box(E.CAM_PCB_X, 0).x
    z_up = p.z_li - 0.5                     # under the lid
    z_lo, x_left = 6.0, 6.6                 # low point of the loop, and how far left it reaches
    r = (z_up - z_lo) / 2
    xc, zc = x_left + r, (z_up + z_lo) / 2
    k = r / math.sqrt(2)
    x_pi, z_pi = pc["x"], pc["top"] - 4.0
    z_v, r_v = pc["top"] + 0.5, 2.5        # the ribbon is vertical from here down, after a bend of this radius
    ctrl = [(x_cam, zr), (x_e, zr), (x_e - 2.0, zr + 0.6), (x_e - 4.0, zr + 2.6), (x_e - 6.0, z_up - 1.0),
            (x_e - 9.0, z_up), (xc + 5.0, z_up), (xc, z_up),
            (xc - k, zc + k), (x_left, zc), (xc - k, zc - k), (xc, z_lo),
            (xc + 8.0, z_lo), (28.0, z_lo), (34.0, z_lo + 1.5), (39.5, z_v + r_v - 0.2), (x_pi - r_v, z_v + r_v),
            (x_pi - r_v + r_v / math.sqrt(2), z_v + r_v / math.sqrt(2)), (x_pi, z_v), (x_pi, z_v - 1.8),
            (x_pi, z_pi)]

    def width(t):       # 15-way at both ends
        return 16.0

    def centre(t):
        return p.lens_y + (11.5 - p.lens_y) * t
    return ctrl, width, centre


def ribbon(p, n=90):
    """The camera FPC as a chain of thin slabs, from the camera connector to the Pi's.

    Returns the whole ribbon; the slabs outside the two connectors; those
    slabs less the first 3 mm out of the camera connector; and the length. Path in the xz plane, width along y. It leaves the camera
    connector with its contacts facing the camera PCB. The ribbon does not
    twist, so its contacts then face +x on the Pi 5 (the Ethernet jack, as the
    Pi 5 connector requires) and -x on the Pi 4 (the micro-HDMI sockets).
    """
    ctrl, width, centre = ribbon_path(p)
    curve = Part.BSplineCurve()
    curve.interpolate([V(x, 0, z) for x, z in ctrl])
    length = curve.length()
    pc = PI_CONNECTOR[p.board]
    x_mouth = p.cam_to_box(23.49, 0).x
    slabs, free, away = [], [], []
    mouth = V(x_mouth, 0, p.cam_zb - 1.3)
    for i in range(n):
        u0 = curve.FirstParameter + (curve.LastParameter - curve.FirstParameter) * i / n
        u1 = curve.FirstParameter + (curve.LastParameter - curve.FirstParameter) * (i + 1) / n
        a, b = curve.value(u0), curve.value(u1)
        t = (i + 0.5) / n
        w = width(t)
        yc = centre(t)
        d = b - a
        seg = Part.makeBox(d.Length, w, 0.25, V(0, -w / 2, -0.125))
        ang = math.degrees(math.atan2(d.z, d.x))
        seg.rotate(V(0, 0, 0), V(0, 1, 0), -ang)
        seg.translate(V(a.x, yc, a.z))
        slabs.append(seg)
        # A slab that reaches into either connector counts as inside it. A
        # tilted slab reaches back past its end by up to half its thickness,
        # hence the 0.3 mm.
        in_cam = any(p.cam_zb - E.CAM_CONN_H < q.z < p.cam_zb
                     and ((q.x > x_mouth - 0.3) if p.cam_flip else (q.x < x_mouth + 0.3)) for q in (a, b))
        in_pi = any(q.z < pc["top"] + 0.3 and abs(q.x - pc["x"]) < 1.5 for q in (a, b))
        if not (in_cam or in_pi):
            free.append(seg)
            if min((a - mouth).Length, (b - mouth).Length) > 3.0:
                away.append(seg)
    return E.fuse_all(slabs), free, away, length


def grown(bb, d=0.05):
    b = App.BoundBox(bb)
    b.enlarge(d)
    return b


def box_gap(p, q):
    """Distance between two bounding boxes, 0 where they overlap."""
    dx = max(0.0, p.XMin - q.XMax, q.XMin - p.XMax)
    dy = max(0.0, p.YMin - q.YMax, q.YMin - p.YMax)
    dz = max(0.0, p.ZMin - q.ZMax, q.ZMin - p.ZMax)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def overlaps(part, others, label, min_vol=1e-3):
    """Boolean overlap of part with each of others.

    A piece can only overlap the part if its box meets the box of one of the
    part's faces, so the expensive boolean runs on those candidates only.
    """
    t0 = time.time()
    found = []
    face_boxes = [grown(f.BoundBox) for f in part.Faces]
    pb = part.BoundBox
    for name, s in others:
        sb = s.BoundBox
        if not pb.intersect(sb) or not any(fb.intersect(sb) for fb in face_boxes):
            continue
        try:
            v = part.common(s).Volume
        except Exception:
            continue
        if v > min_vol:
            bb = s.BoundBox
            found.append({"with": name, "volume_mm3": round(v, 3),
                          "at": [round(bb.Center.x, 1), round(bb.Center.y, 1), round(bb.Center.z, 1)]})
    print(f"{label}: {len(found)} overlaps ({time.time() - t0:.0f} s)", flush=True)
    for f in found:
        print("   ", f, flush=True)
    return found


def gap(a, b, d=5.0):
    """Smallest distance between two shapes, or two lists of shapes, and the point on ``a`` where it occurs.

    Only pairs whose boxes come within d of each other are measured: the
    camera and Pi 5 models have hundreds of solids, and measuring against all
    of one at once takes longer than the rest of the check."""
    a = a if isinstance(a, list) else [a]
    b = b if isinstance(b, list) else [b]
    # The distance between two boxes is a lower bound on the distance between
    # what is in them, so the pairs are measured nearest box first, and the
    # search stops once no box is nearer than the best distance found.
    pairs = sorted((box_gap(sa.BoundBox, sb.BoundBox), i, j) for i, sa in enumerate(a) for j, sb in enumerate(b))
    best = None
    for lower, i, j in pairs:
        if lower > d or (best is not None and lower >= best[0]):
            break
        dist, pts, _ = a[i].distToShape(b[j])
        if best is None or dist < best[0]:
            best = (dist, pts[0][0])
    if best is None:
        return {"mm": None, "note": f"more than {d} mm"}
    q = best[1]
    return {"mm": round(best[0], 3), "at": [round(q.x, 1), round(q.y, 1), round(q.z, 1)]}


def sections(p):
    """Cross-sections for the drawings: normal, offset, label, title."""
    out = [((1, 0, 0), p.lens_x, "x_lens", "section x = lens axis (y horizontal, z up)")]
    if p.board == "pi5":
        out += [((0, 1, 0), 8.5, "y_ribbon", "section y = 8.5 through the camera ribbon (x horizontal, z up)"),
                ((0, 1, 0), 18.4, "y_button", "section y = 18.4 through the power button (x horizontal, z up)")]
    else:
        out += [((0, 1, 0), 11.5, "y_ribbon", "section y = 11.5 through the camera ribbon (x horizontal, z up)"),
                ((0, 1, 0), p.fan_xy[1], "y_fan",
                 f"section y = {p.fan_xy[1]:g} through the fan (x horizontal, z up)"),
                ((0, 1, 0), 53.77, "y_leads",
                 "section y = 53.8 through the fan leads on the GPIO header (x horizontal, z up)")]
    out += [((0, 0, 1), 3.0, "z_ports", "section z = 3 through the ports (x horizontal, y up)"),
            ((0, 1, 0), 60.5, "y_tripod", "section y = 60.5 through the tripod nut (x horizontal, z up)"),
            ((1, 0, 0), p.x_mid, "x_tripod", "section x = tripod axis (y horizontal, z up)")]
    return out


def main():
    t0 = time.time()
    p, parts = E.build(board=BOARD)
    base, lid = parts["base"], parts["lid"]

    pi = load(model_path(BOARD))
    pi_list = pieces(pi) if BOARD == "pi5" else pi4b_pieces(pi)
    pi_pieces = [(f"pi#{i}", s) for i, s in enumerate(pi_list)]
    print(f"models loaded, {len(pi_pieces)} Pi pieces ({time.time() - t0:.0f} s)", flush=True)
    cam = load(model_path("cm3"))
    cam.Placement = p.cam_placement()
    cam = cam.copy()
    cam_pieces = [(f"cam#{i}", s) for i, s in enumerate(cam.Solids)]
    more = extras(p)
    rib, free_slabs, away_slabs, rib_len = ribbon(p)
    rib_free = E.fuse_all(free_slabs)

    report = {"board": BOARD, "params": {"z_top": p.z_top, "z_li": p.z_li, "cam_zb": p.cam_zb}}
    # The Pi goes in USB end first and is then lowered flat, so nothing on the
    # base may overhang the PCB outline.
    column = E.box(0.05, 84.95, 0.05, 55.95, 0.01, p.z_top)
    report["base_over_pcb_mm3"] = round(base.common(column).Volume, 4)
    print("base material above the PCB outline:", report["base_over_pcb_mm3"], "mm3", flush=True)
    for name, part in parts.items():
        report[f"{name}_vs_pi"] = overlaps(part, pi_pieces, f"{name} vs Pi")
    report["base_vs_cam"] = overlaps(base, cam_pieces, "base vs camera")
    report["lid_vs_cam"] = overlaps(lid, cam_pieces, "lid vs camera")
    report["pi_vs_cam"] = overlaps(cam, [(n, s) for n, s in pi_pieces if s.BoundBox.ZMax > 5],
                                   "camera vs tall Pi parts")
    for name, s in more.items():
        report[f"{name}_vs_cam"] = overlaps(s, cam_pieces, f"{name} vs camera")
        report[f"{name}_vs_lid"] = overlaps(lid, [(name, s)], f"lid vs {name}")
        report[f"{name}_vs_base"] = overlaps(base, [(name, s)], f"base vs {name}")
        report[f"ribbon_vs_{name}"] = overlaps(rib_free, [(name, s)], f"ribbon vs {name}")
    if "fan" in more:
        report["fan_vs_pi"] = overlaps(more["fan"], pi_pieces, "fan vs Pi")
        report["fan_vs_leads"] = overlaps(more["fan"], [("leads", more["leads"])], "fan vs leads")
    report["ribbon_vs_lid"] = overlaps(rib_free, [("lid", lid)], "ribbon vs lid")
    # Where the ribbon is inside a connector it overlaps the connector by
    # design; everywhere else it must be clear of the Pi and the camera.
    report["ribbon_vs_pi"] = overlaps(rib_free, pi_pieces, "ribbon outside the connectors vs Pi")
    report["ribbon_vs_cam"] = overlaps(rib_free, cam_pieces, "ribbon outside the connectors vs camera")

    def near(shape, pieces_, d=3.0):
        fbs = [grown(f.BoundBox, d) for f in shape.Faces]
        return Part.makeCompound([s for _, s in pieces_ if any(fb.intersect(s.BoundBox) for fb in fbs)])

    tall_pi = [(n, s) for n, s in pi_pieces if s.BoundBox.ZMax > 1.5 + p.pcb_top - 1.31]
    lo, hi, ymax = PI_CONNECTOR[BOARD]["pieces"]
    not_cam_conn = [(n, s) for n, s in pi_pieces
                    if not (lo < s.BoundBox.XMin and s.BoundBox.XMax < hi and s.BoundBox.YMax < ymax)]
    cam_solids = [s for _, s in cam_pieces]
    gaps = {
        "lid_to_pi_components": gap(lid, near(lid, tall_pi)),
        "base_to_pi": gap(base, near(base, pi_pieces)),
        "camera_to_pi": gap(cam_solids, [s for _, s in tall_pi]),
        "ribbon_to_lid": gap(rib_free, lid),
        "ribbon_to_pi": gap(free_slabs, [s for _, s in not_cam_conn]),
        "ribbon_to_camera": gap(away_slabs, cam_solids),
    }
    if "button" in parts:
        gaps["button_pin_to_plunger"] = gap(parts["button"], near(parts["button"], pi_pieces))
    for name, s in more.items():
        gaps[f"lid_to_{name}"] = gap(lid, s)
        gaps[f"camera_to_{name}"] = gap(cam_solids, s)
        gaps[f"ribbon_to_{name}"] = gap(free_slabs, s)
    if "fan" in more:
        gaps["fan_to_pi"] = gap(more["fan"], [s for _, s in pi_pieces], 12.0)
        gaps["fan_to_leads"] = gap(more["fan"], more["leads"])
    report["gaps"] = gaps
    report["ribbon_path_length_mm"] = round(rib_len, 1)
    print(f"gaps done ({time.time() - t0:.0f} s)", flush=True)
    print(json.dumps({"gaps": gaps, "ribbon_path_length_mm": round(rib_len, 1)}, indent=2))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    meshes = dict(parts)
    meshes.update({"camera": cam, "ribbon": rib, "pi": Part.makeCompound([s for _, s in pi_pieces])})
    meshes.update(more)
    for name, s in meshes.items():
        m = MeshPart.meshFromShape(Shape=s, LinearDeflection=0.08 if name in ("pi", "camera") else 0.03,
                                   AngularDeflection=0.35, Relative=False)
        m.write(os.path.join(OUT, f"{name}.stl"))

    # The Pi and the camera are sliced piece by piece, and only the pieces the
    # plane crosses: slicing the whole Pi compound at once takes longer than
    # the rest of the check.
    to_slice = {name: [s] for name, s in meshes.items()}
    to_slice["pi"] = [s for _, s in pi_pieces]
    to_slice["camera"] = [s for _, s in cam_pieces]
    secs = {}
    for normal, d, label, title in sections(p):
        axis = normal.index(1)
        sec = {}
        for name, shapes in to_slice.items():
            polys = []
            for s in shapes:
                bb = s.BoundBox
                lo, hi = [(bb.XMin, bb.XMax), (bb.YMin, bb.YMax), (bb.ZMin, bb.ZMax)][axis]
                if not lo < d < hi:
                    continue
                try:
                    wires = s.slice(V(*normal), d)
                except Exception:
                    continue
                for w in wires:
                    pts = w.discretize(Deflection=0.02)
                    polys.append([[round(q.x, 3), round(q.y, 3), round(q.z, 3)] for q in pts])
            if polys:
                sec[name] = polys
        secs[label] = {"normal": normal, "offset": d, "title": title, "parts": sec}
        print(f"section {label} ({time.time() - t0:.0f} s)", flush=True)
    with open(os.path.join(OUT, "sections.json"), "w") as f:
        json.dump(secs, f)
    print("wrote", OUT)


main()
