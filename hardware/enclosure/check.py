"""Check the enclosure against the vendor models. Run with ``freecad.cmd check.py``.

Needs Raspberry Pi's STEP models of the Pi 5 and the Camera Module 3. They are
not redistributed here; the script downloads them into ref/ on first run.

Reports every overlap between a printed part and the Pi, the camera, the
Active Cooler envelope and the camera ribbon, and the smallest gaps at the
places where the design is tight. Writes meshes and cross-sections to
out/check/ for render.py.
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
REF = os.environ.get("KVIHTAI_REF", os.path.join(HERE, "ref"))
OUT = os.path.join(HERE, "out", "check")

MODELS = {
    "pi5": ("https://datasheets.raspberrypi.com/rpi5/RaspberryPi5-step.zip", "rpi-5b_no_graphics.step"),
    "cm3": ("https://datasheets.raspberrypi.com/camera/camera-module-3-step.zip",
            "Camera_module_3_std_model_simple.stp"),
}


def model_path(key):
    url, name = MODELS[key]
    for root, _dirs, files in os.walk(REF):
        if name in files and "__MACOSX" not in root:
            return os.path.join(root, name)
    os.makedirs(REF, exist_ok=True)
    print("downloading", url)
    data = urllib.request.urlopen(url).read()
    zipfile.ZipFile(io.BytesIO(data)).extractall(os.path.join(REF, key))
    return model_path(key)


def load(path):
    s = Part.Shape()
    s.read(path)
    return s


def pieces(shape):
    """Solids, plus the shells that the Pi model uses for its micro-HDMI sockets."""
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


def cooler_envelope():
    """Active Cooler as a conservative envelope, from the mechanical drawing."""
    z0 = E.PCB_TOP
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


def ribbon(p, cable_len=38.0):
    """The camera FPC as a chain of thin slabs, from the camera connector to CAM/DISP 0.

    Path in the xz plane, width along y. It leaves the camera connector towards
    +x with its contacts facing the camera PCB, which after the bend down puts
    them facing the Ethernet jack, as the Pi 5 connector requires.
    """
    zr = p.cam_zb - 1.3
    x_cam = p.cam_x0 + 23.49 - 3.5          # end of the ribbon inside the camera connector
    x_pi, z_pi = 55.0, 5.29 - 3.0           # end inside the Pi connector
    # a loop up towards the lid takes up the slack of a 38 mm cable
    x_e = p.cam_x0 + 23.86
    ctrl = [V(x_cam, 0, zr), V(x_e, 0, zr), V(x_e + 1.6, 0, 16.4), V(x_e + 2.0, 0, 19.4), V(x_e + 3.6, 0, 20.9),
            V(56.6, 0, 20.9), V(58.3, 0, 19.0), V(58.4, 0, 15.0), V(57.8, 0, 11.5), V(56.6, 0, 9.4),
            V(55.3, 0, 7.4), V(55.0, 0, 5.2), V(x_pi, 0, z_pi)]
    curve = Part.BSplineCurve()
    curve.interpolate(ctrl)
    length = curve.length()
    n = 60
    slabs = []
    for i in range(n):
        u0 = curve.FirstParameter + (curve.LastParameter - curve.FirstParameter) * i / n
        u1 = curve.FirstParameter + (curve.LastParameter - curve.FirstParameter) * (i + 1) / n
        a, b = curve.value(u0), curve.value(u1)
        t = (i + 0.5) / n
        # 15-way end is 16 mm wide and centred on the camera, 22-way end 11.5 mm on the Pi connector
        w = 16.0 + (11.5 - 16.0) * max(0.0, (t - 0.55) / 0.45)
        yc = p.lens_y + (8.5 - p.lens_y) * t
        d = b - a
        seg = Part.makeBox(d.Length, w, 0.25, V(0, -w / 2, -0.125))
        ang = math.degrees(math.atan2(d.z, d.x))
        seg.rotate(V(0, 0, 0), V(0, 1, 0), -ang)
        seg.translate(V(a.x, yc, a.z))
        slabs.append(seg)
    return E.fuse_all(slabs), length


def grown(bb, d=0.05):
    b = App.BoundBox(bb)
    b.enlarge(d)
    return b


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


def gap(a, b):
    """Smallest distance between two shapes, and the point on ``a`` where it occurs."""
    d, pairs, _ = a.distToShape(b)
    q = pairs[0][0]
    return {"mm": round(d, 3), "at": [round(q.x, 1), round(q.y, 1), round(q.z, 1)]}


def main():
    t0 = time.time()
    p, parts = E.build()
    base, lid, button = parts["base"], parts["lid"], parts["button"]

    pi = load(model_path("pi5"))
    pi_pieces = [(f"pi#{i}", s) for i, s in enumerate(pieces(pi))]
    print(f"models loaded, {len(pi_pieces)} Pi pieces ({time.time() - t0:.0f} s)", flush=True)
    cam = load(model_path("cm3"))
    cam.Placement = p.cam_placement()
    cam = cam.copy()
    cam_pieces = [(f"cam#{i}", s) for i, s in enumerate(cam.Solids)]
    cooler = cooler_envelope()
    rib, rib_len = ribbon(p)

    report = {"params": {"z_top": p.z_top, "z_li": p.z_li, "cam_zb": p.cam_zb}}
    # The Pi goes in USB end first and is then lowered flat, so nothing on the
    # base may overhang the PCB outline.
    column = E.box(0.05, 84.95, 0.05, 55.95, 0.01, p.z_top)
    report["base_over_pcb_mm3"] = round(base.common(column).Volume, 4)
    print("base material above the PCB outline:", report["base_over_pcb_mm3"], "mm3", flush=True)
    report["base_vs_pi"] = overlaps(base, pi_pieces, "base vs Pi 5")
    report["lid_vs_pi"] = overlaps(lid, pi_pieces, "lid vs Pi 5")
    report["button_vs_pi"] = overlaps(button, pi_pieces, "button pin vs Pi 5")
    report["base_vs_cam"] = overlaps(base, cam_pieces, "base vs camera")
    report["lid_vs_cam"] = overlaps(lid, cam_pieces, "lid vs camera")
    report["cooler_vs_cam"] = overlaps(cooler, cam_pieces, "cooler vs camera")
    report["pi_vs_cam"] = overlaps(cam, [(n, s) for n, s in pi_pieces if s.BoundBox.ZMax > 5],
                                   "camera vs tall Pi parts")
    report["cooler_vs_lid"] = overlaps(lid, [("cooler", cooler)], "lid vs cooler")
    report["cooler_vs_base"] = overlaps(base, [("cooler", cooler)], "base vs cooler")
    report["ribbon_vs_lid"] = overlaps(rib, [("lid", lid)], "ribbon vs lid")
    report["ribbon_vs_cooler"] = overlaps(rib, [("cooler", cooler)], "ribbon vs cooler")
    # the ribbon ends inside the two connectors it plugs into; anything else is a collision
    far = [(n, s) for n, s in pi_pieces if s.BoundBox.ZMax > 1.4
           and not (53.0 < s.BoundBox.XMin and s.BoundBox.XMax < 56.5 and s.BoundBox.YMax < 17.0)]
    report["ribbon_vs_pi"] = overlaps(rib, far, "ribbon vs Pi 5 (except CAM/DISP 0)")
    cam_body = [(n, s) for n, s in cam_pieces if not (s.BoundBox.ZMax < p.cam_zb - 0.02
                                                       and s.BoundBox.XMin > p.cam_x0 + 16.0)]
    report["ribbon_vs_cam"] = overlaps(rib, [(n, s) for n, s in cam_body if s.BoundBox.ZMin > p.cam_zb - 0.05],
                                       "ribbon vs camera PCB and lens")

    def near(shape, pieces_, d=3.0):
        fbs = [grown(f.BoundBox, d) for f in shape.Faces]
        return Part.makeCompound([s for _, s in pieces_ if any(fb.intersect(s.BoundBox) for fb in fbs)])

    tall_pi = [(n, s) for n, s in pi_pieces if s.BoundBox.ZMax > 1.5]
    gaps = {
        "lid_to_cooler": gap(lid, cooler),
        "lid_to_pi_components": gap(lid, near(lid, tall_pi)),
        "camera_to_cooler": gap(cam, cooler),
        "base_to_pi": gap(base, near(base, pi_pieces)),
        "button_pin_to_plunger": gap(button, near(button, pi_pieces)),
        "ribbon_to_lid": gap(rib, lid),
        "ribbon_to_cooler": gap(rib, cooler),
    }
    report["gaps"] = gaps
    report["ribbon_path_length_mm"] = round(rib_len, 1)
    print(f"gaps done ({time.time() - t0:.0f} s)", flush=True)
    print(json.dumps({"gaps": gaps, "ribbon_path_length_mm": round(rib_len, 1)}, indent=2))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    meshes = {"base": base, "lid": lid, "button": button, "camera": cam, "cooler": cooler, "ribbon": rib,
              "pi": Part.makeCompound([s for _, s in pi_pieces])}
    for name, s in meshes.items():
        m = MeshPart.meshFromShape(Shape=s, LinearDeflection=0.08 if name in ("pi", "camera") else 0.03,
                                   AngularDeflection=0.35, Relative=False)
        m.write(os.path.join(OUT, f"{name}.stl"))

    # cross-sections for the drawings: (normal, offset, label). The Pi and the
    # camera are sliced piece by piece, and only the pieces the plane crosses:
    # slicing the whole Pi compound at once takes longer than the rest of the check.
    to_slice = {name: [s] for name, s in meshes.items()}
    to_slice["pi"] = [s for _, s in pi_pieces]
    to_slice["camera"] = [s for _, s in cam_pieces]
    cuts = [((1, 0, 0), p.lens_x, "x_lens"), ((0, 1, 0), 8.5, "y_ribbon"), ((0, 1, 0), 18.4, "y_button"),
            ((0, 0, 1), 3.0, "z_ports"), ((0, 1, 0), 60.5, "y_tripod"), ((1, 0, 0), p.x_mid, "x_tripod")]
    sections = {}
    for normal, d, label in cuts:
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
        sections[label] = {"normal": normal, "offset": d, "parts": sec}
        print(f"section {label} ({time.time() - t0:.0f} s)", flush=True)
    with open(os.path.join(OUT, "sections.json"), "w") as f:
        json.dump(sections, f)
    print("wrote", OUT)


main()
