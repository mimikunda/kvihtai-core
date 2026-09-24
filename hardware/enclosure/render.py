"""Render the check meshes to PNG, for looking at the design without a GUI.

Needs numpy and opencv (the core requirements). Run after check.py:

    python render.py

Writes out/render/*.png: shaded views of the assembly, an exploded view, and
the cross-sections that check.py exported.
"""

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "out", "check")
OUT = os.path.join(HERE, "out", "render")

COLORS = {  # BGR
    "base": (150, 148, 145), "lid": (185, 183, 180), "button": (60, 60, 210),
    "pi": (70, 140, 40), "camera": (60, 100, 30), "cooler": (190, 190, 195), "ribbon": (40, 150, 235),
}


def read_stl(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        v = [list(map(float, ln.split()[1:4])) for ln in data.decode().splitlines() if ln.strip().startswith("vertex")]
        return np.array(v, np.float64).reshape(-1, 3, 3)
    n = int(np.frombuffer(data, np.uint32, 1, 80)[0])
    rec = np.frombuffer(data, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n, 84)
    return rec["v"].astype(np.float64)


def view_matrix(azim, elev):
    """Rows are screen right, screen up and the direction towards the viewer."""
    a, e = np.radians(azim), np.radians(elev)
    toward = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    up0 = np.array([0.0, 0.0, 1.0]) if abs(elev) < 89 else np.array([0.0, 1.0, 0.0])
    right = np.cross(up0, toward)
    right /= np.linalg.norm(right)
    up = np.cross(toward, right)
    return np.stack([right, up, toward])


def standing_view(yaw=0.0, pitch=0.0):
    """As it stands on a tripod: GPIO side down, lens towards the viewer.

    yaw turns the enclosure about the vertical, pitch tips it towards the viewer."""
    r0 = np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])
    a, b = np.radians(yaw), np.radians(pitch)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return rx @ ry @ r0


def render(scene, rot, size=(1400, 1000), ss=2, bg=(255, 255, 255), margin=70):
    """Z-buffered flat-shaded render. scene is a list of (triangles, BGR colour)."""
    W, H = size[0] * ss, size[1] * ss
    allv = np.concatenate([t.reshape(-1, 3) for t, _ in scene]) @ rot.T
    lo, hi = allv.min(0), allv.max(0)
    scale = min((W - 2 * margin * ss) / (hi[0] - lo[0]), (H - 2 * margin * ss) / (hi[1] - lo[1]))
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    img = np.full((H, W, 3), bg, np.float32)
    zbuf = np.full((H, W), -np.inf)
    ibuf = np.full((H, W), -1, np.int32)
    light = np.array([-0.35, 0.45, 0.82])
    light /= np.linalg.norm(light)
    tri_id = 0
    for tris, color in scene:
        p = tris @ rot.T
        sx = (p[..., 0] - cx) * scale + W / 2
        sy = H / 2 - (p[..., 1] - cy) * scale
        sz = p[..., 2]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        nl = np.linalg.norm(n, axis=1)
        ok = nl > 1e-12
        n[ok] /= nl[ok, None]
        n[n[:, 2] < 0] *= -1                     # two-sided
        shade = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
        col = np.array(color, np.float32)
        for i in np.nonzero(ok)[0]:
            xs, ys, zs = sx[i], sy[i], sz[i]
            x0, x1 = int(max(np.floor(xs.min()), 0)), int(min(np.ceil(xs.max()), W - 1))
            y0, y1 = int(max(np.floor(ys.min()), 0)), int(min(np.ceil(ys.max()), H - 1))
            if x1 < x0 or y1 < y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
            if abs(d) < 1e-9:
                continue
            l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d
            l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d
            l2 = 1 - l0 - l1
            inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
            if not inside.any():
                continue
            z = l0 * zs[0] + l1 * zs[1] + l2 * zs[2]
            zb = zbuf[y0:y1 + 1, x0:x1 + 1]
            m = inside & (z > zb)
            zb[m] = z[m]
            img[y0:y1 + 1, x0:x1 + 1][m] = col * shade[i]
            ibuf[y0:y1 + 1, x0:x1 + 1][m] = tri_id + i
        tri_id += len(tris)
    # outlines where depth jumps or the triangle's facet changes by a lot
    zf = np.where(np.isfinite(zbuf), zbuf, lo[2] - 10)
    dz = np.abs(cv2.Sobel(zf, cv2.CV_64F, 1, 0)) + np.abs(cv2.Sobel(zf, cv2.CV_64F, 0, 1))
    shade_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dg = np.abs(cv2.Sobel(shade_g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(shade_g, cv2.CV_32F, 0, 1))
    edge = (dz > 1.2 / scale * 8) | (dg > 60)
    img[edge] *= 0.35
    out = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return np.clip(out, 0, 255).astype(np.uint8)


def label(img, text, pos=(20, 40), scale=0.9):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), 2, cv2.LINE_AA)
    return img


def load_scene(names, offsets=None):
    offsets = offsets or {}
    scene = []
    for n in names:
        t = read_stl(os.path.join(CHECK, f"{n}.stl"))
        if n in offsets:
            t = t + np.array(offsets[n])
        scene.append((t, COLORS[n]))
    return scene


def draw_section(sec, axes, title, size=(1400, 700), px_per_mm=None, highlight=None):
    """Filled cross-section; axes picks the two coordinates to plot, e.g. (0, 2) for x and z."""
    order = ["pi", "cooler", "camera", "ribbon", "button", "base", "lid"]
    pts_all = [np.array(poly)[:, axes] for n in order if n in sec["parts"] for poly in sec["parts"][n]]
    if not pts_all:
        return None
    allp = np.concatenate(pts_all)
    lo, hi = allp.min(0), allp.max(0)
    W, H = size
    s = px_per_mm or min((W - 80) / (hi[0] - lo[0]), (H - 110) / (hi[1] - lo[1]))
    img = np.full((H, W, 3), 255, np.uint8)

    def tr(a):
        return np.stack([(a[:, 0] - lo[0]) * s + 40, H - 40 - (a[:, 1] - lo[1]) * s], 1).astype(np.int32)

    # 1 mm grid
    for gx in np.arange(np.floor(lo[0]), hi[0] + 1, 1.0):
        x = int((gx - lo[0]) * s + 40)
        cv2.line(img, (x, 60), (x, H - 30), (238, 238, 238) if gx % 10 else (215, 215, 215), 1)
    for gy in np.arange(np.floor(lo[1]), hi[1] + 1, 1.0):
        y = int(H - 40 - (gy - lo[1]) * s)
        cv2.line(img, (30, y), (W - 30, y), (238, 238, 238) if gy % 10 else (215, 215, 215), 1)
    for n in order:
        if n not in sec["parts"]:
            continue
        polys = [tr(np.array(poly)[:, axes]) for poly in sec["parts"][n]]
        # even-odd fill so holes stay open
        mask = np.zeros((H, W), np.uint8)
        for poly in polys:
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [poly], 1)
            mask ^= m
        c = np.array(COLORS[n], np.float32)
        img[mask > 0] = (0.55 * c + 0.45 * 255).astype(np.uint8)
        cv2.polylines(img, polys, True, tuple(int(v * 0.6) for v in COLORS[n]), 1, cv2.LINE_AA)
    label(img, title, (20, 35), 0.8)
    cv2.putText(img, "grid 1 mm", (W - 160, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    everything = ["base", "lid", "button", "pi", "cooler", "camera", "ribbon"]
    views = {
        "standing": (everything, {}, standing_view(), "standing on its GPIO side, as on a tripod"),
        "standing_3q": (everything, {}, standing_view(-35, 20), "standing, three-quarter view"),
        "front": (everything, {}, view_matrix(-60, 35), "assembled, front and port side"),
        "back": (everything, {}, view_matrix(120, -40), "assembled, back and USB/Ethernet end"),
        "exploded": (everything, {"lid": (0, 0, 45), "camera": (0, 0, 45), "ribbon": (0, 0, 45),
                                  "button": (-14, 0, 0)}, view_matrix(-55, 30), "exploded"),
        "inside": (["base", "button", "pi", "cooler"], {}, view_matrix(-70, 60), "base with the Pi, lid off"),
        "lid_inside": (["lid", "camera", "ribbon"], {}, view_matrix(-60, -50), "lid from inside, with camera"),
        "left_end": (everything, {}, view_matrix(180, 5), "left end: LED, power button, microSD, vents"),
        "port_side": (everything, {}, view_matrix(-90, 5), "port side: USB-C, 2x micro-HDMI, vents"),
        "usb_end": (everything, {}, view_matrix(0, 5), "right end: USB and Ethernet"),
        "gpio_side": (everything, {}, view_matrix(90, 5), "GPIO side: tripod nut"),
    }
    only = set(sys.argv[1:])
    for name, (parts, off, rot, title) in views.items():
        if only and name not in only:
            continue
        img = render(load_scene(parts, off), rot)
        cv2.imwrite(os.path.join(OUT, f"{name}.png"), label(img, title))
        print("wrote", name, flush=True)

    with open(os.path.join(CHECK, "sections.json")) as f:
        sections = json.load(f)
    axes = {"x_lens": (1, 2), "x_tripod": (1, 2), "y_ribbon": (0, 2), "y_button": (0, 2),
            "y_tripod": (0, 2), "z_ports": (0, 1)}
    titles = {"x_lens": "section x = lens axis (y horizontal, z up)",
              "y_ribbon": "section y = 8.5 through the camera ribbon (x horizontal, z up)",
              "y_button": "section y = 18.4 through the power button (x horizontal, z up)",
              "y_tripod": "section y = 60 through the tripod nut (x horizontal, z up)",
              "z_ports": "section z = 3 through the ports (x horizontal, y up)",
              "x_tripod": "section x = tripod axis (y horizontal, z up)"}
    for key, sec in sections.items():
        if only and key not in only:
            continue
        img = draw_section(sec, axes[key], titles[key])
        if img is not None:
            cv2.imwrite(os.path.join(OUT, f"section_{key}.png"), img)
            print("wrote section", key, flush=True)


if __name__ == "__main__":
    main()
