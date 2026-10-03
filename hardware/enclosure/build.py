"""Build the enclosure and export it. Run with ``freecad.cmd build.py``.

Set KVIHTAI_BOARD=pi4b for the Raspberry Pi 4 Model B; the default is the Pi 5.
Writes to out/<board>/: one STEP per part and enclosure.FCStd with all parts in
their assembled positions, and to out/<board>/print/: one STL per part turned
the way it prints, plus enclosure.3mf with all parts on one Core One plate and
the settings they need. Set KVIHTAI_CAMERA=wide for the wide-angle Camera
Module 3, which needs a taller lid.
"""

import json
import os
import sys
import zipfile
from xml.sax.saxutils import quoteattr

import FreeCAD as App
import Mesh
import MeshPart

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import enclosure  # noqa: E402

BOARD = os.environ.get("KVIHTAI_BOARD", "pi5")
OUT = os.path.join(HERE, "out", BOARD)


# How each part lies on the print bed: base on its back, lid on its front face,
# button pin on its collar. None of them needs support.
PRINT_ROTATION = {
    "base": App.Rotation(),
    "lid": App.Rotation(App.Vector(1, 0, 0), 180),
    "button": App.Rotation(App.Vector(0, 1, 0), 90),
}


# Settings the parts need on top of the "0.20mm STRUCTURAL @COREONE 0.4" print
# profile, stored per object in the 3MF so they apply whatever profile is
# selected. The back and the front face are 2 mm, ten layers; the profile's
# four bottom layers would leave one layer of sparse infill in them. On the lid,
# bridging along y is what crosses the strokes of the engraved name without
# long unsupported runs.
PRINT_SETTINGS = {"perimeters": "3", "bottom_solid_layers": "5", "support_material": "0"}
PRINT_SETTINGS_PART = {"lid": {"bridge_angle": "90"}}

# where each part sits on the 250 x 220 mm Core One bed, by its centre
PLATE = {"base": (74.0, 118.0), "lid": (176.0, 118.0), "button": (125.0, 62.0)}


def print_oriented(name, shape):
    s = shape.copy()
    s.Placement = App.Placement(App.Vector(), PRINT_ROTATION[name])
    s = s.copy()
    bb = s.BoundBox
    s.translate(App.Vector(-bb.Center.x, -bb.Center.y, -bb.ZMin))
    return s


def write_3mf(path, meshes):
    """A PrusaSlicer project: the parts on the plate, with PRINT_SETTINGS per object."""
    objects, items, config = [], [], []
    for oid, (name, mesh) in enumerate(meshes.items(), start=1):
        points, facets = mesh.Topology
        verts = "".join(f'<vertex x="{v.x:.4f}" y="{v.y:.4f}" z="{v.z:.4f}"/>' for v in points)
        tris = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in facets)
        objects.append(f'<object id="{oid}" type="model"><mesh><vertices>{verts}</vertices>'
                       f'<triangles>{tris}</triangles></mesh></object>')
        x, y = PLATE[name]
        items.append(f'<item objectid="{oid}" transform="1 0 0 0 1 0 0 0 1 {x} {y} 0" printable="1"/>')
        settings = {"name": name, **PRINT_SETTINGS, **PRINT_SETTINGS_PART.get(name, {})}
        meta = "".join(f'<metadata type="object" key={quoteattr(k)} value={quoteattr(v)}/>'
                       for k, v in settings.items())
        config.append(f'<object id="{oid}" instances_count="1">{meta}'
                      f'<volume firstid="0" lastid="{len(facets) - 1}">'
                      f'<metadata type="volume" key="name" value={quoteattr(name)}/></volume></object>')
    model = ('<?xml version="1.0" encoding="UTF-8"?>\n'
             '<model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
             'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">'
             '<metadata name="slic3rpe:Version3mf">1</metadata>'
             f'<resources>{"".join(objects)}</resources><build>{"".join(items)}</build></model>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8"?>\n'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" '
                   'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="model" '
                   'ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8"?>\n'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
                   'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        z.writestr("3D/3dmodel.model", model)
        z.writestr("Metadata/Slic3r_PE_model.config",
                   f'<?xml version="1.0" encoding="UTF-8"?>\n<config>{"".join(config)}</config>')


def export(p, parts):
    os.makedirs(os.path.join(OUT, "print"), exist_ok=True)
    doc = App.newDocument("enclosure")
    summary = {}
    print_meshes = {}
    for name, shape in parts.items():
        if not shape.isValid() or len(shape.Solids) != 1:
            raise SystemExit(f"{name}: invalid shape ({len(shape.Solids)} solids)")
        # assembled position, for CAD
        shape.exportStep(os.path.join(OUT, f"{name}.step"))
        doc.addObject("Part::Feature", name).Shape = shape
        # print position, for the slicer
        mesh = MeshPart.meshFromShape(Shape=print_oriented(name, shape), LinearDeflection=0.01,
                                      AngularDeflection=0.2, Relative=False)
        mesh.write(os.path.join(OUT, "print", f"{name}.stl"))
        print_meshes[name] = mesh
        bb = shape.BoundBox
        summary[name] = {"size_mm": [round(bb.XLength, 2), round(bb.YLength, 2), round(bb.ZLength, 2)],
                         "volume_cm3": round(shape.Volume / 1000, 2),
                         "mesh_facets": mesh.CountFacets}
    write_3mf(os.path.join(OUT, "print", "enclosure.3mf"), print_meshes)
    fcstd = os.path.join(OUT, "enclosure.FCStd")
    if os.path.exists(fcstd):
        os.remove(fcstd)            # saveAs would leave an .FCBak behind
    doc.saveAs(fcstd)
    summary["params"] = {k: getattr(p, k) for k in
                         ("board", "camera", "z_bot", "z_floor", "z_li", "z_top", "z_wall", "cam_zb", "cam_x0",
                          "lens_x", "lens_y", "x_out", "y_out")}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


p, parts = enclosure.build(board=BOARD, camera=os.environ.get("KVIHTAI_CAMERA", "standard"))
export(p, parts)
