"""Generate the test assets in ``assets/``.

Two shapes, chosen to make the bakes falsifiable:

``beveled_cube``  A p-norm rounded cube. Every edge is a smooth fillet, so
                  curvature should read uniformly positive along the edges and
                  zero on the faces, and AO should show soft darkening in the
                  fillets. Nothing should read concave.
``chamfered_cube`` A cube with broad, single-plane edge chamfers and flat
                   triangular corner cuts. There are no rounded bevel segments.
``sample``         An L-bracket. Sharp convex outer edges, one reflex interior
                   corner. Curvature must go clearly negative in that corner and
                   AO must be measurably darker there than on an open face.

Run with:  python tools/make_sample.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

sys.path.insert(0, str(ROOT))

from core.bevel import bevel  # noqa: E402
from core.params import BevelParams, UnwrapParams  # noqa: E402
from core.uv_unwrap import unwrap  # noqa: E402


def rounded_cube(power: float = 6.0, subdivisions: int = 5) -> trimesh.Trimesh:
    """Project a sphere onto the unit p-norm ball: a cube with rounded edges.

    Higher ``power`` tightens the fillet radius. Manifold and watertight by
    construction, since it is just a radial reparametrisation of an icosphere.
    """
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    directions = np.asarray(sphere.vertices, dtype=np.float64)
    scale = np.power(np.sum(np.abs(directions) ** power, axis=1), 1.0 / power)
    mesh = trimesh.Trimesh(
        vertices=directions / scale[:, np.newaxis],
        faces=sphere.faces,
        process=True,
    )
    trimesh.repair.fix_normals(mesh)
    return mesh


def chamfered_cube(size: float = 2.0, chamfer: float = 0.25) -> trimesh.Trimesh:
    """A cube with a significant, entirely planar chamfer.

    One segment is deliberate: each original edge is replaced by one broad
    plane rather than a rounded bevel profile.  The eight corners become flat
    triangles, leaving 26 distinct planes in total.
    """
    cube = trimesh.creation.box(extents=(size, size, size))
    mesh = bevel(
        cube,
        BevelParams(enabled=True, amount=chamfer, segments=1),
    )
    trimesh.repair.fix_normals(mesh)

    # Unlike the diagnostic assets, this is also the application's startup
    # mesh, so ship it ready to bake into its own UVs.
    atlas = unwrap(
        mesh,
        UnwrapParams(use_source_uvs=False, padding=8),
        resolution=1024,
    )
    return trimesh.Trimesh(
        vertices=atlas.vertices,
        faces=atlas.faces,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=atlas.uvs),
    )


def l_bracket(
    width: float = 1.0,
    height: float = 1.0,
    thickness: float = 0.35,
    depth: float = 0.6,
    subdivisions: int = 4,
) -> trimesh.Trimesh:
    """Extruded L profile. The reflex corner is the concave test feature."""
    # Counter-clockwise, and star-shaped about vertex 0, so the cap
    # triangulates as a simple fan.
    profile = np.array(
        [
            [0.0, 0.0],
            [width, 0.0],
            [width, thickness],
            [thickness, thickness],
            [thickness, height],
            [0.0, height],
        ],
        dtype=np.float64,
    )
    count = len(profile)
    half = depth * 0.5

    back = np.hstack([profile, np.full((count, 1), -half)])
    front = np.hstack([profile, np.full((count, 1), half)])
    vertices = np.vstack([back, front])

    faces: list[tuple[int, int, int]] = []
    for i in range(1, count - 1):
        faces.append((count + 0, count + i, count + i + 1))  # front cap, +Z
        faces.append((0, i + 1, i))                          # back cap, -Z

    for i in range(count):
        j = (i + 1) % count
        b_i, b_j = i, j
        f_i, f_j = count + i, count + j
        faces.append((b_i, b_j, f_j))
        faces.append((b_i, f_j, f_i))

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
    mesh.merge_vertices()

    # Midpoint subdivision keeps the hard edges but gives the per-vertex
    # curvature estimator enough samples to draw a crisp line along them.
    for _ in range(subdivisions):
        mesh = mesh.subdivide()

    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh


def _find_assimp() -> str | None:
    for candidate in ("assimp", "/opt/homebrew/bin/assimp", "/usr/local/bin/assimp"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def write(mesh: trimesh.Trimesh, stem: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    obj_path = ASSETS / f"{stem}.obj"
    mesh.export(obj_path)
    print(f"wrote {obj_path.relative_to(ROOT)}  "
          f"({len(mesh.vertices):,} verts, {len(mesh.faces):,} tris)")

    exe = _find_assimp()
    if exe is None:
        print("  (assimp CLI not found - skipping FBX; the .obj works fine as a test asset)")
        return

    fbx_path = ASSETS / f"{stem}.fbx"
    result = subprocess.run(
        [exe, "export", str(obj_path), str(fbx_path)], capture_output=True, text=True
    )
    if result.returncode == 0 and fbx_path.exists():
        print(f"wrote {fbx_path.relative_to(ROOT)}")
    else:
        print(f"  FBX export failed: {(result.stderr or result.stdout).strip()[:200]}")


def main() -> int:
    write(rounded_cube(), "beveled_cube")
    write(chamfered_cube(), "chamfered_cube")
    write(l_bracket(), "sample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
