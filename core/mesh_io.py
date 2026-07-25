"""FBX (and friends) import, validation and clean-up.

Assimp does the actual FBX parsing, as specified. There are two ways to reach
it from Python and this module tries both:

``assimp`` CLI   The system Assimp binary converts the file to glTF, which
                 trimesh then loads natively. This is the default path -- it
                 uses exactly the same Assimp that ``pyassimp`` would wrap, but
                 without the ctypes layer.
``pyassimp``     Used first when it is importable. Note that the last release
                 (5.2.5) imports ``distutils`` and therefore does not work on
                 Python 3.12+, which is why the CLI path exists.

Either way the system Assimp library needs to be installed -- ``brew install
assimp`` / ``apt install assimp-utils``. See the README.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from .params import MeshInfo

#: Formats trimesh parses on its own, no Assimp round-trip needed.
NATIVE_SUFFIXES = {
    ".obj", ".stl", ".ply", ".glb", ".gltf", ".off", ".dae",
    ".3mf", ".xyz", ".msh", ".stp",
}

#: Everything we advertise support for in the file dialog / drag-and-drop.
SUPPORTED_SUFFIXES = NATIVE_SUFFIXES | {".fbx", ".3ds", ".blend", ".x", ".ase", ".lwo", ".ms3d"}

_ASSIMP_SEARCH_PATHS = (
    "assimp",
    "/opt/homebrew/bin/assimp",
    "/usr/local/bin/assimp",
    "/usr/bin/assimp",
)


class MeshLoadError(RuntimeError):
    """Raised when no import backend could produce usable geometry."""


def find_assimp_cli() -> str | None:
    """Locate the Assimp command line tool, or return None."""
    for candidate in _ASSIMP_SEARCH_PATHS:
        found = shutil.which(candidate) if "/" not in candidate else (
            candidate if Path(candidate).exists() else None
        )
        if found:
            return found
    return None


def _load_via_pyassimp(path: Path) -> trimesh.Trimesh:
    import pyassimp  # noqa: PLC0415 -- optional backend, imported on demand

    flags = (
        pyassimp.postprocess.aiProcess_Triangulate
        | pyassimp.postprocess.aiProcess_JoinIdenticalVertices
        | pyassimp.postprocess.aiProcess_GenSmoothNormals
        | pyassimp.postprocess.aiProcess_PreTransformVertices
    )
    with pyassimp.load(str(path), processing=flags) as scene:
        if not scene.meshes:
            raise MeshLoadError(f"pyassimp found no meshes in {path.name}")
        parts = [
            trimesh.Trimesh(
                vertices=np.asarray(m.vertices, dtype=np.float64),
                faces=np.asarray(m.faces, dtype=np.int64),
                process=False,
            )
            for m in scene.meshes
            if len(m.faces)
        ]
    if not parts:
        raise MeshLoadError(f"pyassimp found no faces in {path.name}")
    return trimesh.util.concatenate(parts)


def _load_via_assimp_cli(path: Path) -> trimesh.Trimesh:
    exe = find_assimp_cli()
    if exe is None:
        raise MeshLoadError(
            f"Cannot read {path.suffix} files: the Assimp command line tool was not "
            "found on PATH. Install it with 'brew install assimp' (macOS) or "
            "'apt install assimp-utils' (Linux)."
        )

    with tempfile.TemporaryDirectory(prefix="meshmap-import-") as tmp:
        converted = Path(tmp) / "converted.glb"
        proc = subprocess.run(
            [exe, "export", str(path), str(converted)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0 or not converted.exists():
            detail = (proc.stderr or proc.stdout or "").strip()[-800:]
            raise MeshLoadError(f"Assimp failed to convert {path.name}:\n{detail}")
        loaded = trimesh.load(converted, force="mesh", process=False)

    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise MeshLoadError(f"Assimp produced no usable geometry from {path.name}")
    return loaded


def _load_native(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise MeshLoadError(f"No usable geometry in {path.name}")
    return loaded


def prepare_mesh(mesh: trimesh.Trimesh, notes: list[str]) -> trimesh.Trimesh:
    """Triangulate, weld, drop junk faces and make normals consistent.

    Materials, rigs and animation are already gone by this point -- we only
    ever build a Trimesh from vertices and faces.
    """
    before_v, before_f = len(mesh.vertices), len(mesh.faces)

    # trimesh only ever represents triangles, so n-gons were already split on
    # import; what is left to do is weld the duplicate vertices that FBX
    # exporters emit per-face.
    #
    # Welding also averages the vertex normals, which matters a great deal
    # here: the curvature bake differentiates the *interpolated* normal, so a
    # flat-shaded mesh whose normals are constant per face would differentiate
    # to zero and bake pure black. Welding is what gives it something to see.
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    if len(mesh.faces) == 0:
        raise MeshLoadError("Mesh has no faces left after clean-up")

    if before_v != len(mesh.vertices) or before_f != len(mesh.faces):
        notes.append(
            f"Cleaned: {before_v}->{len(mesh.vertices)} verts, "
            f"{before_f}->{len(mesh.faces)} faces"
        )

    # Consistent winding matters: AO fires rays along the vertex normal, and a
    # flipped island would bake occlusion from inside the mesh.
    try:
        trimesh.repair.fix_winding(mesh)
        if mesh.is_watertight:
            trimesh.repair.fix_inversion(mesh)
    except Exception as exc:  # pragma: no cover - repair is best-effort
        notes.append(f"Normal repair skipped: {exc}")

    if not mesh.is_watertight:
        notes.append("Mesh is not watertight - AO may leak through open edges")

    return mesh


def _axis_fix(mesh: trimesh.Trimesh, z_up: bool) -> None:
    """Rotate a Z-up source into the viewport's Y-up convention."""
    if not z_up:
        return
    rot = trimesh.transformations.rotation_matrix(-np.pi / 2.0, (1.0, 0.0, 0.0))
    mesh.apply_transform(rot)


def load_mesh(path: str | Path, *, z_up: bool = False) -> tuple[trimesh.Trimesh, MeshInfo]:
    """Load a mesh file and return it alongside a populated :class:`MeshInfo`."""
    path = Path(path).expanduser()
    if not path.exists():
        raise MeshLoadError(f"File not found: {path}")

    suffix = path.suffix.lower()
    notes: list[str] = []

    if suffix in NATIVE_SUFFIXES:
        mesh = _load_native(path)
        backend = "trimesh"
    else:
        try:
            mesh = _load_via_pyassimp(path)
            backend = "pyassimp"
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            notes.append(f"pyassimp unavailable ({type(exc).__name__}), using assimp CLI")
            mesh = _load_via_assimp_cli(path)
            backend = "assimp-cli"

    _axis_fix(mesh, z_up)
    mesh = prepare_mesh(mesh, notes)

    # Touch these once here so the (cached) normals are computed on the main
    # thread rather than lazily from inside a bake worker.
    _ = mesh.vertex_normals

    info = MeshInfo(
        path=str(path),
        backend=backend,
        vertices=len(mesh.vertices),
        faces=len(mesh.faces),
        extents=tuple(float(v) for v in mesh.extents),
        scale=float(np.linalg.norm(mesh.extents)),
        watertight=bool(mesh.is_watertight),
        notes=notes,
    )

    if info.scale <= 0.0:
        raise MeshLoadError("Mesh has zero size")

    # Unit sanity check -- the spec calls this out as a real-file risk.
    if info.scale > 500.0:
        notes.append(f"Large mesh (diagonal {info.scale:.0f}) - likely centimetre units")
    elif info.scale < 0.05:
        notes.append(f"Tiny mesh (diagonal {info.scale:.4f}) - check source units")

    return mesh, info
