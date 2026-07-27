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
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from .params import MeshInfo
from .uv_unwrap import source_uvs, uv_density

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


def _fbx_unit_scale(path: Path) -> float:
    """Return the FBX file-unit size relative to its numeric coordinates.

    FBX records ``UnitScaleFactor`` as the number of centimeters represented by
    one coordinate unit. Assimp preserves the geometry numbers but OBJ has no
    corresponding metadata, so we bake ``UnitScaleFactor / 100`` into vertices
    to express the exported OBJ in Blender's metre-based coordinate convention.
    """
    data = path.read_bytes()

    # ASCII FBX (the number is the final property on the P record).
    if not data.startswith(b"Kaydara FBX Binary"):
        import re  # noqa: PLC0415

        match = re.search(
            rb'P:\s*"UnitScaleFactor"[^\r\n]*,\s*([-+0-9.eE]+)\s*(?:\r?\n|$)',
            data,
        )
        return float(match.group(1)) if match else 1.0

    version = struct.unpack_from("<I", data, 23)[0]
    wide = version >= 7500
    header = 25
    header_size = 25 if wide else 13

    def read_node(offset: int) -> tuple[int, str, list[object]]:
        if wide:
            end, count, prop_len = struct.unpack_from("<QQQ", data, offset)
            name_len = data[offset + 24]
        else:
            end, count, prop_len = struct.unpack_from("<III", data, offset)
            name_len = data[offset + 12]
        if end == 0:
            return 0, "", []
        cursor = offset + header_size
        name = data[cursor:cursor + name_len].decode("utf-8", "replace")
        cursor += name_len
        props: list[object] = []
        for _ in range(count):
            kind = chr(data[cursor])
            cursor += 1
            formats = {
                "Y": ("<h", 2), "C": ("<?", 1), "I": ("<i", 4),
                "F": ("<f", 4), "D": ("<d", 8), "L": ("<q", 8),
            }
            if kind in formats:
                fmt, size = formats[kind]
                props.append(struct.unpack_from(fmt, data, cursor)[0])
                cursor += size
            elif kind in ("S", "R"):
                size = struct.unpack_from("<I", data, cursor)[0]
                cursor += 4
                raw = data[cursor:cursor + size]
                props.append(raw.decode("utf-8", "replace") if kind == "S" else raw)
                cursor += size
            elif kind.lower() in ("f", "d", "l", "i", "b", "c"):
                length, encoding, compressed = struct.unpack_from("<III", data, cursor)
                cursor += 12 + compressed
                props.append(None)
            else:
                return int(end), name, props
        return int(end), name, props

    stack = [header]
    while stack:
        offset = stack.pop()
        while offset + header_size <= len(data):
            end, name, props = read_node(offset)
            if not end:
                break
            if name == "P" and props and props[0] == "UnitScaleFactor":
                numeric = [value for value in props[1:] if isinstance(value, (int, float))]
                return float(numeric[-1]) if numeric else 1.0

            # Children start immediately after this node's properties.
            if wide:
                count, prop_len = struct.unpack_from("<QQ", data, offset + 8)
                name_len = data[offset + 24]
            else:
                count, prop_len = struct.unpack_from("<II", data, offset + 4)
                name_len = data[offset + 12]
            child = offset + header_size + name_len + int(prop_len)
            if child + header_size < end:
                stack.append(child)
            offset = end
    return 1.0


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
        parts = [_from_assimp_mesh(m) for m in scene.meshes if len(m.faces)]
    if not parts:
        raise MeshLoadError(f"pyassimp found no faces in {path.name}")
    return trimesh.util.concatenate(parts)


def _from_assimp_mesh(mesh) -> trimesh.Trimesh:
    """Wrap one Assimp mesh, keeping the first UV channel if it has one.

    Assimp exposes up to eight UV channels; Blender writes the active UV map
    first, and that is the one the bake targets. The UVs have to survive import
    or there is nothing to bake into -- see :mod:`core.uv_unwrap`.
    """
    visual = None
    channels = getattr(mesh, "texturecoords", None)
    if channels is not None and len(channels):
        uv = np.asarray(channels[0], dtype=np.float64)
        # Assimp always stores 3D texture coordinates; W is 0 for a 2D map.
        if uv.ndim == 2 and uv.shape[0] == len(mesh.vertices) and uv.shape[1] >= 2:
            visual = trimesh.visual.TextureVisuals(uv=uv[:, :2])

    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        visual=visual,
        process=False,
    )


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
    #
    # It deliberately stops short of welding across a UV seam (trimesh defaults
    # to merge_tex=False), because those splits are the atlas layout and we bake
    # into it as authored. Normals stay continuous there anyway --
    # :func:`core.uv_unwrap._welded_vertex_normals` re-welds by position for
    # normals only.
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
    #
    # Watertightness has to be judged on the position-welded topology: every UV
    # seam is a vertex split, so a perfectly sealed textured mesh reads as full
    # of holes if you ask the split vertex set. Winding is safe to fix in place
    # because rotating a face's indices leaves its per-vertex UVs alone.
    watertight = bool(is_watertight(mesh))
    try:
        trimesh.repair.fix_winding(mesh)
        if watertight:
            trimesh.repair.fix_inversion(mesh)
    except Exception as exc:  # pragma: no cover - repair is best-effort
        notes.append(f"Normal repair skipped: {exc}")

    if not watertight:
        notes.append("Mesh is not watertight - AO may leak through open edges")

    return mesh


def is_watertight(mesh: trimesh.Trimesh) -> bool:
    """Watertightness of the geometry, ignoring vertex splits made for UVs."""
    if mesh.is_watertight:
        return True
    if source_uvs(mesh) is None:
        return False

    unique, inverse = trimesh.grouping.unique_rows(mesh.vertices)
    if len(unique) == len(mesh.vertices):
        return False
    welded = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices)[unique],
        faces=inverse[np.asarray(mesh.faces)],
        process=False,
    )
    return bool(welded.is_watertight)


#: Y-up source -> the app's Z-up world. FBX and glTF both store Y-up, and this
#: is the same +90-degrees-about-X that Blender's own importers apply, which is
#: why a Blender export comes back in with Z up exactly as it left.
Y_UP_TO_Z_UP = trimesh.transformations.rotation_matrix(np.pi / 2.0, (1.0, 0.0, 0.0))


def _axis_fix(mesh: trimesh.Trimesh, source_z_up: bool) -> np.ndarray:
    """Bring a source into the app's Z-up world, returning what was applied.

    The caller keeps the transform in :attr:`MeshInfo.to_world` so anything that
    needs to speak the source file's coordinates again can undo it.
    """
    if source_z_up:
        return np.eye(4)
    mesh.apply_transform(Y_UP_TO_Z_UP)
    return Y_UP_TO_Z_UP


def load_mesh(
    path: str | Path, *, source_z_up: bool = False
) -> tuple[trimesh.Trimesh, MeshInfo]:
    """Load a mesh file and return it alongside a populated :class:`MeshInfo`.

    ``source_z_up`` says the file is *already* Z-up and needs no conversion.
    Leave it False for anything out of Blender: the FBX and glTF exporters both
    write Y-up.
    """
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

    if suffix == ".fbx":
        unit_scale = _fbx_unit_scale(path)
        if np.isfinite(unit_scale) and unit_scale > 0.0:
            scale_to_metres = unit_scale / 100.0
            if not np.isclose(scale_to_metres, 1.0):
                mesh.apply_scale(scale_to_metres)
                notes.append(
                    f"Converted FBX units to metres (x{scale_to_metres:g})"
                )

    to_world = _axis_fix(mesh, source_z_up)
    mesh = prepare_mesh(mesh, notes)

    # Touch these once here so the (cached) normals are computed on the main
    # thread rather than lazily from inside a bake worker.
    _ = mesh.vertex_normals

    uvs = source_uvs(mesh)
    if uvs is None:
        notes.append("No UV map in the file - unwrap in Blender to bake into your own UVs")
    else:
        outside = int((uvs < -1e-4).any(axis=1).sum() + (uvs > 1.0 + 1e-4).any(axis=1).sum())
        if outside:
            notes.append(
                f"{outside} UVs fall outside 0..1 - anything off the atlas bakes empty"
            )

    info = MeshInfo(
        path=str(path),
        backend=backend,
        vertices=len(mesh.vertices),
        faces=len(mesh.faces),
        extents=tuple(float(v) for v in mesh.extents),
        scale=float(np.linalg.norm(mesh.extents)),
        watertight=is_watertight(mesh),
        has_uvs=uvs is not None,
        uv_density=uv_density(mesh),
        to_world=to_world,
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
