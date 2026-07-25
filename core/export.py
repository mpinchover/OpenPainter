"""Writers for output maps and Blender-friendly textured OBJ packages.

Images are flipped vertically on write. Our textures store row 0 at v=0, while
PNG stores row 0 at the top, so the flip is what makes the exported file line up
with the same UVs in Blender, Substance or any other DCC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .uv_unwrap import UnwrapResult

#: The product: the EdgeWear001 mask.
EDGE_WEAR_NAME = "edge_wear"
#: The Curvature Texture bake it is derived from.
CURVATURE_NAME = "curvature"

MAP_NAMES = (EDGE_WEAR_NAME, CURVATURE_NAME)


def _to_image(array: np.ndarray, bits: int) -> Image.Image:
    data = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    data = np.flipud(data)

    # Pillow infers I;16 from uint16 and L from uint8; passing mode= explicitly
    # is deprecated as of Pillow 12.
    if bits == 16:
        return Image.fromarray((data * 65535.0 + 0.5).astype(np.uint16))
    return Image.fromarray((data * 255.0 + 0.5).astype(np.uint8))


def save_map(path: str | Path, array: np.ndarray, *, bits: int = 8) -> Path:
    """Write a single-channel 0..1 float image as an 8- or 16-bit grayscale PNG."""
    if bits not in (8, 16):
        raise ValueError(f"bits must be 8 or 16, got {bits}")

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    _to_image(array, bits).save(path, format="PNG", optimize=(bits == 8))
    return path


def export_maps(
    output_dir: str | Path,
    edge_wear: np.ndarray,
    curvature: np.ndarray,
    *,
    bits: int = 8,
    prefix: str = "",
) -> list[Path]:
    """Write edge_wear.png and the curvature bake behind it.

    16-bit is worth reaching for if banding shows up in the falloff -- the
    curvature bake is a smooth gradient that 256 levels can struggle with.
    """
    output_dir = Path(output_dir).expanduser()
    arrays = (edge_wear, curvature)
    return [
        save_map(output_dir / f"{prefix}{name}.png", array, bits=bits)
        for name, array in zip(MAP_NAMES, arrays)
    ]


def export_textured_obj(
    output_dir: str | Path,
    name: str,
    mesh: UnwrapResult,
    edge_wear: np.ndarray,
    curvature: np.ndarray,
    *,
    bits: int = 8,
) -> list[Path]:
    """Write an OBJ, MTL, and its texture maps into one portable folder.

    The unwrap result has one position, normal and UV for every exported vertex,
    including xatlas' seam splits. Giving all three OBJ indices the same value
    therefore preserves the exact atlas used by the bake.
    """
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(name).stem or "mesh"
    obj_path = output_dir / f"{safe_name}.obj"
    mtl_path = output_dir / f"{safe_name}.mtl"
    texture_paths = export_maps(
        output_dir, edge_wear, curvature, bits=bits
    )

    material_name = f"{safe_name}_edge_wear"
    obj_lines = [
        f"# Exported by MeshMap\nmtllib {mtl_path.name}\n",
        f"o {safe_name}\nusemtl {material_name}\ns 1\n",
    ]
    obj_lines.extend(
        f"v {float(x):.9g} {float(y):.9g} {float(z):.9g}\n"
        for x, y, z in mesh.vertices
    )
    obj_lines.extend(
        f"vt {float(u):.9g} {float(v):.9g}\n" for u, v in mesh.uvs
    )
    obj_lines.extend(
        f"vn {float(x):.9g} {float(y):.9g} {float(z):.9g}\n"
        for x, y, z in mesh.normals
    )
    for face in mesh.faces:
        indices = [int(index) + 1 for index in face]
        obj_lines.append(
            "f " + " ".join(f"{index}/{index}/{index}" for index in indices) + "\n"
        )
    obj_path.write_text("".join(obj_lines), encoding="utf-8")

    mtl_path.write_text(
        "\n".join(
            (
                "# Exported by MeshMap",
                f"newmtl {material_name}",
                "Ka 0 0 0",
                "Kd 1 1 1",
                "Ks 0 0 0",
                "Ns 1",
                "illum 1",
                f"map_Kd {texture_paths[0].name}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return [obj_path, mtl_path, *texture_paths]
