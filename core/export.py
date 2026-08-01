"""Writers for the output maps.

Images are flipped vertically on write. Our textures store row 0 at v=0, while
PNG stores row 0 at the top, so the flip is what makes the exported file line up
with the same UVs in Blender, Substance or any other DCC.

The maps are the whole product: they are baked into the mesh's own UV layout, so
they apply directly to the model you exported from Blender. There is nothing
else to ship alongside them.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

#: Tangent-space normals, carrying whatever decals are placed.
NORMAL_NAME = "normal"
#: The mask tree resolved to colour, and the two surface channels that travel
#: with it. One file each rather than a packed ORM: they are separate inputs in
#: every shader graph they end up in, and packing is a step to undo.
COLOR_NAME = "color"
METALLIC_NAME = "metallic"
ROUGHNESS_NAME = "roughness"
#: How much sky each texel can see, from the ray-cast bake.
OCCLUSION_NAME = "ao"

#: The whole export, in the order it is written. A standard PBR set and nothing
#: else: the edge wear is *in* the colour map, where the texture put it, and the
#: curvature bake is an input to that rather than a product of it.
MAP_NAMES = (COLOR_NAME, NORMAL_NAME, METALLIC_NAME, ROUGHNESS_NAME, OCCLUSION_NAME)


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


def _write_rgb16_png(path: Path, pixels: np.ndarray) -> None:
    """Write a 16-bit-per-channel RGB PNG.

    Pillow does not: its 16-bit support is the single-channel ``I;16`` mode, and
    handing it a three-channel uint16 array raises. Every DCC reads 16-bit RGB
    PNGs happily, though, and it is the depth a normal map wants -- 8 bits puts
    visible steps in a shallow bevel or a gentle vent lip.

    So this emits the file directly. A PNG is a signature, an IHDR describing
    the raster, one zlib-compressed IDAT holding the rows -- each prefixed with
    a filter byte, 0 for "stored as-is" -- and an IEND, every chunk carrying its
    own CRC. Big-endian samples, per the spec.
    """
    height, width, _ = pixels.shape
    rows = b"".join(b"\x00" + row.tobytes() for row in pixels.astype(">u2"))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)  # 2 = truecolour
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows, 9))
        + chunk(b"IEND", b"")
    )


def save_rgb_map(path: str | Path, array: np.ndarray, *, bits: int = 8) -> Path:
    """Write an (h, w, 3) array of 0..1 values as an RGB PNG.

    Values are written as handed over. The two RGB products differ in what they
    mean -- a normal map is an encoded direction, a colour map is colour -- but
    not in how they are stored, and re-encoding either here would put the
    convention in two places.
    """
    if bits not in (8, 16):
        raise ValueError(f"bits must be 8 or 16, got {bits}")

    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"an RGB map needs shape (h, w, 3), got {array.shape}")

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.flipud(np.clip(array, 0.0, 1.0))
    if bits == 16:
        _write_rgb16_png(path, (data * 65535.0 + 0.5).astype(np.uint16))
    else:
        Image.fromarray((data * 255.0 + 0.5).astype(np.uint8)).save(
            path, format="PNG", optimize=True
        )
    return path


def save_normal_map(path: str | Path, array: np.ndarray, *, bits: int = 8) -> Path:
    """Write already-encoded tangent-space normals, flat being (0.5, 0.5, 1.0)."""
    return save_rgb_map(path, array, bits=bits)


def export_maps(
    output_dir: str | Path,
    *,
    color: np.ndarray | None = None,
    normal: np.ndarray | None = None,
    material: np.ndarray | None = None,
    occlusion: np.ndarray | None = None,
    maps: Iterable[str] = MAP_NAMES,
    bits: int = 8,
    prefix: str = "",
) -> list[Path]:
    """Write whichever maps were handed over, and return what was written.

    :data:`MAP_NAMES`, and nothing besides: colour and its metallic and
    roughness from the texture, normals from the decals, occlusion from the
    bake. They come from independent halves of the app, so any of them may be
    absent and the rest still get written. One export call, everything there is.

    ``maps`` narrows that to the ones asked for. It is a separate question from
    which arrays were handed over: metallic and roughness arrive together in
    ``material`` because they travel together through the compositor, and either
    can be wanted without the other.

    16-bit is worth reaching for if banding shows up in a gradient -- occlusion
    is a smooth falloff that 256 levels can struggle with.
    """
    output_dir = Path(output_dir).expanduser()
    wanted = set(maps)
    written: list[Path] = []

    for name, array in ((COLOR_NAME, color), (NORMAL_NAME, normal)):
        if array is not None and name in wanted:
            written.append(
                save_rgb_map(output_dir / f"{prefix}{name}.png", array, bits=bits)
            )

    if material is not None:
        written.extend(
            _write_material(output_dir, material, wanted, bits=bits, prefix=prefix)
        )

    if occlusion is not None and OCCLUSION_NAME in wanted:
        written.append(
            save_map(output_dir / f"{prefix}{OCCLUSION_NAME}.png", occlusion, bits=bits)
        )
    return written


def _write_material(
    output_dir: Path,
    material: np.ndarray,
    wanted: set[str],
    *,
    bits: int,
    prefix: str,
) -> list[Path]:
    """Metallic and roughness, one map each.

    ``material`` is (h, w, 4): metallic, roughness, alpha, emission, in the
    order :meth:`core.layers.ColorSlot.material` packs them. Only the first two
    are written -- alpha and emission are there for the viewport to shade with,
    not for the export, which is a standard PBR set and stops at that.
    """
    material = np.asarray(material, dtype=np.float32)
    if material.ndim != 3 or material.shape[2] != 4:
        raise ValueError(f"a material needs shape (h, w, 4), got {material.shape}")

    return [
        save_map(output_dir / f"{prefix}{name}.png", material[..., index], bits=bits)
        for index, name in enumerate((METALLIC_NAME, ROUGHNESS_NAME))
        if name in wanted
    ]
