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

import numpy as np
from PIL import Image

#: The product: the EdgeWear001 mask.
EDGE_WEAR_NAME = "edge_wear"
#: The Curvature Texture bake it is derived from.
CURVATURE_NAME = "curvature"
#: Tangent-space normals, carrying whatever decals are placed.
NORMAL_NAME = "normal"

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


def save_normal_map(path: str | Path, array: np.ndarray, *, bits: int = 8) -> Path:
    """Write an (h, w, 3) array of already-encoded 0..1 normals as an RGB PNG.

    Encoded, not raw: the caller hands over what a Normal Map node expects to
    read, with a flat surface at (0.5, 0.5, 1.0). Nothing here re-encodes, so
    there is no second place for the convention to be decided.
    """
    if bits not in (8, 16):
        raise ValueError(f"bits must be 8 or 16, got {bits}")

    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"a normal map needs shape (h, w, 3), got {array.shape}")

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


def export_maps(
    output_dir: str | Path,
    edge_wear: np.ndarray | None = None,
    curvature: np.ndarray | None = None,
    *,
    normal: np.ndarray | None = None,
    bits: int = 8,
    prefix: str = "",
) -> list[Path]:
    """Write whichever maps were handed over, and return what was written.

    edge_wear.png and the curvature bake behind it come from the bake; normal.png
    comes from the decals and needs no bake at all, so any of the three may be
    absent and the others still get written.

    16-bit is worth reaching for if banding shows up in the falloff -- the
    curvature bake is a smooth gradient that 256 levels can struggle with.
    """
    output_dir = Path(output_dir).expanduser()
    written = [
        save_map(output_dir / f"{prefix}{name}.png", array, bits=bits)
        for name, array in zip(MAP_NAMES, (edge_wear, curvature))
        if array is not None
    ]
    if normal is not None:
        written.append(
            save_normal_map(output_dir / f"{prefix}{NORMAL_NAME}.png", normal, bits=bits)
        )
    return written
