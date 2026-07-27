"""Writers for the output maps.

Images are flipped vertically on write. Our textures store row 0 at v=0, while
PNG stores row 0 at the top, so the flip is what makes the exported file line up
with the same UVs in Blender, Substance or any other DCC.

The maps are the whole product: they are baked into the mesh's own UV layout, so
they apply directly to the model you exported from Blender. There is nothing
else to ship alongside them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

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
