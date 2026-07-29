"""Reading a decal image, and the normal map it composites into.

The decal is a tangent-space normal map stamped into a rectangle of the atlas.
Everywhere else the output is flat -- ``(0.5, 0.5, 1.0)``, the encoding of a
normal pointing straight out of the surface -- so the exported PNG can be
plugged into a Normal Map node and change nothing except where the decal is.

Two conventions are worth stating, because getting either wrong is a silently
wrong image rather than an error:

**Row order.** Everything in this app stores row 0 at v=0, and
:mod:`core.export` flips on write. A PNG's row 0 is its *top*, so
:func:`load_decal` flips on read -- the two flips cancel and the decal comes out
of the exporter the way round it went in.

**Slope, not vector.** Intensity scales ``xy/z``, the surface slope the normal
describes, rather than the stored vector. Scaling the vector runs the normal
flat against the surface and past it at high values, where scaling the slope
just makes the bump steeper, which is what the control is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .params import DecalParams

#: Largest edge kept when loading. A decal covering a quarter of a 4K atlas has
#: 1024 texels to fill; anything past this is memory and upload time spent on
#: detail the atlas cannot hold.
MAX_EDGE = 2048

#: Below this spread between the channels an RGB image is really a grayscale
#: one, and decoding it as a normal map would give nonsense (a mid-grey pixel
#: decodes to a normal pointing along the diagonal). Such an image is treated as
#: a height map instead -- which is what a "bump" texture usually is.
_GRAYSCALE_TOLERANCE = 2.0 / 255.0


class DecalLoadError(RuntimeError):
    """The image could not be read, or holds nothing usable."""


@dataclass(frozen=True)
class DecalImage:
    """A decal ready to upload: RGB normals plus coverage, row 0 at v=0."""

    path: str
    normals: np.ndarray
    """(h, w, 3) float32 in 0..1 -- tangent-space normals, encoded."""
    alpha: np.ndarray
    """(h, w) float32 in 0..1. All ones for an image without transparency."""
    from_height: bool
    """True when the source was grayscale and normals were derived from it."""

    @property
    def size(self) -> tuple[int, int]:
        return (self.normals.shape[1], self.normals.shape[0])

    @property
    def aspect(self) -> float:
        """Width over height."""
        return self.normals.shape[1] / max(self.normals.shape[0], 1)

    def rgba(self) -> np.ndarray:
        """Interleaved for upload: (h, w, 4) float32."""
        return np.concatenate([self.normals, self.alpha[..., None]], axis=-1)


def _is_grayscale(rgb: np.ndarray) -> bool:
    spread = rgb.max(axis=-1) - rgb.min(axis=-1)
    return bool(spread.max() <= _GRAYSCALE_TOLERANCE)


def height_to_normals(height: np.ndarray) -> np.ndarray:
    """Encode a 0..1 height field as a tangent-space normal map.

    Central differences, in texels, so the slope is whatever the image's own
    resolution implies -- the same thing every height-to-normal converter does.
    :attr:`DecalParams.intensity` is the knob for how strong it reads, so
    nothing here tries to guess an amplitude.
    """
    # Gradients along v (rows) and u (columns), as height per texel.
    dv, du = np.gradient(height.astype(np.float32))
    normals = np.stack([-du, -dv, np.ones_like(du)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return (normals * 0.5 + 0.5).astype(np.float32)


def load_decal(path: str | Path) -> DecalImage:
    """Read a decal image off disk.

    An RGB image is taken as a normal map. A grayscale one is taken as a height
    map and converted, because that is the only reading of it that produces a
    surface -- see :data:`_GRAYSCALE_TOLERANCE`.
    """
    path = Path(path).expanduser()
    try:
        with Image.open(path) as opened:
            opened.load()
            image = opened.convert("RGBA")
    except Exception as exc:
        raise DecalLoadError(f"Could not read {path.name}: {exc}") from exc

    if image.width < 2 or image.height < 2:
        raise DecalLoadError(f"{path.name} is {image.width}x{image.height}, too small to use")

    if max(image.size) > MAX_EDGE:
        ratio = MAX_EDGE / max(image.size)
        image = image.resize(
            (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
            Image.LANCZOS,
        )

    # Flip now, so this array is in the app's row-0-at-v=0 convention and every
    # later stage -- upload, compositing, export -- can ignore the question.
    data = np.flipud(np.asarray(image, dtype=np.float32) / 255.0)
    rgb, alpha = data[..., :3], data[..., 3]

    from_height = _is_grayscale(rgb)
    normals = height_to_normals(rgb[..., 0]) if from_height else np.ascontiguousarray(rgb)

    return DecalImage(
        path=str(path),
        normals=normals.astype(np.float32),
        alpha=np.ascontiguousarray(alpha, dtype=np.float32),
        from_height=from_height,
    )


def sample_decal(image: DecalImage, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear lookup into the decal, for coordinates in 0..1.

    Matches what the GL sampler does for ``LINEAR`` with clamped edges: texel
    centres sit at ``(i + 0.5) / size``, and sampling outside repeats the edge
    texel rather than wrapping.
    """
    height, width = image.normals.shape[:2]
    x = np.clip(u * width - 0.5, 0.0, width - 1.0)
    y = np.clip(v * height - 0.5, 0.0, height - 1.0)

    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]

    def blend(field: np.ndarray) -> np.ndarray:
        top = field[y0, x0] * (1.0 - fx) + field[y0, x1] * fx
        bottom = field[y1, x0] * (1.0 - fx) + field[y1, x1] * fx
        return top * (1.0 - fy) + bottom * fy

    normals = blend(image.normals)
    alpha = blend(image.alpha[..., None])[..., 0]
    return normals, alpha


def composite_normal_map(
    image: DecalImage | None, params: DecalParams, resolution: int
) -> np.ndarray:
    """The atlas-wide normal map, as a (res, res, 3) float32 array in 0..1.

    The numpy mirror of ``render/shaders/decal.frag``. The shader is what runs
    in the app -- this exists so the placement and the slope maths can be
    checked against something readable, the way :mod:`core.edge_wear` mirrors
    the wear pass.
    """
    flat = np.zeros((resolution, resolution, 3), dtype=np.float32)
    flat[..., :2] = 0.5
    flat[..., 2] = 1.0
    if image is None or not params.active():
        return flat

    # Texel centres, so a decal placed at 0.5 lands on the middle of the sheet.
    axis = (np.arange(resolution, dtype=np.float32) + 0.5) / resolution
    u, v = np.meshgrid(axis, axis)

    offset_u = u - params.center_u
    offset_v = v - params.center_v
    angle = np.radians(params.rotation)
    cos, sin = np.cos(angle), np.sin(angle)
    # Rotate by -angle: the decal turns one way, so the lookup turns the other.
    local_u = offset_u * cos + offset_v * sin
    local_v = -offset_u * sin + offset_v * cos

    width, height = params.size(image.aspect)
    decal_u = local_u / max(width, 1e-6) + 0.5
    decal_v = local_v / max(height, 1e-6) + 0.5

    inside = (decal_u >= 0.0) & (decal_u <= 1.0) & (decal_v >= 0.0) & (decal_v <= 1.0)
    if not inside.any():
        return flat

    encoded, alpha = sample_decal(image, decal_u, decal_v)
    normals = encoded * 2.0 - 1.0
    if params.flip_green:
        normals[..., 1] = -normals[..., 1]

    # Slope in the decal's own frame, scaled, then turned back into a normal.
    slope = normals[..., :2] / np.maximum(normals[..., 2:3], 1e-4)
    slope = slope * (params.intensity * alpha[..., None]) * inside[..., None]

    # The decal's own frame is rotated within the atlas, so its slope has to be
    # rotated with it -- otherwise a turned decal keeps lighting as though it
    # were upright.
    rotated = np.stack(
        [slope[..., 0] * cos - slope[..., 1] * sin,
         slope[..., 0] * sin + slope[..., 1] * cos],
        axis=-1,
    )

    # A slope of (a, b) is the normal (a, b, 1) once renormalised, so scaling
    # the slope and rebuilding is all there is to changing the depth.
    result = np.concatenate(
        [rotated, np.ones(rotated.shape[:2] + (1,), np.float32)], axis=-1
    )
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    return (result * 0.5 + 0.5).astype(np.float32)
