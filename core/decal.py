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

**Square in UV is not square on the model.** A UV unit does not have to cover
the same amount of surface across as it does up: the starter cube's box unwrap
gives each face a 1/3 x 1/2 cell, so one unit of u spans 1.5x the world that one
unit of v does. Stamping a square image into a square UV rectangle therefore
lands it 1.5x too wide. :func:`uv_aspect` measures that ratio off the mesh, and
:meth:`core.params.DecalParams.size` divides it back out, so a round decal comes
out round.

**Slope, not vector.** Intensity scales ``xy/z``, the surface slope the normal
describes, rather than the stored vector. Scaling the vector runs the normal
flat against the surface and past it at high values, where scaling the slope
just makes the bump steeper, which is what the control is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

from .params import MAX_FALLOFF, DecalParams

#: Largest edge kept when loading. A decal covering a quarter of a 4K atlas has
#: 1024 texels to fill; anything past this is memory and upload time spent on
#: detail the atlas cannot hold.
MAX_EDGE = 2048

#: Below this spread between the channels an RGB image is really a grayscale
#: one, and decoding it as a normal map would give nonsense (a mid-grey pixel
#: decodes to a normal pointing along the diagonal). Such an image is treated as
#: a height map instead -- which is what a "bump" texture usually is.
_GRAYSCALE_TOLERANCE = 2.0 / 255.0


#: How far from square the correction will go. Past this the layout is telling
#: us something strange -- a collapsed island, a UV set in the wrong units --
#: and a decal stretched sixteen to one is not what anyone meant either way.
MAX_UV_ASPECT = 16.0


def uv_aspect(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    face: int | None = None,
) -> float:
    """World distance per unit u, over world distance per unit v.

    1.0 means a square in UV space is a square on the surface. Above that, u is
    the stretched axis: the starter cube's box unwrap comes out at 1.5.

    With ``face``, the answer for that one triangle -- which is what a decal
    sitting on it wants, since a layout can be dense in one island and sparse in
    the next. Without, one number for the whole mesh: the geometric mean over
    every triangle, weighted by the UV area each covers, because it is a ratio
    (a face at 2.0 and a face at 0.5 average to 1.0, not to 1.25) and because
    the islands that own most of the atlas should decide.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    uvs = np.asarray(uvs, dtype=np.float64)
    if len(faces) == 0 or len(uvs) == 0:
        return 1.0
    if face is not None:
        faces = faces[int(face):int(face) + 1]

    corners = vertices[faces]
    texels = uvs[faces]
    edge1 = corners[:, 1] - corners[:, 0]
    edge2 = corners[:, 2] - corners[:, 0]
    delta1 = texels[:, 1] - texels[:, 0]
    delta2 = texels[:, 2] - texels[:, 0]

    # The determinant is twice the triangle's UV area, and inverting it is what
    # turns two edges into the surface's du and dv directions.
    determinant = delta1[:, 0] * delta2[:, 1] - delta2[:, 0] * delta1[:, 1]
    usable = np.abs(determinant) > 1e-12
    if not usable.any():
        return 1.0  # every triangle collapsed in UV space; nothing to measure

    inverse = 1.0 / determinant[usable]
    du = (edge1[usable] * delta2[usable][:, 1, None]
          - edge2[usable] * delta1[usable][:, 1, None]) * inverse[:, None]
    dv = (edge2[usable] * delta1[usable][:, 0, None]
          - edge1[usable] * delta2[usable][:, 0, None]) * inverse[:, None]

    per_u = np.linalg.norm(du, axis=1)
    per_v = np.linalg.norm(dv, axis=1)
    alive = (per_u > 1e-12) & (per_v > 1e-12)
    if not alive.any():
        return 1.0

    ratio = per_u[alive] / per_v[alive]
    weight = np.abs(determinant[usable][alive])
    aspect = float(np.exp(np.average(np.log(ratio), weights=weight)))
    if not np.isfinite(aspect) or aspect <= 0.0:
        return 1.0
    return float(np.clip(aspect, 1.0 / MAX_UV_ASPECT, MAX_UV_ASPECT))


#: What the library offers. Everything :func:`load_decal` can read.
LIBRARY_SUFFIXES = (".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff", ".bmp", ".exr")


def library(directory: Optional[Path]) -> list[Path]:
    """Every decal image in a directory, in a stable order.

    Sorted by name and case-insensitively, so the shelf does not rearrange
    itself between launches or between machines. A directory that is missing,
    unreadable or not a directory at all is an empty shelf rather than an
    error -- the path comes from metadata.json, and a typo there should cost
    the library rather than the session.
    """
    if directory is None:
        return []
    try:
        entries = list(Path(directory).iterdir())
    except OSError:
        return []
    return sorted(
        (entry for entry in entries
         if entry.is_file() and entry.suffix.lower() in LIBRARY_SUFFIXES),
        key=lambda entry: entry.name.lower(),
    )


def decal_at_uv(decals, u: float, v: float) -> Optional[int]:
    """Which decal covers a point of the atlas, or None.

    The last one wins, because that is the one stamped last and so the one on
    top. Rectangles only: the alpha and the edge fade decide what a decal
    *looks* like, but what it means to click on one is to click inside its
    placement, including the parts of it that happen to be transparent.
    """
    for index in range(len(decals) - 1, -1, -1):
        params = decals[index]
        if not params.active():
            continue
        if _inside(params, u, v):
            return index
    return None


def _inside(params, u: float, v: float) -> bool:
    """Whether a UV lands within a decal's own rectangle."""
    width, height = params.size()
    if width <= 0.0 or height <= 0.0:
        return False

    angle = np.radians(params.rotation)
    cos, sin = np.cos(angle), np.sin(angle)
    offset_u = u - params.center_u
    offset_v = v - params.center_v
    # Rotate by -angle: the decal turns one way, so the lookup turns the other.
    local_u = (offset_u * cos + offset_v * sin) / width
    local_v = (-offset_u * sin + offset_v * cos) / height
    return abs(local_u) <= 0.5 and abs(local_v) <= 0.5


def outline_uvs(params, samples: int = 12) -> np.ndarray:
    """The decal's own border, as a closed loop of UV points.

    Walked rather than cornered: the four corners alone would draw a straight
    line across a UV island that the surface bends through, so each edge is
    subdivided and every point is looked up on the mesh separately.
    """
    width, height = params.size()
    angle = np.radians(params.rotation)
    cos, sin = np.cos(angle), np.sin(angle)

    corners = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    points = []
    for index, (start_u, start_v) in enumerate(corners):
        end_u, end_v = corners[(index + 1) % len(corners)]
        for step in range(max(int(samples), 1)):
            fraction = step / max(int(samples), 1)
            local_u = (start_u + (end_u - start_u) * fraction) * width
            local_v = (start_v + (end_v - start_v) * fraction) * height
            points.append((
                params.center_u + local_u * cos - local_v * sin,
                params.center_v + local_u * sin + local_v * cos,
            ))
    points.append(points[0])  # closed
    return np.asarray(points, dtype=np.float64)


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

    # A fully transparent texel has no normal, but it still has a colour, and
    # that colour is usually white -- which decodes to a normal tilted 45
    # degrees along both axes. Nothing samples it directly, because the alpha
    # multiplies it out; filtering does, though. Every bilinear tap and every
    # mipmap level around the edge of the artwork averages it in and comes back
    # with a tilt that was never drawn. Flat is what belongs where there is
    # nothing, so filtering there blends towards no change at all.
    normals = np.where(
        alpha[..., None] > 0.0, normals, np.array([0.5, 0.5, 1.0], dtype=np.float32)
    )

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


def edge_fade(decal_u: np.ndarray, decal_v: np.ndarray, falloff: float) -> np.ndarray:
    """How much of the decal survives at each point of its own rectangle.

    Cut, then feathered. The outer ``falloff`` of the way out is gone entirely
    and the fade is spread over the same width again just inside it, so the
    middle is untouched.

    A fade that merely reached zero *at* the border would leave no hard edge but
    would still show most of whatever is out there -- and what is out there is
    usually the reason: a canvas the artwork was drawn on, white or black,
    neither of which is a flat normal. Cutting it means the border is gone
    rather than blurred.

    Measured along whichever axis is nearer its edge, so all four sides lose the
    same width instead of the corners going first.

    The numpy mirror of ``fade()`` in render/shaders/decal.frag.
    """
    falloff = float(np.clip(falloff, 0.0, MAX_FALLOFF))
    if falloff <= 0.0:
        return np.ones_like(decal_u)

    edge = np.maximum(np.abs(decal_u - 0.5), np.abs(decal_v - 0.5)) * 2.0
    ramp = np.clip((edge - (1.0 - 2.0 * falloff)) / falloff, 0.0, 1.0)
    return 1.0 - ramp * ramp * (3.0 - 2.0 * ramp)  # smoothstep


def composite_normal_map(
    image: DecalImage | None, params: DecalParams, resolution: int
) -> np.ndarray:
    """One decal stamped into an atlas-wide normal map. See :func:`composite`."""
    return composite([] if image is None else [(image, params)], resolution)


def composite(
    decals: Iterable[tuple["DecalImage", DecalParams]], resolution: int
) -> np.ndarray:
    """Every decal stamped into one atlas-wide (res, res, 3) map in 0..1.

    The numpy mirror of ``render/shaders/decal.frag``. The shader is what runs
    in the app -- this exists so the placement and the slope maths can be
    checked against something readable, the way :mod:`core.edge_wear` mirrors
    the wear pass.

    Decals accumulate as **slopes**, which is what lets them overlap sensibly: a
    vent stamped across a panel line should read as both, and adding the two
    surfaces' gradients is what "both" means. Adding the encoded normals instead
    would average them towards flat, and the second decal would rub out the
    first as much as it added itself.
    """
    total = np.zeros((resolution, resolution, 2), dtype=np.float32)
    for image, params in decals:
        if image is not None and params.active():
            total += _decal_slope(image, params, resolution)

    # A slope of (a, b) is the normal (a, b, 1) once renormalised, so summing
    # the slopes and rebuilding once at the end is the whole composite.
    result = np.concatenate(
        [total, np.ones(total.shape[:2] + (1,), np.float32)], axis=-1
    )
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    return (result * 0.5 + 0.5).astype(np.float32)


def _decal_slope(
    image: DecalImage, params: DecalParams, resolution: int
) -> np.ndarray:
    """What one decal adds to the atlas's surface gradient, as (res, res, 2)."""
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

    width, height = params.size()
    decal_u = local_u / max(width, 1e-6) + 0.5
    decal_v = local_v / max(height, 1e-6) + 0.5

    inside = (decal_u >= 0.0) & (decal_u <= 1.0) & (decal_v >= 0.0) & (decal_v <= 1.0)
    if not inside.any():
        return np.zeros((resolution, resolution, 2), dtype=np.float32)

    fade = edge_fade(decal_u, decal_v, params.falloff)
    encoded, alpha = sample_decal(image, decal_u, decal_v)
    normals = encoded * 2.0 - 1.0
    if params.flip_green:
        normals[..., 1] = -normals[..., 1]

    # Slope in the decal's own frame, scaled, then turned back into a normal.
    slope = normals[..., :2] / np.maximum(normals[..., 2:3], 1e-4)
    slope = slope * (params.intensity * alpha * fade)[..., None] * inside[..., None]

    # The decal's own frame is rotated within the atlas, so its slope has to be
    # rotated with it -- otherwise a turned decal keeps lighting as though it
    # were upright.
    return np.stack(
        [slope[..., 0] * cos - slope[..., 1] * sin,
         slope[..., 0] * sin + slope[..., 1] * cos],
        axis=-1,
    ).astype(np.float32)
