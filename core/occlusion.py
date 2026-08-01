"""Ambient occlusion, baked into the mesh's own UV layout.

How much of the sky each texel can see. Rays are cast from the surface into the
hemisphere it faces and asked whether anything is in the way -- a corner sees
less than a flat panel, the inside of a recess sees almost nothing -- so what
comes out is the contact shading a renderer cannot work out from a colour map.

This is the one map here that is not derived from the G-buffer arithmetically:
curvature is a derivative of the normals at a texel, which says what the surface
does *there*, while occlusion depends on geometry that may be nowhere near it.
So it is a real ray cast against the same geometry the bake rasterised, through
trimesh's Embree backend where that is installed and its pure-Python fallback
where it is not.

Rays stop at :data:`DEFAULT_DISTANCE` of the model's own size. Without a limit,
occlusion means "is there anything at all in this direction", which on a closed
model darkens every concave surface no matter how far away the far side is; with
one, it means "is there anything *nearby*", which is what reads as contact.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np
import trimesh

#: Rays per texel. Cosine-weighted and stratified, so this buys more than the
#: same number of uniform samples would; the residual speckle is taken out by
#: the blur below rather than by pushing this up, which costs a ray cast each.
DEFAULT_SAMPLES = 32

#: How far a ray travels before it stops caring, as a fraction of the model's
#: bounding-box diagonal.
DEFAULT_DISTANCE = 0.2

#: Texels per batch. Rays are built as arrays, and the whole map at 4096 would
#: be several gigabytes of them at once.
_BATCH = 200_000

#: Ray batches are independent, but each carries large origin and direction
#: arrays. Use the machine without letting a high core count multiply the
#: bake's peak memory without bound.
MAX_WORKERS = 8


def bake_occlusion(
    mesh: trimesh.Trimesh,
    position: np.ndarray,
    normal: np.ndarray,
    mask: np.ndarray,
    *,
    samples: int = DEFAULT_SAMPLES,
    distance: float = DEFAULT_DISTANCE,
    seed: int = 0,
    progress: Optional[Callable[[float], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """Bake ambient occlusion for a rasterised G-buffer.

    ``position`` and ``normal`` are per-texel world-space fields, ``mask`` says
    which texels a chart actually covers. Returns a float32 map in 0..1 where
    **1 is open sky and 0 is fully enclosed** -- the way every renderer expects
    an occlusion map, so it can be multiplied straight into ambient light.

    Uncovered texels come back as 1.0: no surface, nothing to shade, and white
    is what the seam padding should be spreading outwards from.
    """
    height, width = mask.shape
    occlusion = np.ones((height, width), dtype=np.float32)

    covered = np.flatnonzero(mask.reshape(-1))
    if covered.size == 0 or len(mesh.faces) == 0:
        return occlusion

    reach = float(np.linalg.norm(mesh.extents)) * float(distance)
    if reach <= 0.0:
        return occlusion
    # Far enough off the surface not to hit it, small enough not to skip over
    # anything real. Scaled to the model so it holds at any size.
    epsilon = reach * 1e-3

    origins = position.reshape(-1, 3)[covered].astype(np.float64)
    normals = normal.reshape(-1, 3)[covered].astype(np.float64)
    intersector = mesh.ray

    # Built once, not once per ray: the frame is a property of the surface, and
    # the phase is what keeps each texel's spiral of rays turned differently
    # from its neighbour's, so what noise is left looks like grain rather than
    # like a pattern.
    tangent, bitangent = _frame(normals)
    rng = np.random.default_rng(seed)
    phase = rng.random(covered.size)
    jitter = rng.random(covered.size)

    lit = np.zeros(covered.size, dtype=np.float32)
    total = max(int(samples), 1)
    batches = [
        (start, min(start + _BATCH, covered.size))
        for start in range(0, covered.size, _BATCH)
    ]
    worker_count = min(len(batches), MAX_WORKERS, os.cpu_count() or 1)

    # One pool for the whole bake. Each worker only reads the intersector and
    # writes no shared arrays; completed chunks are accumulated on this thread
    # in batch order, keeping the result bit-for-bit deterministic.
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for index in range(total):
            if should_stop is not None and should_stop():
                break
            directions = _hemisphere(
                normals, tangent, bitangent, phase, jitter, index, total
            )

            def trace(bounds: tuple[int, int]) -> np.ndarray:
                start, stop = bounds
                return _occluded(
                    intersector,
                    origins[start:stop] + normals[start:stop] * epsilon,
                    directions[start:stop],
                    reach,
                )

            for (start, stop), blocked in zip(batches, executor.map(trace, batches)):
                lit[start:stop] += blocked
            if progress is not None:
                progress((index + 1) / total)

    ao = 1.0 - lit / float(total)
    occlusion.reshape(-1)[covered] = np.clip(ao, 0.0, 1.0)
    return _denoise(occlusion, mask)


def _occluded(
    intersector, origins: np.ndarray, directions: np.ndarray, reach: float
) -> np.ndarray:
    """How much each ray is blocked: 1 at the surface, falling to 0 at ``reach``.

    A ray that hits something far away is barely occluded at all, which is what
    keeps a wall on the other side of the room out of the shading. Rays that
    miss contribute nothing.
    """
    blocked = np.zeros(len(origins), dtype=np.float32)
    _, index_ray, locations = intersector.intersects_id(
        origins, directions, return_locations=True, multiple_hits=False
    )
    if len(index_ray) == 0:
        return blocked

    hit_distance = np.linalg.norm(locations - origins[index_ray], axis=1)
    falloff = np.clip(1.0 - hit_distance / reach, 0.0, 1.0)
    # np.add.at rather than plain indexing: a ray index can repeat if the
    # backend reports more than one hit for it, and the last write would win.
    np.add.at(blocked, index_ray, falloff.astype(np.float32))
    return np.minimum(blocked, 1.0)


#: The golden angle, in turns. Successive rays step round by this much, which
#: is the least-clumping step there is -- the same reason a sunflower uses it.
_GOLDEN_TURN = 0.6180339887498949


def _hemisphere(
    normals: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
    phase: np.ndarray,
    jitter: np.ndarray,
    index: int,
    total: int,
) -> np.ndarray:
    """One cosine-weighted direction per texel, from stratum ``index``.

    Cosine weighting is what makes an unoccluded surface come out at exactly 1
    without a per-sample ``dot(n, l)`` term: the sampling density carries it.

    The rays are stratified in both directions rather than drawn at random.
    Elevation walks outwards one ring per sample, and azimuth turns by the
    golden angle each time, so a texel's rays end up spread evenly over its
    hemisphere instead of clumping the way chance would leave them. That is
    most of the difference between 32 rays that look smooth and 32 that speckle.
    """
    radial = np.sqrt((index + jitter) / total)
    azimuth = (phase + index * _GOLDEN_TURN) * (2.0 * np.pi)

    x = radial * np.cos(azimuth)
    y = radial * np.sin(azimuth)
    z = np.sqrt(np.clip(1.0 - radial * radial, 0.0, 1.0))

    directions = (
        tangent * x[:, None] + bitangent * y[:, None] + normals * z[:, None]
    )
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    return directions / np.clip(lengths, 1e-12, None)


def _frame(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A tangent and bitangent for each normal. Any pair will do; they only have
    to be perpendicular to it and to each other."""
    # Cross with whichever axis the normal is least aligned to, so the cross
    # product is never degenerate.
    helper = np.zeros_like(normals)
    helper[np.arange(len(normals)), np.argmin(np.abs(normals), axis=1)] = 1.0

    tangent = np.cross(normals, helper)
    tangent /= np.clip(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12, None)
    return tangent, np.cross(normals, tangent)


def _denoise(occlusion: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A 3x3 average over covered texels only.

    Ray casting leaves speckle, and a blur is far cheaper than the extra rays
    that would smooth it out. Uncovered texels are excluded from both the sum
    and the count, so a texel at the edge of a chart is not dragged towards the
    empty space beside it -- the seam padding fills that afterwards.

    The edges of the map do not wrap. An atlas is not a tiling texture: a chart
    against the left border has nothing to do with one against the right, and
    averaging them together drags a shadow across the map to a texel that can
    see the sky perfectly well.
    """
    weight = mask.astype(np.float32)
    values = occlusion * weight

    # One texel of zero-weight border, so a texel at the edge averages the
    # neighbours it has rather than the ones opposite.
    padded_values = np.pad(values, 1)
    padded_weight = np.pad(weight, 1)

    height, width = mask.shape
    total = np.zeros_like(values)
    counts = np.zeros_like(weight)
    for dy in range(3):
        for dx in range(3):
            total += padded_values[dy:dy + height, dx:dx + width]
            counts += padded_weight[dy:dy + height, dx:dx + width]

    smoothed = np.where(counts > 0, total / np.maximum(counts, 1e-6), 1.0)
    return np.where(mask, smoothed, 1.0).astype(np.float32)
