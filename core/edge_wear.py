"""The EdgeWear001 node group, on the CPU.

Decoded from ArmorPaint's ``cloud/materials/Procedural/EdgeWear001.arm`` and
ported node for node. The whole graph reduces to::

    noise = tex_noise(bposition * value * 10, detail, roughness,
                      lacunarity, distortion)
    mask  = clamp((curvature - noise * wear_amount) * contrast, 0, 1)

The GPU path (``render/shaders/edge_wear.frag``) is what the viewport shows and
what :mod:`core.export` reads back, so preview and exported PNG are the same
pixels by construction. This module mirrors it so the formula can be tested and
driven headlessly without a GL context.

The noise is :func:`tex_noise`, a straight port of ArmorPaint's ``str_tex_noise``
in ``nodes_material/noise_texture_node.c``. It is sampled at ``bposition`` --
object-space position normalised over the bounding box -- not at the UV, which
is why the wear pattern is continuous across atlas seams.
"""

from __future__ import annotations

import numpy as np

from .params import EdgeWearParams


def _hash(n: np.ndarray) -> np.ndarray:
    """ArmorPaint: `fun hash(n: float): float { return frac(sin(n) * 10000.0); }`"""
    value = np.sin(n) * np.float32(10000.0)
    return value - np.floor(value)


_STEP = np.array([110.0, 241.0, 171.0], dtype=np.float32)


def tex_noise_f(x: np.ndarray) -> np.ndarray:
    """Value noise on a 3D lattice. Port of ``tex_noise_f``."""
    x = np.asarray(x, dtype=np.float32)
    i = np.floor(x)
    f = x - i
    n = i @ _STEP
    u = f * f * (3.0 - 2.0 * f)

    def corner(dx: float, dy: float, dz: float) -> np.ndarray:
        return _hash(n + float(np.dot(_STEP, (dx, dy, dz))))

    def lerp(a, b, t):
        return a + (b - a) * t

    return lerp(
        lerp(
            lerp(corner(0, 0, 0), corner(1, 0, 0), u[..., 0]),
            lerp(corner(0, 1, 0), corner(1, 1, 0), u[..., 0]),
            u[..., 1],
        ),
        lerp(
            lerp(corner(0, 0, 1), corner(1, 0, 1), u[..., 0]),
            lerp(corner(0, 1, 1), corner(1, 1, 1), u[..., 0]),
            u[..., 1],
        ),
        u[..., 2],
    )


def tex_noise_fbm(
    p: np.ndarray, detail: float, roughness: float, lacunarity: float
) -> np.ndarray:
    """Port of ``tex_noise_fbm``, including its fractional-octave blend."""
    fscale = 1.0
    amp = 1.0
    maxamp = 0.0
    total = np.zeros(p.shape[:-1], dtype=np.float32)

    octaves = int(np.clip(detail, 0.0, 8.0))
    for _ in range(octaves + 1):  # ArmorPaint loops `ii <= n`, so one extra
        total = total + amp * tex_noise_f(p * fscale)
        maxamp += amp
        amp *= roughness
        fscale *= lacunarity

    remainder = detail - np.floor(detail)
    if remainder > 0.0:
        extra = tex_noise_f(p * fscale)
        total2 = total + extra * amp
        maxamp2 = maxamp + amp
        return (total / maxamp) + ((total2 / maxamp2) - (total / maxamp)) * remainder
    return total / maxamp


def tex_noise(
    p: np.ndarray,
    scale: float,
    detail: float,
    roughness: float,
    lacunarity: float,
    distortion: float,
) -> np.ndarray:
    """Port of ``tex_noise``. Returns the Factor output (the ``.x`` component)."""
    pp = np.asarray(p, dtype=np.float32) * np.float32(scale)
    if distortion != 0.0:
        warp = np.stack(
            [
                tex_noise_f(pp),
                tex_noise_f(pp + np.array([0.5, 0.0, 0.0], dtype=np.float32)),
                tex_noise_f(pp + np.array([0.0, 0.5, 0.0], dtype=np.float32)),
            ],
            axis=-1,
        )
        pp = pp + np.float32(distortion) * (warp * 2.0 - 1.0)
    return tex_noise_fbm(pp, detail, roughness, lacunarity)


def edge_wear(
    curvature: np.ndarray, bposition: np.ndarray, params: EdgeWearParams
) -> np.ndarray:
    """CPU mirror of ``edge_wear.frag``.

    ``curvature`` is the baked 0..1 map; ``bposition`` is (h, w, 3) object-space
    position normalised over the bounding box. Returns a float32 image in 0..1.
    """
    curvature = np.asarray(curvature, dtype=np.float32)

    noise = tex_noise(
        bposition,
        params.value * 10.0,
        params.detail,
        params.roughness,
        params.lacunarity,
        params.distortion,
    )
    wear = noise * np.float32(params.wear_amount)
    mask = (curvature - wear) * np.float32(params.contrast)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def normalize_position(position: np.ndarray, lower: np.ndarray, extents: np.ndarray) -> np.ndarray:
    """World position -> ArmorPaint's ``bposition``.

    ``(input.pos.xyz + hdim) / dim`` in ArmorPaint, where hdim/dim are half and
    full model-space bounding-box dimensions. Written here as ``(p - min) / dim``,
    which is the same thing for a mesh centred on its origin and stays correct
    when it is not.
    """
    extents = np.maximum(np.asarray(extents, dtype=np.float32), 1e-9)
    return ((np.asarray(position, dtype=np.float32) - np.asarray(lower, dtype=np.float32))
            / extents).astype(np.float32)
