"""Turning a cursor position into a point on the mesh, and its UV.

This is what lets a decal be placed by pointing at the model rather than by
typing coordinates into the UV sliders. The cursor becomes a ray, the ray hits a
triangle, and the triangle's own UVs are interpolated at the hit -- so the
answer comes back in exactly the coordinates the decal is placed in.

Kept in numpy rather than handed to trimesh's ray backends for two reasons: the
geometry being pointed at is the *rendered* one (seam-split by the unwrap, which
trimesh no longer has), and a full Moller-Trumbore sweep over a few thousand
triangles is a fraction of a millisecond -- cheap enough to run on every mouse
move, which is what makes the decal follow the cursor.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

#: Rays that graze a triangle edge-on divide by nearly zero. Anything below this
#: is treated as parallel; the value is in the units of a normalised cross
#: product, so it is scale-free.
_PARALLEL = 1e-12


def screen_ray(
    inverse_mvp: np.ndarray, ndc_x: float, ndc_y: float
) -> tuple[np.ndarray, np.ndarray]:
    """The world-space ray through a point on the near plane.

    ``inverse_mvp`` is the inverse of projection times view. Unprojecting the
    two ends of the NDC depth range and subtracting works for a perspective and
    an orthographic camera alike -- which matters here, because clicking an axis
    ball puts the viewport into ortho, where rays are parallel and there is no
    single eye point to shoot from.
    """
    near = inverse_mvp @ np.array([ndc_x, ndc_y, -1.0, 1.0])
    far = inverse_mvp @ np.array([ndc_x, ndc_y, 1.0, 1.0])
    near = near[:3] / near[3]
    far = far[:3] / far[3]

    direction = far - near
    length = float(np.linalg.norm(direction))
    if length <= 0.0:
        return near, np.array([0.0, 0.0, 1.0])
    return near, direction / length


def ray_mesh_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Optional[tuple[int, float, float, float]]:
    """The nearest triangle the ray enters, as ``(face, u, v, distance)``.

    Moller-Trumbore, over every triangle at once. ``u`` and ``v`` are the
    barycentric weights of the second and third corners, which is what the
    caller needs to interpolate any per-vertex attribute at the hit.

    Both facings count. A decal should be placeable on a surface whose winding
    happens to point away, and the viewport draws with culling off, so what is
    visible and what is hittable stay the same thing.
    """
    if len(faces) == 0:
        return None

    corners = vertices[faces]  # (n, 3, 3)
    edge1 = corners[:, 1] - corners[:, 0]
    edge2 = corners[:, 2] - corners[:, 0]

    pvec = np.cross(direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, pvec)

    live = np.abs(determinant) > _PARALLEL
    if not live.any():
        return None

    inverse = np.zeros_like(determinant)
    inverse[live] = 1.0 / determinant[live]

    tvec = origin - corners[:, 0]
    u = np.einsum("ij,ij->i", tvec, pvec) * inverse

    qvec = np.cross(tvec, edge1)
    v = np.einsum("ij,ij->i", np.broadcast_to(direction, edge1.shape), qvec) * inverse

    distance = np.einsum("ij,ij->i", edge2, qvec) * inverse

    inside = live & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (distance > 0.0)
    if not inside.any():
        return None

    candidates = np.flatnonzero(inside)
    nearest = candidates[np.argmin(distance[candidates])]
    return int(nearest), float(u[nearest]), float(v[nearest]), float(distance[nearest])


def pick_uv(
    origin: np.ndarray,
    direction: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
) -> Optional[tuple[float, float]]:
    """The UV coordinate under a ray, or None if it misses the mesh."""
    hit = ray_mesh_hit(origin, direction, vertices, faces)
    if hit is None:
        return None

    face, u, v, _ = hit
    corners = uvs[faces[face]]
    # Barycentric weights: the first corner takes whatever the other two leave.
    interpolated = corners[0] * (1.0 - u - v) + corners[1] * u + corners[2] * v
    return float(interpolated[0]), float(interpolated[1])
