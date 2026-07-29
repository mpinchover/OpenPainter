"""Tests for turning a cursor position into a point on the mesh.

This is the machinery behind placing a decal by pointing at the model. Two
things have to hold for that to feel right: the ray has to come out of the
camera the user is actually looking through -- perspective or orthographic --
and the nearest surface has to win, so a decal lands on the face you can see
rather than the one behind it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from pyglm import glm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.picking import pick_uv, ray_mesh_hit, screen_ray  # noqa: E402

#: A unit square on the z=0 plane, UV-mapped corner to corner.
QUAD_VERTICES = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
QUAD_FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
QUAD_UVS = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)

DOWN = np.array([0.0, 0.0, -1.0])


def above(x: float, y: float) -> np.ndarray:
    return np.array([x, y, 5.0])


# --------------------------------------------------------------------------
# the ray against the mesh
# --------------------------------------------------------------------------

def test_a_ray_through_the_middle_hits_the_middle():
    uv = pick_uv(above(0.5, 0.5), DOWN, QUAD_VERTICES, QUAD_FACES, QUAD_UVS)
    assert uv == pytest.approx((0.5, 0.5))


def test_the_uv_follows_the_hit_across_the_surface():
    for point in ((0.1, 0.2), (0.9, 0.4), (0.25, 0.75), (0.99, 0.01)):
        uv = pick_uv(above(*point), DOWN, QUAD_VERTICES, QUAD_FACES, QUAD_UVS)
        assert uv == pytest.approx(point, abs=1e-9), f"at {point}"


def test_a_ray_past_the_edge_misses():
    assert pick_uv(above(1.5, 0.5), DOWN, QUAD_VERTICES, QUAD_FACES, QUAD_UVS) is None
    assert pick_uv(above(-0.01, 0.5), DOWN, QUAD_VERTICES, QUAD_FACES, QUAD_UVS) is None


def test_a_ray_pointing_away_misses():
    """Only what is in front counts -- geometry behind the camera is not a hit."""
    up = np.array([0.0, 0.0, 1.0])
    assert pick_uv(above(0.5, 0.5), up, QUAD_VERTICES, QUAD_FACES, QUAD_UVS) is None


def test_a_ray_parallel_to_the_surface_misses():
    sideways = np.array([1.0, 0.0, 0.0])
    origin = np.array([-1.0, 0.5, 0.0])
    assert pick_uv(origin, sideways, QUAD_VERTICES, QUAD_FACES, QUAD_UVS) is None


def test_the_nearest_surface_wins():
    """Two stacked quads: the decal belongs on the one you can see."""
    lower = QUAD_VERTICES.copy()
    upper = QUAD_VERTICES + np.array([0.0, 0.0, 1.0])
    vertices = np.vstack([lower, upper])
    faces = np.vstack([QUAD_FACES, QUAD_FACES + 4])
    # Mark the two sheets apart through their UVs.
    uvs = np.vstack([QUAD_UVS * 0.5, QUAD_UVS * 0.5 + 0.5])

    hit = ray_mesh_hit(above(0.5, 0.5), DOWN, vertices, faces)
    assert hit is not None
    _, _, _, distance = hit
    assert distance == pytest.approx(4.0), "the near sheet is 4 units down"

    uv = pick_uv(above(0.5, 0.5), DOWN, vertices, faces, uvs)
    assert uv == pytest.approx((0.75, 0.75)), "the upper sheet's UV range"


def test_a_surface_facing_away_is_still_hittable():
    """The viewport draws with culling off, so what is visible is what is
    pickable -- a decal must be placeable on a reversed face."""
    flipped = QUAD_FACES[:, ::-1].copy()
    uv = pick_uv(above(0.5, 0.5), DOWN, QUAD_VERTICES, flipped, QUAD_UVS)
    assert uv is not None


def test_an_empty_mesh_is_a_miss_not_a_crash():
    assert ray_mesh_hit(above(0, 0), DOWN, QUAD_VERTICES, np.zeros((0, 3), np.int64)) is None


def test_the_barycentric_weights_are_the_hits_own():
    """u and v are the second and third corners' weights, so the first corner
    takes the remainder -- get that backwards and every decal lands mirrored."""
    face, u, v, _ = ray_mesh_hit(above(0.9, 0.05), DOWN, QUAD_VERTICES, QUAD_FACES)
    assert face == 0, "the lower-right triangle"
    corners = QUAD_VERTICES[QUAD_FACES[face]]
    point = corners[0] * (1 - u - v) + corners[1] * u + corners[2] * v
    assert point[:2] == pytest.approx((0.9, 0.05))


# --------------------------------------------------------------------------
# the ray out of the camera
# --------------------------------------------------------------------------

def matrices(orthographic: bool = False):
    """A camera looking down -Z at the origin, as glm would build it."""
    view = glm.lookAt(glm.vec3(0.5, 0.5, 4.0), glm.vec3(0.5, 0.5, 0.0), glm.vec3(0, 1, 0))
    if orthographic:
        projection = glm.ortho(-1.0, 1.0, -1.0, 1.0, 0.1, 10.0)
    else:
        projection = glm.perspective(glm.radians(50.0), 1.0, 0.1, 100.0)
    combined = projection * view
    forward = np.array(combined.to_list(), dtype=np.float64).T
    return forward, np.linalg.inv(forward)


def project(forward: np.ndarray, point) -> tuple[float, float]:
    clip = forward @ np.array([*point, 1.0])
    return float(clip[0] / clip[3]), float(clip[1] / clip[3])


@pytest.mark.parametrize("orthographic", [False, True])
def test_the_ray_through_a_projected_point_comes_back_to_it(orthographic):
    """Project a point on the surface, shoot a ray back through it, and the ray
    must hit the same UV. This is the whole contract of pointing at the mesh."""
    forward, inverse = matrices(orthographic)

    for target in ((0.5, 0.5), (0.2, 0.8), (0.85, 0.15)):
        ndc = project(forward, (*target, 0.0))
        origin, direction = screen_ray(inverse, *ndc)
        uv = pick_uv(origin, direction, QUAD_VERTICES, QUAD_FACES, QUAD_UVS)
        assert uv == pytest.approx(target, abs=1e-6), f"{target} at ndc {ndc}"


def test_the_ray_direction_is_a_unit_vector():
    _, inverse = matrices()
    _, direction = screen_ray(inverse, 0.3, -0.7)
    assert np.linalg.norm(direction) == pytest.approx(1.0)


def test_an_orthographic_ray_starts_where_it_points():
    """With no eye point to shoot from, the ray has to start on the near plane
    at the cursor -- parallel rays, offset across the screen."""
    _, inverse = matrices(orthographic=True)
    left, left_direction = screen_ray(inverse, -0.6, 0.0)
    right, right_direction = screen_ray(inverse, 0.6, 0.0)

    assert left_direction == pytest.approx(right_direction), "parallel, in ortho"
    assert right[0] - left[0] == pytest.approx(1.2, rel=1e-6), "but offset apart"


def test_a_perspective_ray_fans_out_from_one_point():
    _, inverse = matrices()
    left, left_direction = screen_ray(inverse, -0.6, 0.0)
    right, right_direction = screen_ray(inverse, 0.6, 0.0)

    assert left_direction != pytest.approx(right_direction)
    # Walk both back to where they came from: the same eye.
    eye_left = left - left_direction * (left[2] - 4.0) / left_direction[2]
    eye_right = right - right_direction * (right[2] - 4.0) / right_direction[2]
    assert eye_left == pytest.approx(eye_right, abs=1e-6)
