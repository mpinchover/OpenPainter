"""The ambient occlusion bake.

Occlusion is the one map here that geometry near a texel cannot answer on its
own: a texel is dark because of something that may be nowhere near it in the
atlas, or on the surface. So this is a real ray cast, and these tests are about
what the rays find -- an open plane sees the whole sky, the floor of a slot sees
almost none of it, and a corner sees roughly half.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.occlusion import DEFAULT_DISTANCE, bake_occlusion  # noqa: E402

#: Small, so the tests stay quick: these check values, not resolution.
GRID = 24


def plane_gbuffer(size: float = 4.0, height: float = 0.0):
    """A patch of ground, sampled on a grid, all facing up."""
    axis = np.linspace(-size / 2, size / 2, GRID, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    position = np.stack([x, y, np.full_like(x, height)], axis=-1)
    normal = np.zeros_like(position)
    normal[..., 2] = 1.0
    return position, normal, np.ones((GRID, GRID), dtype=bool)


def ground(size: float = 4.0) -> trimesh.Trimesh:
    """A single quad on z=0, big enough that the samples sit well inside it."""
    half = size / 2
    return trimesh.Trimesh(
        vertices=np.array([[-half, -half, 0], [half, -half, 0],
                           [half, half, 0], [-half, half, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )


# --------------------------------------------------------------------------
# what the rays find
# --------------------------------------------------------------------------

def test_open_ground_sees_the_whole_sky():
    """Nothing above it, so nothing occludes it: white, all the way across."""
    position, normal, mask = plane_gbuffer()
    ao = bake_occlusion(ground(), position, normal, mask, samples=16)

    assert ao.min() > 0.99, "an unobstructed surface is not shaded at all"


def test_a_lid_just_above_the_ground_shuts_the_sky_out():
    position, normal, mask = plane_gbuffer(size=2.0)
    lid = ground(size=6.0)
    lid.apply_translation((0, 0, 0.1))
    scene = trimesh.util.concatenate([ground(size=6.0), lid])

    ao = bake_occlusion(scene, position, normal, mask, samples=32)
    assert ao.mean() < 0.15, "a surface with a ceiling on it is nearly black"


def test_an_inside_corner_is_shaded_but_not_black():
    """Half the hemisphere is walled off, so roughly half the light is.

    The far end of the patch is beyond the rays' reach of the wall, so it has
    to come back open -- that contrast is the whole point of the distance
    limit, and a corner that shaded the entire floor would be no use.
    """
    position, normal, mask = plane_gbuffer(size=4.0)
    wall = ground(size=6.0)
    wall.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))
    wall.apply_translation((-2.0, 0, 0))  # standing at the left edge of the patch
    scene = trimesh.util.concatenate([ground(size=6.0), wall])

    ao = bake_occlusion(scene, position, normal, mask, samples=32)
    against_the_wall = ao[:, 0].mean()
    out_in_the_open = ao[:, -1].mean()

    assert 0.2 < against_the_wall < 0.75, against_the_wall
    assert out_in_the_open > 0.95, "past the rays' reach, the wall is not there"


def test_distance_is_what_keeps_the_far_wall_out_of_it():
    """Without a limit, occlusion means "is anything in this direction at all",
    which darkens a surface for geometry on the other side of the model."""
    position, normal, mask = plane_gbuffer(size=1.0)
    lid = ground(size=8.0)
    lid.apply_translation((0, 0, 1.5))
    scene = trimesh.util.concatenate([ground(size=8.0), lid])

    near = bake_occlusion(scene, position, normal, mask, samples=16, distance=1.0)
    far = bake_occlusion(scene, position, normal, mask, samples=16, distance=0.05)

    assert near.mean() < 0.5, "a ceiling within reach shades the floor"
    assert far.mean() > 0.95, "the same ceiling out of reach does not"


def test_a_surface_does_not_shade_itself():
    """Rays start on the surface they are cast from, so without an offset every
    one of them would hit it immediately and the whole map would be black."""
    position, normal, mask = plane_gbuffer()
    ao = bake_occlusion(ground(), position, normal, mask, samples=8)
    assert ao.mean() > 0.99


# --------------------------------------------------------------------------
# the shape of what comes back
# --------------------------------------------------------------------------

def test_uncovered_texels_come_back_white():
    """No surface, nothing to shade -- and white is what the seam padding
    should be spreading outwards into the gutter."""
    position, normal, mask = plane_gbuffer()
    mask[:, :GRID // 2] = False

    ao = bake_occlusion(ground(), position, normal, mask, samples=8)
    assert (ao[:, :GRID // 2] == 1.0).all()


def test_the_map_matches_the_gbuffer_it_was_given():
    position, normal, mask = plane_gbuffer()
    ao = bake_occlusion(ground(), position, normal, mask, samples=4)

    assert ao.shape == mask.shape
    assert ao.dtype == np.float32
    assert ao.min() >= 0.0 and ao.max() <= 1.0


def test_an_empty_mask_is_an_empty_bake_not_a_crash():
    position, normal, mask = plane_gbuffer()
    ao = bake_occlusion(ground(), position, normal, np.zeros_like(mask), samples=8)
    assert (ao == 1.0).all()


def test_a_mesh_with_no_faces_occludes_nothing():
    position, normal, mask = plane_gbuffer()
    empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), np.int64))
    assert (bake_occlusion(empty, position, normal, mask, samples=8) == 1.0).all()


# --------------------------------------------------------------------------
# how it is sampled
# --------------------------------------------------------------------------

def test_the_same_bake_twice_gives_the_same_map():
    """Seeded, so a re-bake with nothing changed does not shuffle the noise --
    which would show up as an export that differs from the last one for no
    reason anybody could point at."""
    position, normal, mask = plane_gbuffer(size=2.0)
    wall = ground(size=6.0)
    wall.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))
    wall.apply_translation((-1.0, 0, 0))
    scene = trimesh.util.concatenate([ground(size=6.0), wall])

    first = bake_occlusion(scene, position, normal, mask, samples=8)
    second = bake_occlusion(scene, position, normal, mask, samples=8)
    assert np.array_equal(first, second)


def test_the_blur_does_not_wrap_around_the_atlas():
    """The speckle blur averages a texel with its neighbours -- and the texels
    on the opposite edge of the map are not among them. An atlas is not a
    tiling texture: a chart against the left border has nothing to do with one
    against the right, and wrapping drags a shadow all the way across.
    """
    from core.occlusion import _denoise

    dark = np.ones((8, 8), dtype=np.float32)
    dark[:, 0] = 0.0  # a shadow down the left edge only

    smoothed = _denoise(dark, np.ones((8, 8), dtype=bool))
    assert smoothed[:, -1].min() == pytest.approx(1.0), "the right edge is open"
    assert smoothed[:, 0].max() < 0.7, "and the left edge is still shaded"


def test_more_rays_means_less_noise():
    """The stratified spiral is what makes a modest ray count usable at all;
    this is the check that more of them still converges rather than wandering."""
    position, normal, mask = plane_gbuffer(size=2.0)
    wall = ground(size=6.0)
    wall.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))
    wall.apply_translation((-1.0, 0, 0))
    scene = trimesh.util.concatenate([ground(size=6.0), wall])

    def roughness(samples: int) -> float:
        ao = bake_occlusion(scene, position, normal, mask, samples=samples)
        # How much a texel differs from the one beside it, along a row where
        # the true answer varies smoothly. Noise shows up here; the gradient
        # itself is small over one texel.
        return float(np.abs(np.diff(ao, axis=1)).mean())

    assert roughness(32) < roughness(2)


def test_the_hemisphere_stays_on_the_lit_side():
    """A ray aimed into the surface would report the surface as occluding it,
    which is the classic way an AO bake comes out uniformly grey."""
    from core.occlusion import _frame, _hemisphere

    rng = np.random.default_rng(0)
    normals = rng.normal(size=(200, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    tangent, bitangent = _frame(normals)
    phase = rng.random(200)
    jitter = rng.random(200)

    for index in range(16):
        directions = _hemisphere(normals, tangent, bitangent, phase, jitter, index, 16)
        assert (directions * normals).sum(axis=1).min() >= 0.0
        assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)


def test_the_default_reach_is_a_fraction_of_the_model():
    """A fixed distance in metres would shade a chair and ignore a building."""
    position, normal, mask = plane_gbuffer(size=2.0)

    def shaded(scale: float) -> float:
        lid = ground(size=8.0 * scale)
        lid.apply_translation((0, 0, 0.3 * scale))
        scene = trimesh.util.concatenate([ground(size=8.0 * scale), lid])
        return float(bake_occlusion(
            scene, position * scale, normal, mask, samples=16
        ).mean())

    assert shaded(1.0) == pytest.approx(shaded(10.0), abs=0.05)


def test_stopping_early_leaves_the_bake_alone():
    """Cancellation returns whatever it had; the pipeline throws it away rather
    than recording the stage as done."""
    position, normal, mask = plane_gbuffer()
    ao = bake_occlusion(
        ground(), position, normal, mask, samples=16, should_stop=lambda: True
    )
    assert ao.shape == mask.shape


def test_ray_batches_use_more_than_one_worker(monkeypatch):
    """A production atlas has many batches; they should not queue on one CPU."""
    import core.occlusion as occlusion

    workers: set[int] = set()
    lock = threading.Lock()

    def trace(intersector, origins, directions, reach):
        with lock:
            workers.add(threading.get_ident())
        # Keep jobs overlapping long enough for the executor to populate its
        # pool; the geometry result is irrelevant to this scheduling test.
        time.sleep(0.005)
        return np.zeros(len(origins), dtype=np.float32)

    monkeypatch.setattr(occlusion, "_BATCH", 64)
    monkeypatch.setattr(occlusion, "_occluded", trace)
    position, normal, mask = plane_gbuffer()
    bake_occlusion(ground(), position, normal, mask, samples=1)

    assert len(workers) > 1


def test_the_reach_default_is_a_sane_fraction():
    assert 0.0 < DEFAULT_DISTANCE < 1.0
