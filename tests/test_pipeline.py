"""Verification tests for the CPU half of the pipeline.

Import and unwrap, the numpy mirror of the EdgeWear001 node group, seam padding,
and the PNG writers. Everything here runs headless -- the GPU curvature bake has
its own file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.baking import dilate  # noqa: E402
from core.bevel import bevel  # noqa: E402
from core.edge_wear import (  # noqa: E402
    edge_wear,
    normalize_position,
    tex_noise,
    tex_noise_f,
    tex_noise_fbm,
)
from core.export import export_maps, export_textured_obj  # noqa: E402
from core.mesh_io import _fbx_unit_scale, load_mesh  # noqa: E402
from core.params import BakeParams, BevelParams, EdgeWearParams, UnwrapParams  # noqa: E402
from core.uv_unwrap import (  # noqa: E402
    SourceUVError,
    _welded_vertex_normals,
    UnwrapResult,
    source_uv_layout,
    source_uvs,
    unwrap,
    uv_density,
)

ASSETS = ROOT / "assets"


@pytest.fixture(scope="module")
def bracket() -> trimesh.Trimesh:
    mesh, _ = load_mesh(ASSETS / "sample.obj")
    return mesh


# --------------------------------------------------------------------------
# import
# --------------------------------------------------------------------------

def test_fbx_import_matches_obj(bracket):
    """The FBX's centimeter metadata is converted to Blender-style metres."""
    fbx = ASSETS / "sample.fbx"
    if not fbx.exists():
        pytest.skip("no FBX asset (assimp CLI unavailable when assets were built)")

    from_fbx, info = load_mesh(fbx)
    assert info.backend in ("assimp-cli", "pyassimp")
    assert len(from_fbx.vertices) == len(bracket.vertices)
    assert len(from_fbx.faces) == len(bracket.faces)
    unit_scale = _fbx_unit_scale(fbx)
    assert np.allclose(
        from_fbx.extents,
        bracket.extents * (unit_scale / 100.0),
        atol=1e-4,
    )


def test_import_welds_and_reports(bracket):
    assert bracket.is_watertight
    assert len(bracket.faces) > 0
    unique = np.unique(np.round(bracket.vertices, 6), axis=0)
    assert len(unique) == len(bracket.vertices)


def test_ascii_fbx_unit_scale_is_read(tmp_path):
    path = tmp_path / "units.fbx"
    path.write_text(
        '; FBX 7.4.0 project file\n'
        'P: "UnitScaleFactor", "double", "Number", "",100\n'
    )
    assert _fbx_unit_scale(path) == 100.0


# --------------------------------------------------------------------------
# bevel
# --------------------------------------------------------------------------

CUBE_AXES = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float
)


@pytest.fixture
def cube() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(2.0, 2.0, 2.0))


def test_bevel_is_a_no_op_when_off(cube):
    for params in (
        BevelParams(enabled=False),
        BevelParams(enabled=True, amount=0.0),
        BevelParams(enabled=True, angle=179.0),  # nothing is that sharp
    ):
        assert bevel(cube, params) is cube


def test_bevel_keeps_the_solid_closed(cube):
    for segments in (1, 2, 3, 6):
        result = bevel(cube, BevelParams(enabled=True, amount=0.05, segments=segments))
        assert result.is_watertight, f"segments={segments} left holes"
        assert result.is_winding_consistent
        assert result.volume < cube.volume, "a bevel only removes material"
        # Nothing may poke outside the original solid.
        assert np.abs(result.vertices).max() <= 1.0 + 1e-9


def test_bevel_offset_width_matches_blender(cube):
    """Offset width: each new boundary edge sits `amount` from the original.

    On a cube that means the six original faces survive as squares inset by
    `amount` on every side, so their total area pins the convention exactly.
    """
    amount = 0.05
    result = bevel(cube, BevelParams(enabled=True, amount=amount, segments=3))
    flat = (result.face_normals @ CUBE_AXES.T).max(axis=1) > 1.0 - 1e-9
    assert np.isclose(result.area_faces[flat].sum(), 6 * (2.0 - 2 * amount) ** 2, rtol=1e-9)


def test_bevel_volume_converges_on_the_circular_profile(cube):
    """More segments must approach the true rounded edge from below."""
    volumes = [
        bevel(cube, BevelParams(enabled=True, amount=0.05, segments=s)).volume
        for s in (2, 3, 6, 10)
    ]
    assert volumes == sorted(volumes), "a finer arc encloses more"
    # 12 edges of a quarter-circle fillet, ignoring the eight corner patches.
    exact = 8.0 - 12 * (0.05**2) * (1 - np.pi / 4) * (2.0 - 2 * 0.05)
    assert volumes[-1] == pytest.approx(exact, rel=2e-3)


def test_one_segment_is_a_flat_chamfer(cube):
    """Blender's 1-segment bevel is a chamfer, so the strips must be planar."""
    result = bevel(cube, BevelParams(enabled=True, amount=0.05, segments=1))
    # A cube chamfer has 6 axis faces + 12 edge planes + 8 corners = 26 planes.
    planes = np.unique(np.round(result.face_normals, 6), axis=0)
    assert len(planes) == 26


def test_bevel_only_touches_edges_sharper_than_the_angle():
    """A coplanar edge splitting a flat surface must be left alone."""
    # Two triangles forming a flat square: the shared diagonal is 0 degrees.
    flat = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    assert bevel(flat, BevelParams(enabled=True, amount=0.01, angle=30.0)) is flat


def test_bevel_confines_the_normal_gradient_to_the_strip(cube):
    """The whole reason the bevel exists.

    Without it, welding averages the 90-degree corner normals across the entire
    face and the bake lights the face up. With it, the big faces must come out
    flat to within a rounding error so they bake black.
    """
    before = _welded_vertex_normals(cube)[cube.faces]
    flat_normals = np.repeat(cube.face_normals[:, None, :], 3, axis=1)
    deviation = np.degrees(
        np.arccos(np.clip(np.einsum("fij,fij->fi", before, flat_normals), -1, 1))
    )
    assert deviation.max() > 45.0, "unbeveled, the gradient covers whole faces"

    result = bevel(cube, BevelParams(enabled=True, amount=0.001, segments=3))
    after = _welded_vertex_normals(result)[result.faces]
    flat = (result.face_normals @ CUBE_AXES.T).max(axis=1) > 1.0 - 1e-9
    face_normals = np.repeat(result.face_normals[:, None, :], 3, axis=1)
    deviation = np.degrees(
        np.arccos(np.clip(np.einsum("fij,fij->fi", after, face_normals), -1, 1))
    )
    assert deviation[flat].max() < 0.5, "the big faces must stay flat"
    assert deviation[~flat].max() > 45.0, "the strip carries the whole transition"


def test_uv_density_predicts_the_bevel_width(tmp_path):
    """The width a bevel lands at in the atlas, which is what decides if it bakes.

    A bevel thinner than a texel falls between samples and leaves nothing
    behind, so the UI warns using this number. Pin it against a cube whose
    density is computable by hand.
    """
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0)).unwrap()
    path = tmp_path / "uvbox.obj"
    box.export(path)
    mesh, info = load_mesh(path)

    # Six 2x2 faces packed into the unit square: the atlas covers a bit under
    # 1 UV unit over 24 m2 of surface, so density is a shade under 1/sqrt(24).
    assert info.uv_density == pytest.approx(uv_density(mesh))
    assert 0.9 / np.sqrt(24.0) < info.uv_density <= 1.0 / np.sqrt(24.0)

    # 1 mm on a 2 m cube at 1024 is far below a texel; 10 mm clears it.
    assert 0.001 * info.uv_density * 1024 < 1.0
    assert 0.010 * info.uv_density * 1024 > 2.0


def test_bevel_carries_uvs_into_the_island_rim(tmp_path):
    """New geometry must land inside the original UV islands, not beyond them.

    The bevel is baked but never exported, so its UVs have to fall within the
    layout the original mesh already addresses -- otherwise the wear would bake
    into a gutter no texel of the real mesh ever samples.
    """
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0)).unwrap()
    path = tmp_path / "uvbox.obj"
    box.export(path)
    mesh, _ = load_mesh(path)
    before = source_uvs(mesh)

    result = bevel(mesh, BevelParams(enabled=True, amount=0.02, segments=3))
    after = source_uvs(result)

    assert after is not None, "the bevel must not drop the UV map"
    assert len(result.vertices) > len(mesh.vertices)
    # The atlas must not grow: every new UV lies inside the original footprint.
    assert after.min() >= before.min() - 1e-6
    assert after.max() <= before.max() + 1e-6


# --------------------------------------------------------------------------
# the atlas the bake renders into
# --------------------------------------------------------------------------

def test_uv_seams_survive_import(tmp_path):
    """An OBJ's UV splits must reach the bake -- they are the atlas layout."""
    box = trimesh.creation.box().unwrap()
    path = tmp_path / "uvbox.obj"
    box.export(path)

    mesh, info = load_mesh(path)
    assert info.has_uvs
    assert source_uvs(mesh) is not None
    # Welding must not collapse the seam splits back onto the 8 cube corners.
    assert len(mesh.vertices) > 8
    # It is still a sealed cube underneath, and the note must not claim otherwise.
    assert info.watertight
    assert not any("watertight" in note for note in info.notes)


def test_source_uv_layout_passes_the_mesh_through(tmp_path):
    box = trimesh.creation.box().unwrap()
    path = tmp_path / "uvbox.obj"
    box.export(path)
    mesh, _ = load_mesh(path)

    result = source_uv_layout(mesh, 512)

    assert result.source == "source"
    assert np.array_equal(result.faces, mesh.faces)
    assert np.allclose(result.uvs, mesh.visual.uv, atol=1e-6)
    assert np.array_equal(result.vmapping, np.arange(len(mesh.vertices)))
    # Six flat cube faces, each its own island once the seams split the corners.
    assert result.chart_count == 6
    assert 0.0 < result.utilization <= 1.0
    assert np.allclose(np.linalg.norm(result.normals, axis=1), 1.0, atol=1e-5)


def test_source_uv_layout_refuses_a_mesh_without_uvs(bracket):
    with pytest.raises(SourceUVError, match="no UV map"):
        source_uv_layout(bracket, 512)


def test_unwrap_produces_valid_atlas(bracket):
    result = unwrap(bracket, UnwrapParams(), 512)

    assert result.chart_count > 0
    assert len(result.vertices) == len(result.uvs) == len(result.normals)
    assert result.uvs.min() >= 0.0 and result.uvs.max() <= 1.0
    assert len(result.vertices) >= len(bracket.vertices)
    assert len(result.faces) == len(bracket.faces)
    assert np.allclose(result.vertices, bracket.vertices[result.vmapping], atol=1e-5)
    assert result.uvs.max() - result.uvs.min() > 0.5


# --------------------------------------------------------------------------
# ArmorPaint's noise
# --------------------------------------------------------------------------

def test_noise_is_bounded_and_varies():
    rng = np.random.default_rng(0)
    p = rng.random((2000, 3)).astype(np.float32) * 4.0
    values = tex_noise_f(p)

    assert values.min() >= 0.0 and values.max() <= 1.0
    assert values.std() > 0.15, "a value noise field should actually vary"


def test_noise_is_deterministic():
    p = np.linspace(0.0, 5.0, 300, dtype=np.float32).reshape(100, 3)
    assert np.array_equal(tex_noise_f(p), tex_noise_f(p))


def test_noise_is_continuous_between_lattice_points():
    """Value noise interpolates, so tiny steps must give tiny changes."""
    base = np.array([[1.234, 2.345, 3.456]], dtype=np.float32)
    step = np.array([[1e-4, 0.0, 0.0]], dtype=np.float32)
    moved = float(tex_noise_f(base + step)[0])
    assert abs(moved - float(tex_noise_f(base)[0])) < 0.01


def test_fbm_octave_count_follows_detail():
    """ArmorPaint loops `ii <= n`, so detail 0 still evaluates one octave."""
    p = np.linspace(0.0, 3.0, 300, dtype=np.float32).reshape(100, 3)

    coarse = tex_noise_fbm(p, detail=0.0, roughness=0.7, lacunarity=5.0)
    fine = tex_noise_fbm(p, detail=5.0, roughness=0.7, lacunarity=5.0)

    assert coarse.min() >= 0.0 and coarse.max() <= 1.0
    assert fine.min() >= 0.0 and fine.max() <= 1.0
    # Detail 0 is a single octave, so it must equal the plain noise exactly.
    assert np.allclose(coarse, tex_noise_f(p))
    assert not np.allclose(coarse, fine)


def test_fbm_roughness_controls_the_high_frequency_content():
    p = np.linspace(0.0, 3.0, 900, dtype=np.float32).reshape(300, 3)
    smooth = tex_noise_fbm(p, detail=5.0, roughness=0.1, lacunarity=5.0)
    rough = tex_noise_fbm(p, detail=5.0, roughness=0.9, lacunarity=5.0)
    # Lower roughness damps the later (higher-frequency) octaves faster, so the
    # result tracks the base octave more closely.
    base = tex_noise_f(p)
    assert np.abs(smooth - base).mean() < np.abs(rough - base).mean()


def test_distortion_moves_the_sample_point():
    p = np.linspace(0.0, 3.0, 300, dtype=np.float32).reshape(100, 3)
    plain = tex_noise(p, 1.0, 5.0, 0.7, 5.0, distortion=0.0)
    warped = tex_noise(p, 1.0, 5.0, 0.7, 5.0, distortion=0.15)
    assert not np.allclose(plain, warped)
    assert warped.min() >= 0.0 and warped.max() <= 1.0


# --------------------------------------------------------------------------
# the EdgeWear001 formula
# --------------------------------------------------------------------------

def _flat(curvature: float, params: EdgeWearParams, size: int = 8) -> np.ndarray:
    """Evaluate over a constant position, so the noise is constant too."""
    return edge_wear(
        np.full((size, size), curvature, dtype=np.float32),
        np.zeros((size, size, 3), dtype=np.float32),
        params,
    )


def test_wear_is_curvature_minus_noise_times_contrast():
    """mask = clamp((curvature - noise * wear_amount) * contrast, 0, 1)."""
    params = EdgeWearParams(wear_amount=0.0, contrast=3.0)
    assert float(_flat(0.0, params)[0, 0]) == pytest.approx(0.0)
    assert float(_flat(0.2, params)[0, 0]) == pytest.approx(0.6)
    assert float(_flat(0.5, params)[0, 0]) == pytest.approx(1.0)  # clamped


def test_contrast_scales_and_clamps():
    for contrast in (1.0, 3.0, 8.0):
        params = EdgeWearParams(wear_amount=0.0, contrast=contrast)
        assert float(_flat(0.1, params)[0, 0]) == pytest.approx(min(0.1 * contrast, 1.0))


def test_wear_amount_erodes_the_curvature():
    """More wear amount subtracts more noise, so less survives."""
    curvature = np.full((32, 32), 0.6, dtype=np.float32)
    us = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    gx, gy = np.meshgrid(us, us)
    position = np.stack([gx, gy, np.zeros_like(gx)], axis=-1)

    means = [
        edge_wear(curvature, position, EdgeWearParams(wear_amount=a)).mean()
        for a in (0.0, 0.3, 0.6, 1.2)
    ]
    assert means == sorted(means, reverse=True), f"not monotonic: {means}"


def test_wear_breaks_up_a_uniform_curvature():
    """The whole point of the noise subtract: uniform in, patchy out."""
    curvature = np.full((64, 64), 0.5, dtype=np.float32)
    us = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    gx, gy = np.meshgrid(us, us)
    position = np.stack([gx, gy, np.zeros_like(gx)], axis=-1)

    uniform = edge_wear(curvature, position, EdgeWearParams(wear_amount=0.0))
    broken = edge_wear(curvature, position, EdgeWearParams(wear_amount=0.6))

    assert uniform.std() == pytest.approx(0.0, abs=1e-6)
    assert broken.std() > 0.05


def test_wear_output_is_bounded():
    rng = np.random.default_rng(7)
    curvature = rng.random((32, 32)).astype(np.float32)
    position = rng.random((32, 32, 3)).astype(np.float32)
    for params in (
        EdgeWearParams(),
        EdgeWearParams(contrast=10.0, wear_amount=0.0),
        EdgeWearParams(contrast=0.0),
        EdgeWearParams(wear_amount=2.0, value=5.0),
    ):
        out = edge_wear(curvature, position, params)
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_value_changes_the_noise_scale():
    """Group Input.Value feeds a x10 Math node into the noise Scale.

    Measured on a single octave: at the shipped detail of 5 with lacunarity 5
    the top octave runs at scale * 5^5, far above what a 256-sample scanline can
    resolve, so a frequency metric over the full fBM only sees aliasing.
    """
    line = np.zeros((256, 3), dtype=np.float32)
    line[:, 0] = np.linspace(0.0, 1.0, 256, dtype=np.float32)

    def crossings(value):
        noise = tex_noise(line, value * 10.0, detail=0.0, roughness=0.7,
                          lacunarity=5.0, distortion=0.0)
        centred = noise - noise.mean()
        return int(np.sum(np.diff(np.sign(centred)) != 0))

    assert crossings(0.2) < crossings(1.0) < crossings(3.0)

    # And it reaches the mask: a different Value gives a different result.
    curvature = np.full((64, 64), 0.5, dtype=np.float32)
    us = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    gx, gy = np.meshgrid(us, us)
    position = np.stack([gx, gy, np.zeros_like(gx)], axis=-1)
    assert not np.allclose(
        edge_wear(curvature, position, EdgeWearParams(value=0.2)),
        edge_wear(curvature, position, EdgeWearParams(value=3.0)),
    )


# --------------------------------------------------------------------------
# bposition
# --------------------------------------------------------------------------

def test_normalize_position_maps_the_bounding_box_to_unit_range():
    """ArmorPaint's bposition: (pos + hdim) / dim over the model bounds."""
    lower = np.array([-100.0, -100.0, -100.0], dtype=np.float32)
    extents = np.array([200.0, 200.0, 200.0], dtype=np.float32)
    corners = np.array([[[-100, -100, -100], [100, 100, 100], [0, 0, 0]]], dtype=np.float32)

    out = normalize_position(corners, lower, extents)
    assert np.allclose(out[0, 0], 0.0)
    assert np.allclose(out[0, 1], 1.0)
    assert np.allclose(out[0, 2], 0.5)


def test_normalize_position_is_unit_invariant():
    """A metre mesh and a centimetre mesh give the same bposition."""
    points = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    small = normalize_position(points, np.zeros(3), np.array([4.0, 4.0, 4.0]))
    large = normalize_position(points * 100.0, np.zeros(3), np.array([400.0, 400.0, 400.0]))
    assert np.allclose(small, large)


# --------------------------------------------------------------------------
# seam padding
# --------------------------------------------------------------------------

def test_dilate_fills_the_gutter():
    image = np.zeros((16, 16), dtype=np.float32)
    mask = np.zeros((16, 16), dtype=bool)
    image[6:10, 6:10] = 1.0
    mask[6:10, 6:10] = True

    padded = dilate(image, mask, iterations=2)
    assert padded[5, 6] == pytest.approx(1.0)
    assert padded[4, 6] == pytest.approx(1.0)
    assert padded[3, 6] == pytest.approx(0.0), "must not spread further than asked"
    assert np.allclose(padded[6:10, 6:10], 1.0), "covered texels must be untouched"


def test_dilate_handles_multichannel_and_borders():
    image = np.zeros((8, 8, 3), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=bool)
    image[0, 0] = (1.0, 0.5, 0.25)
    mask[0, 0] = True

    padded = dilate(image, mask, iterations=1)
    assert np.allclose(padded[1, 1], (1.0, 0.5, 0.25))
    assert padded.shape == image.shape


def test_dilate_noop_when_disabled():
    image = np.random.default_rng(0).random((8, 8)).astype(np.float32)
    mask = np.ones((8, 8), dtype=bool)
    assert np.array_equal(dilate(image, mask, 0), image)


# --------------------------------------------------------------------------
# parameter keying
# --------------------------------------------------------------------------

def test_curvature_key_covers_every_bake_parameter():
    """Every knob that changes the bake must invalidate it."""
    base = BakeParams()
    for field, value in (
        ("resolution", 2048), ("strength", 1.7), ("radius", 3.1),
        ("offset", 0.5), ("smooth", 4), ("axis", 2),
    ):
        other = BakeParams(**{field: value})
        assert other.curvature_key() != base.curvature_key(), f"{field} not keyed"

    # Dilation is a padding-stage concern and must NOT force a re-rasterise.
    assert BakeParams(dilation=9).curvature_key() == base.curvature_key()


def test_axis_vector_lookup():
    assert BakeParams(axis=0).axis_vector() == (0.0, 0.0, 0.0)  # XYZ = unmasked
    assert BakeParams(axis=2).axis_vector() == (0.0, 1.0, 0.0)  # Y
    assert BakeParams(axis=5).axis_vector() == (0.0, -1.0, 0.0)  # -Y


def test_shipped_defaults_match_edgewear001():
    """These are the values decoded out of EdgeWear001.arm, not our own taste."""
    bake, wear = BakeParams(), EdgeWearParams()
    assert (bake.strength, bake.radius, bake.offset) == (0.5, 2.0, -2.0)
    assert (wear.value, wear.wear_amount, wear.contrast) == (1.0, 0.6, 3.0)
    assert (wear.detail, wear.roughness, wear.lacunarity, wear.distortion) == (
        5.0, 0.7, 5.0, 0.15)


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def test_export_writes_both_maps(tmp_path):
    from PIL import Image

    gradient = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
    paths = export_maps(tmp_path, gradient, gradient, bits=8)

    assert [p.name for p in paths] == ["edge_wear.png", "curvature.png"]
    for path in paths:
        assert path.exists()

    # PNG row 0 is the top of the image, our arrays start at v=0, so the file
    # must come back flipped.
    written = np.asarray(Image.open(paths[0]), dtype=np.float32) / 255.0
    assert np.allclose(written, np.flipud(gradient), atol=1.0 / 255.0)


def test_export_16_bit_has_more_levels(tmp_path):
    from PIL import Image

    gradient = np.linspace(0.0, 1.0, 256 * 256, dtype=np.float32).reshape(256, 256)
    eight = export_maps(tmp_path / "8", gradient, gradient, bits=8)[0]
    sixteen = export_maps(tmp_path / "16", gradient, gradient, bits=16)[0]

    levels_8 = len(np.unique(np.asarray(Image.open(eight))))
    levels_16 = len(np.unique(np.asarray(Image.open(sixteen))))
    assert levels_8 <= 256
    assert levels_16 > levels_8 * 10


def test_export_rejects_odd_bit_depth(tmp_path):
    with pytest.raises(ValueError):
        export_maps(tmp_path, np.zeros((4, 4)), np.zeros((4, 4)), bits=12)


def test_export_textured_obj_writes_blender_package(tmp_path):
    mesh = UnwrapResult(
        vertices=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32
        ),
        normals=np.array([[0, 0, 1]] * 3, dtype=np.float32),
        uvs=np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        vmapping=np.arange(3, dtype=np.uint32),
        chart_count=1,
        utilization=1.0,
        atlas_size=(4, 4),
    )
    image = np.ones((4, 4), dtype=np.float32)

    paths = export_textured_obj(tmp_path, "sample", mesh, image, image)

    assert [path.name for path in paths] == [
        "sample.obj", "sample.mtl", "edge_wear.png", "curvature.png"
    ]
    obj = paths[0].read_text()
    mtl = paths[1].read_text()
    assert "mtllib sample.mtl" in obj
    assert "usemtl sample_edge_wear" in obj
    assert "f 1/1/1 2/2/2 3/3/3" in obj
    assert "map_Kd edge_wear.png" in mtl
