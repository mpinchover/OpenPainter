"""Tests for the GPU stages: the curvature bake and the EdgeWear001 pass.

These need a real GL context. A standalone one is enough -- no window, no event
loop -- but on a headless CI box without a GPU they will skip rather than fail.

The curvature test is the important one. It builds a quad whose normal varies
linearly across UV space, which makes dFdx/dFdy exactly known, and checks the
shader against ArmorPaint's expression evaluated by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import moderngl  # noqa: E402

from core.baking import UVSpaceBaker, dilate  # noqa: E402
from core.edge_wear import edge_wear, tex_noise  # noqa: E402
from core.mesh_io import load_mesh  # noqa: E402
from core.params import BakeParams, EdgeWearParams, UnwrapParams  # noqa: E402
from core.uv_unwrap import unwrap  # noqa: E402
from render.shaders import load_shader  # noqa: E402

ASSETS = ROOT / "assets"
RESOLUTION = 256


@pytest.fixture(scope="module")
def ctx():
    try:
        context = moderngl.create_standalone_context(require=330)
    except Exception as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"no GL context available: {exc}")
    yield context
    context.release()


def armorpaint_curvature(raw: float, params: BakeParams) -> float:
    """ArmorPaint's expression from make_bake.c, evaluated on the CPU."""
    exponent = (1.0 / max(params.radius, 1e-4)) * 0.25
    return float(np.clip(raw ** exponent * params.strength * 2.0 + params.offset / 10.0, 0.0, 1.0))


@pytest.fixture
def ramp_baker(ctx):
    """A single UV-filling quad whose normal ramps linearly along u.

    Two triangles over the whole 0..1 UV square. The normal goes from
    ``start`` at u=0 to ``end`` at u=1, so after rasterisation the per-texel
    derivative is exactly ``(end - start) / resolution`` in x and zero in y.
    """
    baker = UVSpaceBaker(ctx)

    def run(start, end, resolution, params, smooth=0):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype="f4")
        uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="f4")
        normals = np.array([start, end, end, start], dtype="f4")
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype="u4")

        params.smooth = smooth
        baker.upload(vertices, normals, uvs, faces)
        return baker.rasterize(resolution, params)

    yield run
    baker.release()


# --------------------------------------------------------------------------
# the curvature bake
# --------------------------------------------------------------------------

def test_a_constant_normal_bakes_zero_curvature(ramp_baker):
    """A flat surface has no normal change, so no curvature. Anywhere."""
    params = BakeParams(strength=1.0, radius=1.0, offset=0.0)
    gbuffer = ramp_baker((0, 0, 1), (0, 0, 1), 64, params)

    assert gbuffer.mask.any()
    assert np.all(gbuffer.curvature[gbuffer.mask] == 0.0)


def test_curvature_matches_armorpaints_expression(ramp_baker):
    """The exact formula, against a derivative we know analytically.

    The normal is interpolated (not renormalised) between the two endpoints, so
    across `resolution` texels it changes by `end - start` in total and the
    per-texel derivative is that over the resolution.
    """
    resolution = 64
    start, end = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])

    for params in (
        BakeParams(strength=1.0, radius=1.0, offset=0.0),
        BakeParams(strength=0.5, radius=2.0, offset=-2.0),
        BakeParams(strength=2.0, radius=4.0, offset=1.0),
    ):
        gbuffer = ramp_baker(start, end, resolution, params)
        interior = gbuffer.mask.copy()
        interior[:2, :] = interior[-2:, :] = False  # quads straddling the border
        interior[:, :2] = interior[:, -2:] = False

        # The shader normalises n before differentiating, so do the same here.
        step = (end - start) / resolution
        mid = (start + end) / 2.0
        d_unit = step / np.linalg.norm(mid)  # ~unit-length normal at midpoint
        raw = float(np.dot(d_unit, d_unit))
        expected = armorpaint_curvature(raw, params)

        got = float(np.median(gbuffer.curvature[interior]))
        assert got == pytest.approx(expected, rel=0.35), (
            f"{params}: expected ~{expected:.4f}, got {got:.4f}"
        )


def test_strength_and_offset_lift_the_result(ramp_baker):
    start, end = (0, 0, 1), (0, 1, 0)

    def peak(strength=1.0, offset=0.0):
        params = BakeParams(strength=strength, radius=1.0, offset=offset)
        g = ramp_baker(start, end, 64, params)
        return float(np.median(g.curvature[g.mask]))

    assert peak(strength=0.5) < peak(strength=1.0) < peak(strength=1.5)
    assert peak(offset=-1.0) < peak(offset=0.0) < peak(offset=1.0)
    # Offset is applied as offset/10, exactly.
    assert peak(offset=1.0) - peak(offset=0.0) == pytest.approx(0.1, abs=1e-4)


def test_radius_is_an_exponent_not_a_distance(ramp_baker):
    """Larger radius means a smaller exponent, so a bigger lift."""
    start, end = (0, 0, 1), (0, 1, 0)

    values = []
    for radius in (0.5, 1.0, 2.0, 4.0):
        params = BakeParams(strength=1.0, radius=radius, offset=0.0)
        g = ramp_baker(start, end, 64, params)
        values.append(float(np.median(g.curvature[g.mask])))

    assert values == sorted(values), f"not monotonic in radius: {values}"
    assert values[0] < 0.1 < values[-1], "radius should span most of the range"


def test_axis_masking_multiplies_by_the_facing(ramp_baker):
    """ArmorPaint: `curvature *= dot(n, axis)` when a bake axis is chosen."""
    start, end = (0, 0, 1), (0, 1, 0)
    base = BakeParams(strength=1.0, radius=1.0, offset=0.0, axis=0)
    unmasked = ramp_baker(start, end, 64, base)

    # Axis index 3 is +Z. The normal ramps away from +Z, so the mask attenuates
    # by dot(n, +Z) = n.z -- which varies across the quad, so this has to be
    # compared per texel rather than on a summary statistic.
    masked = ramp_baker(start, end, 64, BakeParams(
        strength=1.0, radius=1.0, offset=0.0, axis=3))

    covered = unmasked.mask
    expected = unmasked.curvature[covered] * unmasked.normal[covered][:, 2]
    assert np.allclose(masked.curvature[covered], expected, atol=2e-3)

    # And the attenuation is real, not a no-op.
    assert masked.curvature[covered].mean() < unmasked.curvature[covered].mean() * 0.95


def test_smoothing_does_not_shift_the_overall_level(ramp_baker):
    """The blur redistributes; it must not brighten or darken the map."""
    start, end = (0, 0, 1), (0, 1, 0)
    params = BakeParams(strength=1.0, radius=2.0, offset=0.0)

    sharp = ramp_baker(start, end, 128, params, smooth=0)
    blurred = ramp_baker(start, end, 128, params, smooth=4)

    assert blurred.curvature[blurred.mask].mean() == pytest.approx(
        sharp.curvature[sharp.mask].mean(), rel=0.05
    )


# --------------------------------------------------------------------------
# real geometry
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def baked_bracket(ctx):
    mesh, _ = load_mesh(ASSETS / "sample.obj")
    result = unwrap(mesh, UnwrapParams(), RESOLUTION)
    baker = UVSpaceBaker(ctx)
    baker.upload(result.vertices, result.normals, result.uvs, result.faces)
    params = BakeParams(strength=1.0, radius=2.0, offset=0.0, smooth=1)
    gbuffer = baker.rasterize(RESOLUTION, params)
    yield mesh, result, gbuffer
    baker.release()


def test_gbuffer_shape_and_coverage(baked_bracket):
    _, _, gbuffer = baked_bracket
    assert gbuffer.position.shape == (RESOLUTION, RESOLUTION, 3)
    assert gbuffer.normal.shape == (RESOLUTION, RESOLUTION, 3)
    assert gbuffer.curvature.shape == (RESOLUTION, RESOLUTION)
    assert 0.3 < gbuffer.coverage < 1.0, "a packed atlas should cover most of the sheet"


def test_gbuffer_positions_land_on_the_mesh(baked_bracket):
    import trimesh

    mesh, _, gbuffer = baked_bracket
    positions = gbuffer.position[gbuffer.mask]
    lower, upper = mesh.bounds
    assert np.all(positions >= lower - 1e-4)
    assert np.all(positions <= upper + 1e-4)

    sample = positions[:: max(1, len(positions) // 300)]
    _, distances, _ = trimesh.proximity.closest_point_naive(mesh, sample)
    assert distances.max() < 1e-4


def test_gbuffer_normals_are_unit_length(baked_bracket):
    _, _, gbuffer = baked_bracket
    lengths = np.linalg.norm(gbuffer.normal[gbuffer.mask], axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-4)


def test_curvature_is_bounded_and_lands_on_the_edges(baked_bracket):
    _, _, gbuffer = baked_bracket
    values = gbuffer.curvature[gbuffer.mask]
    assert 0.0 <= values.min() and values.max() <= 1.0
    assert values.max() > 0.1, "the bracket's edges should register"
    # Edges are a minority of the surface, so most of it stays dark.
    assert float(np.mean(values < 0.1)) > 0.5


def test_rasterizer_reuses_targets_across_calls(baked_bracket, ctx):
    """Re-rasterising at the same resolution must not leak framebuffers."""
    _, result, _ = baked_bracket
    baker = UVSpaceBaker(ctx)
    baker.upload(result.vertices, result.normals, result.uvs, result.faces)
    params = BakeParams()
    first = baker.rasterize(128, params)
    second = baker.rasterize(128, params)
    assert np.array_equal(first.mask, second.mask)
    assert baker.rasterize(64, params).resolution == 64
    baker.release()


# --------------------------------------------------------------------------
# the EdgeWear001 pass
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wear_runner(ctx):
    program = ctx.program(
        vertex_shader=load_shader("fullscreen.vert"),
        fragment_shader=load_shader("edge_wear.frag"),
    )
    program["u_curvature"].value = 0
    program["u_position"].value = 1
    vao = ctx.vertex_array(program, [])

    def run(curvature, position, params, show_curvature=False):
        size = curvature.shape[0]
        tex_c = ctx.texture((size, size), 1, dtype="f4")
        tex_p = ctx.texture((size, size), 3, dtype="f4")
        target = ctx.texture((size, size), 1, dtype="f4")
        fbo = ctx.framebuffer(color_attachments=[target])

        tex_c.write(np.ascontiguousarray(curvature, dtype="f4").tobytes())
        tex_p.write(np.ascontiguousarray(position, dtype="f4").tobytes())
        tex_c.use(0)
        tex_p.use(1)
        program["u_showCurvature"].value = 1 if show_curvature else 0
        for name, value in params.as_uniforms().items():
            program[name].value = value

        fbo.use()
        ctx.viewport = (0, 0, size, size)
        vao.render(moderngl.TRIANGLES, vertices=3)
        out = np.frombuffer(fbo.read(components=1, dtype="f4"), dtype="f4").reshape(size, size)

        for resource in (fbo, target, tex_c, tex_p):
            resource.release()
        return out.copy()

    yield run
    vao.release()
    program.release()


def _grid(size=32):
    rng = np.random.default_rng(4)
    curvature = rng.random((size, size)).astype(np.float32)
    us = np.linspace(0.0, 1.0, size, dtype=np.float32)
    gx, gy = np.meshgrid(us, us)
    position = np.stack([gx, gy, np.full_like(gx, 0.25)], axis=-1)
    return curvature, position


def test_wear_shader_matches_the_numpy_mirror_without_noise(wear_runner):
    """With the noise weighted out, the two must agree to float32 rounding.

    The noise itself is a sine-based hash, which is chaotic enough that GPU and
    CPU transcendentals diverge; it gets its own statistical test below.
    """
    curvature, position = _grid()
    for params in (
        EdgeWearParams(wear_amount=0.0, contrast=3.0),
        EdgeWearParams(wear_amount=0.0, contrast=1.0),
        EdgeWearParams(wear_amount=0.0, contrast=8.0),
    ):
        gpu = wear_runner(curvature, position, params)
        cpu = edge_wear(curvature, position, params)
        assert np.allclose(gpu, cpu, atol=1e-5), f"drift with {params}"


def test_wear_shader_noise_agrees_statistically(wear_runner):
    """A sine hash cannot be compared bit for bit, so compare distributions."""
    curvature, position = _grid(64)
    params = EdgeWearParams()

    gpu = wear_runner(curvature, position, params)
    cpu = edge_wear(curvature, position, params)

    assert gpu.mean() == pytest.approx(cpu.mean(), abs=0.06)
    assert gpu.std() == pytest.approx(cpu.std(), abs=0.06)
    assert 0.0 <= gpu.min() and gpu.max() <= 1.0


def test_wear_shader_passes_curvature_through_when_asked(wear_runner):
    """The Curvature Texture preview mode must show the bake untouched."""
    curvature, position = _grid()
    out = wear_runner(curvature, position, EdgeWearParams(), show_curvature=True)
    assert np.allclose(out, curvature, atol=1e-6)


def test_wear_is_the_edgewear001_formula(wear_runner):
    """mask = clamp((curvature - noise * wear_amount) * contrast, 0, 1)."""
    size = 32
    position = np.zeros((size, size, 3), dtype=np.float32)  # constant -> constant noise
    params = EdgeWearParams(wear_amount=0.0, contrast=2.0)

    for level in (0.0, 0.1, 0.4, 0.9):
        curvature = np.full((size, size), level, dtype=np.float32)
        out = wear_runner(curvature, position, params)
        assert np.allclose(out, min(level * 2.0, 1.0), atol=1e-5)


def test_full_bake_through_the_wear_pass(baked_bracket, wear_runner):
    """End to end on real geometry: edges lit, most of the surface dark."""
    mesh, _, gbuffer = baked_bracket
    curvature = dilate(gbuffer.curvature, gbuffer.mask, 4)
    lower = np.asarray(mesh.bounds[0], dtype=np.float32)
    extents = np.asarray(mesh.extents, dtype=np.float32)
    position = dilate((gbuffer.position - lower) / extents, gbuffer.mask, 4)

    out = wear_runner(curvature, position, EdgeWearParams(contrast=3.0))
    values = out[gbuffer.mask]

    assert 0.0 <= values.min() and values.max() <= 1.0
    assert values.max() > 0.2, "nothing wore at all"
    assert float(np.mean(values < 0.05)) > 0.5, "wear should be a minority of the surface"
