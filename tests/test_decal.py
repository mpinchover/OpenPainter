"""Tests for normal-map decals: reading them, placing them, exporting them.

The decal is stamped into the atlas, so the two things that can go silently
wrong are conventions rather than crashes -- a decal that lands upside down, or
one whose depth control tips the normal past flat instead of steepening the
slope. Both are pinned here, along with the agreement between the shader that
does the work and the numpy mirror that documents it.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.decal import (  # noqa: E402
    MAX_EDGE,
    MAX_UV_ASPECT,
    DecalImage,
    DecalLoadError,
    composite_normal_map,
    edge_fade,
    height_to_normals,
    load_decal,
    uv_aspect,
)
from core.params import MAX_FALLOFF  # noqa: E402
from core.picking import face_at_uv  # noqa: E402
from core.export import export_maps, save_normal_map  # noqa: E402
from core.params import DecalArrayModifier, DecalParams  # noqa: E402

FLAT = np.array([0.5, 0.5, 1.0], dtype=np.float32)


def _projected_array(**changes) -> DecalParams:
    values = dict(
        path="array.png",
        projector_center=(0.0, 0.0, 0.0),
        projector_right=(1.0, 0.0, 0.0),
        projector_up=(0.0, 1.0, 0.0),
        projector_forward=(0.0, 0.0, 1.0),
        projector_size=(1.0, 1.0),
    )
    values.update(changes)
    return DecalParams(**values)


def test_axes_array_is_transient_and_offsets_in_selected_local_axes():
    from render.viewport import MeshMapApp

    source = _projected_array(
        modifiers=[DecalArrayModifier(
            count=2, offset_x=0.5, offset_y=-0.5,
        )],
    )
    instances = MeshMapApp.decal_instances(object.__new__(MeshMapApp), source)

    assert instances[0] is source
    assert [instance.projector_center for instance in instances] == [
        (0.0, 0.0, 0.0), (0.5, -0.5, 0.0), (1.0, -1.0, 0.0)
    ]
    assert not instances[1].modifiers


def test_radial_array_distributes_copies_around_local_axis():
    from render.viewport import MeshMapApp

    source = _projected_array(
        modifiers=[DecalArrayModifier(
            mode="radial", count=4, radius=2.0,
            radial_axis="z",
        )],
    )
    instances = MeshMapApp.decal_instances(object.__new__(MeshMapApp), source)
    centers = np.asarray([instance.projector_center for instance in instances])
    pivot = np.asarray(source.projector_center)

    assert np.linalg.norm(centers - pivot, axis=1) == pytest.approx([2.0] * 5)
    angles = np.unwrap(np.arctan2(centers[:, 1], centers[:, 0]))
    assert np.diff(angles) == pytest.approx([2.0 * np.pi / 5.0] * 4)
    assert centers[0] == pytest.approx((2.0, 0.0, 0.0))
    assert source.projector_center == (0.0, 0.0, 0.0), (
        "the editable source transform remains the circle center"
    )
    for instance in instances:
        inward = pivot - np.asarray(instance.projector_center)
        inward /= np.linalg.norm(inward)
        assert np.asarray(instance.projector_up) == pytest.approx(inward)
    assert instances[0].projector_up != source.projector_up, (
        "the rendered original must face inward instead of keeping its static basis"
    )

    # Equal angular positions at one radius also mean equal straight-line
    # distances between neighbours, including the closing edge to the source.
    closed = np.vstack((centers, centers[0]))
    gaps = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    assert gaps == pytest.approx([gaps[0]] * 5)


def test_radial_array_default_radius_is_point_one():
    assert DecalArrayModifier().radius == pytest.approx(0.1)


@pytest.mark.parametrize("mode", ["move", "scale", "rotate"])
def test_live_radial_transform_evaluates_ring_without_bare_center_decal(mode):
    """G/S/R uses the live transform as the ring pivot, not a rendered copy."""
    from render.viewport import MeshMapApp

    app = object.__new__(MeshMapApp)
    app._decal_transform_mode = mode
    app._live_decal_projector = {
        "center": (3.0, 4.0, 0.0),
        "right": (0.0, 1.0, 0.0),
        "up": (-1.0, 0.0, 0.0),
        "forward": (0.0, 0.0, 1.0),
        "size": (2.0, 3.0),
    }
    source = _projected_array(modifiers=[
        DecalArrayModifier(mode="radial", count=3, radius=0.5)
    ])

    live = app.live_decal_source(source)
    instances = app.decal_instances(live)
    centers = np.asarray([instance.projector_center for instance in instances])

    assert len(instances) == 4
    assert np.linalg.norm(centers - (3.0, 4.0, 0.0), axis=1) == pytest.approx(
        [0.5] * 4
    )
    assert not np.any(np.all(np.isclose(centers, (3.0, 4.0, 0.0)), axis=1))
    assert all(instance.projector_size == (2.0, 3.0) for instance in instances)


def test_array_stack_feeds_each_result_into_the_next_modifier():
    from render.viewport import MeshMapApp

    source = _projected_array(modifiers=[
        DecalArrayModifier(count=100, offset_x=0.1),
        DecalArrayModifier(count=2, offset_x=-0.1),
    ])
    instances = MeshMapApp.decal_instances(object.__new__(MeshMapApp), source)

    assert len(instances) == 303, "101 results, each expanded to three by modifier two"


def test_second_array_operates_on_copies_from_the_first_array():
    from render.viewport import MeshMapApp

    source = _projected_array(modifiers=[
        DecalArrayModifier(count=1, offset_x=1.0),
        DecalArrayModifier(count=1, offset_x=0.0, offset_y=2.0),
    ])
    instances = MeshMapApp.decal_instances(object.__new__(MeshMapApp), source)

    assert {instance.projector_center for instance in instances} == {
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (0.0, 2.0, 0.0), (1.0, 2.0, 0.0),
    }


def write_normal_map(path: Path, size=(32, 32), tilt=(0.0, 0.0)) -> Path:
    """An RGB normal map tilted by a fixed amount, for placement tests."""
    normal = np.array([tilt[0], tilt[1], 1.0], dtype=np.float32)
    normal /= np.linalg.norm(normal)
    encoded = np.round((normal * 0.5 + 0.5) * 255.0).astype(np.uint8)
    data = np.tile(encoded, (size[1], size[0], 1))
    Image.fromarray(data).save(path)
    return path


def slope_of(encoded: np.ndarray) -> np.ndarray:
    """Turn encoded normals back into the surface slope they describe."""
    normal = encoded * 2.0 - 1.0
    return normal[..., :2] / np.maximum(normal[..., 2:3], 1e-6)


def is_flat(pixels: np.ndarray, tolerance: float = 2e-3) -> np.ndarray:
    return np.all(np.abs(pixels - FLAT) < tolerance, axis=-1)


# --------------------------------------------------------------------------
# reading the image
# --------------------------------------------------------------------------

def test_an_rgb_image_is_read_as_a_normal_map(tmp_path):
    path = write_normal_map(tmp_path / "vent.png", tilt=(0.5, 0.0))
    image = load_decal(path)

    assert not image.from_height
    assert image.size == (32, 32)
    assert image.alpha.min() == 1.0
    assert slope_of(image.normals)[..., 0] == pytest.approx(0.5, abs=0.01)


def test_a_grayscale_image_is_read_as_a_height_map(tmp_path):
    """Decoding a grey pixel as a normal gives a vector along the diagonal --
    nonsense. A grayscale image can only sensibly be a height field."""
    path = tmp_path / "bump.png"
    height = np.zeros((32, 32), dtype=np.uint8)
    height[:, 16:] = 255  # a step halfway across
    Image.fromarray(height).save(path)

    image = load_decal(path)
    assert image.from_height
    assert np.allclose(image.normals[:, :4], FLAT, atol=1e-3), "flat where the height is"
    # The step faces the low side: a wall rising to the right tilts toward -u.
    assert slope_of(image.normals)[:, 15].mean() < -0.1


def test_a_normal_map_of_a_flat_surface_stays_flat(tmp_path):
    path = write_normal_map(tmp_path / "flat.png", tilt=(0.0, 0.0))
    assert np.allclose(load_decal(path).normals, FLAT, atol=1e-2)


def test_transparency_is_kept_as_coverage(tmp_path):
    path = tmp_path / "masked.png"
    data = np.zeros((16, 16, 4), dtype=np.uint8)
    data[..., :3] = (128, 128, 255)
    data[..., 3] = 255
    data[:8, :, 3] = 0  # the top half of the PNG is cut away
    Image.fromarray(data).save(path)

    image = load_decal(path)
    # Row 0 is v=0 here, and a PNG's row 0 is its top, so the hole is now high.
    assert image.alpha[-1].max() == 0.0
    assert image.alpha[0].min() == 1.0


def test_the_image_is_flipped_into_the_apps_row_order(tmp_path):
    """A PNG's first row is its top; ours is v=0. Getting this wrong exports a
    decal that is upside down against the same UVs everywhere else."""
    path = tmp_path / "marked.png"
    data = np.tile(np.array([128, 128, 255], np.uint8), (16, 16, 1))
    data[0] = (255, 128, 255)  # a marker along the top edge of the image
    Image.fromarray(data).save(path)

    image = load_decal(path)
    assert image.normals[-1, 0, 0] > 0.9, "the top of the PNG belongs at high v"
    assert image.normals[0, 0, 0] == pytest.approx(0.5, abs=0.01)


def test_an_oversized_image_is_brought_down_to_size(tmp_path):
    path = tmp_path / "huge.png"
    write_normal_map(path, size=(MAX_EDGE * 2, MAX_EDGE))
    assert max(load_decal(path).size) == MAX_EDGE


def test_unreadable_and_tiny_images_are_refused(tmp_path):
    missing = tmp_path / "nope.png"
    with pytest.raises(DecalLoadError):
        load_decal(missing)

    text = tmp_path / "notanimage.png"
    text.write_text("this is not a PNG")
    with pytest.raises(DecalLoadError):
        load_decal(text)

    single = tmp_path / "one.png"
    Image.fromarray(np.zeros((1, 1, 3), np.uint8)).save(single)
    with pytest.raises(DecalLoadError, match="too small"):
        load_decal(single)


def test_height_to_normals_is_unit_length_and_outward():
    rng = np.random.default_rng(3)
    normals = height_to_normals(rng.random((24, 24)).astype(np.float32))
    decoded = normals * 2.0 - 1.0
    assert np.allclose(np.linalg.norm(decoded, axis=-1), 1.0, atol=1e-5)
    assert (decoded[..., 2] > 0.0).all(), "a height field never faces backwards"


# --------------------------------------------------------------------------
# placing it in the atlas
# --------------------------------------------------------------------------

@pytest.fixture
def tilted(tmp_path) -> DecalImage:
    """A decal whose every texel slopes the same way, so placement is legible."""
    return load_decal(write_normal_map(tmp_path / "tilt.png", tilt=(1.0, 0.0)))


def test_with_no_decal_the_map_is_flat(tilted):
    blank = composite_normal_map(None, DecalParams(path="x"), 32)
    assert is_flat(blank).all()

    disabled = composite_normal_map(tilted, DecalParams(path="x", enabled=False), 32)
    assert is_flat(disabled).all()


def test_the_decal_lands_where_it_is_placed(tilted):
    params = DecalParams(path="x", center_u=0.25, center_v=0.75, scale=0.25)
    result = composite_normal_map(tilted, params, 64)

    marked = ~is_flat(result)
    rows, columns = np.nonzero(marked)
    # Row index is v, column is u, both scaled by the resolution.
    assert columns.mean() / 64 == pytest.approx(0.25, abs=0.02)
    assert rows.mean() / 64 == pytest.approx(0.75, abs=0.02)
    assert is_flat(result[0, 0]), "and nowhere near the far corner"


def test_scale_sets_how_much_of_the_atlas_is_covered(tilted):
    for scale in (0.2, 0.5, 0.8):
        # No edge fade: this is about how much of the sheet the rectangle
        # spans, and fading eats into the answer on purpose.
        params = DecalParams(path="x", scale=scale, falloff=0.0)
        covered = ~is_flat(composite_normal_map(tilted, params, 128))
        # A square decal covers scale^2 of the sheet, give or take a texel row.
        assert covered.mean() == pytest.approx(scale ** 2, abs=0.02)


def test_a_wide_decal_keeps_its_aspect_ratio(tmp_path):
    """Scale is the width; the height follows the image, so a vent stays wide."""
    wide = load_decal(write_normal_map(tmp_path / "wide.png", size=(64, 16), tilt=(1, 0)))
    covered = ~is_flat(composite_normal_map(
        wide,
        DecalParams(path="x", scale=0.8, falloff=0.0, image_aspect=wide.aspect),
        128,
    ))

    rows, columns = np.nonzero(covered)
    spread_u = np.ptp(columns) / 128
    spread_v = np.ptp(rows) / 128
    assert spread_u == pytest.approx(0.8, abs=0.03)
    assert spread_v / spread_u == pytest.approx(16 / 64, abs=0.05)


def test_a_transparent_texel_is_given_a_flat_normal(tmp_path):
    """Nothing samples a transparent texel directly -- the alpha multiplies it
    out -- but every bilinear tap and mipmap level around the artwork averages
    it in. Left as the white the artist's canvas happened to be, that averaging
    invents a 45-degree tilt all round the edge of the decal."""
    from PIL import Image

    canvas = np.zeros((16, 16, 4), np.uint8)
    canvas[..., :3] = 255          # white, the usual canvas
    canvas[6:10, 6:10] = [128, 128, 255, 255]   # a flat, opaque patch
    path = tmp_path / "on_white.png"
    Image.fromarray(canvas).save(path)

    image = load_decal(path)
    assert image.alpha[0, 0] == 0.0
    assert image.normals[0, 0] == pytest.approx([0.5, 0.5, 1.0]), "not white"
    assert image.normals[8, 8] == pytest.approx([128 / 255, 128 / 255, 1.0], abs=0.01)


# --------------------------------------------------------------------------
# fading the decal's own edge
# --------------------------------------------------------------------------

def test_the_edge_fade_leaves_the_middle_alone(tilted):
    """It takes the border off, not the subject: whatever is in the middle of
    the image has to come through untouched."""
    middle = 32
    hard = composite_normal_map(tilted, DecalParams(path="x", falloff=0.0), 64)
    faded = composite_normal_map(tilted, DecalParams(path="x", falloff=0.2), 64)

    assert faded[middle, middle] == pytest.approx(hard[middle, middle], abs=1e-6)


def test_the_edge_fade_takes_the_border_off(tilted):
    """The reported symptom: a decal drawn on a white canvas shows its own
    rectangle, because white decodes to a normal tilted 45 degrees each way and
    the surface beside it is flat."""
    params = DecalParams(path="x", scale=0.5, falloff=0.0)
    hard = composite_normal_map(tilted, params, 128)
    faded = composite_normal_map(tilted, replace(params, falloff=0.12), 128)

    # The texels just inside the rectangle's edge, where the seam lives.
    edge = slice(33, 36)
    assert not is_flat(hard[64, edge]).any(), "the canvas is stamped without it"
    assert is_flat(faded[64, edge]).all(), "and the fade has taken it away"
    assert not is_flat(faded[64, 64]), "while the middle is still stamped"


def test_more_falloff_eats_more_of_the_decal(tilted):
    covered = [
        (~is_flat(composite_normal_map(
            tilted, DecalParams(path="x", scale=0.8, falloff=falloff), 128
        ))).mean()
        for falloff in (0.0, 0.1, 0.25, 0.4)
    ]
    assert covered == sorted(covered, reverse=True)
    assert covered[0] > covered[-1] * 1.5


def test_the_fade_is_the_same_width_on_every_side():
    """Measured along whichever axis is nearer its edge, so the corners do not
    go first and leave the decal looking like a circle."""
    axis = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    u, v = np.meshgrid(axis, axis)
    fade = edge_fade(u, v, 0.2)

    assert fade[50, 50] == pytest.approx(1.0), "the middle survives"
    # The same distance in from each of the four sides gives the same answer.
    assert fade[50, 5] == pytest.approx(fade[50, -6])
    assert fade[5, 50] == pytest.approx(fade[-6, 50])
    assert fade[50, 5] == pytest.approx(fade[5, 50])


def test_the_fade_cuts_the_border_rather_than_only_softening_it():
    """A fade that merely reached zero *at* the border would leave no hard edge
    but would still show most of the canvas just inside it -- which is the whole
    reason the border is a problem."""
    axis = np.linspace(0.0, 1.0, 201, dtype=np.float32)
    u, v = np.meshgrid(axis, axis)
    fade = edge_fade(u, v, 0.1)

    # Everything in the outer tenth is gone outright, not merely dimmed.
    outer = np.maximum(np.abs(u - 0.5), np.abs(v - 0.5)) * 2.0 >= 0.9
    assert fade[outer].max() == 0.0
    # And the inner four fifths are untouched.
    inner = np.maximum(np.abs(u - 0.5), np.abs(v - 0.5)) * 2.0 <= 0.8
    assert fade[inner].min() == pytest.approx(1.0)


def test_no_falloff_changes_nothing():
    axis = np.linspace(0.0, 1.0, 33, dtype=np.float32)
    u, v = np.meshgrid(axis, axis)
    assert (edge_fade(u, v, 0.0) == 1.0).all()
    assert (edge_fade(u, v, -1.0) == 1.0).all(), "and nor does a negative one"


def test_the_fade_cannot_eat_the_whole_decal():
    axis = np.linspace(0.0, 1.0, 33, dtype=np.float32)
    u, v = np.meshgrid(axis, axis)
    assert edge_fade(u, v, 5.0)[16, 16] == pytest.approx(
        edge_fade(u, v, MAX_FALLOFF)[16, 16]
    )
    assert edge_fade(u, v, MAX_FALLOFF).max() > 0.0, "the middle always survives"


def test_intensity_scales_the_slope_not_the_vector(tilted):
    """The reason the control works on the slope: doubling it must double the
    steepness, where doubling the stored vector would run the normal flat."""
    slopes = []
    for intensity in (0.5, 1.0, 2.0, 4.0):
        params = DecalParams(path="x", scale=0.5, intensity=intensity)
        result = composite_normal_map(tilted, params, 64)
        middle = result[32, 32]
        slopes.append(float(slope_of(middle)[0]))

    assert slopes == pytest.approx([0.5, 1.0, 2.0, 4.0], rel=0.02)


def test_intensity_zero_is_the_same_as_no_decal(tilted):
    result = composite_normal_map(tilted, DecalParams(path="x", intensity=0.0), 32)
    assert is_flat(result).all()


def test_flipping_green_mirrors_the_v_slope(tmp_path):
    image = load_decal(write_normal_map(tmp_path / "v.png", tilt=(0.0, 0.75)))
    straight = composite_normal_map(image, DecalParams(path="x", scale=0.5), 64)
    flipped = composite_normal_map(
        image, DecalParams(path="x", scale=0.5, flip_green=True), 64
    )

    assert slope_of(straight[32, 32])[1] == pytest.approx(0.75, rel=0.02)
    assert slope_of(flipped[32, 32])[1] == pytest.approx(-0.75, rel=0.02)
    assert slope_of(straight[32, 32])[0] == pytest.approx(slope_of(flipped[32, 32])[0])


def test_rotation_turns_the_decal_and_its_slope_with_it(tilted):
    """A rotated decal has to light as though it were rotated -- the slope lives
    in the decal's frame and has to be carried into the atlas's."""
    upright = composite_normal_map(tilted, DecalParams(path="x", scale=0.5), 64)
    turned = composite_normal_map(
        tilted, DecalParams(path="x", scale=0.5, rotation=90.0), 64
    )

    # The decal slopes along u; turned a quarter turn, it must slope along v.
    assert slope_of(upright[32, 32]) == pytest.approx([1.0, 0.0], abs=0.02)
    assert slope_of(turned[32, 32]) == pytest.approx([0.0, 1.0], abs=0.02)


def test_the_normal_map_is_always_a_unit_normal(tilted):
    params = DecalParams(path="x", scale=0.6, intensity=3.0, rotation=37.0)
    decoded = composite_normal_map(tilted, params, 64) * 2.0 - 1.0
    assert np.allclose(np.linalg.norm(decoded, axis=-1), 1.0, atol=1e-4)
    assert (decoded[..., 2] > 0.0).all()


def test_a_decal_off_the_edge_of_the_atlas_is_simply_absent(tilted):
    params = DecalParams(path="x", center_u=3.0, center_v=3.0, scale=0.2)
    assert is_flat(composite_normal_map(tilted, params, 32)).all()


# --------------------------------------------------------------------------
# the shader that actually runs
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ctx():
    import moderngl

    try:
        context = moderngl.create_standalone_context(require=330)
    except Exception as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"no GL context available: {exc}")
    yield context
    context.release()


@pytest.fixture
def gpu_composite(ctx):
    """Run render/shaders/decal.frag, the way the viewport does."""
    import moderngl

    from render.shaders import load_shader

    program = ctx.program(
        vertex_shader=load_shader("fullscreen.vert"),
        fragment_shader=load_shader("decal.frag"),
    )
    program["u_decal"].value = 0
    vao = ctx.vertex_array(program, [])

    # The decal pass writes a slope; a second pass turns the total into a
    # normal map. Both are needed to compare against the numpy mirror, which
    # does the same two steps.
    encode = ctx.program(
        vertex_shader=load_shader("fullscreen.vert"),
        fragment_shader=load_shader("normal_encode.frag"),
    )
    encode["u_slope"].value = 0
    encode_vao = ctx.vertex_array(encode, [])

    def run(image: DecalImage, params: DecalParams, resolution: int) -> np.ndarray:
        texture = ctx.texture(
            image.size, 4, data=np.ascontiguousarray(image.rgba(), "f4").tobytes(),
            dtype="f4",
        )
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = texture.repeat_y = False
        slope = ctx.texture((resolution, resolution), 2, dtype="f4")
        slope_fbo = ctx.framebuffer(color_attachments=[slope])
        target = ctx.texture((resolution, resolution), 3, dtype="f4")
        fbo = ctx.framebuffer(color_attachments=[target])

        texture.use(0)
        for name, value in params.as_uniforms().items():
            program[name].value = value

        ctx.viewport = (0, 0, resolution, resolution)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        slope_fbo.use()
        slope_fbo.clear(0.0, 0.0, 0.0, 0.0)
        vao.render(moderngl.TRIANGLES, vertices=3)

        fbo.use()
        slope.use(0)
        encode_vao.render(moderngl.TRIANGLES, vertices=3)
        out = np.frombuffer(fbo.read(components=3, dtype="f4"), dtype="f4").reshape(
            resolution, resolution, 3
        )

        for resource in (fbo, target, slope_fbo, slope, texture):
            resource.release()
        return out.copy()

    yield run
    encode_vao.release()
    encode.release()
    vao.release()
    program.release()


def test_the_shader_agrees_with_the_numpy_mirror(gpu_composite, tmp_path):
    """Same placement, same slope maths, on both sides of the API."""
    rng = np.random.default_rng(11)
    height = rng.random((48, 32)).astype(np.float32)
    Image.fromarray((height * 255).astype(np.uint8)).save(tmp_path / "noise.png")
    image = load_decal(tmp_path / "noise.png")

    for params in (
        DecalParams(path="x", scale=0.5, image_aspect=image.aspect),
        DecalParams(path="x", scale=0.3, center_u=0.3, center_v=0.7, intensity=2.5,
                    image_aspect=image.aspect),
        DecalParams(path="x", scale=0.7, rotation=35.0, intensity=0.4,
                    image_aspect=image.aspect),
        DecalParams(path="x", scale=0.45, rotation=-120.0, flip_green=True,
                    image_aspect=image.aspect),
    ):
        gpu = gpu_composite(image, params, 64)
        cpu = composite_normal_map(image, params, 64)
        assert np.allclose(gpu, cpu, atol=6e-3), f"drift with {params}"


def test_the_shader_leaves_the_rest_of_the_atlas_flat(gpu_composite, tmp_path):
    image = load_decal(write_normal_map(tmp_path / "t.png", tilt=(1.0, 0.5)))
    result = gpu_composite(image, DecalParams(path="x", scale=0.25), 64)

    assert is_flat(result[0, 0]) and is_flat(result[-1, -1])
    assert not is_flat(result[32, 32])


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def _read(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.float32) / 255.0


def test_a_normal_map_round_trips_through_the_png(tmp_path, tilted):
    composed = composite_normal_map(tilted, DecalParams(path="x", scale=0.5), 32)
    path = save_normal_map(tmp_path / "normal.png", composed)

    image = Image.open(path)
    assert image.mode == "RGB"
    assert image.size == (32, 32)
    # Written flipped, so reading it back and flipping again returns the source.
    assert np.allclose(np.flipud(_read(path)), composed, atol=1.5 / 255)


def test_the_exported_normal_map_is_upright(tmp_path, tilted):
    """v=0 is the bottom row of our arrays and the *last* row of the PNG."""
    composed = composite_normal_map(
        tilted, DecalParams(path="x", scale=0.3, center_v=0.2), 64
    )
    pixels = _read(save_normal_map(tmp_path / "n.png", composed))

    marked = ~is_flat(pixels, tolerance=4e-3)
    assert marked[40:, :].any(), "a decal low in UV belongs low in the image"
    assert not marked[:24, :].any()


def test_export_writes_the_normal_map_alongside_the_bake(tmp_path, tilted):
    gradient = np.linspace(0.0, 1.0, 16 * 16).reshape(16, 16).astype(np.float32)
    composed = composite_normal_map(tilted, DecalParams(path="x", scale=0.5), 16)

    written = export_maps(tmp_path, normal=composed, occlusion=gradient)
    assert [path.name for path in written] == ["normal.png", "ao.png"]
    assert all(path.exists() for path in written)


def test_a_decal_can_be_exported_without_any_bake(tmp_path, tilted):
    """The decal is placed in UV space, so it owes the geometry pass nothing."""
    composed = composite_normal_map(tilted, DecalParams(path="x", scale=0.5), 16)
    written = export_maps(tmp_path, normal=composed)

    assert [path.name for path in written] == ["normal.png"]
    assert not (tmp_path / "ao.png").exists()


def test_the_normal_map_writer_refuses_the_wrong_shape(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        save_normal_map(tmp_path / "bad.png", np.zeros((8, 8), np.float32))
    with pytest.raises(ValueError, match="bits"):
        save_normal_map(tmp_path / "bad.png", np.zeros((8, 8, 3), np.float32), bits=12)


def read_rgb16_png(path: Path) -> np.ndarray:
    """Decode a 16-bit RGB PNG, since Pillow reads one back as 8-bit.

    Only handles what :func:`core.export.save_normal_map` writes: no filtering,
    no interlacing, one IDAT.
    """
    import struct
    import zlib

    blob = path.read_bytes()
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    chunks: dict[bytes, bytes] = {}
    offset = 8
    while offset < len(blob):
        length = struct.unpack(">I", blob[offset:offset + 4])[0]
        tag = blob[offset + 4:offset + 8]
        chunks[tag] = blob[offset + 8:offset + 8 + length]
        offset += 12 + length

    width, height, depth, colour = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert (depth, colour) == (16, 2), f"expected 16-bit truecolour, got {depth}/{colour}"

    raw = zlib.decompress(chunks[b"IDAT"])
    stride = width * 3 * 2 + 1
    rows = [
        np.frombuffer(raw[y * stride + 1:(y + 1) * stride], dtype=">u2")
        for y in range(height)
    ]
    assert all(raw[y * stride] == 0 for y in range(height)), "unfiltered rows only"
    return np.stack(rows).reshape(height, width, 3).astype(np.float32) / 65535.0


def test_sixteen_bit_really_is_sixteen_bit(tmp_path):
    """A shallow decal is exactly where 8-bit steps show, so the depth setting
    has to reach the normal map too -- which means writing the PNG by hand.

    The decal is a gentle dome: a slope that changes across the whole image, of
    the kind 8 bits reduces to a few flat terraces.
    """
    axis = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    u, v = np.meshgrid(axis, axis)
    dome = np.clip(1.0 - (u ** 2 + v ** 2), 0.0, 1.0)
    Image.fromarray((dome * 255).astype(np.uint8)).save(tmp_path / "dome.png")
    image = load_decal(tmp_path / "dome.png")

    composed = composite_normal_map(
        image, DecalParams(path="x", scale=0.9, intensity=0.02), 64
    )
    eight = save_normal_map(tmp_path / "8.png", composed, bits=8)
    sixteen = save_normal_map(tmp_path / "16.png", composed, bits=16)

    assert Image.open(eight).mode == "RGB"
    coarse = np.flipud(np.asarray(Image.open(eight), dtype=np.float32) / 255.0)
    fine = np.flipud(read_rgb16_png(sixteen))

    assert np.abs(fine - composed).max() < np.abs(coarse - composed).max() / 50
    assert np.abs(fine - composed).max() < 2.0 / 65535
    # 8-bit quantises this decal's whole gradient into a handful of levels.
    assert len(np.unique(fine[..., 1])) > len(np.unique(coarse[..., 1])) * 5


# --------------------------------------------------------------------------
# through the app itself
# --------------------------------------------------------------------------

def _uv_quad(path: Path) -> Path:
    """A square with a full 0..1 UV layout, so placement maps to the surface."""
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
        visual=trimesh.visual.TextureVisuals(
            uv=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        ),
    )
    mesh.export(path)
    return path


@pytest.fixture
def app(ctx, tmp_path):
    """The real window config, headless. Skips wherever GL is unavailable.

    The ImGui context is torn down afterwards. Each app creates its own, and a
    leftover one leaves its font atlas registered against a renderer that the
    next app's frames no longer use -- which surfaces as a texture the renderer
    has never heard of, several tests later.
    """
    import moderngl_window as mglw

    from render.viewport import MeshMapApp

    MeshMapApp.initial_mesh = str(_uv_quad(tmp_path / "quad.obj"))
    MeshMapApp.initial_resolution = 512
    try:
        instance = mglw.create_window_config_instance(MeshMapApp, args=["-wnd", "headless"])
    except Exception as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"no headless window available: {exc}")
    yield instance
    instance.controller.release()
    imgui.destroy_context()
    MeshMapApp.initial_mesh = None


def test_a_decal_reaches_the_export_without_any_bake(app, tmp_path):
    """The path the feature is actually used through: import, place, export."""
    write_normal_map(tmp_path / "vent.png", size=(64, 32), tilt=(0.8, 0.0))
    app.open_decal(tmp_path / "vent.png")
    assert app.selected_decal is not None, app.status

    app.selected_decal.scale = 0.5
    app.selected_decal.intensity = 2.0
    app.mark_normal_dirty()
    app.on_render(0.0, 1 / 60.0)

    composed = app.read_normal_map()
    assert composed is not None
    assert composed.shape == (512, 512, 3)
    assert not is_flat(composed[256, 256]), "the decal belongs in the middle"
    assert is_flat(composed[8, 8]), "and nowhere near the corner"

    app.export_dir = str(tmp_path / "out")
    app.export()
    written = sorted(p.name for p in (tmp_path / "out" / "quad").glob("*.png"))
    assert written == ["normal.png"], app.status
    assert "normal.png" in app.status


def test_switching_the_decal_off_flattens_the_map(app, tmp_path):
    write_normal_map(tmp_path / "vent.png", tilt=(1.0, 0.0))
    app.open_decal(tmp_path / "vent.png")
    app.on_render(0.0, 1 / 60.0)
    assert app.read_normal_map() is not None

    app.selected_decal.enabled = False
    app.mark_normal_dirty()
    app.on_render(0.0, 1 / 60.0)

    assert app.read_normal_map() is None, "nothing to export with the decal off"
    # The target itself is flattened too, so the viewport stops lighting with it.
    pixels = np.frombuffer(
        app.normal_fbo.read(components=3, dtype="f4"), dtype="f4"
    ).reshape(512, 512, 3)
    assert is_flat(pixels).all()


def test_the_normal_map_follows_the_export_resolution(app, tmp_path):
    write_normal_map(tmp_path / "vent.png", tilt=(1.0, 0.0))
    app.open_decal(tmp_path / "vent.png")
    app.on_render(0.0, 1 / 60.0)
    assert app.read_normal_map().shape[0] == 512

    app.controller.bake_params.resolution = 1024
    app.on_render(0.0, 1 / 60.0)
    assert app.read_normal_map().shape[0] == 1024


def test_a_bad_image_is_reported_rather_than_raised(app, tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_text("not a PNG at all")
    app.open_decal(broken)

    assert app.selected_decal is None
    assert app.status_is_error
    assert app.read_normal_map() is None


def test_the_decal_lights_the_mesh_in_the_viewport(app, tmp_path):
    """The preview builds its tangent frame per pixel, so this is the only
    check that the decal actually reaches the shading."""
    write_normal_map(tmp_path / "vent.png", tilt=(1.5, 0.0))
    app.open_decal(tmp_path / "vent.png")
    app.selected_decal.scale = 1.0  # cover the quad entirely
    app.selected_decal.intensity = 3.0
    app.mark_normal_dirty()

    def render() -> np.ndarray:
        for _ in range(2):
            app.on_render(0.0, 1 / 60.0)
        width, height = app.wnd.buffer_size
        return np.frombuffer(app.wnd.fbo.read(components=3), dtype=np.uint8).reshape(
            height, width, 3
        ).astype(np.int16)

    lit = render()
    app.selected_decal.enabled = False
    app.mark_normal_dirty()
    flat = render()

    changed = np.abs(lit - flat).max(axis=-1)
    assert (changed > 2).sum() > 500, "the decal has to change how the mesh lights"


# --------------------------------------------------------------------------
# placing it by pointing at the mesh
# --------------------------------------------------------------------------

@pytest.fixture
def placeable(app, tmp_path):
    """An app with a decal imported and a frame drawn, ready to place."""
    write_normal_map(tmp_path / "vent.png", tilt=(1.0, 0.0))
    app.open_decal(tmp_path / "vent.png")
    app.selected_decal.scale = 0.2
    app.on_render(0.0, 1 / 60.0)
    return app


def viewport_point(app, fraction_x: float, fraction_y: float) -> tuple[int, int]:
    """A cursor position inside the 3D view, as a fraction across it.

    Not across the window: the navigation bar and the sidebar take their share,
    and the model is laid out in what is left.
    """
    from ui.panel import NAVBAR_HEIGHT

    left, _, width, height = app.viewport_rect
    top = NAVBAR_HEIGHT * app.ui_pixel_scale
    ratio = app.wnd.pixel_ratio
    return (
        int((left + width * fraction_x) / ratio),
        int((top + height * fraction_y) / ratio),
    )


def test_the_cursor_reads_the_uv_under_it(placeable):
    """The quad fills the view, so the middle of the screen is the middle of it."""
    middle = placeable.surface_uv_at(viewport_point(placeable, 0.5, 0.5))
    assert middle is not None
    assert middle == pytest.approx((0.5, 0.5), abs=0.05)

    # And the corners of the mesh read as corners of its UV layout.
    lower_left = placeable.surface_uv_at(viewport_point(placeable, 0.5, 0.5))
    upper = placeable.surface_uv_at(viewport_point(placeable, 0.5, 0.42))
    assert upper is not None and upper[1] > lower_left[1], "up the screen is up in v"


def test_pointing_at_nothing_reads_nothing(placeable):
    """The very corner of the window is background, not mesh."""
    assert placeable.surface_uv_at((1.0, 1.0)) is None


def test_placing_moves_the_decal_to_the_cursor_and_drops_it(placeable):
    placeable.begin_decal_placement()
    assert placeable.decal_placing

    x, y = viewport_point(placeable, 0.44, 0.56)
    placeable.on_mouse_position_event(x, y, 0, 0)
    expected = placeable.surface_uv_at((x, y))
    assert expected is not None
    assert (placeable.selected_decal.center_u, placeable.selected_decal.center_v) == \
        pytest.approx(expected)

    placeable.on_mouse_press_event(x, y, placeable.wnd.mouse.left)
    assert not placeable.decal_placing, "the click ends the placement"
    assert (placeable.selected_decal.center_u, placeable.selected_decal.center_v) == \
        pytest.approx(expected), "and leaves it where it was dropped"


def test_pinch_zoom_is_ready_immediately_after_placing_selected_decal(placeable):
    placeable._pending_zoom = -4.0
    placeable._scroll_arrived = 4.0
    placeable._drag_owner = "decal"
    placeable.begin_decal_placement()
    placeable.end_decal_placement(keep=True)

    assert placeable.selected_decal is not None
    assert placeable._drag_owner is None
    assert placeable._pending_zoom == 0.0
    before = placeable.camera.radius
    placeable.on_pinch_zoom(0.2)
    placeable._drain_scroll()
    assert placeable.camera.radius < before


def test_the_placing_click_does_not_also_orbit_the_view(placeable):
    """A stroke that drops a decal must not be read as a camera drag as well."""
    before = np.array(placeable.camera.eye.to_list())

    placeable.begin_decal_placement()
    x, y = viewport_point(placeable, 0.5, 0.5)
    placeable.on_mouse_press_event(x, y, placeable.wnd.mouse.left)
    assert placeable._drag_owner == "decal"

    placeable.on_mouse_drag_event(x + 120, y + 90, 120, 90)
    assert np.allclose(np.array(placeable.camera.eye.to_list()), before)


def test_click_jitter_while_selecting_does_not_move_camera(placeable):
    """Selection's click tolerance must also be a camera dead zone."""
    x, y = viewport_point(placeable, 0.5, 0.5)
    placeable._drag_owner = "camera"
    placeable._press_at = (float(x), float(y))
    placeable.wnd.mouse_states.right = True
    before_radius = placeable.camera.radius

    try:
        placeable.on_mouse_drag_event(x + 2, y + 2, 2, 2)
    finally:
        placeable.wnd.mouse_states.right = False

    assert placeable.camera.radius == pytest.approx(before_radius)


def test_cancelling_puts_the_decal_back(placeable):
    origin = (placeable.selected_decal.center_u, placeable.selected_decal.center_v)

    placeable.begin_decal_placement()
    placeable.on_mouse_position_event(*viewport_point(placeable, 0.42, 0.6), 0, 0)
    assert (placeable.selected_decal.center_u, placeable.selected_decal.center_v) != \
        pytest.approx(origin), "it moved while being placed"

    placeable.end_decal_placement(keep=False)
    assert not placeable.decal_placing
    assert (placeable.selected_decal.center_u, placeable.selected_decal.center_v) == \
        pytest.approx(origin)


def test_escape_does_not_close_the_window(app):
    """moderngl-window closes on its exit_key, which ships as Escape and is
    tested before the key reaches the app at all. Escape has to mean 'cancel',
    so nothing may be left holding that binding."""
    assert app.wnd.exit_key is None
    assert not app.wnd.is_closing

    keys = app.wnd.keys
    app.on_key_event(keys.ESCAPE, keys.ACTION_PRESS, app.wnd.modifiers)
    assert not app.wnd.is_closing


def test_escape_cancels_a_placement(placeable):
    origin = (placeable.selected_decal.center_u, placeable.selected_decal.center_v)
    keys = placeable.wnd.keys

    placeable.begin_decal_placement()
    placeable.on_mouse_position_event(*viewport_point(placeable, 0.42, 0.6), 0, 0)
    placeable.on_key_event(keys.ESCAPE, keys.ACTION_PRESS, placeable.wnd.modifiers)

    assert not placeable.decal_placing
    assert (placeable.selected_decal.center_u, placeable.selected_decal.center_v) == \
        pytest.approx(origin)


def test_a_right_click_cancels_a_placement(placeable):
    origin = (placeable.selected_decal.center_u, placeable.selected_decal.center_v)

    placeable.begin_decal_placement()
    x, y = viewport_point(placeable, 0.42, 0.6)
    placeable.on_mouse_position_event(x, y, 0, 0)
    placeable.on_mouse_press_event(x, y, placeable.wnd.mouse.right)

    assert not placeable.decal_placing
    assert (placeable.selected_decal.center_u, placeable.selected_decal.center_v) == \
        pytest.approx(origin)


def test_dropping_it_off_the_mesh_keeps_the_last_good_spot(placeable):
    """What you see is what you get: the decal is visibly sitting somewhere, and
    a click in the background should not throw that away."""
    placeable.begin_decal_placement()
    x, y = viewport_point(placeable, 0.46, 0.54)
    placeable.on_mouse_position_event(x, y, 0, 0)
    landed = (placeable.selected_decal.center_u, placeable.selected_decal.center_v)

    placeable.on_mouse_position_event(2, 2, 0, 0)  # out over the background
    placeable.on_mouse_press_event(2, 2, placeable.wnd.mouse.left)

    assert not placeable.decal_placing
    assert (placeable.selected_decal.center_u, placeable.selected_decal.center_v) == \
        pytest.approx(landed)


def test_placement_needs_a_decal_and_a_mesh(app, tmp_path):
    app.begin_decal_placement()
    assert not app.decal_placing, "nothing imported yet"
    assert app.status_is_error

    write_normal_map(tmp_path / "vent.png", tilt=(1.0, 0.0))
    app.open_decal(tmp_path / "vent.png")
    app.begin_decal_placement()
    assert app.decal_placing


def test_placing_switches_a_disabled_decal_back_on(placeable):
    """Reaching for the placement tool means wanting to see the thing."""
    placeable.selected_decal.enabled = False
    placeable.begin_decal_placement()
    assert placeable.selected_decal.enabled


def test_decal_tree_name_is_editable_without_renaming_its_image(placeable):
    decal = placeable.selected_decal
    source_path = decal.path
    placeable.begin_decal_rename(placeable.decal_index)
    assert placeable.decal_renaming_opened

    decal.name = "  Riveted Vent  "
    placeable.end_decal_rename()

    assert decal.display_name() == "Riveted Vent"
    assert decal.path == source_path


def test_clearing_the_decal_ends_any_placement(placeable):
    placeable.begin_decal_placement()
    placeable.clear_decals()
    assert not placeable.decal_placing
    assert placeable.selected_decal is None and not placeable.decals


# --------------------------------------------------------------------------
# keyboard transforms
# --------------------------------------------------------------------------

def test_g_follows_surface_hits_from_one_finger_pointer_motion(placeable):
    decal = placeable.selected_decal
    origin = (decal.center_u, decal.center_v)
    placeable.surface_hit_at = lambda mouse: ((0.22, 0.81), 0)

    assert placeable.begin_decal_transform("move")
    placeable.transform_decal_with_pointer(12.0, -8.0)

    assert (decal.center_u, decal.center_v) == pytest.approx((0.22, 0.81))
    placeable.end_decal_transform(keep=False)
    assert (decal.center_u, decal.center_v) == pytest.approx(origin)


def test_g_keeps_moving_on_the_decal_plane_when_pointer_is_off_mesh(placeable):
    decal = placeable.selected_decal
    original = np.asarray(decal.projector_center, dtype=np.float64)
    plane_points = iter([
        np.asarray((1.0, 1.0, 0.0)),
        np.asarray((1.4, 0.8, 0.0)),
    ])
    placeable._cursor_plane_point = lambda *args: next(plane_points)

    assert placeable.begin_decal_transform("move")
    placeable.surface_hit_at = lambda mouse: None
    placeable.world_surface_hit_at = lambda mouse: None
    assert placeable.transform_decal_with_pointer(20.0, -10.0)

    assert placeable._live_decal_projector["center"] == pytest.approx(
        original + (0.4, -0.2, 0.0)
    )


def test_g_does_not_jump_when_pointer_reenters_the_mesh(placeable):
    """A surface hit must preserve the relative grab established off-mesh."""
    original = np.asarray(placeable.selected_decal.projector_center, dtype=np.float64)
    plane_points = iter([
        np.asarray((1.0, 1.0, 0.0)),
        np.asarray((1.4, 0.8, 0.0)),
    ])
    placeable._cursor_plane_point = lambda *args: next(plane_points)
    placeable.world_surface_hit_at = lambda mouse: None

    assert placeable.begin_decal_transform("move")
    placeable.surface_hit_at = lambda mouse: None
    assert placeable.transform_decal_with_pointer(20.0, -10.0)
    moved = original + (0.4, -0.2, 0.0)
    assert placeable._live_decal_projector["center"] == pytest.approx(moved)

    # The cursor has just crossed onto the same plane, but nowhere near the
    # decal. Its first hit establishes an offset instead of teleporting it.
    placeable.surface_hit_at = lambda mouse: ((0.8, 0.8), 0)
    placeable.world_surface_hit_at = lambda mouse: (np.asarray((4.0, 3.0, 0.0)), 0)
    assert placeable.transform_decal_with_pointer(2.0, 0.0)
    assert placeable._live_decal_projector["center"] == pytest.approx(moved)

    # Further surface travel is relative to that grab offset.
    placeable.world_surface_hit_at = lambda mouse: (np.asarray((4.2, 3.1, 0.0)), 0)
    assert placeable.transform_decal_with_pointer(2.0, 1.0)
    assert placeable._live_decal_projector["center"] == pytest.approx(
        moved + (0.2, 0.1, 0.0)
    )


def test_finishing_decal_transform_releases_gesture_owner(placeable):
    assert placeable.begin_decal_transform("move")
    placeable._drag_owner = "decal_transform"
    placeable.end_decal_transform(keep=True)

    assert placeable._drag_owner is None


def test_inspector_scale_and_rotation_update_the_committed_projector(placeable):
    decal = placeable.selected_decal
    previous = (
        decal.center_u, decal.center_v,
        decal.scale, decal.scale_x, decal.scale_y, decal.rotation,
    )
    old_size = np.asarray(decal.projector_size)
    old_right = np.asarray(decal.projector_right)

    decal.scale *= 1.5
    decal.rotation += 30.0
    placeable.sync_decal_inspector_projector(decal, previous)

    assert np.asarray(decal.projector_size) == pytest.approx(old_size * 1.5)
    assert not np.allclose(decal.projector_right, old_right)


def test_g_then_x_moves_only_across_u(placeable):
    decal = placeable.selected_decal
    origin = (decal.center_u, decal.center_v)

    placeable.begin_decal_transform("move")
    placeable.constrain_decal_transform("x")
    placeable.transform_decal_with_pointer(30.0, -20.0)

    assert decal.center_u > origin[0]
    assert decal.center_v == pytest.approx(origin[1])


def test_g_can_cross_to_a_different_mesh_face(placeable):
    decal = placeable.selected_decal
    decal.surface_face = 2
    hits = iter([((0.15, 0.25), 2), ((0.78, 0.66), 9)])
    converted = []
    placeable.surface_hit_at = lambda mouse: next(hits)
    placeable._surface_uv_on_wrap = lambda anchor, face, uv: (
        converted.append((anchor, face)) or uv
    )

    placeable.begin_decal_transform("move")
    placeable.transform_decal_with_pointer(8.0, 0.0)
    placeable.transform_decal_with_pointer(8.0, 0.0)

    assert (decal.center_u, decal.center_v) == pytest.approx((0.78, 0.66))
    assert converted == [(2, 2), (2, 9)]
    assert decal.surface_face == 2, "the chart stays stable while crossing the edge"


def test_s_then_y_scales_only_the_decal_height(placeable):
    decal = placeable.selected_decal
    before = decal.size()

    placeable.begin_decal_transform("scale")
    placeable.constrain_decal_transform("y")
    placeable.transform_decal_with_pointer(0.0, -30.0)
    after = decal.size()

    assert after[0] == pytest.approx(before[0])
    assert after[1] > before[1]
    placeable.end_decal_transform(keep=True)
    assert placeable._decal_transform_mode is None


def test_r_rotates_around_the_surface_normal_and_escape_restores_it(placeable):
    decal = placeable.selected_decal
    original_rotation = decal.rotation

    assert placeable.begin_decal_transform("rotate")
    placeable.transform_decal_with_pointer(30.0, 0.0)

    assert decal.rotation == pytest.approx((original_rotation - 15.0) % 360.0)
    placeable.end_decal_transform(keep=False)
    assert decal.rotation == pytest.approx(original_rotation)


def test_pressing_the_same_scale_constraint_again_returns_to_uniform_scale(placeable):
    decal = placeable.selected_decal
    original_scale = decal.scale

    placeable.begin_decal_transform("scale")
    placeable.constrain_decal_transform("x")
    placeable.transform_decal_with_pointer(30.0, 0.0)
    placeable.constrain_decal_transform("x")

    assert placeable._decal_transform_axis is None
    assert decal.scale > original_scale
    assert decal.scale_x == pytest.approx(1.0)
    assert decal.scale_y == pytest.approx(1.0)


def test_decal_transform_undo_and_redo(placeable):
    decal = placeable.selected_decal
    original_scale = decal.scale

    placeable.begin_decal_transform("scale")
    placeable.transform_decal_with_pointer(30.0, 0.0)
    changed_scale = decal.scale
    placeable.end_decal_transform(keep=True)

    assert changed_scale > original_scale
    assert placeable.undo_decal_action()
    assert placeable.selected_decal.scale == pytest.approx(original_scale)
    assert placeable.redo_decal_action()
    assert placeable.selected_decal.scale == pytest.approx(changed_scale)


def test_escape_restores_a_keyboard_transform(placeable):
    decal = placeable.selected_decal
    origin = (decal.center_u, decal.center_v)
    keys = placeable.wnd.keys

    placeable.on_key_event(keys.G, keys.ACTION_PRESS, placeable.wnd.modifiers)
    x, y = viewport_point(placeable, 0.42, 0.6)
    placeable.on_mouse_position_event(x, y, 10, 10)
    assert (decal.center_u, decal.center_v) != pytest.approx(origin)

    placeable.on_key_event(keys.ESCAPE, keys.ACTION_PRESS, placeable.wnd.modifiers)
    assert (decal.center_u, decal.center_v) == pytest.approx(origin)


def test_sidebar_shortcut_does_not_start_a_decal_transform(placeable):
    keys = placeable.wnd.keys
    placeable._mouse = (10.0, 100.0)
    assert not placeable.mouse_over_viewport()

    placeable.on_key_event(keys.S, keys.ACTION_PRESS, placeable.wnd.modifiers)

    assert placeable._decal_transform_mode is None


def test_transform_started_in_viewport_continues_across_sidebar(placeable):
    keys = placeable.wnd.keys
    decal = placeable.selected_decal
    before = decal.scale
    placeable._mouse = viewport_point(placeable, 0.5, 0.5)

    placeable.on_key_event(keys.S, keys.ACTION_PRESS, placeable.wnd.modifiers)
    assert placeable._decal_transform_mode == "scale"
    placeable.on_mouse_position_event(10, 100, 20, -20)

    assert decal.scale > before


def test_x_requests_confirmation_before_deleting_selected_decal(placeable):
    keys = placeable.wnd.keys
    count = len(placeable.decals)

    placeable.on_key_event(keys.X, keys.ACTION_PRESS, placeable.wnd.modifiers)

    assert placeable._delete_decal_index == placeable.decal_index
    assert len(placeable.decals) == count, "the shortcut itself must not delete"


@pytest.mark.parametrize(
    ("mode", "field", "value"),
    (("move", "center_u", 0.73), ("scale", "scale", 0.81),
     ("rotate", "rotation", 47.0)),
)
def test_completed_decal_transforms_use_app_undo_and_redo(
    placeable, mode, field, value
):
    decal = placeable.selected_decal
    original = getattr(decal, field)

    assert placeable.begin_decal_transform(mode)
    setattr(decal, field, value)
    placeable.end_decal_transform(keep=True)

    assert getattr(placeable.selected_decal, field) == pytest.approx(value)
    assert placeable.undo_action()
    assert getattr(placeable.selected_decal, field) == pytest.approx(original)
    assert placeable.redo_action()
    assert getattr(placeable.selected_decal, field) == pytest.approx(value)


def test_confirming_relative_g_keeps_radial_ring_at_live_preview(placeable):
    decal = placeable.selected_decal
    decal.modifiers = [DecalArrayModifier(mode="radial", count=5, radius=0.1)]
    x, y = viewport_point(placeable, 0.62, 0.54)
    surface = placeable.world_surface_hit_at((x, y))
    expected_uv = placeable.surface_uv_at((x, y))
    assert surface is not None and expected_uv is not None

    assert placeable.begin_decal_transform("move")
    placeable._live_decal_projector["center"] = tuple(surface[0])
    # Relative G can leave the cursor somewhere other than the projector
    # centre. Confirmation must commit the visible centre, not this cursor UV.
    placeable._decal_transform_last_hit = ((0.02, 0.97), int(surface[1]))
    live = placeable.live_decal_source(decal)
    before = np.asarray([
        instance.projector_center for instance in placeable.decal_instances(live)
    ])

    placeable.end_decal_transform(keep=True)
    after = np.asarray([
        instance.projector_center for instance in placeable.decal_instances(decal)
    ])

    assert (decal.center_u, decal.center_v) == pytest.approx(expected_uv, abs=1e-5)
    assert after == pytest.approx(before)


def test_duplicate_enters_move_and_is_one_undoable_action(placeable):
    original = placeable.selected_decal
    original_count = len(placeable.decals)

    assert placeable.duplicate_selected_decal()
    assert len(placeable.decals) == original_count + 1
    assert placeable._decal_transform_mode == "move"
    assert placeable.selected_decal is not original
    assert placeable.selected_decal == original

    placeable.selected_decal.center_u = 0.72
    placeable.end_decal_transform(keep=True)
    assert placeable.undo_action()
    assert len(placeable.decals) == original_count
    assert placeable.redo_action()
    assert len(placeable.decals) == original_count + 1
    assert placeable.selected_decal.center_u == pytest.approx(0.72)


def test_app_history_keeps_the_last_one_hundred_actions(placeable):
    """The stack must retain 100 distinct steps, not one coalesced snapshot."""
    original = placeable.checker_scale
    for value in range(1, 106):
        before = placeable._app_snapshot()
        placeable.checker_scale = float(value)
        placeable._record_app_undo(before)

    assert len(placeable._app_undo) == 100
    for _ in range(100):
        assert placeable.undo_action()
    assert placeable.checker_scale == pytest.approx(5.0)
    assert not placeable.undo_action()

    for _ in range(100):
        assert placeable.redo_action()
    assert placeable.checker_scale == pytest.approx(105.0)
    assert not placeable.redo_action()


def test_cancelling_duplicate_removes_the_unplaced_copy(placeable):
    original_count = len(placeable.decals)
    assert placeable.duplicate_selected_decal()
    placeable.end_decal_transform(keep=False)
    assert len(placeable.decals) == original_count
    assert placeable._decal_transform_mode is None


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

def test_every_field_changes_the_key():
    """The key is what a cache would be keyed on; a field missing from it is a
    control that silently stops updating."""
    base = DecalParams(path="a.png")
    variants = (
        DecalParams(path="a.png", enabled=False),
        DecalParams(path="b.png"),
        DecalParams(path="a.png", center_u=0.1),
        DecalParams(path="a.png", center_v=0.1),
        DecalParams(path="a.png", scale=0.9),
        DecalParams(path="a.png", scale_x=1.2),
        DecalParams(path="a.png", scale_y=1.2),
        DecalParams(path="a.png", rotation=12.0),
        DecalParams(path="a.png", intensity=2.0),
        DecalParams(path="a.png", flip_green=True),
    )
    keys = {base.key()} | {variant.key() for variant in variants}
    assert len(keys) == len(variants) + 1


def test_active_needs_both_an_image_and_the_switch():
    assert not DecalParams().active(), "no path means nothing to stamp"
    assert not DecalParams(path="a.png", enabled=False).active()
    assert DecalParams(path="a.png").active()


def test_size_follows_the_image_aspect():
    assert DecalParams(scale=0.6, image_aspect=1.0).size() == pytest.approx((0.6, 0.6))
    assert DecalParams(scale=0.6, image_aspect=2.0).size() == pytest.approx(
        (0.6, 0.3)
    ), "twice as wide as tall"
    assert DecalParams(scale=0.6, image_aspect=0.5).size() == pytest.approx((0.6, 1.2))


def test_axis_scale_factors_change_width_and_height_independently():
    params = DecalParams(scale=0.4, scale_x=2.0, scale_y=0.5)
    assert params.size() == pytest.approx((0.8, 0.2))


def test_size_undoes_the_surfaces_own_stretch():
    """A UV rectangle is only the same shape on the model when a unit of u
    covers as much of it as a unit of v. Where it does not, the rectangle has to
    be the other shape to come out right."""
    stretched = DecalParams(scale=0.4, surface_aspect=2.0)
    assert stretched.size() == pytest.approx((0.4, 0.8)), "u is the wide axis"

    squeezed = DecalParams(scale=0.4, surface_aspect=0.5)
    assert squeezed.size() == pytest.approx((0.4, 0.2))

    # Both aspects at once: a wide image on a stretched surface.
    assert DecalParams(
        scale=0.4, surface_aspect=1.5, image_aspect=2.0
    ).size() == pytest.approx((0.4, 0.3))


# --------------------------------------------------------------------------
# how stretched the surface is under a decal
# --------------------------------------------------------------------------

def _sheet(uvs, size=(1.0, 1.0)):
    """One world-space quad, mapped to whatever UV rectangle is asked for."""
    width, height = size
    vertices = np.array(
        [[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]], dtype=float
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return vertices, faces, np.asarray(uvs, dtype=float)


SQUARE_UVS = [[0, 0], [1, 0], [1, 1], [0, 1]]


def test_a_square_layout_measures_square():
    assert uv_aspect(*_sheet(SQUARE_UVS)) == pytest.approx(1.0)


def test_a_stretched_layout_is_measured():
    """Half the u range for the same surface means a unit of u covers twice the
    world -- which is exactly the starter cube's box unwrap, at 1.5."""
    narrow = [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]
    assert uv_aspect(*_sheet(narrow)) == pytest.approx(2.0)

    short = [[0, 0], [1, 0], [1, 0.5], [0, 0.5]]
    assert uv_aspect(*_sheet(short)) == pytest.approx(0.5)


def test_the_starter_cubes_own_layout_is_the_one_that_squashed_a_decal():
    """The 3x2 box unwrap gives each face a 1/3 x 1/2 cell of a square atlas, so
    a round decal came out 1.5 times too wide. This is that number."""
    from core.mesh_io import default_mesh

    mesh, _ = default_mesh()
    aspect = uv_aspect(
        np.asarray(mesh.vertices), np.asarray(mesh.faces), np.asarray(mesh.visual.uv)
    )
    assert aspect == pytest.approx(1.5)


def test_one_face_can_be_asked_about_on_its_own():
    """A layout can be dense in one island and stretched in the next, so a decal
    asks about the triangle it is actually sitting on."""
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0],       # square in UV
         [2, 0, 0], [3, 0, 0], [3, 1, 0]],      # stretched
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    uvs = np.array(
        [[0, 0], [1, 0], [1, 1],
         [0, 0], [0.5, 0], [0.5, 1]], dtype=float
    )

    assert uv_aspect(vertices, faces, uvs, face=0) == pytest.approx(1.0)
    assert uv_aspect(vertices, faces, uvs, face=1) == pytest.approx(2.0)


def test_the_mesh_wide_answer_is_a_geometric_mean():
    """It is a ratio: a face at 2 and a face at 0.5 are equal and opposite, and
    average to 1 rather than to 1.25."""
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0],
         [2, 0, 0], [3, 0, 0], [3, 1, 0]], dtype=float
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    # The same triangle stretched each way, over equal UV areas.
    uvs = np.array(
        [[0, 0], [0.5, 0], [0.5, 1],
         [0, 0], [1, 0], [1, 0.5]], dtype=float
    )
    assert uv_aspect(vertices, faces, uvs) == pytest.approx(1.0)


def test_an_absurd_layout_is_clamped_rather_than_obeyed():
    hair = [[0, 0], [1e-6, 0], [1e-6, 1], [0, 1]]
    assert uv_aspect(*_sheet(hair)) == pytest.approx(MAX_UV_ASPECT)


def test_a_layout_with_nothing_to_measure_reads_as_square():
    """Every fallback is 1.0, which changes nothing about the placement --
    better than a decal that vanishes because its rectangle came out zero."""
    collapsed = [[0.5, 0.5]] * 4
    assert uv_aspect(*_sheet(collapsed)) == 1.0

    vertices, faces, uvs = _sheet(SQUARE_UVS)
    assert uv_aspect(vertices, faces[:0], uvs) == 1.0
    assert uv_aspect(vertices, faces, uvs[:0]) == 1.0

    # A degenerate triangle in world space: UVs fine, surface collapsed.
    flat = np.zeros((4, 3))
    assert uv_aspect(flat, faces, uvs) == 1.0


# --------------------------------------------------------------------------
# which triangle a texel belongs to
# --------------------------------------------------------------------------

def test_a_texel_finds_the_triangle_it_is_in():
    _, faces, uvs = _sheet(SQUARE_UVS)
    assert face_at_uv(faces, uvs, 0.8, 0.2) == 0, "the lower-right triangle"
    assert face_at_uv(faces, uvs, 0.2, 0.8) == 1


def test_a_texel_in_the_gutter_belongs_to_nothing():
    """Between the charts there is no surface, and a decal centred there has no
    local answer to fall back on but the mesh's own."""
    _, faces, uvs = _sheet([[0, 0], [0.4, 0], [0.4, 0.4], [0, 0.4]])
    assert face_at_uv(faces, uvs, 0.9, 0.9) is None
    assert face_at_uv(faces, uvs, -0.1, 0.2) is None


def test_an_empty_layout_finds_nothing():
    _, faces, uvs = _sheet(SQUARE_UVS)
    assert face_at_uv(faces[:0], uvs, 0.5, 0.5) is None
    assert face_at_uv(faces, uvs[:0], 0.5, 0.5) is None


# --------------------------------------------------------------------------
# through the app: a round decal has to land round
# --------------------------------------------------------------------------

@pytest.fixture
def cube_app(tmp_path):
    """The app on its starter cube, whose box unwrap stretches u by 1.5."""
    import moderngl_window as mglw

    from render.viewport import MeshMapApp

    MeshMapApp.initial_mesh = None
    try:
        instance = mglw.create_window_config_instance(MeshMapApp, args=["-wnd", "headless"])
    except Exception as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"no headless window available: {exc}")
    yield instance
    instance.controller.release()
    imgui.destroy_context()


def test_a_round_decal_lands_round_on_the_starter_cube(cube_app, tmp_path):
    """The bug this was reported as: a circular vent came out an ellipse, half
    again as wide as it was tall, because a square of UV space is not a square
    of that cube's surface.
    """
    write_normal_map(tmp_path / "vent.png", size=(64, 64), tilt=(0.6, 0.0))
    cube_app.open_decal(tmp_path / "vent.png")
    cube_app.selected_decal.scale = 0.25
    cube_app.selected_decal.center_u = 1 / 6   # the middle of one face's cell
    cube_app.selected_decal.center_v = 0.75
    cube_app.on_render(0.0, 1 / 60.0)

    assert cube_app.selected_decal.surface_aspect == pytest.approx(1.5)

    width, height = cube_app.selected_decal.size()
    # World size is the UV size times what a unit of each axis covers: 1.5
    # across, 1.0 up, for every face of this unwrap.
    assert width * 1.5 == pytest.approx(height * 1.0), "square on the model"


def _two_islands(path: Path) -> Path:
    """Two quads in one atlas: one mapped square, one stretched twice as far in
    v, and empty space around both."""
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],          # island A
            [2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0],          # island B
        ], dtype=float),
        faces=np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]),
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=np.array([
            [0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4],      # square: 1.0
            [0.5, 0.0], [0.9, 0.0], [0.9, 0.2], [0.5, 0.2],      # squat:  0.5
        ], dtype=float)),
    )
    mesh.export(path)
    return path


@pytest.fixture
def islands_app(tmp_path):
    import moderngl_window as mglw

    from render.viewport import MeshMapApp

    MeshMapApp.initial_mesh = str(_two_islands(tmp_path / "islands.obj"))
    try:
        instance = mglw.create_window_config_instance(MeshMapApp, args=["-wnd", "headless"])
    except Exception as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"no headless window available: {exc}")
    yield instance
    instance.controller.release()
    imgui.destroy_context()
    MeshMapApp.initial_mesh = None


def test_the_correction_follows_the_decal_rather_than_being_remembered(
    islands_app, tmp_path
):
    """Measured from wherever the decal is now. Stored at placement time it
    would go quietly wrong the moment the decal was moved to another island --
    a layout can be square in one place and stretched in the next."""
    write_normal_map(tmp_path / "vent.png", size=(32, 32))
    islands_app.open_decal(tmp_path / "vent.png")

    def aspect_at(u: float, v: float) -> float:
        islands_app.selected_decal.center_u = u
        islands_app.selected_decal.center_v = v
        islands_app.on_render(0.0, 1 / 60.0)
        return islands_app.selected_decal.surface_aspect

    assert aspect_at(0.2, 0.2) == pytest.approx(1.0), "the square island"
    assert aspect_at(0.7, 0.1) == pytest.approx(0.5), "the squat one"
    assert aspect_at(0.2, 0.2) == pytest.approx(1.0), "and back again"


def test_a_decal_in_the_gutter_falls_back_to_the_whole_mesh(islands_app, tmp_path):
    """Between the charts there is no surface to measure, and the mesh's own
    average is a better guess than pretending the layout is square."""
    write_normal_map(tmp_path / "vent.png", size=(32, 32))
    islands_app.open_decal(tmp_path / "vent.png")

    islands_app.selected_decal.center_u = 0.98
    islands_app.selected_decal.center_v = 0.98
    islands_app.on_render(0.0, 1 / 60.0)

    whole_mesh = islands_app.measure_uv_aspect()
    assert islands_app.selected_decal.surface_aspect == pytest.approx(whole_mesh)
    assert 0.5 < whole_mesh < 1.0, "somewhere between its two islands"
