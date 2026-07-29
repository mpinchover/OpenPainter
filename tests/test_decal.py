"""Tests for normal-map decals: reading them, placing them, exporting them.

The decal is stamped into the atlas, so the two things that can go silently
wrong are conventions rather than crashes -- a decal that lands upside down, or
one whose depth control tips the normal past flat instead of steepening the
slope. Both are pinned here, along with the agreement between the shader that
does the work and the numpy mirror that documents it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.decal import (  # noqa: E402
    MAX_EDGE,
    DecalImage,
    DecalLoadError,
    composite_normal_map,
    height_to_normals,
    load_decal,
)
from core.export import export_maps, save_normal_map  # noqa: E402
from core.params import DecalParams  # noqa: E402

FLAT = np.array([0.5, 0.5, 1.0], dtype=np.float32)


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
        params = DecalParams(path="x", scale=scale)
        covered = ~is_flat(composite_normal_map(tilted, params, 128))
        # A square decal covers scale^2 of the sheet, give or take a texel row.
        assert covered.mean() == pytest.approx(scale ** 2, abs=0.02)


def test_a_wide_decal_keeps_its_aspect_ratio(tmp_path):
    """Scale is the width; the height follows the image, so a vent stays wide."""
    wide = load_decal(write_normal_map(tmp_path / "wide.png", size=(64, 16), tilt=(1, 0)))
    covered = ~is_flat(composite_normal_map(wide, DecalParams(path="x", scale=0.8), 128))

    rows, columns = np.nonzero(covered)
    spread_u = np.ptp(columns) / 128
    spread_v = np.ptp(rows) / 128
    assert spread_u == pytest.approx(0.8, abs=0.03)
    assert spread_v / spread_u == pytest.approx(16 / 64, abs=0.05)


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

    def run(image: DecalImage, params: DecalParams, resolution: int) -> np.ndarray:
        texture = ctx.texture(
            image.size, 4, data=np.ascontiguousarray(image.rgba(), "f4").tobytes(),
            dtype="f4",
        )
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = texture.repeat_y = False
        target = ctx.texture((resolution, resolution), 3, dtype="f4")
        fbo = ctx.framebuffer(color_attachments=[target])

        texture.use(0)
        for name, value in params.as_uniforms(image.aspect).items():
            program[name].value = value

        fbo.use()
        ctx.viewport = (0, 0, resolution, resolution)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        vao.render(moderngl.TRIANGLES, vertices=3)
        out = np.frombuffer(fbo.read(components=3, dtype="f4"), dtype="f4").reshape(
            resolution, resolution, 3
        )

        for resource in (fbo, target, texture):
            resource.release()
        return out.copy()

    yield run
    vao.release()
    program.release()


def test_the_shader_agrees_with_the_numpy_mirror(gpu_composite, tmp_path):
    """Same placement, same slope maths, on both sides of the API."""
    rng = np.random.default_rng(11)
    height = rng.random((48, 32)).astype(np.float32)
    Image.fromarray((height * 255).astype(np.uint8)).save(tmp_path / "noise.png")
    image = load_decal(tmp_path / "noise.png")

    for params in (
        DecalParams(path="x", scale=0.5),
        DecalParams(path="x", scale=0.3, center_u=0.3, center_v=0.7, intensity=2.5),
        DecalParams(path="x", scale=0.7, rotation=35.0, intensity=0.4),
        DecalParams(path="x", scale=0.45, rotation=-120.0, flip_green=True),
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

    written = export_maps(tmp_path, gradient, gradient, normal=composed)
    assert [path.name for path in written] == [
        "edge_wear.png", "curvature.png", "normal.png"
    ]
    assert all(path.exists() for path in written)


def test_a_decal_can_be_exported_without_any_bake(tmp_path, tilted):
    """The decal is placed in UV space, so it owes the geometry pass nothing."""
    composed = composite_normal_map(tilted, DecalParams(path="x", scale=0.5), 16)
    written = export_maps(tmp_path, normal=composed)

    assert [path.name for path in written] == ["normal.png"]
    assert not (tmp_path / "edge_wear.png").exists()


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
    assert app.decal_image is not None, app.status

    app.decal_params.scale = 0.5
    app.decal_params.intensity = 2.0
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

    app.decal_params.enabled = False
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

    assert app.decal_image is None
    assert app.status_is_error
    assert app.read_normal_map() is None


def test_the_decal_lights_the_mesh_in_the_viewport(app, tmp_path):
    """The preview builds its tangent frame per pixel, so this is the only
    check that the decal actually reaches the shading."""
    write_normal_map(tmp_path / "vent.png", tilt=(1.5, 0.0))
    app.open_decal(tmp_path / "vent.png")
    app.decal_params.scale = 1.0  # cover the quad entirely
    app.decal_params.intensity = 3.0
    app.show_map_inspector = False
    app.mark_normal_dirty()

    def render() -> np.ndarray:
        for _ in range(2):
            app.on_render(0.0, 1 / 60.0)
        width, height = app.wnd.buffer_size
        return np.frombuffer(app.wnd.fbo.read(components=3), dtype=np.uint8).reshape(
            height, width, 3
        ).astype(np.int16)

    lit = render()
    app.decal_params.enabled = False
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
    app.decal_params.scale = 0.2
    app.show_map_inspector = False
    app.on_render(0.0, 1 / 60.0)
    return app


def viewport_point(app, fraction_x: float, fraction_y: float) -> tuple[int, int]:
    """A cursor position, as a fraction across the window."""
    width, height = app.wnd.buffer_size
    return int(width * fraction_x), int(height * fraction_y)


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
    assert (placeable.decal_params.center_u, placeable.decal_params.center_v) == \
        pytest.approx(expected)

    placeable.on_mouse_press_event(x, y, placeable.wnd.mouse.left)
    assert not placeable.decal_placing, "the click ends the placement"
    assert (placeable.decal_params.center_u, placeable.decal_params.center_v) == \
        pytest.approx(expected), "and leaves it where it was dropped"


def test_the_placing_click_does_not_also_orbit_the_view(placeable):
    """A stroke that drops a decal must not be read as a camera drag as well."""
    before = np.array(placeable.camera.eye.to_list())

    placeable.begin_decal_placement()
    x, y = viewport_point(placeable, 0.5, 0.5)
    placeable.on_mouse_press_event(x, y, placeable.wnd.mouse.left)
    assert placeable._drag_owner == "decal"

    placeable.on_mouse_drag_event(x + 120, y + 90, 120, 90)
    assert np.allclose(np.array(placeable.camera.eye.to_list()), before)


def test_cancelling_puts_the_decal_back(placeable):
    origin = (placeable.decal_params.center_u, placeable.decal_params.center_v)

    placeable.begin_decal_placement()
    placeable.on_mouse_position_event(*viewport_point(placeable, 0.42, 0.6), 0, 0)
    assert (placeable.decal_params.center_u, placeable.decal_params.center_v) != \
        pytest.approx(origin), "it moved while being placed"

    placeable.end_decal_placement(keep=False)
    assert not placeable.decal_placing
    assert (placeable.decal_params.center_u, placeable.decal_params.center_v) == \
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
    origin = (placeable.decal_params.center_u, placeable.decal_params.center_v)
    keys = placeable.wnd.keys

    placeable.begin_decal_placement()
    placeable.on_mouse_position_event(*viewport_point(placeable, 0.42, 0.6), 0, 0)
    placeable.on_key_event(keys.ESCAPE, keys.ACTION_PRESS, placeable.wnd.modifiers)

    assert not placeable.decal_placing
    assert (placeable.decal_params.center_u, placeable.decal_params.center_v) == \
        pytest.approx(origin)


def test_a_right_click_cancels_a_placement(placeable):
    origin = (placeable.decal_params.center_u, placeable.decal_params.center_v)

    placeable.begin_decal_placement()
    x, y = viewport_point(placeable, 0.42, 0.6)
    placeable.on_mouse_position_event(x, y, 0, 0)
    placeable.on_mouse_press_event(x, y, placeable.wnd.mouse.right)

    assert not placeable.decal_placing
    assert (placeable.decal_params.center_u, placeable.decal_params.center_v) == \
        pytest.approx(origin)


def test_dropping_it_off_the_mesh_keeps_the_last_good_spot(placeable):
    """What you see is what you get: the decal is visibly sitting somewhere, and
    a click in the background should not throw that away."""
    placeable.begin_decal_placement()
    x, y = viewport_point(placeable, 0.46, 0.54)
    placeable.on_mouse_position_event(x, y, 0, 0)
    landed = (placeable.decal_params.center_u, placeable.decal_params.center_v)

    placeable.on_mouse_position_event(2, 2, 0, 0)  # out over the background
    placeable.on_mouse_press_event(2, 2, placeable.wnd.mouse.left)

    assert not placeable.decal_placing
    assert (placeable.decal_params.center_u, placeable.decal_params.center_v) == \
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
    placeable.decal_params.enabled = False
    placeable.begin_decal_placement()
    assert placeable.decal_params.enabled


def test_clearing_the_decal_ends_any_placement(placeable):
    placeable.begin_decal_placement()
    placeable.clear_decal()
    assert not placeable.decal_placing
    assert placeable.decal_image is None


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
    params = DecalParams(scale=0.6)
    assert params.size(1.0) == pytest.approx((0.6, 0.6))
    assert params.size(2.0) == pytest.approx((0.6, 0.3)), "twice as wide as tall"
    assert params.size(0.5) == pytest.approx((0.6, 1.2))
