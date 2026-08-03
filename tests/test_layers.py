"""The mask tree: its shape, and the passes that render it.

A mask decides between two things, and either of those can be another mask, so
the model is a binary tree of unbounded depth. Two properties carry the whole
feature: edits rebuild the tree rather than mutating it (a panel holding a stale
node must not be able to write into the live one), and rendering costs one pass
per node with a bounded number of targets however deep it goes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataclasses import replace  # noqa: E402

from core.layers import (
    EDGE_WEAR_KIND,
    MASK_KINDS,
    MAX_DEPTH,
    ColorSlot,
    MaskLayer,
    NoiseMaskParams,
    can_nest,
    convert_slot,
    depth,
    describe,
    kind_of,
    mask_at,
    mask_count,
    new_texture,
    path_labels,
    set_slot,
    slot_at,
    texture_key,
    walk,
)
from core.params import EdgeWearParams  # noqa: E402

RED = (1.0, 0.0, 0.0)
BLUE = (0.0, 0.0, 1.0)
GREEN = (0.0, 1.0, 0.0)


def nested_tree() -> MaskLayer:
    """Edge wear over red, with a noise deciding the black side."""
    return MaskLayer(
        kind="edge_wear",
        white=ColorSlot(RED),
        black=MaskLayer(kind="noise", white=ColorSlot(GREEN), black=ColorSlot(BLUE)),
    )


# --------------------------------------------------------------------------
# the shape of the tree
# --------------------------------------------------------------------------

def test_a_new_texture_is_a_single_colour():
    """A texture starts as the simplest thing that is still a texture."""
    texture = new_texture()
    assert isinstance(texture, ColorSlot)
    assert kind_of(texture) == "color"
    assert depth(texture) == 0 and mask_count(texture) == 0


def test_making_a_colour_into_a_mask_grows_the_tree_under_it():
    texture = ColorSlot(RED)
    as_mask = convert_slot(texture, "noise")

    assert isinstance(as_mask, MaskLayer) and as_mask.kind == "noise"
    assert depth(as_mask) == 1 and mask_count(as_mask) == 1
    # Plain white and black underneath, so the model shows the mask itself --
    # the thing just asked for, and the thing to judge before colouring it.
    assert as_mask.white.color == (1.0, 1.0, 1.0)
    assert as_mask.black.color == (0.0, 0.0, 0.0)


def test_a_mask_back_to_a_colour_takes_its_branches_with_it():
    tree = set_slot(convert_slot(ColorSlot(RED), "edge_wear"), ("white",), ColorSlot(RED))
    tree = set_slot(tree, ("black",), MaskLayer(kind="noise"))
    assert depth(tree) == 2

    collapsed = convert_slot(tree, "color")
    assert isinstance(collapsed, ColorSlot)
    assert collapsed.color == RED, "the white side is the colour just described"
    assert depth(collapsed) == 0


def test_switching_between_two_mask_kinds_keeps_the_branches():
    tree = MaskLayer(kind="noise", white=ColorSlot(RED), black=ColorSlot(BLUE))
    switched = convert_slot(tree, "edge_wear")

    assert switched.kind == "edge_wear"
    assert switched.white.color == RED and switched.black.color == BLUE


def test_an_unknown_kind_is_refused():
    with pytest.raises(KeyError):
        convert_slot(ColorSlot(RED), "terrazzo")


def test_paths_address_every_slot():
    tree = nested_tree()
    assert slot_at(tree, ()) is tree
    assert slot_at(tree, ("white",)).color == RED
    assert slot_at(tree, ("black", "white")).color == GREEN
    assert slot_at(tree, ("black", "black")).color == BLUE
    assert mask_at(tree, ("black",)).kind == "noise"


def test_a_path_through_a_colour_is_an_error():
    tree = nested_tree()
    with pytest.raises(KeyError):
        slot_at(tree, ("white", "white"))
    with pytest.raises(KeyError):
        mask_at(tree, ("white",))


def test_only_white_and_black_are_slots():
    with pytest.raises(KeyError):
        MaskLayer().slot("grey")


def test_editing_rebuilds_rather_than_mutates():
    """The panel edits by replacing. A tree handed out earlier must not change
    underneath whoever is holding it."""
    tree = nested_tree()
    edited = set_slot(tree, ("black", "white"), ColorSlot((0.5, 0.5, 0.5)))

    assert slot_at(edited, ("black", "white")).color == (0.5, 0.5, 0.5)
    assert slot_at(tree, ("black", "white")).color == GREEN, "the original stands"
    assert edited is not tree


def test_setting_the_root_replaces_the_whole_texture():
    """Including with a colour: the whole texture may be just one."""
    replacement = MaskLayer(kind="noise")
    assert set_slot(nested_tree(), (), replacement) is replacement

    flat = ColorSlot(RED)
    assert set_slot(nested_tree(), (), flat) is flat


def test_a_mask_can_replace_a_colour_and_a_colour_a_whole_subtree():
    tree = set_slot(MaskLayer(), ("white",), MaskLayer(kind="noise"))
    assert depth(tree) == 2

    flattened = set_slot(tree, ("white",), ColorSlot(RED))
    assert depth(flattened) == 1, "collapsing a branch takes its children with it"


def test_depth_counts_the_deepest_branch():
    assert depth(ColorSlot(RED)) == 0
    assert depth(MaskLayer()) == 1
    assert depth(nested_tree()) == 2

    stacked = MaskLayer()
    for _ in range(4):
        stacked = MaskLayer(white=stacked)
    assert depth(stacked) == 5
    assert mask_count(stacked) == 5


def test_walk_visits_parents_before_children_and_white_before_black():
    paths = [path for path, _ in walk(nested_tree())]
    assert paths == [
        (), ("white",), ("black",), ("black", "white"), ("black", "black"),
    ]


def test_nesting_stops_at_the_cap():
    tree = MaskLayer()
    path: tuple[str, ...] = ()
    for _ in range(MAX_DEPTH - 1):
        path = path + ("white",)
        tree = set_slot(tree, path, MaskLayer(kind="noise"))

    assert can_nest(tree, path + ("white",)) is False, "one more would pass the cap"
    assert can_nest(tree, ("black",)) is True, "a shallow slot is still free"


def test_switching_kind_keeps_both_sets_of_settings():
    """Dialling in wear, trying noise, and going back should find the wear
    settings where they were left."""
    from dataclasses import replace

    node = MaskLayer(kind="edge_wear")
    node.edge_wear.contrast = 7.5
    node.noise.scale = 33.0

    as_noise = replace(node, kind="noise")
    back = replace(as_noise, kind="edge_wear")
    assert back.edge_wear.contrast == 7.5
    assert as_noise.noise.scale == 33.0


def test_a_slot_can_be_named_and_un_named():
    """A name is for the parts worth naming; everything else names itself."""
    colour = ColorSlot((1.0, 0.0, 0.0))
    assert describe(colour) == "#FF0000"

    colour.name = "Rust"
    assert describe(colour) == "Rust"
    assert colour.auto_label == "#FF0000", "what it goes back to"

    colour.name = ""
    assert describe(colour) == "#FF0000"


def test_a_name_survives_a_change_of_kind():
    """It names the part of the texture, not the kind of thing it is today."""
    colour = ColorSlot(RED, name="Rust")

    as_mask = convert_slot(colour, "noise")
    assert as_mask.name == "Rust" and describe(as_mask) == "Rust"
    assert as_mask.auto_label == "Noise"

    as_wear = convert_slot(as_mask, "edge_wear")
    assert as_wear.name == "Rust"

    back = convert_slot(as_wear, "color")
    assert back.name == "Rust"


def test_labels_say_what_a_slot_is():
    assert describe(ColorSlot((1.0, 0.0, 0.0))) == "#FF0000"
    assert describe(MaskLayer(kind="noise")) == MASK_KINDS["noise"]

    crumbs = path_labels(nested_tree(), ("black",))
    assert crumbs == ["Edge wear", "Black: Noise"]


def test_only_edge_wear_needs_the_bake():
    assert MaskLayer(kind="edge_wear").needs_bake
    assert not MaskLayer(kind="noise").needs_bake


def test_every_requested_procedural_generator_is_available():
    requested = {
        "noise", "grunge", "scratches", "brushed_metal", "cells", "clouds",
        "directional_streaks", "gradient", "brick", "wood_grain", "marble",
    }
    assert requested <= set(MASK_KINDS)
    for kind in requested:
        converted = convert_slot(ColorSlot(RED), kind)
        assert isinstance(converted, MaskLayer) and converted.kind == kind


def test_a_colour_leaf_carries_every_requested_material_channel():
    material = ColorSlot(
        RED, metallic=0.8, roughness=0.2, alpha=0.6,
        ambient_occlusion=0.4, emission=3.0,
    )
    assert material.color == RED
    assert material.channels() == (0.8, 0.2, 0.6, 0.4, 3.0)
    assert texture_key(material) != texture_key(replace(material, ambient_occlusion=1.0))


def test_noise_params_reach_the_shader_by_name():
    uniforms = NoiseMaskParams(scale=3.0).as_uniforms()
    assert uniforms["u_noiseScale"] == 3.0
    assert set(uniforms) == {
        "u_noiseScale", "u_noiseDetail", "u_noiseRoughness", "u_noiseLacunarity",
        "u_noiseDistortion", "u_noiseBias", "u_noiseContrast",
    }


def test_generator_specific_values_are_render_state():
    """Every control shown for a specialised generator must invalidate it."""
    import copy

    base = MaskLayer(kind="scratches")
    for field in (
        "scratch_width", "scratch_length", "scratch_irregularity",
        "brush_density", "brush_waviness", "brush_variation",
        "cell_jitter", "cell_edge", "streak_length", "streak_width",
        "mortar_thickness", "brick_aspect", "vein_width",
    ):
        changed = copy.deepcopy(base)
        setattr(changed.noise, field, getattr(changed.noise, field) + 0.01)
        assert texture_key(changed) != texture_key(base), field


# --------------------------------------------------------------------------
# rendering it
# --------------------------------------------------------------------------

RESOLUTION = 64


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
def compositor(ctx):
    from render.composite import LayerCompositor

    instance = LayerCompositor(ctx)
    yield instance
    instance.release()


@pytest.fixture
def inputs(ctx):
    """A curvature ramp along u, and positions spanning the bounding box."""
    curvature = np.tile(np.linspace(0.0, 1.0, RESOLUTION, dtype="f4"), (RESOLUTION, 1))
    axis = np.linspace(0.0, 1.0, RESOLUTION, dtype="f4")
    gx, gy = np.meshgrid(axis, axis)
    position = np.stack([gx, gy, np.full_like(gx, 0.3)], axis=-1).astype("f4")

    tex_curvature = ctx.texture((RESOLUTION, RESOLUTION), 1, data=curvature.tobytes(), dtype="f4")
    tex_position = ctx.texture((RESOLUTION, RESOLUTION), 3, data=position.tobytes(), dtype="f4")
    yield tex_curvature, tex_position
    tex_curvature.release()
    tex_position.release()


def render(compositor, inputs, tree) -> np.ndarray:
    compositor.render(tree, RESOLUTION, *inputs)
    return compositor.read()


def plain_wear(**colors) -> MaskLayer:
    """Wear with the noise weighted out, so the mask is the curvature itself."""
    tree = MaskLayer(kind="edge_wear", **colors)
    tree.edge_wear.wear_amount = 0.0
    tree.edge_wear.contrast = 1.0
    return tree


def test_the_boundary_is_a_boundary_by_default(compositor, inputs):
    """Every texel belongs to one side or the other -- no blend of the two.

    A mask is a continuous field, so mixing by it directly bleeds the colours
    into each other everywhere it sits mid-way. Cut at the threshold, the only
    colours on the surface are the ones that were put there.
    """
    out = render(compositor, inputs, plain_wear(
        white=ColorSlot(RED), black=ColorSlot(BLUE)
    ))

    colours = np.unique(out.round(3).reshape(-1, 3), axis=0)
    assert len(colours) == 2, f"expected two colours, got {colours}"
    assert sorted(map(tuple, colours)) == [BLUE, RED]


def test_softness_is_what_brings_the_blend_back(compositor, inputs):
    tree = plain_wear(white=ColorSlot(RED), black=ColorSlot(BLUE))
    tree.softness = 0.25
    out = render(compositor, inputs, tree)

    colours = np.unique(out.round(2).reshape(-1, 3), axis=0)
    assert len(colours) > 2, "a band of intermediate mixes"
    # And it is a band, not the whole surface: both ends are still pure.
    assert np.any(np.all(np.abs(out - RED) < 1e-3, axis=-1))
    assert np.any(np.all(np.abs(out - BLUE) < 1e-3, axis=-1))


def test_the_threshold_moves_where_the_sides_divide(compositor, inputs):
    """The curvature ramps along u, so the split slides along with it."""
    def white_share(threshold: float) -> float:
        tree = plain_wear(white=ColorSlot(RED), black=ColorSlot(BLUE))
        tree.threshold = threshold
        out = render(compositor, inputs, tree)
        return float(np.mean(np.all(np.abs(out - RED) < 1e-3, axis=-1)))

    assert white_share(0.2) > white_share(0.5) > white_share(0.8)
    assert white_share(0.5) == pytest.approx(0.5, abs=0.05)


GOLD = ColorSlot(RED, metallic=1.0, roughness=0.2, alpha=0.4, emission=2.0)


def test_a_colour_carries_a_whole_surface():
    """Colour is most of a material, not all of it."""
    plain = ColorSlot(RED)
    assert plain.material() == (0.0, 0.5, 1.0, 0.0), "dielectric, half rough, solid, dark"
    assert GOLD.material() == (1.0, 0.2, 0.4, 2.0)


def test_the_surface_channels_are_part_of_what_gets_re_rendered():
    """They change what is drawn, so they belong in the key that notices."""
    plain = ColorSlot(RED)
    for field, value in (
        ("metallic", 1.0), ("roughness", 0.1), ("alpha", 0.5), ("emission", 3.0)
    ):
        changed = replace(plain, **{field: value})
        assert texture_key(changed) != texture_key(plain), field


def test_the_mask_blends_the_surface_the_way_it_blends_the_colour(compositor, inputs):
    """Metallic, roughness, alpha and emission ride through the tree with the
    colour they belong to -- one pass, two attachments."""
    compositor.render(
        plain_wear(white=GOLD, black=ColorSlot(BLUE)), RESOLUTION, *inputs
    )
    material = compositor.read_material()

    assert material.shape == (RESOLUTION, RESOLUTION, 4)
    assert material[32, -1] == pytest.approx(GOLD.material(), abs=0.01), "white's side"
    assert material[32, 0] == pytest.approx(ColorSlot(BLUE).material(), abs=0.01)


def test_a_flat_colour_fills_the_surface_maps_too(compositor, inputs):
    compositor.render(GOLD, RESOLUTION, *inputs)
    material = compositor.read_material()
    assert np.allclose(material, GOLD.material(), atol=0.01)


def test_emission_survives_being_re_rendered(compositor, inputs):
    """Emission is the one channel that goes above 1, and the material map has
    to hold it after the first pass as well as during it.

    Binding any other framebuffer -- the per-node thumbnail blit, for one --
    turns fragment-colour clamping on for a float target, and moderngl exposes
    no way to turn it back off. The channel is stored as a fraction of
    MAX_EMISSION for that reason; this is what notices if that stops happening.
    """
    from core.layers import MAX_EMISSION

    bright = ColorSlot(RED, emission=MAX_EMISSION * 0.875)
    for _ in range(3):
        compositor.render(bright, RESOLUTION, *inputs)
        emission = compositor.read_material()[..., 3]
        assert emission.max() == pytest.approx(MAX_EMISSION * 0.875, rel=0.01)


def test_a_nested_mask_carries_its_surfaces_up(compositor, inputs):
    tree = plain_wear(white=ColorSlot(RED))
    tree = set_slot(tree, ("black",), MaskLayer(
        kind="noise",
        noise=NoiseMaskParams(scale=20.0, contrast=12.0),
        white=GOLD,
        black=ColorSlot(BLUE, roughness=1.0),
    ))
    compositor.render(tree, RESOLUTION, *inputs)

    # The dark end of the curvature is entirely the nested noise, so both of
    # its materials have to show up there.
    metallic = compositor.read_material()[:, 0, 0]
    assert metallic.max() == pytest.approx(1.0, abs=0.01), "the gold underneath"
    assert metallic.min() == pytest.approx(0.0, abs=0.01), "and the other side"


def test_a_mask_picks_between_its_two_colours(compositor, inputs):
    out = render(compositor, inputs, plain_wear(white=ColorSlot(RED), black=ColorSlot(BLUE)))

    assert out.shape == (RESOLUTION, RESOLUTION, 3)
    assert out[32, -1] == pytest.approx(RED, abs=0.05), "high curvature is white's side"
    assert out[32, 0] == pytest.approx(BLUE, abs=0.05), "low curvature is black's"


def test_the_two_sides_swap_with_the_colours(compositor, inputs):
    swapped = render(compositor, inputs, plain_wear(white=ColorSlot(BLUE), black=ColorSlot(RED)))
    assert swapped[32, -1] == pytest.approx(BLUE, abs=0.05)


def test_a_nested_mask_replaces_that_side(compositor, inputs):
    """The black side becomes a noise between two colours, so what was one flat
    colour is now two."""
    flat = render(compositor, inputs, plain_wear(white=ColorSlot(RED), black=ColorSlot(BLUE)))

    tree = plain_wear(white=ColorSlot(RED))
    tree = set_slot(tree, ("black",), MaskLayer(
        kind="noise",
        noise=NoiseMaskParams(scale=20.0, contrast=12.0),
        white=ColorSlot(GREEN),
        black=ColorSlot(BLUE),
    ))
    nested = render(compositor, inputs, tree)

    # The first column is curvature 0, so the mask is fully on black's side and
    # what shows there is entirely whatever black resolves to.
    assert len(np.unique(flat[:, 0].round(2), axis=0)) == 1, "one flat colour"
    assert len(np.unique(nested[:, 0].round(2), axis=0)) > 1, "now a noise"
    # The white side is untouched by what happens under black.
    assert nested[32, -1] == pytest.approx(flat[32, -1], abs=0.02)


def test_every_mask_gets_a_thumbnail_and_colours_do_not(compositor, inputs):
    tree = nested_tree()
    render(compositor, inputs, tree)

    assert sorted(compositor.thumbnails) == [(), ("black",)]
    assert compositor.thumbnail(()) is not None
    assert compositor.thumbnail(("white",)) is None, "a colour needs no preview"


def test_thumbnails_are_dropped_when_their_branch_goes(compositor, inputs):
    tree = nested_tree()
    render(compositor, inputs, tree)
    assert compositor.thumbnail(("black",)) is not None

    render(compositor, inputs, set_slot(tree, ("black",), ColorSlot(BLUE)))
    assert compositor.thumbnail(("black",)) is None
    assert compositor.retired, "and handed over to be unregistered, not just freed"


def test_depth_costs_passes_not_targets(compositor, inputs):
    """The point of rendering bottom-up: a target per level in flight, and the
    pool is reused rather than grown with the tree."""
    tree = MaskLayer(kind="noise")
    for _ in range(MAX_DEPTH - 1):
        tree = MaskLayer(kind="noise", white=tree, black=ColorSlot(BLUE))

    render(compositor, inputs, tree)
    assert depth(tree) == MAX_DEPTH
    assert compositor._pool.size <= MAX_DEPTH + 1
    assert len(compositor.thumbnails) == MAX_DEPTH


def test_a_tree_past_the_cap_is_refused_rather_than_rendered(compositor, inputs):
    tree = MaskLayer(kind="noise")
    for _ in range(MAX_DEPTH + 1):
        tree = MaskLayer(kind="noise", white=tree)

    with pytest.raises(ValueError, match="deeper"):
        render(compositor, inputs, tree)


def test_re_rendering_the_same_tree_is_stable(compositor, inputs):
    tree = nested_tree()
    first = render(compositor, inputs, tree)
    second = render(compositor, inputs, tree)
    assert np.array_equal(first, second)


def test_changing_a_colour_changes_only_that_side(compositor, inputs):
    tree = plain_wear(white=ColorSlot(RED), black=ColorSlot(BLUE))
    before = render(compositor, inputs, tree)
    after = render(compositor, inputs, set_slot(tree, ("black",), ColorSlot(GREEN)))

    assert after[32, -1] == pytest.approx(before[32, -1], abs=0.02)
    assert after[32, 0] == pytest.approx(GREEN, abs=0.05)


def test_a_noise_mask_needs_no_curvature_to_say_something(compositor, inputs):
    """Noise reads the positions, not the bake, so it patterns a surface whose
    curvature is flat everywhere."""
    _, position = inputs
    ctx = compositor.ctx
    flat = ctx.texture((RESOLUTION, RESOLUTION), 1,
                       data=np.zeros((RESOLUTION, RESOLUTION), "f4").tobytes(), dtype="f4")
    try:
        tree = MaskLayer(kind="noise", noise=NoiseMaskParams(scale=15.0, contrast=10.0),
                         white=ColorSlot(RED), black=ColorSlot(BLUE))
        compositor.render(tree, RESOLUTION, flat, position)
        out = compositor.read()
    finally:
        flat.release()

    colors = np.unique(out.round(2).reshape(-1, 3), axis=0)
    assert len(colors) > 1, "the noise has to vary across the surface"


def _wear_layer(params, threshold=0.5, softness=0.25) -> MaskLayer:
    """An edge-wear layer whose composited grey *is* the mask, put through the
    boundary. Softness makes that a smoothstep -- continuous, so the output is a
    known function of the mask rather than a coin toss at the threshold."""
    return MaskLayer(
        kind=EDGE_WEAR_KIND, edge_wear=params, threshold=threshold, softness=softness,
        white=ColorSlot((1.0, 1.0, 1.0)), black=ColorSlot((0.0, 0.0, 0.0)),
    )


def _smoothstep(mask, threshold=0.5, softness=0.25):
    ramp = np.clip((mask - (threshold - softness)) / (2 * softness), 0.0, 1.0)
    return ramp * ramp * (3.0 - 2.0 * ramp)


def _bake_fields():
    """The same ramp and positions the ``inputs`` fixture uploads."""
    curvature = np.tile(np.linspace(0.0, 1.0, RESOLUTION, dtype="f4"), (RESOLUTION, 1))
    axis = np.linspace(0.0, 1.0, RESOLUTION, dtype="f4")
    gx, gy = np.meshgrid(axis, axis)
    return curvature, np.stack([gx, gy, np.full_like(gx, 0.3)], axis=-1).astype("f4")


def test_the_layers_edge_wear_is_the_node_group_it_claims_to_be(compositor, inputs):
    """The formula that renders, against the numpy port of EdgeWear001.

    ``render/shaders/edge_wear.frag`` is the reference port, decoded node for
    node from the ``.arm`` file and checked against ``core/edge_wear.py`` in
    tests/test_gl.py. It is not what draws any more -- ``layer.frag`` is, with
    its own copy of the expression -- so this is what stops the two from
    drifting apart and the reference from quietly becoming fiction.

    With the noise weighted out, the two are the same arithmetic and must agree
    to rounding; the sine hash is compared separately, below, because it cannot
    be matched bit for bit across GLSL and numpy.
    """
    from core.edge_wear import edge_wear

    curvature, position = _bake_fields()
    for params in (
        EdgeWearParams(wear_amount=0.0),
        EdgeWearParams(wear_amount=0.0, contrast=1.0),
        EdgeWearParams(wear_amount=0.0, contrast=6.0),
    ):
        drawn = render(compositor, inputs, _wear_layer(params))[..., 0]
        expected = _smoothstep(edge_wear(curvature, position, params))
        assert np.abs(drawn - expected).max() < 0.01, f"drift with {params}"


def test_the_layers_wear_noise_agrees_with_the_numpy_mirror(compositor, inputs):
    """A sine hash cannot be compared bit for bit, so compare distributions --
    the same way the reference port is checked in tests/test_gl.py."""
    from core.edge_wear import edge_wear

    curvature, position = _bake_fields()
    params = EdgeWearParams()

    drawn = render(compositor, inputs, _wear_layer(params))[..., 0]
    expected = _smoothstep(edge_wear(curvature, position, params))

    assert drawn.mean() == pytest.approx(expected.mean(), abs=0.06)
    assert drawn.std() == pytest.approx(expected.std(), abs=0.06)


# --------------------------------------------------------------------------
# through the app: creating a texture, editing it, and the one export button
# --------------------------------------------------------------------------

@pytest.fixture
def baked_app(ctx, tmp_path):
    """The real app, headless, with a bake behind it -- what the masks read."""
    import time

    import moderngl_window as mglw
    import trimesh
    from imgui_bundle import imgui

    from render.viewport import MeshMapApp

    mesh = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
        visual=trimesh.visual.TextureVisuals(
            uv=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        ),
    )
    mesh.export(tmp_path / "quad.obj")

    MeshMapApp.initial_mesh = str(tmp_path / "quad.obj")
    MeshMapApp.initial_resolution = 256
    try:
        app = mglw.create_window_config_instance(MeshMapApp, args=["-wnd", "headless"])
    except Exception as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"no headless window available: {exc}")

    app.request_bake()
    deadline = time.monotonic() + 30.0
    while app.controller.running and time.monotonic() < deadline:
        app.controller.pump()
        # The CPU stages run on a worker thread; spinning here just fights it
        # for the GIL, which is why the app pumps once a frame rather than in a
        # loop like this one.
        time.sleep(0.002)
    app.on_render(0.0, 1 / 60.0)

    yield app
    app.controller.release()
    imgui.destroy_context()
    MeshMapApp.initial_mesh = None


def test_an_app_opens_with_a_texture_already_there(baked_app):
    """A flat colour, named, selected -- something to start from rather than an
    empty tab."""
    assert isinstance(baked_app.texture, ColorSlot)
    assert baked_app.texture.name == "Texture 01"
    assert baked_app.texture_path == ()
    assert baked_app.read_color_map() is not None


def test_new_textures_count_up_from_the_one_that_is_there(baked_app):
    assert baked_app.texture.name == "Texture 01"

    baked_app.create_texture()
    assert baked_app.texture.name == "Texture 02"
    assert "Texture 02" in baked_app.status
    assert len(baked_app.textures) == 2


def test_a_texture_can_be_renamed(baked_app):
    baked_app.texture.name = "Hull plate"
    assert describe(baked_app.texture) == "Hull plate"

    # And renaming changes nothing about what is rendered.
    before = baked_app.read_color_map()
    baked_app.texture.name = "Something else"
    assert np.array_equal(before, baked_app.read_color_map())


def test_starting_or_switching_texture_ends_any_rename(baked_app):
    baked_app.begin_rename(())
    baked_app.create_texture()
    assert baked_app.renaming_path is None

    baked_app.begin_rename(())
    baked_app.select_texture(0)
    assert baked_app.renaming_path is None


def test_a_new_texture_composites_as_a_flat_colour(baked_app):
    assert isinstance(baked_app.texture, ColorSlot)

    composite = baked_app.read_color_map()
    assert composite is not None
    expected = baked_app.texture.color
    assert np.allclose(composite, expected, atol=2e-3), "one colour, everywhere"


def test_turning_the_texture_into_a_mask_re_composites(baked_app):
    flat = baked_app.read_color_map()

    baked_app.set_texture(convert_slot(baked_app.texture, "noise"))
    patterned = baked_app.read_color_map()

    assert not np.allclose(flat, patterned)
    assert patterned.min() < 0.05 and patterned.max() > 0.95, "black and white"


def test_removing_the_last_texture_leaves_nothing_to_composite(baked_app):
    assert baked_app.read_color_map() is not None

    baked_app.remove_texture()
    assert baked_app.texture is None and baked_app.textures == []
    assert baked_app.read_color_map() is None
    assert baked_app.compositor.texture is None


def test_textures_are_kept_and_can_be_switched_between(baked_app):
    """Every texture made stays around; the picker is how one is gone back to."""
    baked_app.set_texture(MaskLayer(
        kind="noise", name="Speckle", white=ColorSlot(RED), black=ColorSlot(BLUE)
    ))
    speckled = baked_app.read_color_map()

    baked_app.create_texture()
    baked_app.set_texture(ColorSlot(GREEN, name="Flat green"))
    assert len(baked_app.textures) == 2
    assert np.allclose(baked_app.read_color_map(), GREEN, atol=2e-3)

    baked_app.select_texture(0)
    assert describe(baked_app.texture) == "Speckle"
    assert np.array_equal(baked_app.read_color_map(), speckled)


def test_removing_one_texture_falls_back_to_a_neighbour(baked_app):
    baked_app.create_texture()
    second = describe(baked_app.textures[1])

    baked_app.select_texture(0)
    baked_app.remove_texture()

    assert len(baked_app.textures) == 1
    assert describe(baked_app.texture) == second, "what is left is what is shown"
    assert baked_app.texture_path == ()


def test_selecting_a_texture_that_is_not_there_is_ignored(baked_app):
    baked_app.select_texture(7)
    assert baked_app.texture_index == 0


def test_a_colour_slot_can_be_selected_as_well_as_a_mask(baked_app):
    """The panel edits whichever is selected, so both have to be reachable."""
    baked_app.set_texture(convert_slot(baked_app.texture, "edge_wear"))

    baked_app.select_slot(("white",))
    assert baked_app.texture_path == ("white",)
    assert isinstance(baked_app.selected_slot, ColorSlot)

    baked_app.select_slot(())
    assert isinstance(baked_app.selected_slot, MaskLayer)


def test_selection_follows_the_tree_when_a_branch_is_removed(baked_app):
    """The panel stands on a path. Collapsing that branch has to move it
    somewhere that still exists, not leave it pointing at nothing."""
    baked_app.set_texture(convert_slot(baked_app.texture, "edge_wear"))
    baked_app.set_texture(
        set_slot(baked_app.texture, ("white",), MaskLayer(kind="noise"))
    )
    baked_app.select_slot(("white", "black"))
    assert baked_app.texture_path == ("white", "black")

    baked_app.set_texture(set_slot(baked_app.texture, ("white",), ColorSlot(RED)))
    assert baked_app.texture_path == ("white",), "up to the nearest slot left"
    assert isinstance(baked_app.selected_slot, ColorSlot)


def test_selecting_a_path_that_does_not_exist_is_ignored(baked_app):
    baked_app.select_slot(("white", "black"))  # a flat colour has no slots
    assert baked_app.texture_path == ()


def test_one_export_writes_the_colour_and_the_normal_map(baked_app, tmp_path):
    """One button, everything there is."""
    from PIL import Image

    decal = tmp_path / "decal.png"
    normal = np.array([0.9, 0.5, 1.0], dtype=np.float32)
    Image.fromarray(
        np.tile((normal * 255).astype(np.uint8), (32, 32, 1))
    ).save(decal)
    baked_app.open_decal(decal)
    baked_app.on_render(0.0, 1 / 60.0)

    baked_app.export_dir = str(tmp_path / "out")
    baked_app.export()

    written = sorted(p.name for p in (tmp_path / "out" / "quad").glob("*.png"))
    assert written == [
        "ao.png", "color.png", "metallic.png", "normal.png", "roughness.png",
    ]
    assert "color.png" in baked_app.status and "normal.png" in baked_app.status


def test_no_texture_means_no_colour_map_is_written(baked_app, tmp_path):
    baked_app.remove_texture()
    baked_app.export_dir = str(tmp_path / "out")
    baked_app.export()

    written = sorted(p.name for p in (tmp_path / "out" / "quad").glob("*.png"))
    assert written == ["ao.png"], "the bake's own map, and nothing from a texture"


def test_the_exported_colours_are_the_ones_on_screen(baked_app, tmp_path):
    from PIL import Image

    baked_app.set_texture(convert_slot(baked_app.texture, "noise"))
    baked_app.set_texture(
        set_slot(baked_app.texture, ("black",), ColorSlot((0.2, 0.6, 0.9)))
    )
    baked_app.export_dir = str(tmp_path / "out")
    baked_app.export()

    on_screen = baked_app.read_color_map()
    written = np.asarray(
        Image.open(tmp_path / "out" / "quad" / "color.png"), dtype=np.float32
    ) / 255.0
    assert np.abs(np.flipud(written) - on_screen).max() < 2.0 / 255


def test_the_colour_map_follows_the_bake_resolution(baked_app):
    import time

    assert baked_app.read_color_map().shape[0] == 256

    baked_app.controller.bake_params.resolution = 512
    baked_app.request_bake()
    while baked_app.controller.running:
        baked_app.controller.pump()
        time.sleep(0.002)
    baked_app.on_render(0.0, 1 / 60.0)

    assert baked_app.read_color_map().shape[0] == 512


def test_the_shaded_view_is_the_texture(baked_app):
    """Shaded shows what the texture resolves to, which is the whole point of
    setting colours under the masks."""
    from render.viewport import PREVIEW_MODES, SHADED_INDEX

    assert PREVIEW_MODES[SHADED_INDEX].texture == "composite"
    baked_app.preview_index = SHADED_INDEX
    baked_app.on_render(0.0, 1 / 60.0)
    assert baked_app._current_texture() is baked_app.compositor.texture


def test_a_fresh_mask_reads_as_black_and_white(baked_app):
    """Nothing has been coloured yet, so the shaded model shows the mask itself
    -- white where the mask is white, black where it is black."""
    baked_app.set_texture(convert_slot(baked_app.texture, "noise"))

    tree = baked_app.texture
    assert tree.white.color == (1.0, 1.0, 1.0)
    assert tree.black.color == (0.0, 0.0, 0.0)

    composite = baked_app.read_color_map()
    # Black and white, and nothing in between -- the boundary is a boundary.
    assert np.allclose(composite[..., 0], composite[..., 1], atol=1e-3)
    assert np.allclose(composite[..., 1], composite[..., 2], atol=1e-3)
    assert composite.min() < 0.05 and composite.max() > 0.95, "it spans the range"
    assert set(np.unique(composite.round(3))) <= {0.0, 1.0}


def test_colours_under_the_mask_are_what_shaded_shows(baked_app):
    baked_app.set_texture(MaskLayer(
        kind="noise", white=ColorSlot(RED), black=ColorSlot(BLUE)
    ))
    composite = baked_app.read_color_map()

    flat = composite.reshape(-1, 3)
    assert np.any(np.all(np.abs(flat - RED) < 0.05, axis=1)), "the white side"
    assert np.any(np.all(np.abs(flat - BLUE) < 0.05, axis=1)), "the black side"
    assert not np.any(flat[:, 1] > 0.5), "and nothing green anywhere"


def test_the_picker_searches_by_name(baked_app):
    """Half-remembered names are as often remembered by their end as their
    start, so the search is unanchored -- and case does not matter."""
    from ui.panel import filter_textures

    textures = [
        ColorSlot(RED, name="Hull plate 01"),
        ColorSlot(RED, name="Hull plate 02"),
        ColorSlot(RED, name="Rusted vent"),
        MaskLayer(name="Panel wear"),
    ]

    assert filter_textures(textures, "") == [
        (0, "Hull plate 01"), (1, "Hull plate 02"),
        (2, "Rusted vent"), (3, "Panel wear"),
    ]
    assert [index for index, _ in filter_textures(textures, "hull")] == [0, 1]
    assert [index for index, _ in filter_textures(textures, "  VENT ")] == [2]
    assert [index for index, _ in filter_textures(textures, "wear")] == [3]
    assert filter_textures(textures, "nothing like it") == []


def test_the_picker_searches_what_a_texture_is_called_now(baked_app):
    """An unnamed texture is searchable by the label it shows instead."""
    from ui.panel import filter_textures

    assert [index for index, _ in filter_textures([ColorSlot(RED)], "FF00")] == [0]


def test_an_edit_anywhere_in_the_tree_re_composites(baked_app):
    """Every control edits the tree in place, so the composite follows the tree
    rather than a flag somebody has to remember to set. A control that forgot
    would silently do nothing."""
    baked_app.set_texture(MaskLayer(
        kind="noise", white=ColorSlot(RED), black=ColorSlot(BLUE)
    ))
    before = baked_app.read_color_map()

    # Mutated in place and nothing told: exactly what a slider does.
    baked_app.texture.white.color = GREEN
    assert not np.array_equal(baked_app.read_color_map(), before)

    before = baked_app.read_color_map()
    baked_app.texture.noise.scale *= 3.0
    assert not np.array_equal(baked_app.read_color_map(), before)

    before = baked_app.read_color_map()
    baked_app.texture.threshold = 0.8
    assert not np.array_equal(baked_app.read_color_map(), before)


def test_renaming_alone_does_not_re_composite(baked_app):
    """Names are for reading, not rendering."""
    baked_app.set_texture(MaskLayer(kind="noise", name="Speckle"))
    before = baked_app.read_color_map()

    baked_app.texture.name = "Something else"
    assert np.array_equal(baked_app.read_color_map(), before)


def test_a_texture_that_comes_out_flat_is_noticed(baked_app):
    """A mask whose every texel lands on one side paints the model in a single
    colour, which reads as broken rather than as empty -- so the panel has to
    be able to tell the difference."""
    baked_app.set_texture(MaskLayer(
        kind="noise", white=ColorSlot(RED), black=ColorSlot(BLUE)
    ))
    baked_app.read_color_map()
    assert not baked_app.texture_is_flat, "a noise divides the surface"

    # No contrast leaves the mask a flat grey, so every texel lands on the
    # same side of the threshold and the whole model takes one colour.
    baked_app.texture.noise.contrast = 0.0
    baked_app.mark_texture_dirty()
    baked_app.read_color_map()
    assert baked_app.texture_is_flat

    baked_app.texture.noise.contrast = 8.0
    baked_app.mark_texture_dirty()
    baked_app.read_color_map()
    assert not baked_app.texture_is_flat


def test_a_plain_colour_is_flat_but_not_a_complaint(baked_app):
    """It is flat because it is a colour, which is not something to warn about
    -- the panel only mentions it for a texture that has a mask in it."""
    baked_app.read_color_map()
    assert baked_app.texture_is_flat
    assert mask_count(baked_app.texture) == 0


def test_removing_the_texture_mid_frame_does_not_take_the_panel_with_it(baked_app):
    """Remove is a control inside the tab it deletes the subject of.

    An ImGui frame is drawn top to bottom, so everything below the button --
    the mask count, the inspector, the tree -- would go on describing a texture
    that stopped existing halfway down. It crashed on the first row of the
    tree.
    """
    from imgui_bundle import imgui

    from ui import panel

    real_button = imgui.button

    def press_remove(label, *args, **kwargs):
        pressed = real_button(label, *args, **kwargs)
        return True if label == "Remove" else pressed

    imgui.new_frame()
    imgui.begin("Parameters")
    imgui.button = press_remove
    try:
        panel._draw_texture_tab(baked_app)  # must not raise
    finally:
        imgui.button = real_button
    imgui.end()
    imgui.end_frame()

    assert baked_app.texture is None and baked_app.textures == []

    # And the frame after it draws the empty state rather than the wreckage.
    imgui.new_frame()
    imgui.begin("Parameters")
    panel._draw_texture_tab(baked_app)
    imgui.end()
    imgui.end_frame()


def test_the_texture_tab_draws_in_every_state(baked_app):
    """The panel's own begin/end pairing, through each shape a texture takes.

    Cheap to run and the only thing that catches an unbalanced child region or
    a control asking a colour for something only a mask has.
    """
    from imgui_bundle import imgui

    from ui import panel

    def draw() -> None:
        imgui.new_frame()
        imgui.begin("Parameters")
        panel._draw_texture_tab(baked_app)
        imgui.end()
        imgui.end_frame()

    draw()  # the flat colour it opens with

    baked_app.remove_texture()
    draw()  # and with no texture at all

    baked_app.create_texture()
    draw()  # a fresh one

    baked_app.set_texture(convert_slot(baked_app.texture, "edge_wear"))
    draw()  # a mask, selected

    baked_app.set_texture(
        set_slot(baked_app.texture, ("black",), MaskLayer(kind="noise"))
    )
    baked_app.select_slot(("black", "white"))
    draw()  # a colour nested two levels down

    baked_app.select_slot(("black",))
    draw()  # and the mask above it

    baked_app.begin_rename(("black",))
    draw()  # a row being renamed in place

    baked_app.end_rename()
    for split in (0.15, 0.5, 0.85):
        baked_app.set_texture_split(split)
        draw()  # the two panes at each end of the splitter's travel

    baked_app.end_rename()
    for _ in range(6):
        baked_app.create_texture()
    baked_app.texture_filter = "02"
    draw()  # several textures, with the picker's search box in play


def test_shaded_falls_back_to_plain_grey_before_a_bake(ctx, tmp_path):
    """The tree has nothing to stand on until the bake exists, and a viewport
    that renders nothing at all would read as a broken load."""
    import moderngl_window as mglw
    import trimesh
    from imgui_bundle import imgui

    from render.viewport import FLAT_MODE, PREVIEW_MODES, SHADED_INDEX, MeshMapApp

    trimesh.creation.box().export(tmp_path / "box.obj")
    MeshMapApp.initial_mesh = str(tmp_path / "box.obj")
    MeshMapApp.initial_resolution = 256
    try:
        app = mglw.create_window_config_instance(MeshMapApp, args=["-wnd", "headless"])
    except Exception as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"no headless window available: {exc}")

    try:
        app.preview_index = SHADED_INDEX
        assert app.compositor.texture is None, "nothing baked"
        assert app._current_texture() is None
        assert FLAT_MODE.texture is None and FLAT_MODE.shader_mode == 3

        app.on_render(0.0, 1 / 60.0)  # must draw rather than raise
        assert PREVIEW_MODES[app.preview_index].texture == "composite"
    finally:
        app.controller.release()
        imgui.destroy_context()
        MeshMapApp.initial_mesh = None
