"""Several decals on one mesh: the shelf, placing, selecting, and the outline.

A decal used to be a single thing the app owned. It is a list now, which changes
what most of the questions mean: not "where is the decal" but "which one is this
click about", not "is there a decal" but "what do several of them add up to".
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

from core.decal import decal_at_uv, library, make_text_decal, outline_uvs  # noqa: E402
from core.decal_thumbnail import load_thumbnail, thumbnail_cache_key  # noqa: E402
from core.params import DecalParams  # noqa: E402

FLAT = np.array([0.5, 0.5, 1.0], dtype=np.float32)


def write_decal(path: Path, tilt=(0.8, 0.0), size=(32, 32)) -> Path:
    """A normal map with a constant tilt, opaque, saved as a PNG."""
    width, height = size
    normal = np.array([tilt[0], tilt[1], 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    encoded = ((normal * 0.5 + 0.5) * 255).astype(np.uint8)
    Image.fromarray(np.tile(encoded, (height, width, 1))).save(path)
    return path


@pytest.fixture
def app():
    """The app on its starter cube, headless. No bake: placing a decal reads
    the mesh's own UVs, and owes the curvature pass nothing.

    Preferences are already pointed at a throwaway directory by the suite's own
    autouse fixture, so nothing here touches the real one.
    """
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


def tilt_map(app) -> np.ndarray:
    """How far the composited normal map leans, per texel."""
    app.on_render(0.0, 1 / 60.0)
    composed = app.read_normal_map()
    if composed is None:
        return np.zeros((4, 4), dtype=np.float32)
    return np.abs(composed - FLAT).max(axis=2)


# --------------------------------------------------------------------------
# the shelf
# --------------------------------------------------------------------------

def test_the_library_lists_the_images_in_a_folder(tmp_path):
    for name in ("b.png", "a.PNG", "c.jpg"):
        write_decal(tmp_path / name)
    (tmp_path / "notes.txt").write_text("not a decal")
    (tmp_path / "nested").mkdir()

    found = [path.name for path in library(tmp_path)]
    assert found == ["a.PNG", "b.png", "c.jpg"], "sorted, and images only"


def test_a_library_that_is_not_there_is_an_empty_shelf(tmp_path):
    """The path comes from metadata.json, so a typo should cost the shelf
    rather than the session."""
    assert library(None) == []
    assert library(tmp_path / "absent") == []
    assert library(write_decal(tmp_path / "a.png")) == [], "a file is not a folder"


def test_the_shipped_library_is_the_one_metadata_points_at():
    from core.metadata import METADATA

    if METADATA.decals is None:
        pytest.skip("this checkout ships no decal library")
    assert METADATA.decals.is_dir()
    assert library(METADATA.decals), "and it has something on it"


def test_the_app_offers_that_shelf(app):
    from core.metadata import METADATA

    assert [path.name for path in app.decal_library] == [
        path.name for path in library(METADATA.decals)
    ]


def test_text_decals_are_transparent_generated_height_maps():
    text = make_text_decal("Text")
    same = make_text_decal("Text")
    changed = make_text_decal("Another label")

    assert text.path == same.path
    assert text.path != changed.path
    assert text.from_height
    assert text.aspect > 1.0
    assert text.alpha.min() == 0.0 and text.alpha.max() == 1.0
    assert np.any(np.abs(text.normals - FLAT) > 0.01)


def test_add_text_creates_selected_placeable_white_decal(app):
    from core.layers import ColorSlot

    index = app.add_text_decal()
    decal = app.decals[index]

    assert app.selected_decal is decal
    assert decal.source_type == "text"
    assert decal.text == "Text"
    assert decal.path in app.decal_images
    assert decal.path in app.decal_textures
    assert isinstance(app.textures[decal.texture_index], ColorSlot)
    assert app.textures[decal.texture_index].color == (1.0, 1.0, 1.0)
    assert app.decal_placing, "the button picks the generated decal up for placement"

    previous_path = decal.path
    assert app.update_text_decal(decal, "Label")
    assert decal.text == "Label"
    assert decal.path != previous_path
    assert decal.path in app.decal_images
    assert decal.path in app.decal_textures


def test_library_thumbnails_are_small_and_persisted(tmp_path):
    source = write_decal(tmp_path / "large.png", size=(600, 300))
    cache = tmp_path / "cache"

    thumbnail = load_thumbnail(source, cache)

    assert thumbnail.size == (256, 128)
    assert len(list(cache.glob("*.png"))) == 1


def test_thumbnail_cache_key_changes_with_the_source(tmp_path):
    source = write_decal(tmp_path / "changing.png")
    before = thumbnail_cache_key(source)

    write_decal(source, size=(48, 32))

    assert thumbnail_cache_key(source) != before


# --------------------------------------------------------------------------
# several at once
# --------------------------------------------------------------------------

def test_two_decals_both_reach_the_normal_map(app, tmp_path):
    one = write_decal(tmp_path / "one.png")
    app.add_decal(one, center=(0.2, 0.25))
    app.decals[-1].scale = 0.1
    app.add_decal(one, center=(0.8, 0.75))
    app.decals[-1].scale = 0.1

    tilt = tilt_map(app)
    resolution = tilt.shape[0]
    left = tilt[:, : resolution // 2]
    right = tilt[:, resolution // 2 :]
    assert left.max() > 0.05 and right.max() > 0.05, "one on each side of the atlas"


def test_one_picture_placed_twice_is_read_once(app, tmp_path):
    """A shelf of large images stamped all over a model should cost one upload
    each, not one per placement."""
    path = write_decal(tmp_path / "vent.png")
    app.add_decal(path, center=(0.2, 0.2))
    app.add_decal(path, center=(0.8, 0.8))

    assert len(app.decals) == 2
    assert len(app.decal_images) == 1
    assert len(app.decal_textures) == 1


def test_overlapping_decals_add_up_rather_than_replacing(app, tmp_path):
    """Slopes accumulate: a vent stamped across a panel line reads as both.
    Adding the encoded normals instead would average them towards flat, and the
    second decal would rub out the first as much as it added itself."""
    path = write_decal(tmp_path / "tilt.png", tilt=(0.8, 0.0))

    app.add_decal(path, center=(0.5, 0.5))
    app.decals[-1].scale = 0.3
    alone = tilt_map(app).max()

    app.add_decal(path, center=(0.5, 0.5))
    app.decals[-1].scale = 0.3
    both = tilt_map(app).max()

    assert both > alone + 0.02, "the second stamp deepens the first"


def test_removing_one_leaves_the_others(app, tmp_path):
    path = write_decal(tmp_path / "a.png")
    for centre in ((0.2, 0.2), (0.5, 0.5), (0.8, 0.8)):
        app.add_decal(path, center=centre)

    app.select_decal(1)
    app.remove_decal()

    assert len(app.decals) == 2
    assert [(d.center_u, d.center_v) for d in app.decals] == [(0.2, 0.2), (0.8, 0.8)]
    assert app.decal_index == 1, "the neighbour takes the selection"


def test_removing_the_last_one_selects_nothing(app, tmp_path):
    app.add_decal(write_decal(tmp_path / "a.png"))
    app.remove_decal()

    assert app.decals == []
    assert app.selected_decal is None
    assert app.decal_index == -1


def test_a_decal_switched_off_leaves_the_others_alone(app, tmp_path):
    path = write_decal(tmp_path / "a.png")
    app.add_decal(path, center=(0.25, 0.25))
    app.decals[-1].scale = 0.15
    app.add_decal(path, center=(0.75, 0.75))
    app.decals[-1].scale = 0.15

    app.decals[0].enabled = False
    app.mark_normal_dirty()
    tilt = tilt_map(app)

    resolution = tilt.shape[0]
    assert tilt[: resolution // 2, : resolution // 2].max() < 0.02, "the one turned off"
    assert tilt[resolution // 2 :, resolution // 2 :].max() > 0.05, "and the one left on"


def test_no_decals_at_all_is_a_flat_map(app, tmp_path):
    app.add_decal(write_decal(tmp_path / "a.png"))
    assert tilt_map(app).max() > 0.05

    app.clear_decals()
    assert app.read_normal_map() is None, "nothing to write, nothing to export"


# --------------------------------------------------------------------------
# which one a click is about
# --------------------------------------------------------------------------

def test_the_topmost_decal_wins_an_overlap():
    """Last placed is last stamped, so it is the one on top and the one a click
    on that spot means."""
    decals = [
        DecalParams(path="under", center_u=0.5, center_v=0.5, scale=0.4),
        DecalParams(path="over", center_u=0.5, center_v=0.5, scale=0.2),
    ]
    assert decal_at_uv(decals, 0.5, 0.5) == 1, "the smaller one, on top"
    assert decal_at_uv(decals, 0.35, 0.5) == 0, "past its edge, the one beneath"
    assert decal_at_uv(decals, 0.05, 0.05) is None


def test_a_decal_switched_off_cannot_be_clicked():
    decals = [DecalParams(path="a", center_u=0.5, center_v=0.5, scale=0.4)]
    assert decal_at_uv(decals, 0.5, 0.5) == 0

    decals[0].enabled = False
    assert decal_at_uv(decals, 0.5, 0.5) is None


def test_a_rotated_decal_is_hit_where_it_is_drawn():
    """The rectangle turns with the decal, so the corners of the unrotated one
    are outside it and the corners of the rotated one are in."""
    decals = [
        DecalParams(path="a", center_u=0.5, center_v=0.5, scale=0.4, rotation=45.0)
    ]
    assert decal_at_uv(decals, 0.5, 0.5) == 0
    # A point beyond the unrotated corner, but inside the turned one.
    assert decal_at_uv(decals, 0.5, 0.5 + 0.27) == 0
    assert decal_at_uv(decals, 0.5 + 0.19, 0.5 + 0.19) is None


def test_clicking_the_mesh_selects_the_decal_under_it(app, tmp_path):
    """The whole point of selecting in the viewport: what you click on is what
    the inspector fills with."""
    centre = viewport_centre(app)
    uv = app.surface_uv_at(centre)
    assert uv is not None, "the cube should be under the middle of the view"

    app.add_decal(write_decal(tmp_path / "a.png"), center=uv)
    app.decals[-1].scale = 0.2
    app.select_decal(None)

    assert app.select_decal_at(centre) == 0
    assert app.selected_decal is app.decals[0]


def test_clicking_bare_surface_deselects(app, tmp_path):
    centre = viewport_centre(app)
    uv = app.surface_uv_at(centre)
    app.add_decal(write_decal(tmp_path / "a.png"), center=uv)
    app.decals[-1].scale = 0.01  # far too small to be under the cursor's texel

    app.decals[0].center_u = (uv[0] + 0.4) % 1.0
    assert app.select_decal_at(centre) is None
    assert app.selected_decal is None


def viewport_centre(app) -> tuple[float, float]:
    """The middle of the 3D view, in the units mouse events arrive in."""
    left, bottom, width, height = app.viewport_rect
    ratio = float(app.wnd.pixel_ratio)
    buffer_height = app.wnd.buffer_size[1]
    return (
        (left + width / 2) / ratio,
        (buffer_height - (bottom + height / 2)) / ratio,
    )


def test_a_drag_across_the_model_does_not_change_the_selection(app, tmp_path):
    """Orbiting starts on the model and often ends on it. Selecting on release,
    only when the pointer stayed put, is what keeps a look-around from being a
    click."""
    centre = viewport_centre(app)
    uv = app.surface_uv_at(centre)
    app.add_decal(write_decal(tmp_path / "a.png"), center=uv)
    app.decals[-1].scale = 0.3
    app.select_decal(None)

    x, y = int(centre[0]), int(centre[1])
    app.on_mouse_press_event(x, y, app.wnd.mouse.left)
    app.on_mouse_drag_event(x + 60, y + 20, 60, 20)
    app.on_mouse_release_event(x + 60, y + 20, app.wnd.mouse.left)

    assert app.selected_decal is None, "that was an orbit, not a click"


def test_a_click_on_the_model_does_change_it(app, tmp_path):
    centre = viewport_centre(app)
    uv = app.surface_uv_at(centre)
    app.add_decal(write_decal(tmp_path / "a.png"), center=uv)
    app.decals[-1].scale = 0.3
    app.select_decal(None)

    x, y = int(centre[0]), int(centre[1])
    app.on_mouse_press_event(x, y, app.wnd.mouse.left)
    app.on_mouse_release_event(x, y, app.wnd.mouse.left)

    assert app.selected_decal is app.decals[0]


def test_a_wrapped_decal_can_be_selected_on_its_neighboring_face(app, tmp_path):
    app.add_decal(write_decal(tmp_path / "a.png"), center=(0.5, 0.5), face=2)
    app.decals[0].scale = 0.3
    app.select_decal(None)
    app.surface_hit_at = lambda mouse: ((0.9, 0.9), 7)
    app._surface_uv_on_wrap = lambda anchor, face, uv: (0.5, 0.5)

    assert app.select_decal_at((100.0, 100.0)) == 0
    assert app.selected_decal is app.decals[0]


# --------------------------------------------------------------------------
# dragging one off the shelf
# --------------------------------------------------------------------------

def test_a_shelf_drag_previews_on_the_mesh_before_drop(app, tmp_path):
    centre = viewport_centre(app)
    path = write_decal(tmp_path / "vent.png")

    app.begin_decal_drag(path)
    app.on_mouse_position_event(int(centre[0]), int(centre[1]), 0, 0)

    preview = app.dragging_decal_preview
    assert preview is not None
    assert app.decals == [], "the preview is not committed before release"
    assert (preview.center_u, preview.center_v) == pytest.approx(
        app.surface_uv_at((float(int(centre[0])), float(int(centre[1]))))
    )
    assert app._normal_dirty, "the live normal-map preview must be redrawn"


def test_a_shelf_drag_hides_its_preview_when_it_leaves_the_mesh(app, tmp_path):
    centre = viewport_centre(app)
    app.begin_decal_drag(write_decal(tmp_path / "vent.png"))
    app.on_mouse_position_event(int(centre[0]), int(centre[1]), 0, 0)
    assert app.dragging_decal_preview is not None

    app.on_mouse_position_event(2, 2, 0, 0)
    assert app.dragging_decal_preview is None


def test_a_shelf_drag_has_a_cursor_thumbnail_over_empty_viewport(app, tmp_path):
    path = write_decal(tmp_path / "vent.png", size=(64, 32))
    app.begin_decal_drag(path)

    ratio = float(app.wnd.pixel_ratio)
    rect_x, rect_y, rect_width, rect_height = app.viewport_rect
    top = app.wnd.buffer_size[1] - (rect_y + rect_height)
    app._mouse = ((rect_x + rect_width - 20) / ratio,
                  (top + 10) / ratio)
    rect = app.dragged_decal_cursor_rect()

    assert rect is not None
    assert (rect[2] - rect[0]) / (rect[3] - rect[1]) == pytest.approx(2.0)
    app.dragging_decal_preview = DecalParams(path=str(path))
    assert app.dragged_decal_cursor_rect() is None, \
        "the projected surface preview replaces the cursor thumbnail"


def test_a_decal_dragged_onto_the_model_lands_where_it_is_let_go(app, tmp_path):
    centre = viewport_centre(app)
    x, y = int(centre[0]), int(centre[1])
    # From the same coordinates the events carry, not the unrounded middle.
    expected = app.surface_uv_at((float(x), float(y)))
    assert expected is not None

    app.begin_decal_drag(write_decal(tmp_path / "vent.png"))
    app.on_mouse_press_event(x, y, app.wnd.mouse.left)
    app.on_mouse_release_event(x, y, app.wnd.mouse.left)

    assert len(app.decals) == 1
    assert app.dragging_decal_preview is None
    placed = app.decals[0]
    assert (placed.center_u, placed.center_v) == pytest.approx(expected, abs=1e-6)
    assert app.selected_decal is placed, "and it is the one being inspected"


def test_a_decal_dropped_off_the_model_is_not_placed(app, tmp_path):
    """A drag abandoned, not a decal at the origin of the atlas."""
    app.begin_decal_drag(write_decal(tmp_path / "vent.png"))
    app.on_mouse_press_event(2, 2, app.wnd.mouse.left)
    app.on_mouse_release_event(2, 2, app.wnd.mouse.left)

    assert app.decals == []
    assert app.dragging_decal is None, "and the drag is over either way"


def test_dropping_does_not_also_count_as_a_click(app, tmp_path):
    """One release, one meaning. The release that places a decal must not then
    be read as a click on the surface as well -- the two would race to say what
    just happened, and the pick would win and report finding what the drop had
    only that moment put there."""
    centre = viewport_centre(app)
    app.begin_decal_drag(write_decal(tmp_path / "vent.png"))
    x, y = int(centre[0]), int(centre[1])
    app.on_mouse_press_event(x, y, app.wnd.mouse.left)
    app.on_mouse_release_event(x, y, app.wnd.mouse.left)

    assert app.selected_decal is not None
    assert "Placed" in app.status, app.status


# --------------------------------------------------------------------------
# the outline
# --------------------------------------------------------------------------

def test_the_outline_walks_the_border_rather_than_cornering_it():
    """Four corners would draw straight lines across a UV island the surface
    bends through. Each edge is subdivided so every point is looked up."""
    params = DecalParams(path="a", center_u=0.5, center_v=0.5, scale=0.4)
    points = outline_uvs(params, samples=8)

    assert len(points) == 4 * 8 + 1, "four edges, subdivided, closed"
    assert points[0] == pytest.approx(points[-1]), "the loop closes"
    assert points[:, 0].min() == pytest.approx(0.3)
    assert points[:, 0].max() == pytest.approx(0.7)


def test_the_outline_turns_with_the_decal():
    upright = outline_uvs(DecalParams(path="a", scale=0.4))
    turned = outline_uvs(DecalParams(path="a", scale=0.4, rotation=45.0))

    assert turned[:, 0].max() > upright[:, 0].max() + 0.05, "a turned square is wider"


def test_the_selected_decal_has_an_outline_on_the_mesh(app, tmp_path):
    centre = viewport_centre(app)
    uv = app.surface_uv_at(centre)
    app.add_decal(write_decal(tmp_path / "a.png"), center=uv)
    app.decals[-1].scale = 0.15
    app.on_render(0.0, 1 / 60.0)

    outline = app.decal_outline()
    assert outline is not None and len(outline) > 4
    assert outline.shape[1] == 3, "world-space points, to project like any other"


def test_nothing_selected_is_no_outline(app, tmp_path):
    app.add_decal(write_decal(tmp_path / "a.png"))
    app.on_render(0.0, 1 / 60.0)
    assert app.decal_outline() is not None

    app.select_decal(None)
    assert app.decal_outline() is None


def test_the_outline_follows_the_selection(app, tmp_path):
    path = write_decal(tmp_path / "a.png")
    app.add_decal(path, center=(1 / 6, 0.25))
    app.decals[-1].scale = 0.1
    app.add_decal(path, center=(1 / 2, 0.25))
    app.decals[-1].scale = 0.1
    app.on_render(0.0, 1 / 60.0)

    app.select_decal(0)
    first = app.decal_outline()
    app.select_decal(1)
    second = app.decal_outline()

    assert first is not None and second is not None
    assert not np.allclose(first.mean(axis=0), second.mean(axis=0))


def test_no_outline_while_the_decal_is_being_moved(app, tmp_path):
    """It is glued to the cursor, the border would be rebuilt every frame, and
    an outline around something already following the pointer says nothing."""
    app.add_decal(write_decal(tmp_path / "a.png"), center=(1 / 6, 0.25))
    app.on_render(0.0, 1 / 60.0)
    assert app.decal_outline() is not None

    app.begin_decal_placement()
    assert app.decal_outline() is None


# --------------------------------------------------------------------------
# the panel's own state
# --------------------------------------------------------------------------

def test_the_decal_tabs_divider_is_remembered(app):
    from render.viewport import _load_prefs

    app.set_decal_split(0.3)
    app.save_prefs()

    assert _load_prefs()["decal_split"] == pytest.approx(0.3)
    assert app.decal_split == pytest.approx(0.3)


def test_the_decal_tab_draws_with_and_without_a_selection(app, tmp_path):
    """Both halves, in one frame, either way -- an unbalanced ImGui stack shows
    up as a later window in the wrong place rather than as an exception."""
    from ui import panel

    for _ in range(2):
        imgui.new_frame()
        panel.draw_panel(app)
        imgui.end_frame()

    app.add_decal(write_decal(tmp_path / "a.png"))
    for _ in range(2):
        imgui.new_frame()
        panel.draw_panel(app)
        imgui.end_frame()
