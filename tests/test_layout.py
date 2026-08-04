"""The window's frame: a navigation bar, a sidebar, and what is left for the model.

The chrome is fixed rather than floating, so the 3D view is no longer the whole
window. Three things have to agree about that rect or the app lies about where
things are: what is drawn, the projection it is drawn with, and the ray a cursor
turns into.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui.panel import NAVBAR_HEIGHT, PANEL_WIDTH, STATUS_BAR_HEIGHT  # noqa: E402


@pytest.fixture
def app(tmp_path):
    import moderngl_window as mglw
    import trimesh
    from imgui_bundle import imgui

    from render.viewport import MeshMapApp

    trimesh.creation.box().export(tmp_path / "box.obj")
    MeshMapApp.initial_mesh = str(tmp_path / "box.obj")
    MeshMapApp.initial_resolution = 256
    try:
        instance = mglw.create_window_config_instance(MeshMapApp, args=["-wnd", "headless"])
    except Exception as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"no headless window available: {exc}")
    yield instance
    instance.controller.release()
    imgui.destroy_context()
    MeshMapApp.initial_mesh = None


def test_the_view_starts_where_the_chrome_ends(app):
    scale = app.ui_pixel_scale
    width, height = app.wnd.buffer_size
    x, y, view_width, view_height = app.viewport_rect

    assert x == pytest.approx(PANEL_WIDTH * scale, abs=1), "left of the sidebar"
    assert y == pytest.approx(STATUS_BAR_HEIGHT * scale, abs=1), "above the status bar"
    assert x + view_width == width, "and out to the right edge"
    assert view_height == pytest.approx(
        height - (NAVBAR_HEIGHT + STATUS_BAR_HEIGHT) * scale, abs=2
    )


def test_material_view_reserves_a_right_inspector(app):
    """The inspector is docked beside the viewport rather than laid over it."""
    window_width = app.wnd.buffer_size[0]
    app.set_right_inspector_collapsed(False)
    app.sidebar_view = 1

    x, _, view_width, _ = app.viewport_rect
    inspector_width = app.right_inspector_pixels
    assert inspector_width > 0
    assert x + view_width + inspector_width == window_width

    app.sidebar_view = 0
    assert app.right_inspector_pixels == 0
    assert app.viewport_rect[0] + app.viewport_rect[2] == window_width


def test_decal_inspector_never_reserves_a_right_sidebar(app):
    from core.params import DecalParams

    app.set_right_inspector_collapsed(False)
    app.sidebar_view = 2
    assert app.right_inspector_kind is None
    assert app.right_inspector_pixels == 0

    app.decals.append(DecalParams(path="vent.png"))
    app.select_decal(0)
    assert app.right_inspector_kind is None
    assert app.right_inspector_pixels == 0

    app.select_decal(None)
    assert app.right_inspector_kind is None
    assert app.right_inspector_pixels == 0


def test_right_inspector_can_collapse_without_reserving_viewport_space(app):
    app.sidebar_view = 1
    app.set_right_inspector_collapsed(False)
    assert app.right_inspector_pixels > 0

    app.set_right_inspector_collapsed(True)
    assert app.right_inspector_pixels == 0
    assert app.viewport_rect[0] + app.viewport_rect[2] == app.wnd.buffer_size[0]


def test_the_sidebar_never_takes_more_than_half_the_window(app):
    """A panel wider than the thing it describes is not a layout to honour."""
    app.set_ui_scale(3.0)
    x, _, view_width, _ = app.viewport_rect
    assert x <= app.wnd.buffer_size[0] * 0.5
    assert view_width >= app.wnd.buffer_size[0] * 0.5


def test_the_projection_matches_the_rect_not_the_window(app):
    app.on_render(0.0, 1 / 60.0)
    _, _, width, height = app.viewport_rect

    assert app.camera.projection.aspect_ratio == pytest.approx(width / height)
    assert app.camera.projection.aspect_ratio != pytest.approx(app.wnd.aspect_ratio), (
        "the window's own aspect would stretch the model"
    )


def test_the_projection_follows_a_change_of_ui_scale(app):
    app.on_render(0.0, 1 / 60.0)
    before = app.camera.projection.aspect_ratio

    app.set_ui_scale(app.ui_scale * 1.6)  # wider chrome, narrower view
    app.on_render(0.0, 1 / 60.0)

    _, _, width, height = app.viewport_rect
    assert app.camera.projection.aspect_ratio == pytest.approx(width / height)
    assert app.camera.projection.aspect_ratio < before


def test_the_cursor_ray_is_built_from_the_same_rect(app):
    """The middle of the 3D view has to hit the model. Measured from the whole
    window it would land left of it, over the sidebar."""
    x, _, width, height = app.viewport_rect
    ratio = app.wnd.pixel_ratio
    top = NAVBAR_HEIGHT * app.ui_pixel_scale

    middle = ((x + width / 2) / ratio, (top + height / 2) / ratio)
    assert app.surface_uv_at(middle) is not None, "the box is in the middle of the view"

    over_the_sidebar = ((x / 4) / ratio, (top + height / 2) / ratio)
    assert app.surface_uv_at(over_the_sidebar) is None


def test_the_gizmo_sits_in_the_view_not_the_window_corner(app):
    app.on_render(0.0, 1 / 60.0)
    center_x, center_y = app._gizmo_center()
    x, _, width, _ = app.viewport_rect

    assert center_x <= x + width, "inside the right edge of the view"
    assert center_y >= NAVBAR_HEIGHT * app.ui_pixel_scale, "below the navigation bar"


def test_every_preview_mode_has_a_number_key_and_no_more(app):
    """A shortcut for a mode that does not exist indexes off the end."""
    from render.viewport import PREVIEW_MODES

    keys = app.wnd.keys
    numbers = (keys.NUMBER_1, keys.NUMBER_2, keys.NUMBER_3, keys.NUMBER_4)
    if len(set(numbers)) != len(numbers):
        # The headless backend defines no number keys, so they all compare
        # equal and there is nothing here to tell apart.
        pytest.skip("this window backend has no number keys")

    for index, key_code in enumerate(numbers):
        app.preview_index = 0
        app.on_key_event(key_code, keys.ACTION_PRESS, app.wnd.modifiers)
        expected = index if index < len(PREVIEW_MODES) else 0
        assert app.preview_index == expected
        assert app.preview_index < len(PREVIEW_MODES)


@pytest.fixture
def starter_app():
    """The app as it opens with no mesh argument at all."""
    import moderngl_window as mglw
    from imgui_bundle import imgui

    from render.viewport import MeshMapApp

    MeshMapApp.initial_mesh = None
    MeshMapApp.initial_resolution = 256
    try:
        instance = mglw.create_window_config_instance(MeshMapApp, args=["-wnd", "headless"])
    except Exception as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"no headless window available: {exc}")
    yield instance
    instance.controller.release()
    imgui.destroy_context()


def test_the_app_opens_on_the_starter_cube(starter_app):
    """No mesh argument, no empty viewport -- something to work on straight
    away, without importing anything."""
    assert starter_app.mesh is not None
    assert starter_app.mesh_info.path == "Chamfered Cube"
    assert starter_app.mesh_info.has_uvs, "so the bake goes into its own UVs"
    assert not starter_app.controller.bevel_params.enabled, "sharp, as asked"
    assert "cube" in starter_app.status.lower()


def test_the_app_opens_with_a_texture_to_work_on(starter_app):
    """Both ends ready: a mesh to bake, and a texture to put on it."""
    assert starter_app.texture is not None
    assert starter_app.texture.name == "Texture 01"


def test_new_material_joins_the_project_without_changing_assignment(starter_app):
    before = len(starter_app.textures)
    assigned = starter_app.mesh_material_index
    starter_app.create_texture()

    assert len(starter_app.textures) == before + 1
    assert starter_app.texture_index == before
    assert starter_app.mesh_material_index == assigned


def test_selecting_a_material_does_not_assign_it_to_the_mesh(starter_app):
    assigned = starter_app.mesh_material_index
    starter_app.create_texture()
    starter_app.select_texture(len(starter_app.textures) - 1)

    assert starter_app.mesh_material_index == assigned


def test_mesh_material_changes_only_through_explicit_assignment(starter_app):
    starter_app.create_texture()
    new_material = starter_app.texture_index
    starter_app.select_scene_mesh(0)
    starter_app.assign_mesh_material(new_material)

    assert starter_app.mesh_material_index == new_material

    starter_app.assign_mesh_material(-1)
    assert starter_app.mesh_material_index == -1


def test_scene_mesh_selection_has_a_viewport_outline(starter_app):
    starter_app.select_scene_mesh(0)

    assert starter_app.mesh_selected_index == 0
    assert starter_app.decal_index == -1
    outline = starter_app.mesh_selection_outline()
    assert outline is not None
    assert len(outline) == 12


def test_mesh_tree_name_is_editable_without_renaming_the_source(starter_app):
    source_path = starter_app.mesh_info.path
    starter_app.begin_mesh_rename()
    assert starter_app.mesh_renaming and starter_app.mesh_renaming_opened

    starter_app.mesh_name = "  Hero Cube  "
    starter_app.end_mesh_rename()

    assert starter_app.mesh_name == "Hero Cube"
    assert starter_app.mesh_info.path == source_path


def test_the_double_click_thresholds_are_loosened(starter_app):
    """Renaming is a double-click, and ImGui's defaults make it a hard one:
    6 physical pixels of travel is a third of a hand's on a HiDPI display,
    and 0.30s is quicker than a comfortable double-click."""
    from imgui_bundle import imgui

    io = imgui.get_io()
    assert io.mouse_double_click_time >= 0.5
    assert io.mouse_double_click_max_dist >= 6.0 * starter_app.ui_pixel_scale - 1e-6

    starter_app.set_ui_scale(2.5)
    assert io.mouse_double_click_max_dist == pytest.approx(
        6.0 * starter_app.ui_pixel_scale
    ), "and the distance follows the scale, since ImGui is driven in pixels"


def test_the_sidebar_can_be_resized_and_the_view_follows(starter_app):
    """One number decides the sidebar's width: the panel draws itself that
    wide and the 3D view starts where it ends, so the two cannot disagree."""
    from ui.panel import PANEL_WIDTH

    assert starter_app.sidebar_width == pytest.approx(PANEL_WIDTH)
    before_x, _, before_width, _ = starter_app.viewport_rect

    starter_app.set_sidebar_width(PANEL_WIDTH - 120)
    x, _, width, _ = starter_app.viewport_rect
    assert x == starter_app.sidebar_pixels
    assert x < before_x and width > before_width, "the view takes what it gives up"

    starter_app.on_render(0.0, 1 / 60.0)
    assert starter_app.camera.projection.aspect_ratio == pytest.approx(
        width / starter_app.viewport_rect[3]
    )


def test_the_edge_is_grabbable_from_either_side_of_it(starter_app):
    """Nothing is drawn there: hovering the sidebar's own border is what makes
    it live. The band straddles the edge, biased outward so a press meant for
    the edge does not land on a control near the panel's border."""
    edge = starter_app.sidebar_pixels
    middle = starter_app.wnd.buffer_size[1] * 0.5

    assert starter_app.over_sidebar_edge((edge, middle))
    assert starter_app.over_sidebar_edge((edge - 3, middle)), "just inside"
    assert starter_app.over_sidebar_edge((edge + 3, middle)), "just outside"
    assert not starter_app.over_sidebar_edge((edge - 40, middle)), "over the panel"
    assert not starter_app.over_sidebar_edge((edge + 40, middle)), "over the model"

    # And it does not reach far into the view: a resize arrow while the cursor
    # is plainly over the model reads as the app losing track of the panel.
    from ui.panel import SIDEBAR_GRAB_OUTSIDE

    assert SIDEBAR_GRAB_OUTSIDE * starter_app.ui_pixel_scale <= 16

    # Not above the navigation bar or below the status bar either.
    assert not starter_app.over_sidebar_edge((edge, 2))
    assert not starter_app.over_sidebar_edge((edge, starter_app.wnd.buffer_size[1] - 2))


def test_hovering_the_edge_asks_for_the_resize_cursor(starter_app):
    """ImGui only records a cursor request and expects the platform layer to
    act on it -- moderngl-window's integration has none, so the app applies it
    itself. Without that, the edge changes nothing on screen and reads as dead.
    """
    from imgui_bundle import imgui

    asked: list = []
    starter_app.gui.sync_mouse_cursor = lambda override=None: asked.append(override)
    middle = int(starter_app.wnd.buffer_size[1] * 0.5)

    starter_app.on_mouse_position_event(int(starter_app.sidebar_pixels), middle, 0, 0)
    starter_app.on_render(0.0, 1 / 60.0)
    assert asked[-1] == imgui.MouseCursor_.resize_ew

    starter_app.on_mouse_position_event(int(starter_app.sidebar_pixels * 0.5), middle, 0, 0)
    starter_app.on_render(0.0, 1 / 60.0)
    assert asked[-1] is None, "elsewhere, whatever ImGui asked for stands"


def test_the_edge_is_claimed_before_imgui_sees_the_press(starter_app):
    """Half the band lies over the panel's last pixels, and ImGui captures
    anything inside its own window -- so asking it first leaves the edge
    working only from the outside, which is not where a hand aims. Nothing is
    forwarded either, or a control near the border reacts to the grab.
    """
    middle = int(starter_app.wnd.buffer_size[1] * 0.5)
    inside = int(starter_app.sidebar_pixels - 4)

    seen: list[str] = []
    starter_app.gui.mouse_press_event = lambda *a: seen.append("press")
    starter_app.gui.mouse_drag_event = lambda *a: seen.append("drag")
    starter_app.gui.mouse_release_event = lambda *a: seen.append("release")

    before = starter_app.sidebar_width
    starter_app.on_mouse_position_event(inside, middle, 0, 0)
    starter_app.on_mouse_press_event(inside, middle, starter_app.wnd.mouse.left)
    assert starter_app._drag_owner == "sidebar", "not handed to the panel"

    starter_app.on_mouse_drag_event(inside + 20, middle, 20, 0)
    starter_app.on_mouse_release_event(inside + 20, middle, starter_app.wnd.mouse.left)

    assert starter_app.sidebar_width > before
    assert seen == [], "ImGui was told nothing about the grab"


def test_a_press_well_inside_the_panel_still_belongs_to_the_panel(starter_app):
    middle = int(starter_app.wnd.buffer_size[1] * 0.5)
    inside = int(starter_app.sidebar_pixels * 0.5)
    before = starter_app.sidebar_width

    starter_app.on_mouse_position_event(inside, middle, 0, 0)
    starter_app.on_render(0.0, 1 / 60.0)
    starter_app.on_mouse_press_event(inside, middle, starter_app.wnd.mouse.left)
    starter_app.on_mouse_drag_event(inside + 30, middle, 30, 0)

    assert starter_app._drag_owner != "sidebar"
    assert starter_app.sidebar_width == pytest.approx(before)


def test_dragging_the_edge_resizes_and_nothing_else(starter_app):
    """The press belongs to the edge: it must not also orbit the camera or
    land on whatever the panel has near its border."""
    middle = int(starter_app.wnd.buffer_size[1] * 0.5)
    before_width = starter_app.sidebar_width
    before_eye = tuple(starter_app.camera.eye.to_list())

    starter_app.on_mouse_position_event(int(starter_app.sidebar_pixels), middle, 0, 0)
    starter_app.on_mouse_press_event(int(starter_app.sidebar_pixels), middle,
                                     starter_app.wnd.mouse.left)
    assert starter_app._drag_owner == "sidebar"

    for _ in range(4):
        starter_app.on_mouse_drag_event(
            int(starter_app.sidebar_pixels + 12), middle, 12, 3
        )
    starter_app.on_mouse_release_event(int(starter_app.sidebar_pixels), middle,
                                       starter_app.wnd.mouse.left)

    assert starter_app.sidebar_width > before_width
    assert tuple(starter_app.camera.eye.to_list()) == pytest.approx(before_eye)
    assert starter_app._drag_owner is None


def test_the_sidebar_stays_between_useful_widths(starter_app):
    from render.viewport import SIDEBAR_MAX, SIDEBAR_MIN

    starter_app.set_sidebar_width(10.0)
    assert starter_app.sidebar_width == pytest.approx(SIDEBAR_MIN)

    starter_app.set_sidebar_width(5000.0)
    assert starter_app.sidebar_width == pytest.approx(SIDEBAR_MAX)
    # And however wide it is asked to be, it never takes more than half the
    # window -- the model is the thing being worked on.
    assert starter_app.sidebar_pixels <= starter_app.wnd.buffer_size[0] * 0.5


def test_the_sidebar_width_is_measured_in_unscaled_pixels(starter_app):
    """So it means the same thing at any UI scale, like every other panel
    dimension in the app."""
    starter_app.set_sidebar_width(400.0)
    narrow = starter_app.sidebar_pixels

    starter_app.set_ui_scale(starter_app.ui_scale * 1.5)
    assert starter_app.sidebar_width == pytest.approx(400.0), "unchanged in itself"
    assert starter_app.sidebar_pixels > narrow, "but bigger on screen"


def test_the_texture_panes_start_even_and_stay_usable(starter_app):
    """Half each, and neither draggable shut -- a pane too short to show a row
    has been closed by accident, and the splitter is the only way back."""
    from render.viewport import MIN_SPLIT

    assert starter_app.texture_split == pytest.approx(0.5)
    assert starter_app.mesh_split == pytest.approx(0.5)
    assert starter_app.explorer_split == pytest.approx(1.0 / 3.0)

    starter_app.set_texture_split(0.9)
    assert starter_app.texture_split == pytest.approx(1.0 - MIN_SPLIT)
    starter_app.set_texture_split(-2.0)
    assert starter_app.texture_split == pytest.approx(MIN_SPLIT)

    starter_app.set_texture_split(0.7)
    assert starter_app.texture_split == pytest.approx(0.7)
    starter_app.set_mesh_split(0.75)
    assert starter_app.mesh_split == pytest.approx(0.75)
    starter_app.set_explorer_split(0.42)
    assert starter_app.explorer_split == pytest.approx(0.42)
    starter_app.set_explorer_split(0.99)
    assert starter_app.explorer_split == pytest.approx(1.0 - MIN_SPLIT)


def test_the_layout_is_remembered_between_runs(starter_app, isolated_settings):
    """Preferences like the UI scale, not something to set up again every
    session."""
    import json

    starter_app.set_texture_split(0.32)
    starter_app.set_mesh_split(0.43)
    starter_app.set_explorer_split(0.37)
    starter_app.set_sidebar_width(512.0)
    starter_app.set_right_inspector_collapsed(True)
    starter_app.save_prefs()

    stored = json.loads((isolated_settings / "prefs.json").read_text())
    assert stored["texture_split"] == pytest.approx(0.32)
    assert stored["mesh_split"] == pytest.approx(0.43)
    assert stored["explorer_split"] == pytest.approx(0.37)
    assert stored["sidebar_width"] == pytest.approx(512.0)
    assert stored["right_inspector_collapsed"] is True

    from render.viewport import _load_prefs

    assert _load_prefs()["texture_split"] == pytest.approx(0.32)


def test_console_is_always_on_and_deduplicated(starter_app):
    starter_app.set_status("ordinary failure", error=True)
    starter_app.console_log("WARNING", "same warning", key="flat")
    starter_app.console_log("WARNING", "same warning", key="flat")

    assert sum(message == "same warning" for _, message in starter_app.console_messages) == 1
    assert ("ERROR", "ordinary failure") in starter_app.console_messages

    starter_app.clear_console()
    assert starter_app.console_messages == []


def test_scene_explorer_nests_materials_and_decals_under_the_mesh(monkeypatch):
    """The persistent upper pane mirrors what is actually in the viewport."""
    from types import SimpleNamespace

    from core.layers import ColorSlot
    from core.params import DecalParams
    from ui import panel

    labels = []

    def tree_node(label, _flags=0):
        labels.append(label.split("##", 1)[0])
        return True

    monkeypatch.setattr(panel.imgui, "tree_node_ex", tree_node)
    monkeypatch.setattr(panel.imgui, "tree_pop", lambda: None)
    monkeypatch.setattr(panel.imgui, "is_item_clicked", lambda: False)
    monkeypatch.setattr(panel.imgui, "is_item_toggled_open", lambda: False)
    monkeypatch.setattr(panel.imgui, "push_id", lambda _value: None)
    monkeypatch.setattr(panel.imgui, "pop_id", lambda: None)
    monkeypatch.setattr(panel.imgui, "same_line", lambda *args: None)
    monkeypatch.setattr(panel.imgui, "text_colored", lambda *args: None)
    monkeypatch.setattr(panel, "_draw_decal_visibility_icon", lambda _enabled: False)
    monkeypatch.setattr(panel, "_tooltip", lambda _text: None)

    app = SimpleNamespace(
        mesh=object(), mesh_info=object(), mesh_name="Chamfered Cube",
        mesh_selected_index=-1, mesh_renaming=False,
        textures=[ColorSlot(name="Steel"), ColorSlot(name="Red paint")],
        mesh_material_index=0,
        decals=[DecalParams(path="vent.png", name="Vent", texture_index=1)],
        decal_index=-1, decal_renaming_index=-1, sidebar_view=0,
    )
    panel._draw_scene_explorer(app)

    assert labels == [
        "Chamfered Cube",
        "Materials", "Steel",
        "Decals (1)", "Vent", "Material: Red paint",
    ]


def test_material_view_shows_only_the_material_tree(monkeypatch):
    """Layer details live in the independent right sidebar."""
    from types import SimpleNamespace

    from core.layers import ColorSlot
    from ui import panel

    drawn = []
    material = ColorSlot(name="Steel")
    app = SimpleNamespace(texture=material, textures=[material])

    monkeypatch.setattr(panel, "_draw_texture_picker", lambda _app: None)
    monkeypatch.setattr(panel, "_draw_texture_warnings", lambda _app: None)
    monkeypatch.setattr(panel, "_begin_panel", lambda name, _size: drawn.append(name))
    monkeypatch.setattr(panel, "_end_panel", lambda: None)
    monkeypatch.setattr(panel, "_draw_texture_tree", lambda _app: drawn.append("tree"))
    monkeypatch.setattr(panel.imgui, "text_colored", lambda *args: None)

    panel._draw_texture_tab(app)

    assert drawn == ["texture_tree", "tree"]


def test_material_tree_draws_only_the_selected_material(monkeypatch):
    """The picker owns material selection; the tree owns only its layers."""
    from types import SimpleNamespace

    from core.layers import ColorSlot
    from ui import panel

    steel = ColorSlot(name="Steel")
    paint = ColorSlot(name="Paint")
    rows = []
    app = SimpleNamespace(
        texture=paint,
        texture_index=1,
        textures=[steel, paint],
        texture_path=(),
        renaming_path=None,
        ui_pixel_scale=1.0,
    )

    monkeypatch.setattr(panel.imgui, "push_id", lambda _value: None)
    monkeypatch.setattr(panel.imgui, "pop_id", lambda: None)
    monkeypatch.setattr(panel.imgui, "color_button", lambda *_args: None)
    monkeypatch.setattr(panel.imgui, "same_line", lambda *args: None)
    monkeypatch.setattr(
        panel, "_draw_tree_row",
        lambda _app, material_index, path, slot: rows.append(
            (material_index, path, slot.name)
        ),
    )

    panel._draw_texture_tree(app)

    assert rows == [(1, (), "Paint")]


def test_decal_menu_has_a_direct_import_action(monkeypatch):
    """Import lives in the top-level Decal menu beside File."""
    from types import SimpleNamespace

    from ui import panel

    opened = []
    dialog = object()
    app = SimpleNamespace(
        decal_dialog=None,
    )

    monkeypatch.setattr(panel.imgui, "button", lambda *_args: False)
    monkeypatch.setattr(panel.imgui, "get_item_rect_min", lambda: panel.imgui.ImVec2(0, 0))
    monkeypatch.setattr(panel.imgui, "get_item_rect_max", lambda: panel.imgui.ImVec2(0, 20))
    monkeypatch.setattr(panel.imgui, "set_next_window_pos", lambda *_args: None)
    monkeypatch.setattr(panel.imgui, "begin_popup", lambda *_args: True)
    monkeypatch.setattr(panel.imgui, "end_popup", lambda: None)
    monkeypatch.setattr(
        panel.imgui, "menu_item_simple",
        lambda label, *_args: label == "Import decal...",
    )
    monkeypatch.setattr(panel, "_tooltip", lambda _text: None)
    monkeypatch.setattr(
        panel.pfd, "open_file",
        lambda title, start, filters: opened.append((title, start, filters)) or dialog,
    )

    panel._draw_decal_menu(app)

    assert app.decal_dialog is dialog
    assert opened and opened[0][0] == "Import decal"


def test_decal_menu_has_a_direct_add_text_action(monkeypatch):
    """Text creation is the second action in the top-level Decal menu."""
    from types import SimpleNamespace

    from ui import panel

    created = []
    app = SimpleNamespace(
        decal_dialog=None,
        add_text_decal=lambda: created.append("Text"),
    )

    monkeypatch.setattr(panel.imgui, "button", lambda *_args: False)
    monkeypatch.setattr(panel.imgui, "get_item_rect_min", lambda: panel.imgui.ImVec2(0, 0))
    monkeypatch.setattr(panel.imgui, "get_item_rect_max", lambda: panel.imgui.ImVec2(0, 20))
    monkeypatch.setattr(panel.imgui, "set_next_window_pos", lambda *_args: None)
    monkeypatch.setattr(panel.imgui, "begin_popup", lambda *_args: True)
    monkeypatch.setattr(panel.imgui, "end_popup", lambda: None)
    monkeypatch.setattr(
        panel.imgui, "menu_item_simple",
        lambda label, *_args: label == "Add text",
    )
    monkeypatch.setattr(panel, "_tooltip", lambda _text: None)

    panel._draw_decal_menu(app)

    assert created == ["Text"]


def test_decal_view_draws_selected_details_on_the_left(monkeypatch):
    """The Decal tool owns its inspector instead of delegating it rightward."""
    from types import SimpleNamespace

    from ui import panel

    drawn = []
    app = SimpleNamespace(selected_decal=object())
    monkeypatch.setattr(
        panel, "_begin_panel", lambda name, _size: drawn.append(("begin", name))
    )
    monkeypatch.setattr(panel, "_draw_decal_inspector", lambda _app: drawn.append("details"))
    monkeypatch.setattr(panel, "_end_panel", lambda: drawn.append("end"))

    panel._draw_decal_tab(app)

    assert drawn == [("begin", "decal_inspector"), "details", "end"]


def test_a_rename_cannot_get_stuck(starter_app):
    """The field takes focus once, when it opens. Asking every frame is what
    makes a rename impossible to leave: if something else held focus at the
    time, the field would never activate, never report being deactivated, and
    grab at the keyboard for the rest of the session."""
    from imgui_bundle import imgui

    from ui import panel

    def draw_texture_tab() -> None:
        imgui.new_frame()
        imgui.begin("Parameters")
        panel._draw_texture_tab(starter_app)
        imgui.end()
        imgui.end_frame()

    starter_app.begin_rename(())
    assert starter_app.renaming_opened, "focus is owed"

    draw_texture_tab()
    assert not starter_app.renaming_opened, "and taken exactly once"
    assert starter_app.renaming_path == (), "still renaming, now holding focus"

    starter_app.end_rename()
    assert starter_app.renaming_path is None


def test_an_explicit_mesh_wins_over_the_starter_cube(tmp_path):
    import moderngl_window as mglw
    import trimesh
    from imgui_bundle import imgui

    from render.viewport import MeshMapApp

    trimesh.creation.icosphere(subdivisions=1).export(tmp_path / "ball.obj")
    MeshMapApp.initial_mesh = str(tmp_path / "ball.obj")
    try:
        instance = mglw.create_window_config_instance(MeshMapApp, args=["-wnd", "headless"])
    except Exception as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"no headless window available: {exc}")

    try:
        assert instance.mesh_info.path.endswith("ball.obj")
        assert instance.mesh_info.faces > 12
    finally:
        instance.controller.release()
        imgui.destroy_context()
        MeshMapApp.initial_mesh = None


def test_the_starter_cube_bakes_into_its_own_uvs(starter_app):
    """The whole point of unwrapping it: no xatlas atlas, so anything exported
    fits the cube itself."""
    import time

    starter_app.request_bake()
    deadline = time.monotonic() + 30.0
    while starter_app.controller.running and time.monotonic() < deadline:
        starter_app.controller.pump()
        time.sleep(0.002)

    assert starter_app.controller.error is None
    assert starter_app.controller.unwrap_result.source == "source"
    assert starter_app.controller.curvature_map is not None


def test_the_shaded_and_normals_modes_are_the_whole_list():
    from render.viewport import PREVIEW_MODES

    assert [mode.label for mode in PREVIEW_MODES] == ["Shaded", "Normals"]


def test_color_input_tooltip_waits_for_a_quiet_second(monkeypatch):
    from ui import panel

    clock = [10.0]
    shown: list[str] = []
    monkeypatch.setattr(panel.imgui, "get_item_id", lambda: 71)
    monkeypatch.setattr(panel.imgui, "get_time", lambda: clock[0])
    monkeypatch.setattr(panel.imgui, "is_item_hovered", lambda _flags: True)
    monkeypatch.setattr(panel.imgui, "is_item_active", lambda: False)
    monkeypatch.setattr(panel.imgui, "is_mouse_dragging", lambda _button: False)
    monkeypatch.setattr(panel.imgui, "set_item_tooltip", shown.append)
    panel._input_tooltip_item = None

    panel._delayed_input_tooltip("help")
    clock[0] = 10.99
    panel._delayed_input_tooltip("help")
    assert shown == []

    clock[0] = 11.0
    panel._delayed_input_tooltip("help")
    assert shown == ["help"]

    # Updating the field cancels the visible tooltip and restarts its delay.
    clock[0] = 11.1
    panel._delayed_input_tooltip("help", changed=True)
    clock[0] = 12.0
    panel._delayed_input_tooltip("help")
    assert shown == ["help"]


def test_inline_color_value_editor_survives_its_activation_frame(monkeypatch):
    """The preceding slider deactivates as it becomes an input with the same ID."""
    from ui import panel

    focused: list[bool] = []
    monkeypatch.setattr(panel.imgui, "get_id", lambda _label: 91)
    monkeypatch.setattr(panel.imgui, "set_keyboard_focus_here", lambda: focused.append(True))
    monkeypatch.setattr(
        panel.imgui, "input_float",
        lambda *_args, **_kwargs: (False, 0.5),
    )
    monkeypatch.setattr(panel.imgui, "is_item_deactivated", lambda: True)
    panel._filled_slider_edit_item = 91
    panel._filled_slider_edit_opened = True

    panel._filled_slider_float("Metallic", 0.5, 0.0, 1.0)

    assert focused == [True]
    assert panel._filled_slider_edit_item == 91, "typing must be possible next frame"
    assert not panel._filled_slider_edit_opened

    # A later deactivation really is clicking away and should end the editor.
    panel._filled_slider_float("Metallic", 0.5, 0.0, 1.0)
    assert panel._filled_slider_edit_item is None
