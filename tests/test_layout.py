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


def test_the_shaded_and_normals_modes_are_the_whole_list():
    from render.viewport import PREVIEW_MODES

    assert [mode.label for mode in PREVIEW_MODES] == ["Shaded", "Normals"]
