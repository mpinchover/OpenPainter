"""ImGui control panel.

Sliders are grouped by which half of the pipeline they belong to, because that
is the distinction that matters when using the tool: everything under "Bake"
costs seconds and needs an explicit re-bake, everything under "Edge wear"
updates the moment the handle moves.

The names, ranges and defaults are ArmorPaint's: the bake block is its
Curvature Texture node plus the bake tool's Smooth and Axis, and the live block
is the EdgeWear001 node group.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd

from core.layers import (
    MAX_DEPTH,
    MAX_EMISSION,
    SLOT_KINDS,
    ColorSlot,
    MaskLayer,
    convert_slot,
    depth,
    describe,
    kind_of,
    mask_count,
    path_labels,
    set_slot,
    walk,
)
from core.params import (
    BAKE_AXES,
    RESOLUTIONS,
    DecalArrayModifier,
    DecalMirrorModifier,
)

if TYPE_CHECKING:  # pragma: no cover
    from render.viewport import MeshMapApp

#: Panel geometry in unscaled reference pixels; multiplied by the app's
#: ui_pixel_scale, because ImGui is driven in physical pixels here.
PANEL_WIDTH = 430
#: Width of the clickable decal preview in the Decal tab.
DECAL_THUMBNAIL = 92
STATUS_BAR_HEIGHT = 34
#: Two rows across the top: the menu bar, then the viewport toolbar below it.
#: Everything downstream (the viewport rect, the sidebar's top, the cursor
#: ray) only needs where the 3D view starts, so NAVBAR_HEIGHT stays their
#: combined height rather than forcing every one of those call sites to add
#: the two together itself.
NAVBAR_MENU_HEIGHT = 46
NAVBAR_TOOLBAR_HEIGHT = 40
NAVBAR_HEIGHT = NAVBAR_MENU_HEIGHT + NAVBAR_TOOLBAR_HEIGHT
#: How far either side of the sidebar's edge counts as grabbing it, in unscaled
#: pixels. Narrow and even: reaching further into the view means the cursor
#: turns into a resize arrow while it is plainly over the model, which reads as
#: the app being confused about where the panel ends.
SIDEBAR_GRAB_INSIDE = 6
SIDEBAR_GRAB_OUTSIDE = 6
#: Width of the vertical view switcher on the sidebar's left edge.
SIDEBAR_ICON_RAIL = 44
#: Width of the material inspector docked on the right. It is capped
#: again by ``MeshMapApp.right_inspector_pixels`` on narrow windows.
RIGHT_INSPECTOR_WIDTH = 360
#: Backwards-compatible name for layout code and external theme experiments.
MATERIAL_INSPECTOR_WIDTH = RIGHT_INSPECTOR_WIDTH
#: Textures beyond which the picker grows a search box. Below it, a list this
#: short is quicker to read than to filter.
_SEARCHABLE_FROM = 5
ERROR_COLOR = imgui.ImVec4(1.0, 0.45, 0.42, 1.0)
MUTED_COLOR = imgui.ImVec4(0.62, 0.64, 0.68, 1.0)
WARN_COLOR = imgui.ImVec4(1.0, 0.78, 0.35, 1.0)

# Sidebar cards are deliberately close to the surrounding chrome: enough
# contrast to group controls without turning the inspector into a stack of
# heavy boxes.  Alpha is kept opaque because these sit over the rendered
# viewport on some platforms.
SECTION_BG = imgui.ImVec4(0.105, 0.115, 0.135, 1.0)
SECTION_HEADER_BG = imgui.ImVec4(0.135, 0.150, 0.175, 1.0)

# Color-node input tooltips deliberately wait for a settled hover. Keeping one
# timer is enough because the pointer can only hover one field at a time.
_input_tooltip_item: int | None = None
_input_tooltip_since = 0.0
_INPUT_TOOLTIP_DELAY = 1.0
#: How long a text decal waits after the last keystroke before regenerating.
#: Long enough that a fast typist does not mint a GPU texture per letter,
#: short enough to read as live rather than as a field you have to confirm.
_TEXT_DECAL_LIVE_DELAY = 0.35
_filled_slider_edit_item: int | None = None
_filled_slider_edit_opened = False
_generator_scrub_edit_item: int | None = None
_generator_scrub_edit_opened = False

MESH_FILTERS = [
    "Meshes", "*.fbx *.obj *.glb *.gltf *.stl *.ply *.dae *.3ds *.off",
    "All files", "*",
]

DECAL_FILTERS = [
    "Images", "*.png *.jpg *.jpeg *.tga *.tif *.tiff *.bmp *.exr",
    "All files", "*",
]


def _tooltip(text: str) -> None:
    imgui.set_item_tooltip(text)


def apply_theme() -> None:
    """Apply the project-wide shape and spacing language to ImGui widgets.

    This runs before the scalable style snapshot is taken, so every value is
    subsequently scaled with the rest of the interface.
    """
    style = imgui.get_style()
    style.frame_rounding = 3.0
    style.grab_rounding = 3.0
    style.child_rounding = 4.0
    style.popup_rounding = 4.0
    style.tab_rounding = 3.0
    style.scrollbar_rounding = 4.0
    style.frame_padding = imgui.ImVec2(6.0, 4.0)
    style.item_spacing = imgui.ImVec2(6.0, 5.0)


def _section_heading(title: str) -> None:
    """Draw a plain title on a softly rounded section header.

    ``separator_text`` puts a rule through every title.  A quiet filled header
    reads as the top of a panel instead and leaves more breathing room before
    the first control.
    """
    scale = max(imgui.get_font_size() / 13.0, 1.0)
    draw = imgui.get_window_draw_list()
    cursor = imgui.get_cursor_screen_pos()
    width = max(imgui.get_content_region_avail().x, 1.0)
    height = imgui.get_text_line_height() + 8.0 * scale
    draw.add_rect_filled(
        cursor,
        imgui.ImVec2(cursor.x + width, cursor.y + height),
        imgui.get_color_u32(SECTION_HEADER_BG),
        4.0 * scale,
    )
    imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 4.0 * scale)
    imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + 7.0 * scale)
    imgui.text_unformatted(title)
    imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() - 7.0 * scale)
    imgui.dummy(imgui.ImVec2(0.0, 6.0 * scale))


def _begin_panel(name: str, size: imgui.ImVec2) -> None:
    """Begin one of the sidebar's inspector/tree panels."""
    scale = max(imgui.get_font_size() / 13.0, 1.0)
    imgui.push_style_color(imgui.Col_.child_bg, SECTION_BG)
    imgui.push_style_var(
        imgui.StyleVar_.window_padding, imgui.ImVec2(7.0 * scale, 7.0 * scale)
    )
    imgui.begin_child(
        name,
        size,
        imgui.ChildFlags_.borders | imgui.ChildFlags_.always_use_window_padding,
    )


def _end_panel() -> None:
    imgui.end_child()
    imgui.pop_style_var()
    imgui.pop_style_color()


def _left_label(label: str, width: float = 142.0) -> str:
    """Place a control label before its field and return a hidden ImGui ID."""
    # Small unit tests exercise individual controls without constructing an
    # ImGui frame. The hidden ID is still useful there, while layout calls are
    # only valid during a real frame.
    if imgui.get_current_context() is None:
        return f"##{label}"
    scale = max(imgui.get_font_size() / 13.0, 1.0)
    imgui.align_text_to_frame_padding()
    imgui.text_unformatted(label)
    imgui.same_line(width * scale)
    imgui.set_next_item_width(-1.0)
    return f"##{label}"


def _labeled_combo(label: str, value: int, choices) -> tuple[bool, int]:
    return imgui.combo(_left_label(label), value, choices)


def _labeled_checkbox(label: str, value: bool) -> tuple[bool, bool]:
    hidden = _left_label(label)
    return imgui.checkbox(hidden, value)


def _toolbar_checkbox(label: str, value: bool) -> tuple[bool, bool]:
    """Compact label-first checkbox for the single-line viewport toolbar."""
    imgui.align_text_to_frame_padding()
    imgui.text_unformatted(label)
    imgui.same_line(0.0, 4.0)
    return imgui.checkbox(f"##{label}", value)


def _labeled_drag_float(
    label: str, value: float, minimum: float, maximum: float,
    fmt: str = "%.3f", flags=0,
) -> tuple[bool, float]:
    speed = max((maximum - minimum) / 300.0, 1e-6)
    changed, value = imgui.drag_float(
        _left_label(label), value, speed, minimum, maximum, fmt,
        flags | imgui.SliderFlags_.always_clamp,
    )
    if imgui.is_item_hovered():
        imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
    return changed, value


def _labeled_drag_int(
    label: str, value: int, minimum: int, maximum: int,
) -> tuple[bool, int]:
    changed, value = imgui.drag_int(
        _left_label(label), value, 0.1, minimum, maximum, "%d",
        imgui.SliderFlags_.always_clamp,
    )
    if imgui.is_item_hovered():
        imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
    return changed, value


def _delayed_input_tooltip(text: str, changed: bool = False) -> None:
    """Show help after one quiet second over the current numeric input.

    Clicking, dragging, or changing the value resets the delay. This keeps the
    tooltip out of the way while the user is actively adjusting the control.
    """
    global _input_tooltip_item, _input_tooltip_since
    item = int(imgui.get_item_id())
    now = float(imgui.get_time())
    interacting = changed or imgui.is_item_active() or imgui.is_mouse_dragging(0)
    if not imgui.is_item_hovered(imgui.HoveredFlags_.delay_none) or interacting:
        if _input_tooltip_item == item:
            _input_tooltip_item = None
            _input_tooltip_since = 0.0
        return
    if _input_tooltip_item != item or now < _input_tooltip_since:
        _input_tooltip_item = item
        _input_tooltip_since = now
        return
    if now - _input_tooltip_since >= _INPUT_TOOLTIP_DELAY:
        imgui.set_item_tooltip(text)


def _muted_wrapped(text: str) -> None:
    """Muted text that folds at the panel's edge rather than running past it.

    For lines whose length is not known when they are written -- a list of file
    names, say, which grows with what there is to export.
    """
    imgui.push_style_color(imgui.Col_.text, MUTED_COLOR)
    imgui.push_text_wrap_pos(0.0)
    imgui.text_wrapped(text)
    imgui.pop_text_wrap_pos()
    imgui.pop_style_color()


def draw_panel(app: "MeshMapApp") -> None:
    _draw_navbar(app)
    _draw_parameters(app)
    _draw_right_inspector_sidebar(app)
    _draw_status_bar(app)
    _draw_delete_confirmation(app)
    _pump_dialogs(app)


def _draw_delete_confirmation(app: "MeshMapApp") -> None:
    """Modal confirmation for the keyboard delete shortcut."""
    popup = "Delete selected decal?"
    if app._delete_decal_index is not None and not imgui.is_popup_open(popup):
        imgui.open_popup(popup)

    opened, _ = imgui.begin_popup_modal(
        popup, None, imgui.WindowFlags_.always_auto_resize
    )
    if not opened:
        return

    index = app._delete_decal_index
    valid = index is not None and 0 <= index < len(app.decals)
    name = Path(app.decals[index].path).name if valid else "the selected decal"
    imgui.text(f"Delete {name}?")
    imgui.text_colored(MUTED_COLOR, "This cannot be undone.")
    imgui.spacing()

    confirm = imgui.button("OK", imgui.ImVec2(100 * app.ui_pixel_scale, 0))
    confirm = confirm or imgui.is_key_pressed(imgui.Key.enter, False)
    confirm = confirm or imgui.is_key_pressed(imgui.Key.keypad_enter, False)
    imgui.set_item_default_focus()
    imgui.same_line()
    cancel = imgui.button("Cancel", imgui.ImVec2(100 * app.ui_pixel_scale, 0))
    cancel = cancel or imgui.is_key_pressed(imgui.Key.escape, False)

    if confirm:
        if valid:
            app.remove_decal(index)
        app._delete_decal_index = None
        imgui.close_current_popup()
    elif cancel:
        app._delete_decal_index = None
        imgui.close_current_popup()
    imgui.end_popup()


#: Windows that make up the frame rather than float inside it: fixed where the
#: app puts them, and not something to drag, resize or remember the position of.
_CHROME_FLAGS = (
    imgui.WindowFlags_.no_move
    | imgui.WindowFlags_.no_resize
    | imgui.WindowFlags_.no_collapse
    | imgui.WindowFlags_.no_saved_settings
    | imgui.WindowFlags_.no_bring_to_front_on_focus
)


def exportable_maps(app: "MeshMapApp") -> list[str]:
    """What an export would write right now. See ``MeshMapApp.exportable``."""
    return app.exportable()


def _draw_file_menu(app: "MeshMapApp") -> None:
    """Opening a mesh and writing the maps out: the two ends of the app.

    Both are here rather than in the tabs because neither belongs to one. The
    mesh is what every tab is about, and an export takes whatever each of them
    has produced -- putting either inside one tab makes it look like that tab's
    business.
    """
    if imgui.button("File"):
        imgui.open_popup("##file")
    # Hung under the button, the way a menu is. Left to itself the popup opens
    # at the cursor, which up here means straight over the bar it came from.
    corner = imgui.get_item_rect_min()
    bottom = imgui.get_item_rect_max().y
    imgui.set_next_window_pos(imgui.ImVec2(corner.x, bottom))

    if imgui.begin_popup("##file"):
        if imgui.menu_item_simple("Import mesh..."):
            app.file_dialog = pfd.open_file("Import mesh", str(Path.home()), MESH_FILTERS)

        imgui.separator()

        maps = exportable_maps(app)
        if imgui.menu_item_simple("Export textures...", "E", False, bool(maps)):
            app.request_export()
        if maps:
            _tooltip(
                "Writes " + ", ".join(f"{name}.png" for name in maps) + ".\n\n"
                "All of them in one go: there is nothing to answer here because\n"
                "the answering is done in Settings, where the maps to write, the\n"
                "depth and the default folder live. They all address your\n"
                "original mesh directly, since they all use its own UV map."
            )
        else:
            _tooltip("Nothing to write yet - bake a mesh, or import a decal.")
        imgui.end_popup()


def _draw_decal_menu(app: "MeshMapApp") -> None:
    """Creation actions for image and generated-text decals."""
    if imgui.button("Decal"):
        imgui.open_popup("##decal_menu")
    corner = imgui.get_item_rect_min()
    bottom = imgui.get_item_rect_max().y
    imgui.set_next_window_pos(imgui.ImVec2(corner.x, bottom))

    if imgui.begin_popup("##decal_menu"):
        if imgui.menu_item_simple("Import decal..."):
            app.decal_dialog = pfd.open_file(
                "Import decal", str(Path.home()), DECAL_FILTERS
            )
        _tooltip("Import a normal map or height map as a decal.")

        if imgui.menu_item_simple("Add text"):
            app.add_text_decal()
        _tooltip("Create a white text decal and place it on the mesh.")
        imgui.end_popup()


def _draw_navbar(app: "MeshMapApp") -> None:
    """The two rows across the top: the menu bar, and the viewport toolbar.

    Separate windows rather than one tall one, the same way the status bar
    and the sidebar are their own windows -- each is its own fixed strip of
    chrome, stacked rather than sharing layout state.
    """
    _draw_menu_bar(app)
    _draw_toolbar(app)


def _draw_menu_bar(app: "MeshMapApp") -> None:
    """File and Decal menus, across the very top."""
    scale = app.ui_pixel_scale
    width = app.wnd.buffer_size[0]
    height = NAVBAR_MENU_HEIGHT * scale

    imgui.set_next_window_pos(imgui.ImVec2(0, 0))
    imgui.set_next_window_size(imgui.ImVec2(width, height))
    # No border: the menu bar and the toolbar sit flush against each other,
    # and ImGui's default 1px window border would otherwise draw a visible
    # seam along that shared edge.
    imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)
    imgui.begin("##menu_bar", None, _CHROME_FLAGS | imgui.WindowFlags_.no_decoration)

    _draw_file_menu(app)
    imgui.same_line()
    _draw_decal_menu(app)

    imgui.end()
    imgui.pop_style_var()


def _toolbar_checkbox_width(label: str) -> float:
    """How much horizontal room one ``_toolbar_checkbox`` actually takes,
    for right-aligning a row of them before any of them are drawn."""
    return (
        imgui.calc_text_size(label).x
        + 4.0
        + imgui.get_frame_height()
    )


#: What the Object/Edit dropdown offers, in display order.
_DECAL_TRANSFORM_SPACES = ("Object", "Edit")


def _draw_toolbar(app: "MeshMapApp") -> None:
    """Object/Edit on the left, Lighting/Wireframe/Gizmo right-aligned.

    View state, not parameters -- which is why it lives out here rather than
    inside the sidebar's tabs, where changing what you are looking at would
    mean leaving the thing you are tuning.
    """
    scale = app.ui_pixel_scale
    width = app.wnd.buffer_size[0]
    top = NAVBAR_MENU_HEIGHT * scale
    height = NAVBAR_TOOLBAR_HEIGHT * scale
    margin = 12.0 * scale

    imgui.set_next_window_pos(imgui.ImVec2(0, top))
    imgui.set_next_window_size(imgui.ImVec2(width, height))
    imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)
    imgui.begin("##toolbar", None, _CHROME_FLAGS | imgui.WindowFlags_.no_decoration)

    imgui.set_cursor_pos_x(margin)
    imgui.set_next_item_width(130.0 * scale)
    space_index = 0 if app.decal_transform_space == "object" else 1
    changed, space_index = imgui.combo(
        "##decal_transform_space", space_index, list(_DECAL_TRANSFORM_SPACES)
    )
    if changed:
        app.set_decal_transform_space(_DECAL_TRANSFORM_SPACES[space_index].lower())
    _tooltip(
        "Object: Scale and Rotate move a decal's whole modifier stack as one\n"
        "rigid unit -- an Array's copies grow and spread apart together.\n"
        "Edit: they act on this one decal alone -- every copy grows or spins\n"
        "in place without moving. Move is identical either way.\n"
        "Tab toggles between them."
    )

    orthographic = bool(app.camera.orthographic)
    spacing = imgui.get_style().item_spacing.x
    labels = ("Lighting", "Wireframe", "Gizmo")
    controls_width = sum(_toolbar_checkbox_width(label) for label in labels)
    controls_width += spacing * (len(labels) - 1)
    if orthographic:
        controls_width += spacing + imgui.calc_text_size("orthographic").x

    imgui.same_line(max(margin, width - controls_width - margin))

    _, app.lighting = _toolbar_checkbox("Lighting", app.lighting)
    imgui.same_line()
    _, app.wireframe = _toolbar_checkbox("Wireframe", app.wireframe)
    imgui.same_line()
    _, app.show_gizmo = _toolbar_checkbox("Gizmo", app.show_gizmo)
    _tooltip(
        "The axis balls in the top-right corner. Click one to look straight down\n"
        "that axis in orthographic projection; orbiting returns to perspective."
    )
    if orthographic:
        imgui.same_line()
        imgui.text_colored(MUTED_COLOR, "orthographic")

    imgui.end()
    imgui.pop_style_var()


# --------------------------------------------------------------------------

def _draw_sidebar_icon(kind: int, selected: bool, size: float) -> bool:
    """Draw one glyph-only navigation button without relying on icon fonts."""
    clicked = imgui.invisible_button(f"##sidebar_view_{kind}", imgui.ImVec2(size, size))
    start = imgui.get_item_rect_min()
    end = imgui.get_item_rect_max()
    hovered = imgui.is_item_hovered()
    draw = imgui.get_window_draw_list()

    if selected:
        background = imgui.ImVec4(0.10, 0.48, 0.58, 0.92)
    elif hovered:
        background = imgui.ImVec4(0.30, 0.32, 0.36, 0.85)
    else:
        background = imgui.ImVec4(0.16, 0.17, 0.20, 0.55)
    draw.add_rect_filled(start, end, imgui.get_color_u32(background), 5.0)

    color = imgui.get_color_u32(
        imgui.ImVec4(0.95, 0.98, 1.0, 1.0) if selected
        else imgui.ImVec4(0.72, 0.75, 0.80, 1.0)
    )
    cx = (start.x + end.x) * 0.5
    cy = (start.y + end.y) * 0.5
    radius = size * 0.23
    thickness = max(1.5, size * 0.055)

    if kind == 0:  # Bake: a small mesh.
        left = imgui.ImVec2(cx - radius, cy + radius * 0.72)
        right = imgui.ImVec2(cx + radius, cy + radius * 0.72)
        top = imgui.ImVec2(cx, cy - radius)
        draw.add_line(left, right, color, thickness)
        draw.add_line(right, top, color, thickness)
        draw.add_line(top, left, color, thickness)
        draw.add_line(top, imgui.ImVec2(cx, cy + radius * 0.72), color, thickness)
    elif kind == 1:  # Material: a colour swatch.
        draw.add_circle(imgui.ImVec2(cx, cy), radius, color, 24, thickness)
        draw.add_circle_filled(imgui.ImVec2(cx, cy), radius * 0.48, color, 20)
    elif kind == 2:  # Decal: a projected diamond.
        top = imgui.ImVec2(cx, cy - radius)
        right = imgui.ImVec2(cx + radius, cy)
        bottom = imgui.ImVec2(cx, cy + radius)
        left = imgui.ImVec2(cx - radius, cy)
        draw.add_line(top, right, color, thickness)
        draw.add_line(right, bottom, color, thickness)
        draw.add_line(bottom, left, color, thickness)
        draw.add_line(left, top, color, thickness)
        draw.add_circle_filled(imgui.ImVec2(cx, cy), radius * 0.18, color, 10)
    elif kind == 3:  # Modifiers: repeated copies, staggered like an array.
        step = radius * 0.5
        box = radius * 0.62
        for offset in (-1, 0, 1):
            box_cx = cx + offset * step
            box_cy = cy - offset * step * 0.6
            draw.add_rect(
                imgui.ImVec2(box_cx - box * 0.5, box_cy - box * 0.5),
                imgui.ImVec2(box_cx + box * 0.5, box_cy + box * 0.5),
                color, 2.0, thickness=thickness,
            )
    elif kind == 4:  # Mesh: a small wireframe cube.
        inset = radius * 0.34
        front_min = imgui.ImVec2(cx - radius, cy - radius * 0.58)
        front_max = imgui.ImVec2(cx + radius * 0.48, cy + radius)
        back_min = imgui.ImVec2(front_min.x + inset, front_min.y - inset)
        back_max = imgui.ImVec2(front_max.x + inset, front_max.y - inset)
        draw.add_rect(front_min, front_max, color, 1.0, thickness=thickness)
        draw.add_rect(back_min, back_max, color, 1.0, thickness=thickness)
        for first, second in (
            (front_min, back_min),
            (imgui.ImVec2(front_max.x, front_min.y),
             imgui.ImVec2(back_max.x, back_min.y)),
            (front_max, back_max),
        ):
            draw.add_line(first, second, color, thickness)
    elif kind == 5:  # Library: a shelf of four assets.
        cell = radius * 0.72
        gap = radius * 0.20
        for row in (-1, 1):
            for column in (-1, 1):
                cell_cx = cx + column * (cell + gap) * 0.5
                cell_cy = cy + row * (cell + gap) * 0.5
                draw.add_rect(
                    imgui.ImVec2(cell_cx - cell * 0.5, cell_cy - cell * 0.5),
                    imgui.ImVec2(cell_cx + cell * 0.5, cell_cy + cell * 0.5),
                    color, 1.5, thickness=thickness,
                )
    elif kind == 6:  # Console: a small command prompt.
        draw.add_rect(
            imgui.ImVec2(cx - radius, cy - radius * 0.78),
            imgui.ImVec2(cx + radius, cy + radius * 0.78),
            color, 2.0, thickness=thickness,
        )
        draw.add_line(
            imgui.ImVec2(cx - radius * 0.58, cy - radius * 0.30),
            imgui.ImVec2(cx - radius * 0.12, cy), color, thickness,
        )
        draw.add_line(
            imgui.ImVec2(cx - radius * 0.12, cy),
            imgui.ImVec2(cx - radius * 0.58, cy + radius * 0.30), color, thickness,
        )
        draw.add_line(
            imgui.ImVec2(cx + radius * 0.02, cy + radius * 0.32),
            imgui.ImVec2(cx + radius * 0.58, cy + radius * 0.32), color, thickness,
        )
    else:  # Settings: sliders.
        for offset, knob in ((-0.55, -0.35), (0.0, 0.42), (0.55, -0.05)):
            y = cy + radius * offset
            draw.add_line(
                imgui.ImVec2(cx - radius, y), imgui.ImVec2(cx + radius, y),
                color, thickness,
            )
            draw.add_circle_filled(
                imgui.ImVec2(cx + radius * knob, y), thickness * 1.45, color, 10
            )
    return clicked

def _draw_parameters(app: "MeshMapApp") -> None:
    """The sidebar: everything that is a parameter, filling the left edge.

    Docked rather than floating, and sized to the window every frame, so it is
    part of the frame the same way the navigation and status bars are. The 3D
    view is laid out around it (``MeshMapApp.viewport_rect``), so nothing is
    ever hidden behind it.
    """
    width, height = app.wnd.buffer_size
    scale = app.ui_pixel_scale
    top = NAVBAR_HEIGHT * scale

    imgui.set_next_window_pos(imgui.ImVec2(0, top))
    imgui.set_next_window_size(
        imgui.ImVec2(app.sidebar_pixels, height - top - STATUS_BAR_HEIGHT * scale)
    )
    # The frame itself stays nearly flush to the window; individual sections
    # provide the readable inset instead of compounding several padding layers.
    imgui.push_style_var(
        imgui.StyleVar_.window_padding, imgui.ImVec2(3.0 * scale, 3.0 * scale)
    )
    imgui.begin("Parameters", None, _CHROME_FLAGS | imgui.WindowFlags_.no_title_bar)

    labels = (
        "Bake", "Material", "Decal", "Modifiers", "Mesh", "Library", "Console",
        "Settings",
    )
    drawers = (
        _draw_bake_tab,
        _draw_texture_tab,
        _draw_decal_tab,
        _draw_decal_modifiers_tab,
        _draw_mesh_tab,
        _draw_decal_library_tab,
        _draw_console_tab,
        _draw_settings_tab,
    )
    app.sidebar_view = max(0, min(int(getattr(app, "sidebar_view", 0)), len(labels) - 1))

    # The scene is the stable context for every tool. It starts at one third of
    # the sidebar and the divider below it can be dragged to give either the
    # hierarchy or the active tool more room.
    splitter = 8.0 * scale
    available = max(imgui.get_content_region_avail().y - splitter, 2.0)
    explorer_height = available * app.explorer_split
    _begin_panel("scene_explorer", imgui.ImVec2(0, explorer_height))
    _draw_scene_explorer(app)
    _end_panel()
    _draw_splitter(
        app, splitter, available, "explorer", app.set_explorer_split
    )

    rail_width = SIDEBAR_ICON_RAIL * scale
    button_size = 34 * scale
    imgui.begin_child("##sidebar_tools", imgui.ImVec2(0, 0))
    imgui.push_style_var(
        imgui.StyleVar_.window_padding, imgui.ImVec2(3.0 * scale, 4.0 * scale)
    )
    imgui.begin_child(
        "##sidebar_icon_rail", imgui.ImVec2(rail_width, 0),
        imgui.ChildFlags_.always_use_window_padding,
    )
    for index, label in enumerate(labels):
        if _draw_sidebar_icon(index, app.sidebar_view == index, button_size):
            app.sidebar_view = index
        _tooltip(label)
        imgui.spacing()
    imgui.end_child()
    imgui.pop_style_var()

    imgui.same_line()
    imgui.push_style_var(
        imgui.StyleVar_.window_padding, imgui.ImVec2(4.0 * scale, 4.0 * scale)
    )
    imgui.begin_child(
        "##sidebar_view", imgui.ImVec2(0, 0),
        imgui.ChildFlags_.always_use_window_padding,
    )
    _section_heading(labels[app.sidebar_view])
    drawers[app.sidebar_view](app)
    imgui.end_child()
    imgui.pop_style_var()
    imgui.end_child()
    imgui.end()
    imgui.pop_style_var()


def _draw_scene_explorer(app: "MeshMapApp") -> None:
    """Persistent hierarchy of everything currently rendered in the viewport.

    The renderer currently owns one mesh, but the hierarchy is deliberately
    expressed as mesh -> materials / decals so adding more renderable meshes
    does not require another sidebar redesign. Modifier-generated decal copies
    stay represented by their source decal, matching their non-object status.
    """
    if app.mesh is None or app.mesh_info is None:
        _muted_wrapped("No meshes in the viewport.")
        return

    node_flags = (
        imgui.TreeNodeFlags_.default_open
        | imgui.TreeNodeFlags_.open_on_arrow
        | imgui.TreeNodeFlags_.span_avail_width
    )
    selected_flags = (
        imgui.TreeNodeFlags_.selected if app.mesh_selected_index == 0 else 0
    )
    mesh_open = imgui.tree_node_ex(
        f"{app.mesh_name}##explorer_mesh_0", node_flags | selected_flags
    )
    if imgui.is_item_clicked() and not imgui.is_item_toggled_open():
        app.select_scene_mesh(0)
        app.sidebar_view = 4  # Mesh
        if imgui.is_mouse_double_clicked(0):
            app.begin_mesh_rename()

    if not mesh_open:
        return

    if app.mesh_renaming:
        _draw_explorer_mesh_rename(app)

    material_open = imgui.tree_node_ex(
        "Materials##explorer_materials", node_flags
    )
    if material_open:
        material_index = app.mesh_material_index
        if 0 <= material_index < len(app.textures):
            material = app.textures[material_index]
            flags = (
                imgui.TreeNodeFlags_.leaf
                | imgui.TreeNodeFlags_.no_tree_push_on_open
                | imgui.TreeNodeFlags_.span_avail_width
            )
            if app.sidebar_view == 1 and app.texture_index == material_index:
                flags |= imgui.TreeNodeFlags_.selected
            imgui.tree_node_ex(
                f"{describe(material)}##explorer_mesh_material", flags
            )
            if imgui.is_item_clicked():
                app.select_decal(None)
                app.select_texture(material_index)
                app.sidebar_view = 1
        else:
            imgui.text_colored(MUTED_COLOR, "No material assigned")
        imgui.tree_pop()

    decals_open = imgui.tree_node_ex(
        f"Decals ({len(app.decals)})##explorer_decals", node_flags
    )
    if decals_open:
        if not app.decals:
            imgui.text_colored(MUTED_COLOR, "No decals")
        for index, decal in enumerate(app.decals):
            _draw_explorer_decal(app, index, decal, node_flags)
        imgui.tree_pop()

    imgui.tree_pop()


def _draw_explorer_mesh_rename(app: "MeshMapApp") -> None:
    """Edit the mesh name without removing its hierarchy from the explorer."""
    just_opened = app.mesh_renaming_opened
    if just_opened:
        imgui.set_keyboard_focus_here()
        app.mesh_renaming_opened = False
    imgui.set_next_item_width(-1)
    entered, app.mesh_name = imgui.input_text(
        "##explorer_mesh_rename", app.mesh_name,
        imgui.InputTextFlags_.enter_returns_true
        | imgui.InputTextFlags_.auto_select_all,
    )
    if entered or imgui.is_item_deactivated_after_edit():
        app.end_mesh_rename()
    elif not just_opened and not imgui.is_item_active():
        app.end_mesh_rename()


def _draw_explorer_decal(app: "MeshMapApp", index: int, decal, node_flags: int) -> None:
    """One decal object and its single assigned-material child."""
    imgui.push_id(f"explorer_decal_{index}")
    if _draw_decal_visibility_icon(decal.enabled):
        decal.enabled = not decal.enabled
        app.mark_normal_dirty()
    _tooltip("Show or hide this decal.")
    imgui.same_line()

    flags = node_flags
    if index == app.decal_index:
        flags |= imgui.TreeNodeFlags_.selected
    decal_open = imgui.tree_node_ex(f"{decal.display_name()}##node", flags)
    if imgui.is_item_clicked() and not imgui.is_item_toggled_open():
        app.select_decal(index)
        app.sidebar_view = 2
        if imgui.is_mouse_double_clicked(0):
            app.begin_decal_rename(index)

    if decal_open:
        if app.decal_renaming_index == index:
            _draw_explorer_decal_rename(app, decal)

        material_index = decal.texture_index
        material_name = (
            describe(app.textures[material_index])
            if 0 <= material_index < len(app.textures) else "None"
        )
        material_flags = (
            imgui.TreeNodeFlags_.leaf
            | imgui.TreeNodeFlags_.no_tree_push_on_open
            | imgui.TreeNodeFlags_.span_avail_width
        )
        imgui.tree_node_ex(f"Material: {material_name}##material", material_flags)
        imgui.tree_pop()
    imgui.pop_id()


def _draw_explorer_decal_rename(app: "MeshMapApp", decal) -> None:
    """Inline rename field kept beneath an open decal row."""
    just_opened = app.decal_renaming_opened
    if just_opened:
        imgui.set_keyboard_focus_here()
        app.decal_renaming_opened = False
    imgui.set_next_item_width(-1)
    entered, decal.name = imgui.input_text(
        "##explorer_decal_rename", decal.name,
        imgui.InputTextFlags_.enter_returns_true
        | imgui.InputTextFlags_.auto_select_all,
    )
    if entered or imgui.is_item_deactivated_after_edit():
        app.end_decal_rename()
    elif not just_opened and not imgui.is_item_active():
        app.end_decal_rename()


def _draw_bake_tab(app: "MeshMapApp") -> None:
    scale = app.ui_pixel_scale
    # Labels sit to the right of each widget; give them a fixed share so long
    # names ("Threshold falloff") are never clipped at any scale.
    imgui.push_item_width(-170 * scale)

    controller = app.controller
    bake = controller.bake_params

    # -- mesh ------------------------------------------------------------
    changed, app.source_z_up = _labeled_checkbox("Source is Z-up", app.source_z_up)
    _tooltip(
        "Tick only if the file itself stores Z as up, in which case it is used\n"
        "as-is. Leave it off for anything out of Blender: the FBX and glTF\n"
        "exporters both write Y-up, and the importer rotates that into this\n"
        "app's Z-up world the same way Blender's own importer does.\n"
        "Reload the file to apply."
    )

    info = app.mesh_info
    if info is not None:
        imgui.text_colored(
            MUTED_COLOR,
            f"{Path(info.path).name}\n"
            f"{info.vertices:,} verts  {info.faces:,} tris  via {info.backend}\n"
            f"size {info.extents[0]:.3g} x {info.extents[1]:.3g} x {info.extents[2]:.3g}"
            f"  (diagonal {info.scale:.3g})\n"
            f"UV map: {'yes' if info.has_uvs else 'none'}",
        )
        for index, note in enumerate(info.notes):
            app.console_log("WARNING", note, key=f"mesh_note_{index}")
    else:
        imgui.text_colored(MUTED_COLOR, "No mesh loaded.")

    # -- bake stage ------------------------------------------------------
    _section_heading("Bake  (re-bake required)")

    resolution_index = RESOLUTIONS.index(bake.resolution) if bake.resolution in RESOLUTIONS else 1
    changed, resolution_index = _labeled_combo(
        "Bake resolution", resolution_index, [f"{r} x {r}" for r in RESOLUTIONS]
    )
    if changed:
        bake.resolution = RESOLUTIONS[resolution_index]
    _tooltip("Output texture size. Also re-packs the atlas, since padding is in texels.")

    changed, bake.strength = _labeled_drag_float("Strength", bake.strength, 0.0, 2.0)
    _tooltip(
        "Multiplies the lifted normal derivative.\n"
        "ArmorPaint: curvature = pow(...) * strength * 2.0 + offset / 10.0"
    )

    changed, bake.radius = _labeled_drag_float("Radius", bake.radius, 0.0, 4.0)
    _tooltip(
        "Not a distance -- an exponent: pow(curvature, (1 / radius) * 0.25).\n"
        "Larger values lift the derivative harder, which widens the band.\n"
        "ArmorPaint's node caps this at 2; the EdgeWear001 group exposes 0..4."
    )

    changed, bake.offset = _labeled_drag_float("Offset", bake.offset, -2.0, 2.0)
    _tooltip(
        "Added as offset / 10 after the lift. Negative crushes near-flat areas\n"
        "to black -- EdgeWear001 ships it at -2.0 for exactly that."
    )

    changed, bake.smooth = _labeled_drag_int("Smooth", bake.smooth, 0, 5)
    _tooltip(
        "Blur iterations over the baked curvature. Each one bounces the map\n"
        "through a 95%-size target and back, which is how ArmorPaint blurs it."
    )

    changed, bake.axis = _labeled_combo("Axis", bake.axis, list(BAKE_AXES))
    _tooltip(
        "Anything but XYZ multiplies the result by dot(normal, axis), so only\n"
        "edges facing that way wear. Z is up as in Blender, so -Z is gravity."
    )

    # -- bevel ------------------------------------------------------------
    _section_heading("Bevel  (re-bake required)")
    bevel = controller.bevel_params
    _, bevel.enabled = _labeled_checkbox("Bevel sharp edges", bevel.enabled)
    _tooltip(
        "Blender's Bevel modifier, approximated: replaces sharp edges with a\n"
        "narrow strip before baking. A perfectly sharp edge has no width for\n"
        "the curvature bake to put a gradient in, so without this the gradient\n"
        "smears across whole faces and they bake grey instead of the edge\n"
        "baking white. The bevel is never exported -- it inherits your UVs, so\n"
        "the texture still fits your original unbeveled mesh."
    )

    imgui.begin_disabled(not bevel.enabled)
    _, bevel.amount = _labeled_drag_float(
        "Amount (m)", bevel.amount, 0.0001, 0.05, "%.4f", imgui.SliderFlags_.logarithmic
    )
    _tooltip(
        "Blender's Amount under the Offset width type: how far the new boundary\n"
        "edge sits from the original edge. 0.001 (1 mm) is the subtle bevel that\n"
        "catches a highlight without reading as visibly rounded."
    )
    _, bevel.segments = _labeled_drag_int("Segments", bevel.segments, 1, 8)
    _tooltip(
        "Subdivisions across the bevel. 1 is a flat chamfer, 2-3 lightly\n"
        "rounded, more is smoother but adds geometry."
    )
    _, bevel.angle = _labeled_drag_float("Angle", bevel.angle, 1.0, 180.0, "%.0f deg")
    _tooltip(
        "Limit Method: Angle. Only edges whose two faces diverge by more than\n"
        "this get beveled, so box corners qualify and coplanar edges splitting\n"
        "a flat surface are left alone. 30 degrees is Blender's usual choice."
    )

    # A sub-texel bevel is useful diagnostic information, but it should not
    # interrupt the normal inspector. Send it to Console instead.
    if info is not None and info.uv_density > 0.0:
        texels = bevel.amount * info.uv_density * bake.resolution
        if texels < 2.0:
            app.console_log(
                "WARNING",
                f"Bevel is {texels:.1f} texels wide - too thin to bake; "
                "raise Amount or the resolution.",
                key="bevel_too_thin",
            )
    imgui.end_disabled()

    if imgui.collapsing_header("Advanced bake settings"):
        _, bake.dilation = _labeled_drag_int("Seam padding (texels)", bake.dilation, 0, 16)
        _tooltip("Pads chart borders outward so bilinear filtering never samples empty gutter.")

        unwrap_params = controller.unwrap_params
        _, unwrap_params.use_source_uvs = _labeled_checkbox(
            "Bake into source UVs", unwrap_params.use_source_uvs
        )
        _tooltip(
            "Bake into the UV map already on the mesh, so the PNG drops straight\n"
            "onto the model you exported from Blender. Turn this off only for a\n"
            "mesh with no UVs -- xatlas then packs a throwaway atlas that fits\n"
            "nothing but the triangulated OBJ this app writes."
        )

        imgui.begin_disabled(unwrap_params.use_source_uvs)
        _, unwrap_params.padding = _labeled_drag_int("Atlas padding", unwrap_params.padding, 0, 16)
        _, unwrap_params.normal_deviation_weight = _labeled_drag_float(
            "Seam eagerness", unwrap_params.normal_deviation_weight, 0.5, 8.0
        )
        _tooltip("How readily xatlas cuts a seam where the surface bends.")
        _, unwrap_params.brute_force = _labeled_checkbox("Brute-force packing", unwrap_params.brute_force)
        imgui.end_disabled()

    _draw_bake_controls(app)

    imgui.pop_item_width()


def _draw_texture_tab(app: "MeshMapApp") -> None:
    """Material collection and layer tree shown in the left tool pane.

    A texture starts as one flat colour. Changing its type to a mask grows two
    slots underneath -- white and black -- and either of those can be a colour
    or another mask, so the shape is a binary tree of unbounded depth.

    Indenting all of that runs out of panel before it runs out of tree, so this
    pane is devoted to selection while the independent right sidebar edits the
    selected layer. Depth costs a line rather than a column.
    """

    if app.texture is None:
        imgui.text_colored(
            MUTED_COLOR,
            "No material yet.\n\n"
            "A material starts as a single colour. Change its type to a mask --\n"
            "edge wear or noise -- and it grows a white and a black side, each\n"
            "of which can be another colour or another mask.",
        )
        if imgui.button("New material", imgui.ImVec2(-1, 0)):
            app.create_texture()
        return

    _draw_texture_picker(app)
    if app.texture is None:
        # Remove was just pressed. Everything below describes the texture that
        # has this moment stopped existing, and an ImGui frame is drawn top to
        # bottom -- so stop here and let the next frame draw the empty state.
        return

    masks = mask_count(app.texture)
    imgui.text_colored(
        MUTED_COLOR,
        f"{masks} mask{'' if masks == 1 else 's'}, {depth(app.texture)} deep"
        if masks else "a flat colour",
    )

    _draw_texture_warnings(app)

    _begin_panel("texture_tree", imgui.ImVec2(0, 0))
    _draw_texture_tree(app)
    _end_panel()


def _draw_right_inspector_sidebar(app: "MeshMapApp") -> None:
    """The selected material layer, docked on the window's right."""
    kind = app.right_inspector_kind
    if kind is None:
        return

    scale = app.ui_pixel_scale
    window_width, window_height = app.wnd.buffer_size
    top = NAVBAR_HEIGHT * scale
    height = window_height - top - STATUS_BAR_HEIGHT * scale
    if app.right_inspector_collapsed:
        _draw_collapsed_right_inspector(app, window_width, top, height, scale)
        return

    inspector_width = app.right_inspector_pixels
    if inspector_width <= 0:
        return
    imgui.set_next_window_pos(imgui.ImVec2(window_width - inspector_width, top))
    imgui.set_next_window_size(imgui.ImVec2(inspector_width, height))
    imgui.push_style_var(
        imgui.StyleVar_.window_padding, imgui.ImVec2(3.0 * scale, 3.0 * scale)
    )
    imgui.begin(
        "##right_inspector_sidebar", None,
        _CHROME_FLAGS | imgui.WindowFlags_.no_title_bar,
    )
    if imgui.button(">##collapse_right_inspector"):
        app.set_right_inspector_collapsed(True)
    _tooltip("Collapse the properties sidebar to the right edge.")

    _begin_panel(f"{kind}_right_inspector", imgui.ImVec2(0, 0))
    if app.texture is None or app.selected_slot is None:
        _muted_wrapped("Select a material layer from the tree on the left.")
    else:
        _draw_slot_params(app)
    _end_panel()
    imgui.end()
    imgui.pop_style_var()


def _draw_collapsed_right_inspector(
    app: "MeshMapApp", window_width: float, top: float,
    available_height: float, scale: float,
) -> None:
    """Small edge tab that restores a collapsed right inspector."""
    width = 32.0 * scale
    height = 42.0 * scale
    imgui.set_next_window_pos(imgui.ImVec2(
        window_width - width,
        top + max(0.0, (available_height - height) * 0.5),
    ))
    imgui.set_next_window_size(imgui.ImVec2(width, height))
    imgui.push_style_var(
        imgui.StyleVar_.window_padding, imgui.ImVec2(3.0 * scale, 3.0 * scale)
    )
    imgui.begin(
        "##collapsed_right_inspector", None,
        _CHROME_FLAGS | imgui.WindowFlags_.no_title_bar
        | imgui.WindowFlags_.no_scrollbar
        | imgui.WindowFlags_.no_scroll_with_mouse,
    )
    if imgui.button("<##expand_right_inspector", imgui.ImVec2(-1, -1)):
        app.set_right_inspector_collapsed(False)
    _tooltip("Open the properties sidebar.")
    imgui.end()
    imgui.pop_style_var()


def _draw_splitter(
    app: "MeshMapApp", height: float, usable: float,
    name: str = "texture", setter=None,
) -> None:
    """The draggable divider between a tab's two panes.

    ImGui has no splitter of its own; the idiom is an invisible button that
    reports being dragged, which is all a splitter is. The line is drawn by
    hand so there is something to aim at, and the cursor changes on hover so it
    is discoverable without a tooltip nobody reads mid-drag.
    """
    setter = setter or app.set_texture_split
    share = getattr(app, f"{name}_split")
    imgui.invisible_button(f"##{name}_split", imgui.ImVec2(-1, height))

    hovered = imgui.is_item_hovered()
    active = imgui.is_item_active()
    if hovered or active:
        imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ns)
    if active:
        setter(share + imgui.get_io().mouse_delta.y / max(usable, 1.0))
    if imgui.is_item_deactivated():
        app.save_prefs()  # on release, rather than on every pixel of the drag

    top_left, bottom_right = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    middle = (top_left.y + bottom_right.y) * 0.5
    colour = imgui.get_color_u32(
        imgui.Col_.separator_active if active
        else imgui.Col_.separator_hovered if hovered
        else imgui.Col_.separator
    )
    imgui.get_window_draw_list().add_line(
        imgui.ImVec2(top_left.x, middle), imgui.ImVec2(bottom_right.x, middle),
        colour, max(1.0, height * 0.25),
    )


def filter_textures(textures, needle: str) -> list[tuple[int, str]]:
    """The textures whose name contains ``needle``, as ``(index, label)``.

    Case-insensitive and unanchored, because the names being searched are
    things like "Hull plate 02" and the half-remembered part is as often the
    end as the beginning. An empty search matches everything.
    """
    wanted = needle.strip().lower()
    return [
        (index, describe(texture))
        for index, texture in enumerate(textures)
        if not wanted or wanted in describe(texture).lower()
    ]


def _draw_texture_warnings(app: "MeshMapApp") -> None:
    """Send material diagnostics to Console without cluttering the inspector."""
    if depth(app.texture) >= MAX_DEPTH:
        message = f"Nesting stops at {MAX_DEPTH} levels."
        app.console_log("WARNING", message, key="material_max_depth")

    if app.controller.curvature_map is None:
        message = "Bake first - material masks read curvature and position maps."
        app.console_log("WARNING", message, key="material_needs_bake")
    elif app.texture_is_flat and mask_count(app.texture) > 0:
        # Every texel landing on the same side of the mask paints the whole
        # model in one colour, which reads as broken rather than as empty.
        wears = any(
            isinstance(slot, MaskLayer) and slot.kind == "edge_wear"
            for _, slot in walk(app.texture)
        )
        remedy = (
            "  Lower its Threshold, or turn on Bevel:\n"
            "  wear needs a rounded edge to grip." if wears else
            "  Lower its Threshold, or raise Contrast."
        )
        message = "One flat colour: every texel lands on the same side of its mask."
        app.console_log("WARNING", message + remedy.replace("\n", " "), key="material_flat")


def _draw_texture_picker(app: "MeshMapApp") -> None:
    """Which texture is being worked on, out of the ones made so far.

    Every texture stays around once made, so this is how one is gone back to.
    The search box earns its place once there are more than a handful; it is
    skipped below that, where a list of three does not need filtering.
    """
    scale = app.ui_pixel_scale

    if imgui.button("New"):
        app.create_texture()
    _tooltip("Add a material: a single flat colour, to build up from.")
    imgui.same_line()
    if imgui.button("Remove"):
        app.remove_texture()
        return
    _tooltip(
        "Drop this material. The others stay; with none left, nothing is\n"
        "written to color.png."
    )

    imgui.same_line()
    imgui.set_next_item_width(-1)
    if imgui.begin_combo("##textures", describe(app.texture)):
        if len(app.textures) > _SEARCHABLE_FROM:
            if imgui.is_window_appearing():
                imgui.set_keyboard_focus_here()
            imgui.set_next_item_width(-1)
            _, app.texture_filter = imgui.input_text_with_hint(
                "##filter", "Search", app.texture_filter
            )
            imgui.separator()

        matches = filter_textures(app.textures, app.texture_filter)
        for index, label in matches:
            if imgui.selectable(f"{label}##texture{index}", index == app.texture_index)[0]:
                app.select_texture(index)
        if not matches:
            imgui.text_colored(MUTED_COLOR, "Nothing matches.")
        imgui.end_combo()
    else:
        app.texture_filter = ""
    _tooltip(
        "Every material made this session. Double-click its row in the tree\n"
        "below to rename it."
    )


def _draw_slot_params(app: "MeshMapApp") -> None:
    """Everything about the selected slot: what it is, and what it is set to."""
    scale = app.ui_pixel_scale
    path = app.texture_path
    slot = app.selected_slot
    if slot is None:
        return

    imgui.push_item_width(-150 * scale)
    imgui.text_colored(MUTED_COLOR, " > ".join(path_labels(app.texture, path)))

    # No preview of the mask here. The model in the viewport is the preview,
    # at the size and on the surface it will actually be seen at.
    kinds = list(SLOT_KINDS)
    index = kinds.index(kind_of(slot))
    changed, index = _labeled_combo(
        "Type", index, [SLOT_KINDS[kind] for kind in kinds]
    )
    if changed:
        app.set_texture(set_slot(app.texture, path, convert_slot(slot, kinds[index])))
        slot = app.selected_slot
    _tooltip(
        "Colour is a flat fill and the end of a branch.\n"
        "Edge wear reads the curvature bake -- white on the edges.\n"
        "Noise is the field the wear pass breaks up with, thresholded.\n\n"
        "Choosing a mask grows a white and a black side underneath it; going\n"
        "back to a colour takes them with it."
    )
    if isinstance(slot, MaskLayer):
        imgui.text_colored(MUTED_COLOR, f"White: {describe(slot.white)}")
        imgui.text_colored(MUTED_COLOR, f"Black: {describe(slot.black)}")

    imgui.separator()
    if isinstance(slot, ColorSlot):
        # The swatch is the button: clicking it opens the picker in a popup,
        # and clicking away or pressing Escape closes it again. A picker is as
        # tall as it is wide, so leaving one open permanently would spend half
        # the sidebar on a control that is used in bursts.
        imgui.align_text_to_frame_padding()
        imgui.text_unformatted(f"Base color  {slot.auto_label}")
        imgui.same_line(142.0 * scale)
        changed, color = imgui.color_edit3(
            "##color", list(slot.color),
            imgui.ColorEditFlags_.no_inputs | imgui.ColorEditFlags_.no_label,
        )
        if changed:
            slot.color = (float(color[0]), float(color[1]), float(color[2]))
        _tooltip(
            "Click the swatch to open the picker.\n"
            "Click away or press Escape to close it again."
        )

        _draw_surface_params(app, slot)
    elif _draw_mask_params(app, slot):
        app.mark_texture_dirty()

    imgui.pop_item_width()


def _filled_slider_float(
    label: str,
    value: float,
    minimum: float,
    maximum: float,
    fmt: str = "%.3f",
    flags=0,
) -> tuple[bool, float]:
    """Slider whose field background is the value indicator.

    Color nodes have several scalar channels in a compact block. A separate
    grab marker adds visual noise there, so the darker/lighter division of the
    input itself shows how much of its allowed range is in use.
    """
    global _filled_slider_edit_item, _filled_slider_edit_opened
    item = int(imgui.get_id(label))
    hidden_label = _left_label(label)
    editing = _filled_slider_edit_item == item
    if editing:
        just_opened = _filled_slider_edit_opened
        if just_opened:
            imgui.set_keyboard_focus_here()
            _filled_slider_edit_opened = False
        changed, value = imgui.input_float(
            hidden_label, float(value), 0.0, 0.0, fmt,
            imgui.InputTextFlags_.auto_select_all
            | imgui.InputTextFlags_.enter_returns_true
            | imgui.InputTextFlags_.chars_decimal,
        )
        value = float(min(max(value, minimum), maximum))
        # The slider with this same ID was active on the preceding frame.
        # ImGui reports that old widget as deactivated while the replacement
        # input is being focused; treating that transition as focus loss makes
        # the editor disappear after one frame, before a key can reach it.
        if changed or (not just_opened and imgui.is_item_deactivated()):
            _filled_slider_edit_item = None
        return changed, value

    start = imgui.get_cursor_screen_pos()
    width = imgui.calc_item_width()
    height = imgui.get_frame_height()
    fraction = 0.0 if maximum <= minimum else (
        (float(value) - minimum) / (maximum - minimum)
    )
    fraction = min(max(fraction, 0.0), 1.0)
    end = imgui.ImVec2(start.x + width, start.y + height)
    fill_end = imgui.ImVec2(start.x + width * fraction, end.y)
    draw = imgui.get_window_draw_list()
    rounding = imgui.get_style().frame_rounding
    draw.add_rect_filled(
        start, end,
        imgui.get_color_u32(imgui.get_style_color_vec4(imgui.Col_.frame_bg)),
        rounding,
    )
    if fraction > 0.0:
        fill_flags = (
            imgui.ImDrawFlags_.round_corners_all if fraction >= 1.0
            else imgui.ImDrawFlags_.round_corners_left
        )
        draw.add_rect_filled(
            start, fill_end,
            imgui.get_color_u32(
                imgui.get_style_color_vec4(imgui.Col_.frame_bg_active)
            ),
            rounding,
            fill_flags,
        )

    transparent = imgui.ImVec4(0.0, 0.0, 0.0, 0.0)
    for color in (
        imgui.Col_.frame_bg,
        imgui.Col_.frame_bg_hovered,
        imgui.Col_.frame_bg_active,
        imgui.Col_.slider_grab,
        imgui.Col_.slider_grab_active,
    ):
        imgui.push_style_color(color, transparent)
    try:
        changed, value = imgui.slider_float(
            hidden_label, value, minimum, maximum, fmt,
            flags | imgui.SliderFlags_.no_input,
        )
    finally:
        imgui.pop_style_color(5)
    if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
        _filled_slider_edit_item = item
        _filled_slider_edit_opened = True
    return changed, value


def _draw_surface_params(app: "MeshMapApp", slot) -> None:
    """What the surface is made of, beyond its colour.

    These ride through the tree exactly as the colour does -- the mask blends
    them the same way -- and each comes out as its own exported map, so what is
    dialled in here is what the renderer at the other end receives.
    """
    _section_heading("Surface")

    changed, slot.metallic = _filled_slider_float("Metallic", slot.metallic, 0.0, 1.0)
    _delayed_input_tooltip(
        "0 is a dielectric -- paint, plastic, rust. 1 is bare metal.\n"
        "In between is for a surface partly covered by something else, not for\n"
        "a material that is half a metal.", changed
    )
    changed, slot.roughness = _filled_slider_float("Roughness", slot.roughness, 0.0, 1.0)
    _delayed_input_tooltip(
        "How wide the highlight spreads: 0 is a mirror, 1 is chalk.", changed
    )
    changed, slot.alpha = _filled_slider_float("Opacity", slot.alpha, 0.0, 1.0)
    _delayed_input_tooltip(
        "1 is solid. Below 1 the surface lets what is behind it through.\n"
        "This replaces the old separate transparency control.", changed
    )
    changed, slot.ambient_occlusion = _filled_slider_float(
        "Ambient occlusion", slot.ambient_occlusion, 0.0, 1.0
    )
    _delayed_input_tooltip(
        "Material-authored occlusion. 1 leaves ambient light alone; lower\n"
        "values darken it. It is multiplied by the baked mesh AO on export.",
        changed,
    )
    changed, slot.emission = _filled_slider_float(
        "Emission", slot.emission, 0.0, MAX_EMISSION, "%.2f",
        imgui.SliderFlags_.logarithmic,
    )
    _delayed_input_tooltip(
        "Light the surface gives off, in multiples of its own colour. Past 1\n"
        "the surface is brighter than white can show, and the extra turns into\n"
        "the glow spilling past its edge -- which is what carries the difference\n"
        "between a lit surface and a light.", changed
    )


def _generator_slider(
    params,
    attribute: str,
    label: str,
    minimum: float,
    maximum: float,
    fmt: str = "%.3f",
    flags=0,
) -> bool:
    """Scrubbable generator value with exact inline entry and default reset."""
    global _generator_scrub_edit_item, _generator_scrub_edit_opened
    value = float(getattr(params, attribute))
    item = int(imgui.get_id(label))
    hidden_label = _left_label(label)
    editing = _generator_scrub_edit_item == item
    if editing:
        just_opened = _generator_scrub_edit_opened
        if just_opened:
            imgui.set_keyboard_focus_here()
            _generator_scrub_edit_opened = False
        edited, value = imgui.input_float(
            hidden_label, value, 0.0, 0.0, fmt,
            imgui.InputTextFlags_.auto_select_all
            | imgui.InputTextFlags_.enter_returns_true,
        )
        value = float(min(max(value, minimum), maximum))
        committed = edited or imgui.is_item_deactivated_after_edit()
        if edited or (not just_opened and imgui.is_item_deactivated()):
            _generator_scrub_edit_item = None
        if committed:
            setattr(params, attribute, value)
        _tooltip(
            "Type an exact value, then press Enter or click away.\n"
            "Cmd-click after editing to restore the default."
        )
        return committed

    speed = max((maximum - minimum) / 300.0, 1e-6)
    changed, value = imgui.drag_float(
        hidden_label, value, speed, minimum, maximum, fmt,
        imgui.SliderFlags_.always_clamp,
    )
    hovered = imgui.is_item_hovered()
    if hovered:
        imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
    io = imgui.get_io()
    reset = hovered and io.key_super and imgui.is_mouse_clicked(0)

    if reset:
        value = float(getattr(type(params)(), attribute))
        changed = True
    elif hovered and imgui.is_mouse_double_clicked(0):
        _generator_scrub_edit_item = item
        _generator_scrub_edit_opened = True

    _tooltip(
        "Drag horizontally to adjust. Double-click to type in place.\n"
        "Cmd-click to restore the default."
    )

    if changed:
        setattr(params, attribute, float(min(max(value, minimum), maximum)))
    return changed


def _draw_generator_params(node) -> bool:
    """Only the controls that the selected procedural generator consumes."""
    params = node.noise
    kind = node.kind
    dirty = False

    dirty |= _generator_slider(
        params, "scale", "Scale", 0.5, 80.0, "%.1f",
        imgui.SliderFlags_.logarithmic,
    )

    if kind == "scratches":
        dirty |= _generator_slider(params, "scratch_width", "Scratch width", 0.002, 0.2)
        dirty |= _generator_slider(params, "scratch_length", "Scratch length", 0.02, 1.0)
        dirty |= _generator_slider(
            params, "scratch_irregularity", "Irregularity", 0.0, 1.0
        )
    elif kind == "brushed_metal":
        dirty |= _generator_slider(params, "brush_density", "Line density", 2.0, 120.0, "%.1f")
        dirty |= _generator_slider(params, "brush_waviness", "Waviness", 0.0, 1.0)
        dirty |= _generator_slider(params, "brush_variation", "Variation", 0.0, 1.0)
    elif kind == "cells":
        dirty |= _generator_slider(params, "cell_jitter", "Cell jitter", 0.0, 1.0)
        dirty |= _generator_slider(params, "cell_edge", "Cell edge", 0.0, 0.8)
    elif kind == "directional_streaks":
        dirty |= _generator_slider(params, "streak_length", "Streak length", 0.1, 30.0, "%.2f")
        dirty |= _generator_slider(params, "streak_width", "Streak width", 0.01, 1.0)
        for attribute, label, low, high in (
            ("detail", "Detail", 0.0, 8.0),
            ("roughness", "Roughness", 0.0, 1.0),
            ("lacunarity", "Lacunarity", 1.0, 8.0),
            ("distortion", "Distortion", 0.0, 2.0),
        ):
            dirty |= _generator_slider(params, attribute, label, low, high)
    elif kind == "brick":
        dirty |= _generator_slider(params, "brick_aspect", "Brick aspect", 0.25, 6.0)
        dirty |= _generator_slider(
            params, "mortar_thickness", "Mortar thickness", 0.001, 0.45
        )
    elif kind in ("wood_grain", "marble"):
        label = "Ring thickness" if kind == "wood_grain" else "Vein thickness"
        dirty |= _generator_slider(params, "vein_width", label, 0.02, 1.0)
        dirty |= _generator_slider(params, "detail", "Turbulence detail", 0.0, 8.0)
        dirty |= _generator_slider(params, "distortion", "Distortion", 0.0, 2.0)
    elif kind == "gradient":
        pass
    else:
        # Noise, grunge and clouds are all fBM-based, so these controls are
        # genuinely shared by those three and affect every visible octave.
        dirty |= _generator_slider(params, "detail", "Detail", 0.0, 8.0)
        dirty |= _generator_slider(params, "roughness", "Roughness", 0.0, 1.0)
        dirty |= _generator_slider(params, "lacunarity", "Lacunarity", 1.0, 8.0)
        dirty |= _generator_slider(params, "distortion", "Distortion", 0.0, 2.0)

    if kind not in ("noise", "grunge", "clouds", "cells", "marble"):
        dirty |= _generator_slider(params, "rotation", "Rotation", -180.0, 180.0, "%.1f deg")
    if kind != "gradient":
        dirty |= _generator_slider(params, "seed", "Seed", 0.0, 100.0, "%.1f")

    _section_heading("Output")
    dirty |= _generator_slider(params, "bias", "Bias", 0.0, 1.0)
    dirty |= _generator_slider(params, "contrast", "Contrast", 0.0, 20.0)
    return dirty


def _draw_mask_params(app: "MeshMapApp", node) -> bool:
    """The selected mask's own knobs. Live, like everything in this tab.

    All of them, laid out flat. The finer settings sat behind a collapsing
    header, which buys back a few rows of a sidebar that already has the room
    and costs a click to find out what is in there.
    """
    dirty = False

    _section_heading("Boundary")
    dirty |= _generator_slider(node, "threshold", "Threshold", 0.0, 1.0)
    _tooltip(
        "Where the mask divides its two sides. A mask is a continuous field --\n"
        "edge wear ramps up as the surface curves, noise wanders through every\n"
        "value -- and this is the level that decides which side a texel is on.\n"
        "Lower gives white's side more of the surface."
    )
    dirty |= _generator_slider(node, "softness", "Softness", 0.0, 0.5)
    _tooltip(
        "How wide the crossing is. 0 is a clean division: every texel belongs\n"
        "to one side or the other. Raise it to blend them across a band."
    )

    _section_heading(SLOT_KINDS[node.kind])
    if node.kind != "edge_wear":
        return dirty | _draw_generator_params(node)

    _muted_wrapped(
        "The EdgeWear001 node group, on this layer. What it produces reaches "
        "the export the way every layer does: through color.png."
    )

    wear = node.edge_wear
    changed, wear.value = _labeled_drag_float("Value", wear.value, 0.0, 5.0)
    dirty |= changed
    _tooltip(
        "Group Input.Value. Feeds a x10 Math node into the noise Scale, so the\n"
        "noise runs at value * 10. Higher = finer, busier break-up."
    )
    changed, wear.wear_amount = _labeled_drag_float("Wear amount", wear.wear_amount, 0.0, 2.0)
    dirty |= changed
    _tooltip(
        "The \"Wear Amount\" multiply. How hard the noise erodes the curvature\n"
        "before the subtract -- raise it and the wear gets patchier."
    )
    changed, wear.contrast = _labeled_drag_float("Contrast", wear.contrast, 0.0, 10.0)
    dirty |= changed
    _tooltip(
        "The \"Contrast\" multiply, clamped to 0..1 after.\n"
        "mask = clamp((curvature - noise * wear_amount) * contrast, 0, 1)"
    )

    _section_heading("Wear noise")
    changed, wear.detail = _labeled_drag_float("Detail", wear.detail, 0.0, 8.0)
    dirty |= changed
    _tooltip("fBM octaves.")
    changed, wear.roughness = _labeled_drag_float("Roughness", wear.roughness, 0.0, 1.0)
    dirty |= changed
    _tooltip("Amplitude falloff per octave.")
    changed, wear.lacunarity = _labeled_drag_float("Lacunarity", wear.lacunarity, 0.0, 8.0)
    dirty |= changed
    _tooltip("Frequency step per octave.")
    changed, wear.distortion = _labeled_drag_float("Distortion", wear.distortion, 0.0, 2.0)
    dirty |= changed
    _tooltip("Warps the sample position by the noise itself before evaluating.")
    return dirty


def _draw_texture_tree(app: "MeshMapApp") -> None:
    """The selected material followed by all of its child slots.

    The picker above changes which material this tree represents. Showing the
    other project materials here as well made the tree look like an assignment
    list and duplicated the picker; this pane is only the selected material's
    layer hierarchy.
    """
    material = app.texture
    material_index = app.texture_index
    if material is None or not 0 <= material_index < len(app.textures):
        return

    scale = app.ui_pixel_scale
    for path, slot in walk(material):
        imgui.push_id(f"material{material_index}/{'/'.join(path)}")
        indent = len(path) * 12 * scale
        if indent:
            imgui.indent(indent)

        if app.renaming_path == path:
            _draw_rename_field(app, slot)
        else:
            if isinstance(slot, ColorSlot):
                imgui.color_button("##swatch", imgui.ImVec4(*slot.color, 1.0))
            else:
                thumbnail = app.compositor.thumbnail(path)
                size = imgui.get_frame_height()
                if thumbnail is not None:
                    imgui.image(imgui.ImTextureRef(thumbnail.glo), imgui.ImVec2(size, size))
                else:
                    imgui.dummy(imgui.ImVec2(size, size))
            imgui.same_line()
            _draw_tree_row(app, material_index, path, slot)

        if indent:
            imgui.unindent(indent)
        imgui.pop_id()


def _draw_tree_row(app: "MeshMapApp", material_index, path, slot) -> None:
    """One row: which side of its mask it is, and what it is called.

    What it is called and nothing else -- a named slot shows its name, not the
    hex code or the kind it happens to be. That is what the Type dropdown above
    is for, and a row that argues with the name given to it reads as the rename
    not having taken.
    """
    side = f"{path[-1][0].upper()}  " if path else ""

    clicked, _ = imgui.selectable(
        f"{side}{describe(slot)}##row",
        material_index == app.texture_index and path == app.texture_path,
        imgui.SelectableFlags_.allow_double_click,
    )
    if clicked:
        if material_index != app.texture_index:
            app.select_texture(material_index)
        app.select_slot(path)
        if imgui.is_mouse_double_clicked(0):
            app.begin_rename(path)
    _tooltip("Double-click to rename.")


def _draw_rename_field(app: "MeshMapApp", slot) -> None:
    """The row being renamed, as a text field.

    Seeded with the name rather than the automatic label, so a slot that has
    never been named starts empty and typing replaces nothing. Clearing it puts
    the automatic label back.

    Focus is taken once, on the frame the field appears. Asking for it every
    frame is what makes a rename impossible to leave: if anything else held
    focus when the field opened it would never activate, never report being
    deactivated, and sit there grabbing at the keyboard for the rest of the
    session. Taking it once and closing the moment it is lost cannot stick.
    """
    just_opened = app.renaming_opened
    if just_opened:
        imgui.set_keyboard_focus_here()
        app.renaming_opened = False

    imgui.set_next_item_width(-1)
    entered, text = imgui.input_text(
        "##rename", slot.name,
        imgui.InputTextFlags_.enter_returns_true | imgui.InputTextFlags_.auto_select_all,
    )
    if entered or imgui.is_item_deactivated_after_edit():
        slot.name = text.strip()
        app.end_rename()
    elif not just_opened and not imgui.is_item_active():
        app.end_rename()  # focus went somewhere else


def _draw_decal_tab(app: "MeshMapApp") -> None:
    """The selected decal's complete inspector in the left tool pane."""
    if app.selected_decal is None:
        _muted_wrapped(
            "Select a decal in the Explorer or viewport to edit it here. Use "
            "Decal > Import decal or Decal > Add text to create one."
        )
        return

    _begin_panel("decal_inspector", imgui.ImVec2(0, 0))
    _draw_decal_inspector(app)
    _end_panel()


def _draw_decal_modifiers_tab(app: "MeshMapApp") -> None:
    """Array modifiers for whichever decal is selected, in their own pane.

    Split out of the Decal tab: a stack of modifiers can run long, and mixing
    it in with the decal's own transform and appearance controls meant
    scrolling past one to reach the other. Reads ``app.selected_decal`` the
    same way the Decal tab does, so picking a different decal here shows that
    decal's own stack without any extra wiring.
    """
    decal = app.selected_decal
    if decal is None:
        _muted_wrapped(
            "Select a decal in the Explorer or viewport to see its modifiers."
        )
        return

    _begin_panel("decal_modifiers_inspector", imgui.ImVec2(0, 0))
    imgui.text_colored(MUTED_COLOR, decal.display_name())
    _draw_decal_modifiers(app, decal)
    _end_panel()


def _draw_decal_array_modifier(app: "MeshMapApp", modifier_index: int, modifier) -> bool:
    """Line or Radial repetition controls for one Array modifier. Returns dirty."""
    dirty = False
    distance_limit = max(
        float(app.mesh_info.scale) * 2.0 if app.mesh_info is not None else 2.0,
        0.1,
    )
    modifier_defaults = DecalArrayModifier()
    distance_speed = max(distance_limit / 500.0, 0.001)

    modes = ("Line", "Radial")
    mode_index = 1 if modifier.mode == "radial" else 0
    changed, mode_index = _labeled_combo("Mode", mode_index, list(modes))
    if changed:
        modifier.mode = "radial" if mode_index == 1 else "axes"
        dirty = True

    changed, copies = _decal_number(
        app, "Copies", float(modifier.count), 1.0, 100.0, 1.0,
        float(modifier_defaults.count), "%.0f",
        state_key=f"modifier:{modifier_index}:count",
    )
    if changed:
        modifier.count = int(round(copies))
        dirty = True
    _tooltip("Generated copies; the selected decal is the first element.")

    if modifier.mode == "axes":
        for axis_name, attribute in (
            ("X distance", "offset_x"),
            ("Y distance", "offset_y"),
            ("Z distance", "offset_z"),
        ):
            changed, value = _decal_number(
                app, axis_name, float(getattr(modifier, attribute)),
                -distance_limit, distance_limit, distance_speed,
                float(getattr(modifier_defaults, attribute)), "%.3f",
                state_key=f"modifier:{modifier_index}:{attribute}",
            )
            if changed:
                setattr(modifier, attribute, float(value))
                dirty = True
        _tooltip(
            "Signed world-space offset per copy along each local decal axis. "
            "Negative values travel in the negative direction."
        )
        if not (modifier.offset_x or modifier.offset_y or modifier.offset_z):
            imgui.text_colored(WARN_COLOR, "Set at least one non-zero distance.")
    else:
        radial_axes = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
        radial_values = ("x", "-x", "y", "-y", "z", "-z")
        radial_index = (
            radial_values.index(modifier.radial_axis)
            if modifier.radial_axis in radial_values else 4
        )
        changed, radial_index = _labeled_combo(
            "Rotation axis", radial_index, list(radial_axes)
        )
        if changed:
            modifier.radial_axis = radial_values[radial_index]
            dirty = True
        changed, modifier.radius = _decal_number(
            app, "Radius", modifier.radius, 0.001, distance_limit,
            distance_speed, modifier_defaults.radius, "%.3f",
            state_key=f"modifier:{modifier_index}:radius",
        )
        dirty |= changed
        _tooltip(
            "Distance from the circle's center to every decal. The selected "
            "decal's transform marks the center; its rendered original and "
            "all copies are spaced evenly around the complete circle."
        )
    return dirty


def _draw_decal_mirror_modifier(modifier) -> bool:
    """A single world-axis reflection. Returns whether it changed."""
    axis_labels = ("X", "Y", "Z")
    axis_values = ("x", "y", "z")
    axis_index = (
        axis_values.index(modifier.axis) if modifier.axis in axis_values else 0
    )
    changed, axis_index = _labeled_combo("Mirror axis", axis_index, list(axis_labels))
    if changed:
        modifier.axis = axis_values[axis_index]
    _tooltip(
        "Reflects the decal across this world axis, through the mesh's own\n"
        "center -- the same plane regardless of which way the source decal\n"
        "happens to be facing."
    )
    return changed


#: What "New" offers, and what it adds when chosen -- kept in one place so the
#: popup menu and the factory it drives cannot drift apart.
_NEW_DECAL_MODIFIERS = (
    ("Array", "Repeated copies in a line or a circle.", DecalArrayModifier),
    (
        "Mirror", "One reflected copy across a world axis, through the mesh's center.",
        DecalMirrorModifier,
    ),
)


def _draw_new_decal_modifier_button(decal) -> bool:
    """The "New" button and its type menu. Returns whether one was added."""
    added = False
    if imgui.button("New"):
        imgui.open_popup("##new_decal_modifier")
    corner = imgui.get_item_rect_min()
    bottom = imgui.get_item_rect_max().y
    imgui.set_next_window_pos(imgui.ImVec2(corner.x, bottom))

    if imgui.begin_popup("##new_decal_modifier"):
        for label, tooltip, factory in _NEW_DECAL_MODIFIERS:
            if imgui.menu_item_simple(label):
                decal.modifiers.append(factory())
                added = True
            _tooltip(tooltip)
        imgui.end_popup()
    return added


def _draw_decal_modifiers(app: "MeshMapApp", decal) -> None:
    """Non-destructive Array and Mirror modifiers stacked on one decal."""
    scale = app.ui_pixel_scale
    imgui.push_item_width(-170 * scale)
    projector_input = (
        decal.center_u, decal.center_v,
        decal.scale, decal.scale_x, decal.scale_y, decal.rotation,
    )

    imgui.begin_disabled(not decal.enabled)
    # Always the first thing in the pane, not appended after the stack, so it
    # never wanders off as the list of modifiers grows and the pane scrolls.
    dirty = _draw_new_decal_modifier_button(decal)
    if not decal.modifiers:
        _muted_wrapped("No modifiers on this decal yet.")

    remove_modifier = None
    for modifier_index, modifier in enumerate(decal.modifiers):
        imgui.push_id(f"decal_modifier_{modifier_index}")
        is_mirror = isinstance(modifier, DecalMirrorModifier)
        _section_heading(f"{'Mirror' if is_mirror else 'Array'} {modifier_index + 1}")
        if imgui.small_button("Remove"):
            remove_modifier = modifier_index

        if is_mirror:
            dirty |= _draw_decal_mirror_modifier(modifier)
        else:
            dirty |= _draw_decal_array_modifier(app, modifier_index, modifier)
        imgui.pop_id()

    if remove_modifier is not None:
        del decal.modifiers[remove_modifier]
        dirty = True

    imgui.end_disabled()

    if dirty:
        app.sync_decal_inspector_projector(decal, projector_input)
        app.mark_normal_dirty()

    imgui.pop_item_width()


def _draw_mesh_tab(app: "MeshMapApp") -> None:
    """Details for the mesh selected in the persistent Explorer or viewport."""
    _begin_panel("mesh_inspector", imgui.ImVec2(0, 0))
    _draw_mesh_inspector(app)
    _end_panel()


def _draw_mesh_inspector(app: "MeshMapApp") -> None:
    """Properties and material assignment for the selected scene mesh."""
    if app.mesh is None or app.mesh_info is None:
        _muted_wrapped("There are no meshes in the scene. Open or drop a mesh to add one.")
        return

    info = app.mesh_info
    name = app.mesh_name
    if app.mesh_selected_index != 0:
        _muted_wrapped("Select a mesh in the Explorer above to inspect it and assign its material.")
        return

    imgui.text(name)
    imgui.text_colored(MUTED_COLOR, f"{info.vertices:,} vertices  ·  {info.faces:,} faces")
    imgui.text_colored(
        MUTED_COLOR,
        "Size  " + " × ".join(f"{extent:.3g}" for extent in info.extents),
    )

    _section_heading("Appearance")
    if app.textures:
        labels = ["None"] + [describe(material) for material in app.textures]
        current = (
            app.mesh_material_index + 1
            if 0 <= app.mesh_material_index < len(app.textures)
            else 0
        )
        changed, current = _labeled_combo("Material", current, labels)
        if changed:
            app.assign_mesh_material(current - 1)
        _tooltip("Assign a material from the Material view to this mesh.")
    else:
        imgui.text_colored(MUTED_COLOR, "Material: none")
        _muted_wrapped("Create a material in the Material view before assigning one.")

    if imgui.button("Frame selected"):
        app.camera.frame(app.mesh.bounds.mean(axis=0), info.scale)


def _draw_decal_visibility_icon(enabled: bool) -> bool:
    """Compact eye-style visibility control used by decal tree rows."""
    size = imgui.get_frame_height()
    clicked = imgui.invisible_button("##visibility", imgui.ImVec2(size, size))
    start, end = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    draw = imgui.get_window_draw_list()
    center = imgui.ImVec2((start.x + end.x) * 0.5, (start.y + end.y) * 0.5)
    radius = size * 0.27
    color = imgui.get_color_u32(
        imgui.ImVec4(0.78, 0.84, 0.90, 1.0) if enabled
        else imgui.ImVec4(0.42, 0.44, 0.48, 1.0)
    )
    draw.add_circle(center, radius, color, 18, max(1.2, size * 0.07))
    if enabled:
        draw.add_circle_filled(center, radius * 0.38, color, 12)
    else:
        draw.add_line(
            imgui.ImVec2(center.x - radius, center.y + radius),
            imgui.ImVec2(center.x + radius, center.y - radius),
            color, max(1.2, size * 0.08),
        )
    return clicked


def _draw_decal_library_tab(app: "MeshMapApp") -> None:
    """The decal shelf, kept separate from decals already on the mesh."""
    _draw_decal_library(app)


def _draw_decal_library(app: "MeshMapApp") -> None:
    """The shelf: every image in the folder metadata.json points at.

    Dragging one onto the model places it there. Dragging rather than clicking
    because a decal has to land *somewhere*, and the somewhere is on the mesh --
    a click would have to invent a position and then ask you to move it.
    """
    scale = app.ui_pixel_scale
    if not app.decal_library:
        _muted_wrapped(
            'No decal library. Point "decals" in metadata.json at a folder of '
            "normal maps and they will appear here."
        )
        if imgui.button("Import one instead...", imgui.ImVec2(-1, 0)):
            app.decal_dialog = pfd.open_file(
                "Import decal", str(Path.home()), DECAL_FILTERS
            )
        return

    _muted_wrapped("Drag one onto the model to place it.")
    app.pump_decal_library()

    thumbnail = DECAL_THUMBNAIL * scale
    across = max(1, int(imgui.get_content_region_avail().x // (thumbnail + 8 * scale)))
    for index, path in enumerate(app.decal_library):
        if index % across:
            imgui.same_line()
        imgui.push_id(f"library{index}")
        _draw_library_item(app, path, thumbnail)
        imgui.pop_id()

    imgui.dummy(imgui.ImVec2(0, 4 * scale))
    if imgui.button("Import another...", imgui.ImVec2(-1, 0)):
        app.decal_dialog = pfd.open_file("Import decal", str(Path.home()), DECAL_FILTERS)


def _draw_library_item(app: "MeshMapApp", path: Path, size: float) -> None:
    """Draw one picture once the background library loader has prepared it."""
    image_size = app.decal_thumbnail_sizes.get(str(path))
    texture = app.decal_thumbnail_textures.get(str(path))
    if texture is None or image_size is None:
        imgui.dummy(imgui.ImVec2(size, size))
        return

    width, height = image_size
    imgui.image_button(
        "##thumb",
        imgui.ImTextureRef(texture.glo),
        imgui.ImVec2(size, size * height / max(width, 1)),
    )
    # Dragged out of the panel and onto the model: the app watches for the
    # release, because ImGui's own drag-and-drop only reaches ImGui targets and
    # the 3D view is not one.
    if imgui.is_item_active() and imgui.is_mouse_dragging(0):
        app.begin_decal_drag(path)
    _tooltip(f"{path.name}\nDrag onto the model to place it.")


def _decal_number(
    app: "MeshMapApp", label: str, value: float,
    minimum: float, maximum: float, speed: float, default: float,
    fmt: str = "%.3f", *, state_key: str | None = None,
) -> tuple[bool, float]:
    """Blender-style numeric field shared by all decal inspector values.

    A stable ``state_key`` distinguishes identically labelled controls in a
    stacked modifier list while the user transitions from clicking to typing.
    """
    scale = app.ui_pixel_scale
    button = imgui.get_frame_height()
    field_width = max(
        70.0 * scale,
        imgui.get_content_region_avail().x - 150.0 * scale - button * 2.0,
    )
    changed = False
    original = float(value)
    reset = False
    io = imgui.get_io()
    interaction_key = state_key or label
    imgui.push_id(interaction_key)

    imgui.align_text_to_frame_padding()
    imgui.text_unformatted(label)
    imgui.same_line(142.0 * scale)

    if imgui.button("-", imgui.ImVec2(button, 0)):
        value -= speed
        changed = True
    if imgui.is_item_hovered():
        imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
        reset |= io.key_super and imgui.is_mouse_clicked(0)
    if imgui.is_item_active() and imgui.is_mouse_dragging(0):
        value += io.mouse_delta.x * speed
        changed = True

    imgui.same_line(0.0, 0.0)
    imgui.set_next_item_width(field_width)
    editing = app.decal_transform_edit_field == interaction_key
    if editing:
        if app.decal_transform_edit_opened:
            imgui.set_keyboard_focus_here()
            app.decal_transform_edit_opened = False
        edited, value = imgui.input_float(
            "##value", float(value), 0.0, 0.0, fmt,
            imgui.InputTextFlags_.auto_select_all
            | imgui.InputTextFlags_.enter_returns_true,
        )
        changed |= edited
        if edited or imgui.is_item_deactivated_after_edit():
            app.decal_transform_edit_field = None
        elif not imgui.is_item_active() and not app.decal_transform_edit_opened:
            app.decal_transform_edit_field = None
    else:
        edited, value = imgui.drag_float(
            "##value", float(value), speed, minimum, maximum, fmt,
            imgui.SliderFlags_.always_clamp,
        )
        changed |= edited
        hovered = imgui.is_item_hovered()
        if hovered:
            imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
        if hovered and imgui.is_mouse_clicked(0):
            app.decal_transform_pressed_field = interaction_key
            app.decal_transform_field_dragged = False
            if io.key_super:
                reset = True
        if (imgui.is_item_active() and imgui.is_mouse_dragging(0)):
            app.decal_transform_field_dragged = True
        if (app.decal_transform_pressed_field == interaction_key
                and imgui.is_mouse_released(0)):
            if hovered and not app.decal_transform_field_dragged and not reset:
                app.decal_transform_edit_field = interaction_key
                app.decal_transform_edit_opened = True
            app.decal_transform_pressed_field = None
            app.decal_transform_field_dragged = False
    _tooltip(
        "Click to type a value. Drag the field horizontally to adjust it.\n"
        "Cmd-click to restore the default."
    )

    imgui.same_line(0.0, 0.0)
    if imgui.button("+", imgui.ImVec2(button, 0)):
        value += speed
        changed = True
    if imgui.is_item_hovered():
        imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
        reset |= io.key_super and imgui.is_mouse_clicked(0)
    if imgui.is_item_active() and imgui.is_mouse_dragging(0):
        value += io.mouse_delta.x * speed
        changed = True
    imgui.pop_id()
    if reset:
        value = float(default)
        changed = True
    value = float(min(max(value, minimum), maximum))
    return changed or value != original, value


def _draw_decal_transform(app: "MeshMapApp", decal) -> bool:
    """Position, rotation and scale grouped at the inspector's top."""
    if not imgui.collapsing_header("Transform", imgui.TreeNodeFlags_.default_open):
        return False

    dirty = False
    blank = type(decal)()
    _section_heading("Location")
    changed, decal.center_u = _decal_number(
        app, "U", decal.center_u, 0.0, 1.0, 0.002, blank.center_u
    )
    dirty |= changed
    changed, decal.center_v = _decal_number(
        app, "V", decal.center_v, 0.0, 1.0, 0.002, blank.center_v
    )
    dirty |= changed

    _section_heading("Rotation")
    # decal.rotation is stored wrapped into [0, 360) (see
    # transform_decal_with_pointer's ``% 360.0``), not [-180, 180]. A
    # narrower range here made this field's own trailing clamp fire the
    # instant a viewport R-drag pushed the angle past 180 -- every frame
    # after that clamped the live value back down and reported it as a
    # genuine edit, corrupting the drag and, in Edit space, spuriously
    # padding edit_spin every single frame for as long as the drag
    # continued past that point.
    changed, decal.rotation = _decal_number(
        app, "Angle", decal.rotation, 0.0, 360.0, 0.5,
        blank.rotation, "%.1f"
    )
    dirty |= changed

    _section_heading("Scale")
    width = decal.scale * decal.scale_x
    height = decal.scale * decal.scale_y
    changed, width = _decal_number(
        app, "Width", width, 0.00004, 200.0, 0.005,
        blank.scale * blank.scale_x,
    )
    if changed:
        decal.scale_x = width / max(decal.scale, 1e-6)
    dirty |= changed
    changed, height = _decal_number(
        app, "Height", height, 0.00004, 200.0, 0.005,
        blank.scale * blank.scale_y,
    )
    if changed:
        decal.scale_y = height / max(decal.scale, 1e-6)
    dirty |= changed

    if imgui.button("Reset transform"):
        decal.center_u, decal.center_v = blank.center_u, blank.center_v
        decal.scale, decal.scale_x, decal.scale_y = (
            blank.scale, blank.scale_x, blank.scale_y
        )
        decal.rotation = blank.rotation
        dirty = True
    return dirty


def _draw_decal_inspector(app: "MeshMapApp") -> None:
    """The selected decal's own controls."""
    scale = app.ui_pixel_scale
    imgui.push_item_width(-170 * scale)
    decal = app.selected_decal
    dirty = False

    if decal is None:
        _muted_wrapped(
            "Nothing selected. Choose a decal from the Explorer above, click one "
            "on the model, or drag a new one from the Library view."
        )
        if app.decals:
            _muted_wrapped(f"{len(app.decals)} on the mesh.")
        imgui.pop_item_width()
        return

    projector_input = (
        decal.center_u, decal.center_v,
        decal.scale, decal.scale_x, decal.scale_y, decal.rotation,
    )

    image = app.decal_image_for(decal)
    origin = (
        "generated text"
        if decal.source_type == "text"
        else "height map, converted" if image and image.from_height else "normal map"
    )
    imgui.text_colored(
        MUTED_COLOR,
        f"{decal.display_name()}\n"
        f"{decal.image_aspect:.2f} : 1  ({origin})\n"
        f"{app.decal_index + 1} of {len(app.decals)} on the mesh",
    )

    if decal.source_type == "text":
        _section_heading("Text")
        if app.decal_text_edit_index != app.decal_index:
            app.decal_text_edit_index = app.decal_index
            app.decal_text_edit_value = decal.text
        submitted, draft = imgui.input_text(
            _left_label("Content"), app.decal_text_edit_value,
            imgui.InputTextFlags_.enter_returns_true
            | imgui.InputTextFlags_.auto_select_all,
        )
        draft = draft[:128]
        if draft != app.decal_text_edit_value:
            app.decal_text_edit_value = draft
            app.decal_text_edit_changed_at = time.monotonic()
        pending = (
            app.decal_text_edit_changed_at is not None
            and app.decal_text_edit_value != decal.text
        )
        settled = (
            pending
            and time.monotonic() - app.decal_text_edit_changed_at >= _TEXT_DECAL_LIVE_DELAY
        )
        if submitted or imgui.is_item_deactivated_after_edit() or settled:
            app.update_text_decal(decal, app.decal_text_edit_value)
            app.decal_text_edit_changed_at = None
        _tooltip("Edit the text -- it updates on the model as you type.")

    dirty |= _draw_decal_transform(app, decal)

    if imgui.button("Remove"):
        app.remove_decal()
        imgui.pop_item_width()
        return
    imgui.same_line()
    if app.decal_placing:
        if imgui.button("Cancel move"):
            app.end_decal_placement(keep=False)
    else:
        imgui.begin_disabled(app.mesh is None)
        if imgui.button("Move on the mesh"):
            app.begin_decal_placement()
        imgui.end_disabled()
        _tooltip(
            "Pick it up and put it somewhere else. It rides the surface under\n"
            "the cursor; click to drop it, Esc to put it back."
        )
    if app.decal_placing:
        imgui.text_colored(
            WARN_COLOR, "Moving: click on the mesh to drop it, Esc to cancel"
        )

    imgui.begin_disabled(not decal.enabled)

    _section_heading("Appearance")
    texture_labels = ["None"] + [describe(texture) for texture in app.textures]
    current_texture = (
        decal.texture_index + 1
        if 0 <= decal.texture_index < len(app.textures)
        else 0
    )
    changed, current_texture = _labeled_combo(
        "Material", current_texture, texture_labels
    )
    if changed:
        decal.texture_index = current_texture - 1
        dirty = True
    _tooltip(
        "Use a material created in the Material tab to colour this decal.\n"
        "None changes only the surface normal."
    )

    _section_heading("Depth")
    decal_defaults = type(decal)()
    changed, decal.intensity = _decal_number(
        app, "Height intensity", decal.intensity, 0.0, 10.0, 0.05,
        decal_defaults.intensity, "%.2f", state_key="appearance:intensity",
    )
    dirty |= changed
    _tooltip(
        "How deep the bump reads. This scales the surface slope the map\n"
        "describes, not the stored vector, so 2.0 is genuinely twice as steep\n"
        "instead of tipping the normal flat against the surface.\n"
        "0 is flat, 1 is the map exactly as it was authored."
    )

    changed, decal.flip_green = _labeled_checkbox("Flip green (DirectX)", decal.flip_green)
    dirty |= changed
    _tooltip(
        "Tick for a map baked in DirectX convention (-Y). Everything here is\n"
        "OpenGL (+Y up), which is what Blender expects. If the decal reads\n"
        "inside-out -- raised where it should be recessed -- this is why."
    )

    _section_heading("Edges")
    changed, decal.falloff = _decal_number(
        app, "Edge falloff", decal.falloff, 0.0, 1.0, 0.01,
        decal_defaults.falloff, "%.2f", state_key="appearance:falloff",
    )
    dirty |= changed
    _tooltip(
        "Fades the decal out towards its own border, as a fraction of its\n"
        "half-width.\n\n"
        "A decal is a rectangle of an image and the surface does not stop\n"
        "there, so unless the image fades to a flat normal by its own edge the\n"
        "rectangle shows up as a seam. Most do not -- a great many normal maps\n"
        "are drawn on a white or a black canvas, and neither of those is flat.\n"
        "Raise this until the join disappears; 0 turns it off for an image that\n"
        "needs no help."
    )

    imgui.end_disabled()

    # Where it lands on the mesh, in the units the artist thinks in.
    info = app.mesh_info
    if info is not None and info.uv_density > 0.0:
        span = decal.scale / info.uv_density
        imgui.text_colored(
            MUTED_COLOR, f"About {span:.3g} m across on the mesh at this scale"
        )

    if info is not None and not info.has_uvs:
        app.console_log(
            "WARNING",
            "This mesh carries no UVs, so there is no decal layout yet. Bake "
            "once to generate an atlas; that atlas fits the exported OBJ, not "
            "the original mesh.",
            key="decal_mesh_has_no_uvs",
        )

    if dirty:
        app.sync_decal_inspector_projector(decal, projector_input)
        app.mark_normal_dirty()

    imgui.pop_item_width()


#: The five maps, and what to call them in the interface.
_MAP_LABELS = (
    ("color", "Colour"),
    ("normal", "Normal"),
    ("metallic", "Metallic"),
    ("roughness", "Roughness"),
    ("ao", "Ambient occlusion"),
)


def _draw_map_switches(app: "MeshMapApp") -> None:
    """Which of the five an export writes.

    Four of them cost nothing to produce -- colour, metallic and roughness are
    one composite pass over the texture, normals one pass over the decal -- so
    switching those off only stops a file being written. Occlusion is the one
    stage that traces rays and costs more than the whole rest of the bake, so
    switching it off takes it out of the bake as well. That is the difference
    the note below is for: the others are free either way.
    """
    imgui.text_colored(MUTED_COLOR, "Maps to write")

    for name, label in _MAP_LABELS:
        changed, enabled = _labeled_checkbox(label, app.map_enabled(name))
        if changed:
            app.set_map_enabled(name, enabled)

    if not app.map_enabled("ao"):
        _muted_wrapped(
            "Occlusion is switched off, so the bake skips it -- which is most "
            "of what a bake costs. Switching it back on needs a re-bake."
        )


def _draw_console_tab(app: "MeshMapApp") -> None:
    """Diagnostics collected without interrupting the normal workflow."""
    if imgui.button("Clear"):
        app.clear_console()
    imgui.same_line()
    imgui.text_colored(MUTED_COLOR, f"{len(app.console_messages)} messages")
    imgui.spacing()

    _begin_panel("console_output", imgui.ImVec2(0, 0))
    for index, (level, message) in enumerate(app.console_messages):
        color = ERROR_COLOR if level == "ERROR" else WARN_COLOR \
            if level == "WARNING" else MUTED_COLOR
        imgui.push_id(f"console_{index}")
        imgui.text_colored(color, f"[{level}]")
        imgui.same_line()
        imgui.push_text_wrap_pos(0.0)
        imgui.text_wrapped(message)
        imgui.pop_text_wrap_pos()
        imgui.pop_id()
    if not app.console_messages:
        imgui.text_colored(MUTED_COLOR, "No diagnostics recorded.")
    _end_panel()


def _draw_settings_tab(app: "MeshMapApp") -> None:
    """Viewport navigation, world lighting and interface scale, all persisted."""
    from render.viewport import (  # local import avoids a cycle
        MAX_WORLD_STRENGTH,
        bearing,
    )

    scale = app.ui_pixel_scale
    imgui.push_item_width(-170 * scale)

    nav = app.navigation
    dirty = False

    _section_heading("Viewport navigation")
    imgui.text_colored(
        MUTED_COLOR,
        "Multipliers on the base rates.\n"
        "1.0 orbits at 0.4 deg/pixel, Blender's default.",
    )

    changed, nav.orbit_speed = _labeled_drag_float(
        "Orbit speed", nav.orbit_speed, 0.05, 3.0, "%.2fx",
        imgui.SliderFlags_.logarithmic,
    )
    dirty |= changed
    _tooltip("Degrees turned per pixel of drag. Applies to scroll-orbit too.")

    changed, nav.pan_speed = _labeled_drag_float(
        "Pan speed", nav.pan_speed, 0.05, 3.0, "%.2fx", imgui.SliderFlags_.logarithmic
    )
    dirty |= changed

    changed, nav.zoom_speed = _labeled_drag_float(
        "Zoom speed", nav.zoom_speed, 0.05, 3.0, "%.2fx", imgui.SliderFlags_.logarithmic
    )
    dirty |= changed

    changed, nav.scroll_speed = _labeled_drag_float(
        "Trackpad / wheel", nav.scroll_speed, 0.05, 3.0, "%.2fx",
        imgui.SliderFlags_.logarithmic,
    )
    dirty |= changed
    _tooltip(
        "Extra multiplier for scroll and trackpad gestures only, on top of the\n"
        "three above. A trackpad reports a gesture as a stream of small deltas\n"
        "and a wheel reports whole notches, so the speed that suits one is\n"
        "usually wrong for the other. Turn this down if two-finger orbiting\n"
        "overshoots."
    )

    changed, nav.smoothing = _labeled_drag_float(
        "Smoothing", nav.smoothing, 0.0, 1.0, "%.2f"
    )
    dirty |= changed
    _tooltip(
        "Moves the view at the speed of the gesture, rather than in the steps\n"
        "the trackpad reports.\n\n"
        "macOS reports scrolling a pixel of finger travel at a time. Move\n"
        "slowly and a pixel takes several frames to cross, so what arrives is a\n"
        "run of nothing punctuated by a step -- the view sticks, then hops.\n"
        "This measures how fast the steps are arriving and moves that much each\n"
        "frame instead, which costs about one step of delay at a crawl and\n"
        "nothing at all once the steps arrive every frame.\n\n"
        "1.00 follows the gesture exactly; lower runs ahead of it, and 0 applies\n"
        "each step whole the moment it lands. Mouse dragging is never smoothed."
    )

    changed, nav.invert_orbit_x = _labeled_checkbox("Invert orbit X", nav.invert_orbit_x)
    dirty |= changed
    changed, nav.invert_orbit_y = _labeled_checkbox("Invert orbit Y", nav.invert_orbit_y)
    dirty |= changed
    _tooltip("For natural-scrolling trackpads, or simple preference.")

    if imgui.button("Reset navigation"):
        app.navigation = type(nav)()
        dirty = True

    if dirty:
        app.apply_navigation()
    dirty = False

    _section_heading("World lighting")
    imgui.text_colored(
        MUTED_COLOR,
        "Where the key light stands, and the light the\nmodel sits in beyond it.",
    )

    world = app.world
    changed, world.rotation = _labeled_drag_float(
        "Rotation", world.rotation, 0.0, 360.0, "%.0f deg"
    )
    dirty |= changed
    # While it is being moved, and for a moment after: an arrow in the viewport
    # standing where the light is and pointing at the model. Asked for on hover
    # as well as on drag, so it appears before the first pixel of movement --
    # otherwise you are turning something you cannot see yet.
    if changed or imgui.is_item_active() or imgui.is_item_hovered():
        app.flash_light_gizmo()
    _tooltip(
        "Walks the key light around the model. 0 puts it on the +X axis and it\n"
        "turns towards +Y, counting the way the gizmo in the corner does.\n\n"
        "The light is anchored to the model, not to the camera, so orbiting\n"
        "moves your view of the lighting rather than the lighting itself --\n"
        "which is the only way putting the light somewhere can mean anything."
    )

    stands, shines = bearing(world.rotation)
    _muted_wrapped(f"Light at {stands}, shining toward {shines}.")

    changed, world.strength = _labeled_drag_float(
        "Strength", world.strength, 0.0, MAX_WORLD_STRENGTH, "%.2f"
    )
    dirty |= changed
    _tooltip(
        "How much of the world reaches the surface. A single lamp leaves a\n"
        "metal with nothing to reflect and a rough surface with nothing to\n"
        "scatter, so both go black at 0 however their own sliders are set.\n"
        "Turn it up for an overcast look, down for a dramatic one."
    )

    changed, color = imgui.color_edit3(
        _left_label("Colour"), list(world.color), imgui.ColorEditFlags_.no_inputs
    )
    if changed:
        world.color = (float(color[0]), float(color[1]), float(color[2]))
        dirty = True
    _tooltip("What colour that light is. Tints every surface it falls on.")

    if imgui.button("Reset lighting"):
        app.world = type(world)()
        dirty = True

    if dirty:
        app.save_prefs()

    _section_heading("Export")
    imgui.text_colored(
        MUTED_COLOR,
        "What File > Export textures writes, and where\nit starts looking.",
    )

    _, app.export_dir = imgui.input_text(_left_label("Folder"), app.export_dir)
    _tooltip(
        "Where the folder chooser opens. Exporting somewhere else moves this\n"
        "here, so the next export starts from where the last one went. The maps\n"
        "land in a folder named after the mesh."
    )
    imgui.same_line()
    if imgui.button("..."):
        app.folder_dialog = pfd.select_folder("Export folder", app.export_dir)

    bit_index = 0 if app.export_bits == 8 else 1
    changed, bit_index = _labeled_combo("Depth", bit_index, ["8-bit PNG", "16-bit PNG"])
    if changed:
        app.export_bits = 8 if bit_index == 0 else 16
    _tooltip("Reach for 16-bit if you see banding in the gradients.")

    _draw_map_switches(app)

    maps = exportable_maps(app)
    _muted_wrapped(
        ("Ready to write " + ", ".join(f"{name}.png" for name in maps) + ".") if maps
        else "Nothing to write yet - bake a mesh, import a decal, or switch a "
             "map back on above."
    )

    _section_heading("Interface")
    changed, requested = _labeled_drag_float("UI scale", app.ui_scale, 0.6, 3.0, "%.2fx")
    if changed:
        app.set_ui_scale(requested)
    _tooltip("Size of this panel and its text. Remembered between launches.")

    if dirty:
        app.save_prefs()

    imgui.text_colored(MUTED_COLOR, "Settings are saved between launches.")
    imgui.pop_item_width()


def _draw_bake_controls(app: "MeshMapApp") -> None:
    controller = app.controller

    if controller.running:
        imgui.progress_bar(controller.progress, imgui.ImVec2(-1, 0), controller.stage)
        if imgui.button("Cancel", imgui.ImVec2(-1, 0)):
            controller.cancel()
        return

    pending = controller.pending_stages()
    imgui.begin_disabled(app.mesh is None or not pending)
    label = "Bake" if not pending else f"Bake  ({len(pending)} stage{'s' if len(pending) != 1 else ''})"
    if imgui.button(label, imgui.ImVec2(-1, 0)):
        app.request_bake()
    imgui.end_disabled()

    if pending:
        imgui.text_colored(MUTED_COLOR, "Stale: " + ", ".join(pending))

    if controller.error:
        message = controller.error.strip()
        app.console_log("ERROR", message, key="bake_error")


# --------------------------------------------------------------------------

def _draw_status_bar(app: "MeshMapApp") -> None:
    width, height = app.wnd.buffer_size
    bar_height = STATUS_BAR_HEIGHT * app.ui_pixel_scale
    imgui.set_next_window_pos(imgui.ImVec2(0, height - bar_height))
    imgui.set_next_window_size(imgui.ImVec2(width, bar_height))
    flags = (
        imgui.WindowFlags_.no_decoration
        | imgui.WindowFlags_.no_move
        | imgui.WindowFlags_.no_saved_settings
        | imgui.WindowFlags_.no_nav
    )
    imgui.begin("##status", None, flags)
    line = app.status.strip().splitlines()[-1] if app.status.strip() else ""
    if app.status_is_error:
        imgui.text_colored(ERROR_COLOR, "Error - see Console")
    else:
        imgui.text_colored(MUTED_COLOR, line)
    imgui.end()


def _pump_dialogs(app: "MeshMapApp") -> None:
    """Native file dialogs are async, so poll them once a frame."""
    # Asked for from the File menu or the E key. Where to put them is the only
    # question an export has, so it is asked first and answering it writes them.
    if app.export_pending:
        app.export_pending = False
        if exportable_maps(app):
            app.export_dialog = pfd.select_folder("Export textures to", app.export_dir)
        else:
            app.set_status(
                "Nothing to export - bake a mesh, or import a decal", error=True
            )

    if app.file_dialog is not None and app.file_dialog.ready():
        selection = app.file_dialog.result()
        app.file_dialog = None
        if selection:
            app.open_mesh(selection[0])

    if app.folder_dialog is not None and app.folder_dialog.ready():
        selection = app.folder_dialog.result()
        app.folder_dialog = None
        if selection:
            app.export_dir = selection

    # Chosen from the File menu: where to put them was the only question, so
    # answering it is the whole gesture -- the export follows immediately.
    if app.export_dialog is not None and app.export_dialog.ready():
        selection = app.export_dialog.result()
        app.export_dialog = None
        if selection:
            app.export_dir = selection
            app.export()

    if app.decal_dialog is not None and app.decal_dialog.ready():
        selection = app.decal_dialog.result()
        app.decal_dialog = None
        if selection:
            app.open_decal(selection[0])
