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
from core.params import BAKE_AXES, RESOLUTIONS

if TYPE_CHECKING:  # pragma: no cover
    from render.viewport import MeshMapApp

#: Panel geometry in unscaled reference pixels; multiplied by the app's
#: ui_pixel_scale, because ImGui is driven in physical pixels here.
PANEL_WIDTH = 430
#: Width of the clickable decal preview in the Decal tab.
DECAL_THUMBNAIL = 92
STATUS_BAR_HEIGHT = 34
#: The bar across the top of the window: what to look at, and how to draw it.
NAVBAR_HEIGHT = 46
#: How far either side of the sidebar's edge counts as grabbing it, in unscaled
#: pixels. Narrow and even: reaching further into the view means the cursor
#: turns into a resize arrow while it is plainly over the model, which reads as
#: the app being confused about where the panel ends.
SIDEBAR_GRAB_INSIDE = 6
SIDEBAR_GRAB_OUTSIDE = 6
#: Width of the vertical view switcher on the sidebar's left edge.
SIDEBAR_ICON_RAIL = 44
#: Textures beyond which the picker grows a search box. Below it, a list this
#: short is quicker to read than to filter.
_SEARCHABLE_FROM = 5
ERROR_COLOR = imgui.ImVec4(1.0, 0.45, 0.42, 1.0)
MUTED_COLOR = imgui.ImVec4(0.62, 0.64, 0.68, 1.0)
WARN_COLOR = imgui.ImVec4(1.0, 0.78, 0.35, 1.0)

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


def _draw_navbar(app: "MeshMapApp") -> None:
    """The bar across the top: what to look at, and how it is drawn.

    View state, not parameters -- which is why it lives out here across the top
    rather than inside the sidebar's tabs, where changing what you are looking
    at would mean leaving the thing you are tuning.
    """
    from render.viewport import PREVIEW_MODES  # local import avoids a cycle

    scale = app.ui_pixel_scale
    width = app.wnd.buffer_size[0]
    height = NAVBAR_HEIGHT * scale

    imgui.set_next_window_pos(imgui.ImVec2(0, 0))
    imgui.set_next_window_size(imgui.ImVec2(width, height))
    imgui.begin("##navbar", None, _CHROME_FLAGS | imgui.WindowFlags_.no_decoration)

    _draw_file_menu(app)
    imgui.same_line()

    imgui.set_next_item_width(220 * scale)
    _, app.preview_index = imgui.combo(
        "##preview", app.preview_index, [mode.label for mode in PREVIEW_MODES]
    )
    _tooltip(
        "Shaded is the mask tree, lit: the mask itself in black and white until\n"
        "you put colours -- or another mask -- under it in the Material tab.\n"
        "Normals is the decal normal map. Keys 1 and 2 switch between them."
    )

    imgui.same_line()
    _, app.lighting = imgui.checkbox("Lighting", app.lighting)
    imgui.same_line()
    _, app.wireframe = imgui.checkbox("Wireframe", app.wireframe)
    imgui.same_line()
    _, app.show_gizmo = imgui.checkbox("Gizmo", app.show_gizmo)
    _tooltip(
        "The axis balls in the top-right corner. Click one to look straight down\n"
        "that axis in orthographic projection; orbiting returns to perspective."
    )
    if app.camera.orthographic:
        imgui.same_line()
        imgui.text_colored(MUTED_COLOR, "orthographic")

    imgui.end()


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
    elif kind == 3:  # Library: a shelf of four assets.
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
    imgui.begin("Parameters", None, _CHROME_FLAGS | imgui.WindowFlags_.no_title_bar)

    rail_width = SIDEBAR_ICON_RAIL * scale
    button_size = 34 * scale
    labels = ("Bake", "Material", "Decal", "Library", "Settings")
    drawers = (
        _draw_bake_tab,
        _draw_texture_tab,
        _draw_decal_tab,
        _draw_decal_library_tab,
        _draw_settings_tab,
    )
    app.sidebar_view = max(0, min(int(getattr(app, "sidebar_view", 0)), len(labels) - 1))

    imgui.begin_child("##sidebar_icon_rail", imgui.ImVec2(rail_width, 0))
    for index, label in enumerate(labels):
        if _draw_sidebar_icon(index, app.sidebar_view == index, button_size):
            app.sidebar_view = index
        _tooltip(label)
        imgui.spacing()
    imgui.end_child()

    imgui.same_line()
    imgui.begin_child("##sidebar_view", imgui.ImVec2(0, 0))
    imgui.separator_text(labels[app.sidebar_view])
    drawers[app.sidebar_view](app)
    imgui.end_child()
    imgui.end()


def _draw_bake_tab(app: "MeshMapApp") -> None:
    scale = app.ui_pixel_scale
    # Labels sit to the right of each widget; give them a fixed share so long
    # names ("Threshold falloff") are never clipped at any scale.
    imgui.push_item_width(-170 * scale)

    controller = app.controller
    bake = controller.bake_params

    # -- mesh ------------------------------------------------------------
    changed, app.source_z_up = imgui.checkbox("Source is Z-up", app.source_z_up)
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
        for note in info.notes:
            imgui.text_colored(WARN_COLOR, f"! {note}")
    else:
        imgui.text_colored(MUTED_COLOR, "No mesh loaded.")

    # -- bake stage ------------------------------------------------------
    imgui.separator_text("Bake  (re-bake required)")

    resolution_index = RESOLUTIONS.index(bake.resolution) if bake.resolution in RESOLUTIONS else 1
    changed, resolution_index = imgui.combo(
        "Bake resolution", resolution_index, [f"{r} x {r}" for r in RESOLUTIONS]
    )
    if changed:
        bake.resolution = RESOLUTIONS[resolution_index]
    _tooltip("Output texture size. Also re-packs the atlas, since padding is in texels.")

    changed, bake.strength = imgui.slider_float("Strength", bake.strength, 0.0, 2.0)
    _tooltip(
        "Multiplies the lifted normal derivative.\n"
        "ArmorPaint: curvature = pow(...) * strength * 2.0 + offset / 10.0"
    )

    changed, bake.radius = imgui.slider_float("Radius", bake.radius, 0.0, 4.0)
    _tooltip(
        "Not a distance -- an exponent: pow(curvature, (1 / radius) * 0.25).\n"
        "Larger values lift the derivative harder, which widens the band.\n"
        "ArmorPaint's node caps this at 2; the EdgeWear001 group exposes 0..4."
    )

    changed, bake.offset = imgui.slider_float("Offset", bake.offset, -2.0, 2.0)
    _tooltip(
        "Added as offset / 10 after the lift. Negative crushes near-flat areas\n"
        "to black -- EdgeWear001 ships it at -2.0 for exactly that."
    )

    changed, bake.smooth = imgui.slider_int("Smooth", bake.smooth, 0, 5)
    _tooltip(
        "Blur iterations over the baked curvature. Each one bounces the map\n"
        "through a 95%-size target and back, which is how ArmorPaint blurs it."
    )

    changed, bake.axis = imgui.combo("Axis", bake.axis, list(BAKE_AXES))
    _tooltip(
        "Anything but XYZ multiplies the result by dot(normal, axis), so only\n"
        "edges facing that way wear. Z is up as in Blender, so -Z is gravity."
    )

    # -- bevel ------------------------------------------------------------
    imgui.separator_text("Bevel  (re-bake required)")
    bevel = controller.bevel_params
    _, bevel.enabled = imgui.checkbox("Bevel sharp edges", bevel.enabled)
    _tooltip(
        "Blender's Bevel modifier, approximated: replaces sharp edges with a\n"
        "narrow strip before baking. A perfectly sharp edge has no width for\n"
        "the curvature bake to put a gradient in, so without this the gradient\n"
        "smears across whole faces and they bake grey instead of the edge\n"
        "baking white. The bevel is never exported -- it inherits your UVs, so\n"
        "the texture still fits your original unbeveled mesh."
    )

    imgui.begin_disabled(not bevel.enabled)
    _, bevel.amount = imgui.slider_float(
        "Amount (m)", bevel.amount, 0.0001, 0.05, "%.4f", imgui.SliderFlags_.logarithmic
    )
    _tooltip(
        "Blender's Amount under the Offset width type: how far the new boundary\n"
        "edge sits from the original edge. 0.001 (1 mm) is the subtle bevel that\n"
        "catches a highlight without reading as visibly rounded."
    )
    _, bevel.segments = imgui.slider_int("Segments", bevel.segments, 1, 8)
    _tooltip(
        "Subdivisions across the bevel. 1 is a flat chamfer, 2-3 lightly\n"
        "rounded, more is smoother but adds geometry."
    )
    _, bevel.angle = imgui.slider_float("Angle", bevel.angle, 1.0, 180.0, "%.0f deg")
    _tooltip(
        "Limit Method: Angle. Only edges whose two faces diverge by more than\n"
        "this get beveled, so box corners qualify and coplanar edges splitting\n"
        "a flat surface are left alone. 30 degrees is Blender's usual choice."
    )

    # A bevel narrower than a texel falls between samples and bakes nothing, so
    # say how wide this one lands rather than let it silently do nothing.
    if info is not None and info.uv_density > 0.0:
        texels = bevel.amount * info.uv_density * bake.resolution
        if texels < 2.0:
            imgui.text_colored(
                WARN_COLOR,
                f"! {texels:.1f} texels wide - too thin to bake, raise Amount "
                f"or the resolution",
            )
        else:
            imgui.text_colored(MUTED_COLOR, f"{texels:.1f} texels wide in the atlas")
    imgui.end_disabled()

    if imgui.collapsing_header("Advanced bake settings"):
        _, bake.dilation = imgui.slider_int("Seam padding (texels)", bake.dilation, 0, 16)
        _tooltip("Pads chart borders outward so bilinear filtering never samples empty gutter.")

        unwrap_params = controller.unwrap_params
        _, unwrap_params.use_source_uvs = imgui.checkbox(
            "Bake into source UVs", unwrap_params.use_source_uvs
        )
        _tooltip(
            "Bake into the UV map already on the mesh, so the PNG drops straight\n"
            "onto the model you exported from Blender. Turn this off only for a\n"
            "mesh with no UVs -- xatlas then packs a throwaway atlas that fits\n"
            "nothing but the triangulated OBJ this app writes."
        )

        imgui.begin_disabled(unwrap_params.use_source_uvs)
        _, unwrap_params.padding = imgui.slider_int("Atlas padding", unwrap_params.padding, 0, 16)
        _, unwrap_params.normal_deviation_weight = imgui.slider_float(
            "Seam eagerness", unwrap_params.normal_deviation_weight, 0.5, 8.0
        )
        _tooltip("How readily xatlas cuts a seam where the surface bends.")
        _, unwrap_params.brute_force = imgui.checkbox("Brute-force packing", unwrap_params.brute_force)
        imgui.end_disabled()

    _draw_bake_controls(app)

    imgui.pop_item_width()


def _draw_texture_tab(app: "MeshMapApp") -> None:
    """The texture: what it is made of, and what the selected piece looks like.

    A texture starts as one flat colour. Changing its type to a mask grows two
    slots underneath -- white and black -- and either of those can be a colour
    or another mask, so the shape is a binary tree of unbounded depth.

    Indenting all of that runs out of panel before it runs out of tree, so the
    two halves do different jobs: the top edits whatever is *selected*, at a
    fixed size no matter how deep it sits, and the bottom is the tree itself,
    one line per slot, for selecting with. Depth costs a line rather than a
    column, and the controls never move.
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

    # Two panes, each with its own scrollbar and its own header, splitting the
    # room evenly until the divider between them is dragged. Fixed shares
    # rather than "whatever the contents need": the inspector's height changes
    # with what is selected, and a tree that jumped about as it did would be a
    # tree you could not keep your place in.
    scale = app.ui_pixel_scale
    splitter = 8.0 * scale
    header = imgui.get_frame_height_with_spacing()
    usable = max(imgui.get_content_region_avail().y - 2 * header - splitter,
                 4 * header)
    inspector_height = usable * app.texture_split

    imgui.separator_text("Inspector")
    imgui.begin_child("texture_inspector", imgui.ImVec2(0, inspector_height))
    _draw_slot_params(app)
    imgui.end_child()

    _draw_splitter(app, splitter, usable)

    imgui.separator_text("Material tree")
    imgui.begin_child("texture_tree", imgui.ImVec2(0, 0))
    _draw_texture_tree(app)
    imgui.end_child()


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
    share = app.texture_split if name == "texture" else app.decal_split
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
    """Why the model might not look like the texture says it should.

    Above the tree rather than below it: a texture that renders black because
    its mask found nothing is exactly when the panel needs to be read, and the
    tree scrolls.
    """
    if depth(app.texture) >= MAX_DEPTH:
        imgui.text_colored(WARN_COLOR, f"! Nesting stops at {MAX_DEPTH} levels.")

    if app.controller.curvature_map is None:
        imgui.text_colored(
            WARN_COLOR, "! Bake first - the masks read the curvature and positions."
        )
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
        imgui.text_colored(
            WARN_COLOR,
            "! One flat colour: every texel lands on\n"
            "  the same side of its mask.\n" + remedy,
        )


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
    changed, index = imgui.combo(
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
        changed, color = imgui.color_edit3(
            "##color", list(slot.color),
            imgui.ColorEditFlags_.no_inputs | imgui.ColorEditFlags_.no_label,
        )
        if changed:
            slot.color = (float(color[0]), float(color[1]), float(color[2]))
        imgui.same_line()
        imgui.text(f"Colour  {slot.auto_label}")
        _tooltip(
            "Click the swatch to open the picker.\n"
            "Click away or press Escape to close it again."
        )

        _draw_surface_params(app, slot)
    elif _draw_mask_params(app, slot):
        app.mark_texture_dirty()

    imgui.pop_item_width()


def _draw_surface_params(app: "MeshMapApp", slot) -> None:
    """What the surface is made of, beyond its colour.

    These ride through the tree exactly as the colour does -- the mask blends
    them the same way -- and each comes out as its own exported map, so what is
    dialled in here is what the renderer at the other end receives.
    """
    imgui.separator_text("Surface")

    _, slot.metallic = imgui.slider_float("Metallic", slot.metallic, 0.0, 1.0)
    _tooltip(
        "0 is a dielectric -- paint, plastic, rust. 1 is bare metal.\n"
        "In between is for a surface partly covered by something else, not for\n"
        "a material that is half a metal."
    )
    _, slot.roughness = imgui.slider_float("Roughness", slot.roughness, 0.0, 1.0)
    _tooltip("How wide the highlight spreads: 0 is a mirror, 1 is chalk.")
    _, slot.alpha = imgui.slider_float("Alpha", slot.alpha, 0.0, 1.0)
    _tooltip(
        "Opacity. Below 1 the surface lets what is behind it through.\n"
        "Shading only -- the export is a PBR set of five, and this is not\n"
        "one of them."
    )
    _, slot.emission = imgui.slider_float(
        "Emission", slot.emission, 0.0, MAX_EMISSION, "%.2f",
        imgui.SliderFlags_.logarithmic,
    )
    _tooltip(
        "Light the surface gives off, in multiples of its own colour. Past 1\n"
        "the surface is brighter than white can show, and the extra turns into\n"
        "the glow spilling past its edge -- which is what carries the difference\n"
        "between a lit surface and a light. Shading only, like Alpha: the\n"
        "export is a PBR set of five, and this is not one of them."
    )


def _draw_mask_params(app: "MeshMapApp", node) -> bool:
    """The selected mask's own knobs. Live, like everything in this tab.

    All of them, laid out flat. The finer settings sat behind a collapsing
    header, which buys back a few rows of a sidebar that already has the room
    and costs a click to find out what is in there.
    """
    dirty = False

    imgui.separator_text("Boundary")
    changed, node.threshold = imgui.slider_float("Threshold", node.threshold, 0.0, 1.0)
    dirty |= changed
    _tooltip(
        "Where the mask divides its two sides. A mask is a continuous field --\n"
        "edge wear ramps up as the surface curves, noise wanders through every\n"
        "value -- and this is the level that decides which side a texel is on.\n"
        "Lower gives white's side more of the surface."
    )
    changed, node.softness = imgui.slider_float("Softness", node.softness, 0.0, 0.5)
    dirty |= changed
    _tooltip(
        "How wide the crossing is. 0 is a clean division: every texel belongs\n"
        "to one side or the other. Raise it to blend them across a band."
    )

    imgui.separator_text(SLOT_KINDS[node.kind])
    if node.kind == "noise":
        noise = node.noise
        changed, noise.scale = imgui.slider_float(
            "Scale", noise.scale, 0.5, 80.0, "%.1f", imgui.SliderFlags_.logarithmic
        )
        dirty |= changed
        _tooltip("Cycles across the model's bounding box. Higher is finer.")
        changed, noise.bias = imgui.slider_float("Bias", noise.bias, 0.0, 1.0)
        dirty |= changed
        _tooltip("The noise level that lands on the midpoint. Lower gives more white.")
        changed, noise.contrast = imgui.slider_float("Contrast", noise.contrast, 0.0, 20.0)
        dirty |= changed
        _tooltip("How hard the crossing is. 0 is flat grey, high is a clean edge.")

        imgui.separator_text("Noise detail")
        changed, noise.detail = imgui.slider_float("Detail", noise.detail, 0.0, 8.0)
        dirty |= changed
        _tooltip("fBM octaves.")
        changed, noise.roughness = imgui.slider_float("Roughness", noise.roughness, 0.0, 1.0)
        dirty |= changed
        _tooltip("Amplitude falloff per octave.")
        changed, noise.lacunarity = imgui.slider_float("Lacunarity", noise.lacunarity, 0.0, 8.0)
        dirty |= changed
        _tooltip("Frequency step per octave.")
        changed, noise.distortion = imgui.slider_float("Distortion", noise.distortion, 0.0, 2.0)
        dirty |= changed
        _tooltip("Warps the sample position by the noise itself before evaluating.")
        return dirty

    _muted_wrapped(
        "The EdgeWear001 node group, on this layer. What it produces reaches "
        "the export the way every layer does: through color.png."
    )

    wear = node.edge_wear
    changed, wear.value = imgui.slider_float("Value", wear.value, 0.0, 5.0)
    dirty |= changed
    _tooltip(
        "Group Input.Value. Feeds a x10 Math node into the noise Scale, so the\n"
        "noise runs at value * 10. Higher = finer, busier break-up."
    )
    changed, wear.wear_amount = imgui.slider_float("Wear amount", wear.wear_amount, 0.0, 2.0)
    dirty |= changed
    _tooltip(
        "The \"Wear Amount\" multiply. How hard the noise erodes the curvature\n"
        "before the subtract -- raise it and the wear gets patchier."
    )
    changed, wear.contrast = imgui.slider_float("Contrast", wear.contrast, 0.0, 10.0)
    dirty |= changed
    _tooltip(
        "The \"Contrast\" multiply, clamped to 0..1 after.\n"
        "mask = clamp((curvature - noise * wear_amount) * contrast, 0, 1)"
    )

    imgui.separator_text("Wear noise")
    changed, wear.detail = imgui.slider_float("Detail", wear.detail, 0.0, 8.0)
    dirty |= changed
    _tooltip("fBM octaves.")
    changed, wear.roughness = imgui.slider_float("Roughness", wear.roughness, 0.0, 1.0)
    dirty |= changed
    _tooltip("Amplitude falloff per octave.")
    changed, wear.lacunarity = imgui.slider_float("Lacunarity", wear.lacunarity, 0.0, 8.0)
    dirty |= changed
    _tooltip("Frequency step per octave.")
    changed, wear.distortion = imgui.slider_float("Distortion", wear.distortion, 0.0, 2.0)
    dirty |= changed
    _tooltip("Warps the sample position by the noise itself before evaluating.")
    return dirty


def _draw_texture_tree(app: "MeshMapApp") -> None:
    """The whole texture, one selectable line per slot.

    Selection and naming -- the controls live above -- so it can afford to
    indent: a line per node stays legible far deeper than a stack of sliders
    would. Double-clicking a row renames it in place.
    """
    scale = app.ui_pixel_scale
    for path, slot in walk(app.texture):
        imgui.push_id(f"tree{'/'.join(path)}")
        indent = len(path) * 12 * scale
        if indent:
            imgui.indent(indent)

        if app.renaming_path == path:
            # No swatch, no thumbnail: while a row is being renamed it is the
            # name and nothing else, so what is typed is what is read back.
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
            _draw_tree_row(app, path, slot)

        if indent:
            imgui.unindent(indent)
        imgui.pop_id()


def _draw_tree_row(app: "MeshMapApp", path, slot) -> None:
    """One row: which side of its mask it is, and what it is called.

    What it is called and nothing else -- a named slot shows its name, not the
    hex code or the kind it happens to be. That is what the Type dropdown above
    is for, and a row that argues with the name given to it reads as the rename
    not having taken.
    """
    side = f"{path[-1][0].upper()}  " if path else ""

    clicked, _ = imgui.selectable(
        f"{side}{describe(slot)}##row", path == app.texture_path,
        imgui.SelectableFlags_.allow_double_click,
    )
    if clicked:
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


def _draw_tree_row(app: "MeshMapApp", path, slot) -> None:
    """One row: which side of its mask it is, and what it is called.

    What it is called and nothing else -- a named slot shows its name, not the
    hex code or the kind it happens to be. That is what the Type dropdown above
    is for, and a row that argues with the name given to it reads as the rename
    not having taken.
    """
    side = f"{path[-1][0].upper()}  " if path else ""

    clicked, _ = imgui.selectable(
        f"{side}{describe(slot)}##row", path == app.texture_path,
        imgui.SelectableFlags_.allow_double_click,
    )
    if clicked:
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
    """The selected decal's controls and every decal currently on the mesh.

    The two panes share selection with the viewport: clicking a row below makes
    that decal the inspector's subject and gives it the viewport outline.
    """
    scale = app.ui_pixel_scale

    splitter = 8.0 * scale
    header = imgui.get_frame_height_with_spacing()
    usable = max(imgui.get_content_region_avail().y - header - splitter,
                 4 * header)
    inspector_height = usable * app.decal_split

    imgui.begin_child("decal_inspector", imgui.ImVec2(0, inspector_height))
    _draw_decal_inspector(app)
    imgui.end_child()

    _draw_splitter(app, splitter, usable, "decal", app.set_decal_split)

    imgui.separator_text(f"In use  ({len(app.decals)})")
    imgui.begin_child("decals_in_use", imgui.ImVec2(0, 0))
    _draw_decals_in_use(app)
    imgui.end_child()


def _draw_decals_in_use(app: "MeshMapApp") -> None:
    """Selectable scene list backed by the viewport's decal selection."""
    if not app.decals:
        _muted_wrapped("No decals on the mesh. Drag one onto it from the Library view.")
        return

    for index, decal in enumerate(app.decals):
        imgui.push_id(f"decal_in_use_{index}")
        name = Path(decal.path).name or f"Decal {index + 1}"
        state = "" if decal.enabled else "  (disabled)"
        clicked, _ = imgui.selectable(
            f"{index + 1}.  {name}{state}", index == app.decal_index
        )
        if clicked:
            app.select_decal(index)
        _tooltip("Select this decal in the inspector and viewport.")
        imgui.pop_id()


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


def _draw_decal_inspector(app: "MeshMapApp") -> None:
    """The selected decal's own controls."""
    scale = app.ui_pixel_scale
    imgui.push_item_width(-170 * scale)
    decal = app.selected_decal
    dirty = False

    if decal is None:
        _muted_wrapped(
            "Nothing selected. Choose a decal from the list below, click one "
            "on the model, or drag a new one from the Library view."
        )
        if app.decals:
            _muted_wrapped(f"{len(app.decals)} on the mesh.")
        imgui.pop_item_width()
        return

    image = app.decal_image_for(decal)
    texture = app.decal_textures.get(decal.path)
    if image is not None and texture is not None:
        thumbnail = DECAL_THUMBNAIL * scale
        width, height = image.size
        imgui.image(
            imgui.ImTextureRef(texture.glo),
            imgui.ImVec2(thumbnail, thumbnail * height / max(width, 1)),
        )
        imgui.same_line()

    imgui.begin_group()
    origin = "height map, converted" if image and image.from_height else "normal map"
    imgui.text_colored(
        MUTED_COLOR,
        f"{Path(decal.path).name}\n"
        f"{decal.image_aspect:.2f} : 1  ({origin})\n"
        f"{app.decal_index + 1} of {len(app.decals)} on the mesh",
    )
    changed, decal.enabled = imgui.checkbox("Enabled", decal.enabled)
    dirty |= changed
    _tooltip("Off leaves this decal out of the normal map, without removing it.")
    imgui.end_group()

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

    imgui.separator_text("Appearance")
    texture_labels = ["None"] + [describe(texture) for texture in app.textures]
    current_texture = (
        decal.texture_index + 1
        if 0 <= decal.texture_index < len(app.textures)
        else 0
    )
    changed, current_texture = imgui.combo(
        "Material", current_texture, texture_labels
    )
    if changed:
        decal.texture_index = current_texture - 1
        dirty = True
    _tooltip(
        "Use a material created in the Material tab to colour this decal.\n"
        "None changes only the surface normal."
    )

    imgui.separator_text("Depth")
    changed, decal.intensity = imgui.slider_float(
        "Height intensity", decal.intensity, 0.0, 4.0, "%.2f"
    )
    dirty |= changed
    _tooltip(
        "How deep the bump reads. This scales the surface slope the map\n"
        "describes, not the stored vector, so 2.0 is genuinely twice as steep\n"
        "instead of tipping the normal flat against the surface.\n"
        "0 is flat, 1 is the map exactly as it was authored."
    )

    changed, decal.flip_green = imgui.checkbox("Flip green (DirectX)", decal.flip_green)
    dirty |= changed
    _tooltip(
        "Tick for a map baked in DirectX convention (-Y). Everything here is\n"
        "OpenGL (+Y up), which is what Blender expects. If the decal reads\n"
        "inside-out -- raised where it should be recessed -- this is why."
    )

    imgui.separator_text("Surface placement")
    changed, decal.scale = imgui.slider_float(
        "Scale", decal.scale, 0.01, 1.0, "%.3f", imgui.SliderFlags_.logarithmic
    )
    dirty |= changed
    _tooltip(
        "Size on the anchor face. Its height follows from\n"
        "two things: the image's own aspect ratio, so a wide vent stays wide,\n"
        "and how stretched the mesh's UVs are where the decal sits, so a round\n"
        "one comes out round. Across a UV seam it continues on the connected face."
    )

    changed, decal.scale_x = imgui.slider_float(
        "Width factor", decal.scale_x, 0.02, 4.0, "%.3f", imgui.SliderFlags_.logarithmic
    )
    dirty |= changed
    changed, decal.scale_y = imgui.slider_float(
        "Height factor", decal.scale_y, 0.02, 4.0, "%.3f", imgui.SliderFlags_.logarithmic
    )
    dirty |= changed
    _tooltip("Independent factors changed by S then X or Y. 1.0 preserves the image shape.")

    stretch = float(decal.surface_aspect)
    if abs(stretch - 1.0) > 0.02:
        wider = "across" if stretch > 1.0 else "up"
        _muted_wrapped(
            f"UV here is {max(stretch, 1 / stretch):.2f}x stretched {wider}; "
            "the decal's shape is corrected for it."
        )

    changed, decal.falloff = imgui.slider_float(
        "Edge falloff", decal.falloff, 0.0, 1.0, "%.2f"
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

    changed, decal.center_u = imgui.slider_float("Position U", decal.center_u, 0.0, 1.0, "%.3f")
    dirty |= changed
    changed, decal.center_v = imgui.slider_float("Position V", decal.center_v, 0.0, 1.0, "%.3f")
    dirty |= changed
    _tooltip(
        "Where the middle of the decal sits in the UV layout. Usually easier to\n"
        "set by pointing: click the decal above, or 'Place on the mesh', and put\n"
        "it where you want it on the model."
    )

    changed, decal.rotation = imgui.slider_float(
        "Rotation", decal.rotation, -180.0, 180.0, "%.0f deg"
    )
    dirty |= changed

    if imgui.button("Reset placement"):
        blank = type(decal)()
        decal.center_u, decal.center_v = blank.center_u, blank.center_v
        decal.scale, decal.scale_x, decal.scale_y = blank.scale, blank.scale_x, blank.scale_y
        decal.rotation = blank.rotation
        dirty = True

    imgui.end_disabled()

    # Where it lands on the mesh, in the units the artist thinks in.
    info = app.mesh_info
    if info is not None and info.uv_density > 0.0:
        span = decal.scale / info.uv_density
        imgui.text_colored(
            MUTED_COLOR, f"About {span:.3g} m across on the mesh at this scale"
        )

    if info is not None and not info.has_uvs:
        imgui.text_colored(
            WARN_COLOR,
            "! This mesh carries no UVs, so there is no layout to place into\n"
            "  yet. Bake once to generate an atlas -- but that atlas fits only\n"
            "  the OBJ this app exports, not your original mesh.",
        )

    if dirty:
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

    scale = app.ui_pixel_scale
    for index, (name, label) in enumerate(_MAP_LABELS):
        if index % 2:
            imgui.same_line(200 * scale)
        changed, enabled = imgui.checkbox(label, app.map_enabled(name))
        if changed:
            app.set_map_enabled(name, enabled)

    if not app.map_enabled("ao"):
        _muted_wrapped(
            "Occlusion is switched off, so the bake skips it -- which is most "
            "of what a bake costs. Switching it back on needs a re-bake."
        )


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

    imgui.separator_text("Viewport navigation")
    imgui.text_colored(
        MUTED_COLOR,
        "Multipliers on the base rates.\n"
        "1.0 orbits at 0.4 deg/pixel, Blender's default.",
    )

    changed, nav.orbit_speed = imgui.slider_float(
        "Orbit speed", nav.orbit_speed, 0.05, 3.0, "%.2fx",
        imgui.SliderFlags_.logarithmic,
    )
    dirty |= changed
    _tooltip("Degrees turned per pixel of drag. Applies to scroll-orbit too.")

    changed, nav.pan_speed = imgui.slider_float(
        "Pan speed", nav.pan_speed, 0.05, 3.0, "%.2fx", imgui.SliderFlags_.logarithmic
    )
    dirty |= changed

    changed, nav.zoom_speed = imgui.slider_float(
        "Zoom speed", nav.zoom_speed, 0.05, 3.0, "%.2fx", imgui.SliderFlags_.logarithmic
    )
    dirty |= changed

    changed, nav.scroll_speed = imgui.slider_float(
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

    changed, nav.smoothing = imgui.slider_float(
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

    changed, nav.invert_orbit_x = imgui.checkbox("Invert orbit X", nav.invert_orbit_x)
    dirty |= changed
    imgui.same_line()
    changed, nav.invert_orbit_y = imgui.checkbox("Invert orbit Y", nav.invert_orbit_y)
    dirty |= changed
    _tooltip("For natural-scrolling trackpads, or simple preference.")

    if imgui.button("Reset navigation"):
        app.navigation = type(nav)()
        dirty = True

    if dirty:
        app.apply_navigation()
    dirty = False

    imgui.separator_text("World lighting")
    imgui.text_colored(
        MUTED_COLOR,
        "Where the key light stands, and the light the\nmodel sits in beyond it.",
    )

    world = app.world
    changed, world.rotation = imgui.slider_float(
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

    changed, world.strength = imgui.slider_float(
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
        "Colour", list(world.color), imgui.ColorEditFlags_.no_inputs
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

    imgui.separator_text("Export")
    imgui.text_colored(
        MUTED_COLOR,
        "What File > Export textures writes, and where\nit starts looking.",
    )

    _, app.export_dir = imgui.input_text("Folder", app.export_dir)
    _tooltip(
        "Where the folder chooser opens. Exporting somewhere else moves this\n"
        "here, so the next export starts from where the last one went. The maps\n"
        "land in a folder named after the mesh."
    )
    imgui.same_line()
    if imgui.button("..."):
        app.folder_dialog = pfd.select_folder("Export folder", app.export_dir)

    bit_index = 0 if app.export_bits == 8 else 1
    changed, bit_index = imgui.combo("Depth", bit_index, ["8-bit PNG", "16-bit PNG"])
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

    imgui.separator_text("Interface")
    changed, requested = imgui.slider_float("UI scale", app.ui_scale, 0.6, 3.0, "%.2fx")
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
        imgui.text_colored(ERROR_COLOR, controller.error.strip().splitlines()[-1])


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
        imgui.text_colored(ERROR_COLOR, line)
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
