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

from core.params import BAKE_AXES, RESOLUTIONS

if TYPE_CHECKING:  # pragma: no cover
    from render.viewport import MeshMapApp

#: Panel geometry in unscaled reference pixels; multiplied by the app's
#: ui_pixel_scale, because ImGui is driven in physical pixels here.
PANEL_WIDTH = 430
INSPECTOR_WIDTH = 380
#: Width of the clickable decal preview in the Decal tab.
DECAL_THUMBNAIL = 92
STATUS_BAR_HEIGHT = 34
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


def draw_panel(app: "MeshMapApp") -> None:
    _draw_parameters(app)
    if app.show_map_inspector:
        _draw_map_inspector(app)
    _draw_status_bar(app)
    _pump_dialogs(app)


# --------------------------------------------------------------------------

def _draw_parameters(app: "MeshMapApp") -> None:
    scale = app.ui_pixel_scale
    margin = 12 * scale
    # Fill the window height minus the status bar, so a long parameter list
    # scrolls inside the panel instead of disappearing underneath it.
    height = app.wnd.buffer_size[1] - STATUS_BAR_HEIGHT * scale - 2 * margin
    imgui.set_next_window_pos(imgui.ImVec2(margin, margin), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size(
        imgui.ImVec2(PANEL_WIDTH * scale, max(240 * scale, height)),
        imgui.Cond_.first_use_ever,
    )
    imgui.begin("Parameters")
    if imgui.begin_tab_bar("panel_tabs"):
        selected, _ = imgui.begin_tab_item("Bake")
        if selected:
            _draw_bake_tab(app)
            imgui.end_tab_item()
        selected, _ = imgui.begin_tab_item("Decal")
        if selected:
            _draw_decal_tab(app)
            imgui.end_tab_item()
        selected, _ = imgui.begin_tab_item("Settings")
        if selected:
            _draw_settings_tab(app)
            imgui.end_tab_item()
        imgui.end_tab_bar()
    imgui.end()


def _draw_bake_tab(app: "MeshMapApp") -> None:
    from render.viewport import PREVIEW_MODES  # local import avoids a cycle

    scale = app.ui_pixel_scale
    # Labels sit to the right of each widget; give them a fixed share so long
    # names ("Threshold falloff") are never clipped at any scale.
    imgui.push_item_width(-170 * scale)

    controller = app.controller
    bake = controller.bake_params

    # -- mesh ------------------------------------------------------------
    if imgui.button("Open mesh..."):
        app.file_dialog = pfd.open_file("Open mesh", str(Path.home()), MESH_FILTERS)
    imgui.same_line()
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

    # -- preview ---------------------------------------------------------
    imgui.separator_text("Preview")
    labels = [mode.label for mode in PREVIEW_MODES]
    changed, app.preview_index = imgui.combo("Show", app.preview_index, labels)
    if changed:
        app._thumbnail_dirty = True
    _tooltip("Keys 1-5 switch modes.")

    _, app.lighting = imgui.checkbox("Lighting", app.lighting)
    imgui.same_line()
    _, app.wireframe = imgui.checkbox("Wireframe", app.wireframe)
    imgui.same_line()
    if PREVIEW_MODES[app.preview_index].label == "UV checker":
        _, app.checker_scale = imgui.slider_float("Checker density", app.checker_scale, 2.0, 96.0, "%.0f")

    _, app.show_map_inspector = imgui.checkbox("Map inspector", app.show_map_inspector)
    imgui.same_line()
    _, app.show_gizmo = imgui.checkbox("Axis gizmo", app.show_gizmo)
    _tooltip(
        "The axis balls in the top-right corner. Click one to look straight down\n"
        "that axis in orthographic projection; orbiting returns to perspective.\n"
        "Z is up, matching Blender."
    )
    if app.camera.orthographic:
        imgui.text_colored(MUTED_COLOR, "Orthographic - orbit to return to perspective")

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

    # -- EdgeWear001 node group ------------------------------------------
    imgui.separator_text("Edge wear  (live)")
    wear = app.wear_params
    dirty = False

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

    if imgui.collapsing_header("Wear Noise (TEX_NOISE)"):
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

    if imgui.button("Reset to EdgeWear001"):
        app.wear_params = type(wear)()
        dirty = True
    _tooltip("Back to the values shipped in EdgeWear001.arm.")

    if dirty:
        app.mark_output_dirty()
        app._thumbnail_dirty = True

    # -- export ----------------------------------------------------------
    imgui.separator_text("Export")
    _, app.export_dir = imgui.input_text("Folder", app.export_dir)
    imgui.same_line()
    if imgui.button("..."):
        app.folder_dialog = pfd.select_folder("Export folder", app.export_dir)

    bit_index = 0 if app.export_bits == 8 else 1
    changed, bit_index = imgui.combo("Depth", bit_index, ["8-bit PNG", "16-bit PNG"])
    if changed:
        app.export_bits = 8 if bit_index == 0 else 16
    _tooltip("Reach for 16-bit if you see banding in the gradients.")

    baked = controller.curvature_map is not None
    decal = app.decal_params.active()
    names = [name for name, present in
             (("edge_wear / curvature", baked), ("normal", decal)) if present]
    imgui.begin_disabled(not names)
    label = f"Export {' + '.join(names)}" if names else "Export"
    if imgui.button(label, imgui.ImVec2(-1, 0)):
        app.export()
    _tooltip(
        "Writes edge_wear.png and curvature.png from the bake, and normal.png\n"
        "for any decal. They apply to your original mesh directly, because both\n"
        "use its own UV map."
    )
    imgui.end_disabled()

    imgui.pop_item_width()


def _draw_decal_tab(app: "MeshMapApp") -> None:
    """Place an imported normal map into the mesh's UV layout.

    Every control here is live: the composite is one full-screen pass over the
    atlas, so the mesh in the viewport re-lights as the handle moves.
    """
    from render.viewport import PREVIEW_MODES  # local import avoids a cycle

    scale = app.ui_pixel_scale
    imgui.push_item_width(-170 * scale)

    decal = app.decal_params
    image = app.decal_image
    dirty = False

    if imgui.button("Import normal map..."):
        app.decal_dialog = pfd.open_file("Import decal", str(Path.home()), DECAL_FILTERS)
    _tooltip(
        "A tangent-space normal map -- a scifi vent, a hatch, a panel seam.\n"
        "A grayscale image is read as a height map and converted, since that is\n"
        "the only reading of it that describes a surface."
    )
    if image is not None:
        imgui.same_line()
        if imgui.button("Clear"):
            app.clear_decal()
            image = None

    if image is None:
        imgui.text_colored(
            MUTED_COLOR,
            "No decal loaded.\n"
            "The decal is stamped into the mesh's UV layout and exported as\n"
            "normal.png, alongside the bake's own maps.",
        )
        imgui.pop_item_width()
        return

    width, height = image.size
    origin = "height map, converted" if image.from_height else "normal map"

    # The map itself, clickable: picking it up is the quickest way to place it,
    # and pointing at the model beats reasoning about UV coordinates.
    if app.tex_decal is not None:
        thumbnail = DECAL_THUMBNAIL * scale
        imgui.begin_group()
        if imgui.image_button(
            "##decal_pick",
            imgui.ImTextureRef(app.tex_decal.glo),
            imgui.ImVec2(thumbnail, thumbnail * height / max(width, 1)),
        ):
            app.begin_decal_placement()
        _tooltip(
            "Click to pick the decal up, then move the cursor over the mesh --\n"
            "it follows the surface under it. Click again to drop it there.\n"
            "Esc or right-click puts it back where it was."
        )
        imgui.end_group()
        imgui.same_line()

    imgui.begin_group()
    imgui.text_colored(
        MUTED_COLOR,
        f"{Path(decal.path).name}\n{width} x {height}\n({origin})",
    )
    changed, decal.enabled = imgui.checkbox("Enabled", decal.enabled)
    dirty |= changed
    _tooltip("Off leaves the normal map flat, and stops normal.png being written.")
    imgui.end_group()

    if app.decal_placing:
        imgui.text_colored(
            WARN_COLOR, "Placing: click on the mesh to drop it, Esc to cancel"
        )
        if imgui.button("Cancel placement", imgui.ImVec2(-1, 0)):
            app.end_decal_placement(keep=False)
    else:
        imgui.begin_disabled(app.mesh is None)
        if imgui.button("Place on the mesh", imgui.ImVec2(-1, 0)):
            app.begin_decal_placement()
        imgui.end_disabled()
        _tooltip(
            "Point at the model instead of typing coordinates. The decal rides\n"
            "the surface under the cursor; click to drop it, Esc to cancel."
        )

    imgui.begin_disabled(not decal.enabled)

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

    imgui.separator_text("Placement  (UV space)")
    changed, decal.scale = imgui.slider_float(
        "Scale", decal.scale, 0.01, 1.0, "%.3f", imgui.SliderFlags_.logarithmic
    )
    dirty |= changed
    _tooltip(
        "Fraction of the atlas the decal spans across. Its height follows from\n"
        "the image's own aspect ratio, so a wide vent stays wide."
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
        decal.scale, decal.rotation = blank.scale, blank.rotation
        dirty = True

    imgui.end_disabled()

    # Where it lands on the mesh, in the units the artist thinks in.
    info = app.mesh_info
    if info is not None and info.uv_density > 0.0:
        span = decal.scale / info.uv_density
        imgui.text_colored(
            MUTED_COLOR, f"About {span:.3g} m across on the mesh at this scale"
        )

    imgui.separator_text("Preview")
    decal_mode = next(
        (index for index, mode in enumerate(PREVIEW_MODES) if mode.texture == "normal"),
        None,
    )
    if decal_mode is not None and imgui.button("Show the normal map", imgui.ImVec2(-1, 0)):
        app.preview_index = decal_mode
        app._thumbnail_dirty = True
    _tooltip(
        "The composited map itself, as colour. The decal lights the mesh in\n"
        "every other mode too, exactly as it will once exported."
    )

    imgui.text_colored(
        MUTED_COLOR,
        "Exported as normal.png with the other maps. The bake is not involved:\n"
        "the decal is placed in UV space, so it needs no geometry pass.",
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


def _draw_settings_tab(app: "MeshMapApp") -> None:
    """Viewport navigation speeds and interface scale, all persisted."""
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

    imgui.separator_text("Interface")
    changed, requested = imgui.slider_float("UI scale", app.ui_scale, 0.6, 3.0, "%.2fx")
    if changed:
        app.set_ui_scale(requested)
    _tooltip("Size of this panel and its text. Remembered between launches.")

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

def _draw_map_inspector(app: "MeshMapApp") -> None:
    from render.viewport import PREVIEW_MODES

    # Anchored by its own left edge -- set_next_window_pos's pivot is ignored
    # under Cond_.first_use_ever, so the offset is computed here instead.
    scale = app.ui_pixel_scale
    width = INSPECTOR_WIDTH * scale
    margin = 12 * scale
    imgui.set_next_window_pos(
        imgui.ImVec2(
            max((PANEL_WIDTH + 24) * scale, app.wnd.buffer_size[0] - width - margin), margin
        ),
        imgui.Cond_.first_use_ever,
    )
    imgui.set_next_window_size(imgui.ImVec2(width, 460 * scale), imgui.Cond_.first_use_ever)
    expanded, app.show_map_inspector = imgui.begin("Map inspector", app.show_map_inspector)

    # Hand the axis gizmo this window's real rect so it can stay clear of it.
    position, size = imgui.get_window_pos(), imgui.get_window_size()
    app._inspector_rect = (position.x, position.y, size.x, size.y)

    if expanded:
        mode = PREVIEW_MODES[app.preview_index]
        if mode.texture is None:
            imgui.text_colored(MUTED_COLOR, "Pick a map preview mode to inspect it here.")
        elif app.controller.curvature_map is None:
            imgui.text_colored(MUTED_COLOR, "Nothing baked yet.")
        else:
            imgui.text_colored(MUTED_COLOR, f"{mode.label}  -  {app._texture_resolution}px atlas")
            width = max(64.0, imgui.get_content_region_avail().x)
            imgui.image(imgui.ImTextureRef(app.thumbnail.glo), imgui.ImVec2(width, width))
            if app.controller.gbuffer is not None:
                gbuffer = app.controller.gbuffer
                result = app.controller.unwrap_result
                charts = result.chart_count if result else 0
                origin = "source UVs" if result and result.source == "source" else "xatlas"
                imgui.text_colored(
                    MUTED_COLOR,
                    f"{charts} charts, {gbuffer.coverage * 100:.1f}% texel coverage "
                    f"({origin})",
                )

    imgui.end()


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

    if app.decal_dialog is not None and app.decal_dialog.ready():
        selection = app.decal_dialog.result()
        app.decal_dialog = None
        if selection:
            app.open_decal(selection[0])
