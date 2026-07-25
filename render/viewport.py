"""The application window: 3D viewport, live EdgeWear001 pass, event routing."""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pyglm import glm
import moderngl
import moderngl_window as mglw
import numpy as np
import trimesh
from imgui_bundle import imgui
from moderngl_window.scene.camera import OrbitCamera

from core.export import export_maps, export_textured_obj
from core.mesh_io import SUPPORTED_SUFFIXES, MeshLoadError, load_mesh
from core.params import EdgeWearParams, MeshInfo
from core.pipeline import BakeController
from render.imgui_renderer import ImGuiRenderer
from render.shaders import load_shader
from ui.panel import draw_panel

THUMBNAIL_SIZE = 384

#: Index of the fallback mode used when the selected map has not been baked.
SHADED_INDEX = 4


@dataclass(frozen=True)
class PreviewMode:
    label: str
    shader_mode: int
    texture: Optional[str]  # which bake target to sample, if any


PREVIEW_MODES = (
    PreviewMode("Edge wear", 0, "output"),
    PreviewMode("Curvature texture", 0, "curvature"),
    PreviewMode("UV checker", 1, None),
    PreviewMode("Normals", 2, None),
    PreviewMode("Shaded", 3, None),
)


def _settings_dir() -> Path:
    """Per-user config directory for panel layout and preferences.

    Left to itself ImGui drops an ``imgui.ini`` into whatever directory the app
    was launched from, so point it at the platform's config location instead.
    """
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))

    directory = base / "MeshMap"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _load_ui_scale(default: float) -> float:
    """Read the persisted UI scale, falling back to ``default``."""
    try:
        stored = json.loads((_settings_dir() / "prefs.json").read_text())
        return float(np.clip(float(stored["ui_scale"]), 0.6, 3.0))
    except Exception:
        return default


def _save_ui_scale(value: float) -> None:
    try:
        (_settings_dir() / "prefs.json").write_text(json.dumps({"ui_scale": round(value, 3)}))
    except OSError:
        pass  # a read-only home directory is not worth failing over


#: Style fields that are not pixel dimensions and must never be scaled:
#: opacities, tessellation tolerances, hover timings, angles, the font-scale
#: knobs themselves, and the 0..1 alignment ratios.
_STYLE_SCALE_EXCLUDE = frozenset({
    "alpha", "disabled_alpha",
    "circle_tessellation_max_error", "curve_tessellation_tol",
    "font_scale_dpi", "font_scale_main", "font_size_base",
    "hover_delay_normal", "hover_delay_short", "hover_stationary_delay",
    "mouse_cursor_scale", "table_angled_headers_angle", "layout_align",
    "button_text_align", "selectable_text_align", "separator_text_align",
    "table_angled_headers_text_align", "window_title_align",
})


def snapshot_style() -> dict[str, object]:
    """Record the unscaled style metrics, before anything touches them.

    Every scale change is re-derived from this snapshot rather than applied as
    a delta to the current style. ImGui's own ``ScaleAllSizes`` truncates to
    integers, which makes it lossy and *not* invertible: scaling up then back
    down erodes values, and repeated shrinking drives border sizes to zero,
    which trips ``IM_ASSERT(WindowBorderHoverPadding > 0)`` and kills the app.
    """
    style = imgui.get_style()
    snapshot: dict[str, object] = {}
    for name in dir(style):
        if name.startswith("_") or name in _STYLE_SCALE_EXCLUDE:
            continue
        value = getattr(style, name, None)
        if isinstance(value, float):
            snapshot[name] = value
        elif hasattr(value, "x") and hasattr(value, "y"):
            snapshot[name] = (float(value.x), float(value.y))
    return snapshot


def apply_style_scale(defaults: dict[str, object], scale: float) -> None:
    """Set every style metric to its default times ``scale``. Idempotent."""
    style = imgui.get_style()
    for name, base in defaults.items():
        if isinstance(base, tuple):
            setattr(style, name, imgui.ImVec2(base[0] * scale, base[1] * scale))
        else:
            setattr(style, name, float(base) * scale)

    # ImGui asserts this one is strictly positive; keep a floor for tiny scales.
    if hasattr(style, "window_border_hover_padding"):
        style.window_border_hover_padding = max(1.0, style.window_border_hover_padding)

    style.font_scale_main = scale


def _mat_bytes(matrix: glm.mat4) -> bytes:
    """glm stores columns; flattening the column list is already GL's order."""
    return np.array(matrix.to_list(), dtype="f4").tobytes()


class PanOrbitCamera(OrbitCamera):
    """OrbitCamera plus panning, and a zoom that suits any mesh scale.

    The stock ``zoom_state`` is additive and clamps the radius at 1.0, which
    makes it useless for a 5cm bolt or a 40m building. This one is
    multiplicative and clamps relative to the subject.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subject_scale = 1.0
        self.pan_sensitivity = 1.0

    @property
    def eye(self) -> glm.vec3:
        angle_x = glm.radians(self.angle_x)
        angle_y = glm.radians(self.angle_y)
        return glm.vec3(
            glm.cos(angle_x) * glm.sin(angle_y) * self.radius + self.target[0],
            glm.cos(angle_y) * self.radius + self.target[1],
            glm.sin(angle_x) * glm.sin(angle_y) * self.radius + self.target[2],
        )

    def _basis(self) -> tuple[glm.vec3, glm.vec3]:
        forward = glm.normalize(self.target - self.eye)
        right = glm.normalize(glm.cross(forward, self.up))
        return right, glm.normalize(glm.cross(right, forward))

    def pan(self, dx: float, dy: float) -> None:
        right, up = self._basis()
        step = self.radius * 0.0018 * self.pan_sensitivity
        self.target = self.target + (-right * float(dx) + up * float(dy)) * step

    def zoom_state(self, y_offset: float) -> None:
        self.radius *= float(np.exp(-y_offset * 0.12 * self._zoom_sensitivity))
        self.radius = float(np.clip(self.radius, self.subject_scale * 0.02, self.subject_scale * 60.0))

    def frame(self, center, scale: float) -> None:
        """Point the camera at a subject of the given size."""
        self.subject_scale = max(float(scale), 1e-6)
        self.target = glm.vec3(*[float(v) for v in center])
        self.radius = self.subject_scale * 1.5
        self.angle_x, self.angle_y = 45.0, -60.0
        self.projection.update(near=self.subject_scale * 0.005, far=self.subject_scale * 200.0)


class MeshMapApp(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "MeshMap - Edge Wear"
    window_size = (1680, 960)
    aspect_ratio = None
    resizable = True
    vsync = True

    #: Set by main.py before the window is created.
    initial_mesh: Optional[str] = None
    initial_z_up: bool = False
    initial_resolution: int = 1024
    initial_ui_scale: float = 1.35
    #: True when --ui-scale was passed, so it overrides the saved preference.
    initial_ui_scale_explicit: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        imgui.create_context()
        imgui.get_io().set_ini_filename(str(_settings_dir() / "layout.ini"))
        self.gui = ImGuiRenderer(self.wnd)

        self.controller = BakeController(self.ctx)
        self.controller.bake_params.resolution = int(self.initial_resolution)
        self.wear_params = EdgeWearParams()
        self.mesh_info: Optional[MeshInfo] = None
        self.mesh: Optional[trimesh.Trimesh] = None

        self.preview_index = 0
        self.lighting = True
        self.checker_scale = 24.0
        self.wireframe = False
        self.z_up_import = bool(self.initial_z_up)
        self.export_dir = str(Path.cwd() / "output")
        self.export_bits = 8
        self.show_map_inspector = True
        self.status = "Drop an FBX onto the window, or click 'Open mesh...'."
        self.status_is_error = False
        self.file_dialog = None
        self.folder_dialog = None

        self.ui_scale = (
            self.initial_ui_scale
            if self.initial_ui_scale_explicit
            else _load_ui_scale(self.initial_ui_scale)
        )
        # Snapshot before the first scale is applied, so the baseline is pristine.
        self._style_defaults = snapshot_style()
        self.apply_ui_scale()

        self.preview_program = self.ctx.program(
            vertex_shader=load_shader("preview.vert"),
            fragment_shader=load_shader("preview.frag"),
        )
        self.shape_program = self.ctx.program(
            vertex_shader=load_shader("fullscreen.vert"),
            fragment_shader=load_shader("edge_wear.frag"),
        )
        self.background_program = self.ctx.program(
            vertex_shader=load_shader("fullscreen.vert"),
            fragment_shader=load_shader("background.frag"),
        )
        self.blit_program = self.ctx.program(
            vertex_shader=load_shader("fullscreen.vert"),
            fragment_shader=load_shader("blit.frag"),
        )

        self.shape_program["u_curvature"].value = 0
        self.shape_program["u_position"].value = 1
        self.preview_program["u_map"].value = 0
        self.blit_program["u_tex"].value = 0

        self.shape_vao = self.ctx.vertex_array(self.shape_program, [])
        self.background_vao = self.ctx.vertex_array(self.background_program, [])
        self.blit_vao = self.ctx.vertex_array(self.blit_program, [])

        self.mesh_vao: Optional[moderngl.VertexArray] = None
        self._mesh_buffers: list = []
        self._vao_token: tuple = ()

        self.tex_curvature: Optional[moderngl.Texture] = None
        self.tex_position: Optional[moderngl.Texture] = None
        self.tex_output: Optional[moderngl.Texture] = None
        self.output_fbo: Optional[moderngl.Framebuffer] = None
        self._texture_resolution = 0
        self._uploaded_maps_version = -1
        self._output_dirty = True

        # Bound whenever the active preview mode samples no map, so unit 0 is
        # never left pointing at nothing.
        self._blank_texture = self.ctx.texture((1, 1), 1, data=np.ones(1, dtype="f4").tobytes(), dtype="f4")

        self.thumbnail = self.ctx.texture((THUMBNAIL_SIZE, THUMBNAIL_SIZE), 4)
        self.thumbnail.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.thumbnail_fbo = self.ctx.framebuffer(color_attachments=[self.thumbnail])
        self.gui.register_texture(self.thumbnail)
        self._thumbnail_dirty = True

        self.camera = PanOrbitCamera(
            target=(0.0, 0.0, 0.0), radius=3.0, aspect_ratio=self.wnd.aspect_ratio,
            fov=50.0, near=0.01, far=1000.0,
        )
        self.camera.mouse_sensitivity = 3.0

        if self.initial_mesh:
            self.open_mesh(self.initial_mesh)

    # -- UI scaling -------------------------------------------------------

    @property
    def ui_pixel_scale(self) -> float:
        """Total multiplier from ImGui's base units to physical pixels.

        The renderer feeds ImGui physical pixels, so on a 2x Retina display a
        13px font would otherwise be physically tiny. Folding the pixel ratio in
        here keeps the panel the same apparent size on every display, with
        ``ui_scale`` on top as the user's own preference.
        """
        return float(self.wnd.pixel_ratio) * float(self.ui_scale)

    def apply_ui_scale(self) -> None:
        """Resize the font and every style metric to the current scale."""
        apply_style_scale(self._style_defaults, self.ui_pixel_scale)

    def set_ui_scale(self, value: float) -> None:
        self.ui_scale = float(np.clip(value, 0.6, 3.0))
        self.apply_ui_scale()
        _save_ui_scale(self.ui_scale)

    # -- mesh loading -----------------------------------------------------

    def open_mesh(self, path: str | Path) -> None:
        try:
            mesh, info = load_mesh(path, z_up=self.z_up_import)
        except MeshLoadError as exc:
            self.set_status(str(exc), error=True)
            return
        except Exception:
            self.set_status(traceback.format_exc(limit=3), error=True)
            return

        self.mesh = mesh
        self.mesh_info = info
        self.controller.set_mesh(mesh)

        # Show the raw geometry immediately; the unwrap pass replaces this VAO
        # with the seam-split version once it finishes.
        self._build_mesh_vao(
            np.asarray(mesh.vertices, dtype="f4"),
            np.asarray(mesh.vertex_normals, dtype="f4"),
            np.zeros((len(mesh.vertices), 2), dtype="f4"),
            np.asarray(mesh.faces, dtype="u4"),
        )
        self._vao_token = ("raw", self.controller.mesh_token)

        self.camera.frame(mesh.bounds.mean(axis=0), info.scale)
        if PREVIEW_MODES[self.preview_index].texture is not None:
            self.preview_index = SHADED_INDEX  # nothing baked yet

        note = f" ({'; '.join(info.notes)})" if info.notes else ""
        self.set_status(
            f"Loaded {Path(info.path).name} via {info.backend}: "
            f"{info.vertices:,} verts / {info.faces:,} tris{note}"
        )

    def set_status(self, message: str, *, error: bool = False) -> None:
        self.status = message
        self.status_is_error = error

    # -- GL resources -----------------------------------------------------

    def _build_mesh_vao(
        self, vertices: np.ndarray, normals: np.ndarray, uvs: np.ndarray, faces: np.ndarray
    ) -> None:
        self._release_mesh_vao()
        interleaved = np.hstack([vertices, normals, uvs]).astype("f4")
        vbo = self.ctx.buffer(interleaved.tobytes())
        ibo = self.ctx.buffer(np.ascontiguousarray(faces, dtype="u4").tobytes())
        self._mesh_buffers = [vbo, ibo]
        self.mesh_vao = self.ctx.vertex_array(
            self.preview_program,
            [(vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv")],
            index_buffer=ibo,
            index_element_size=4,
        )

    def _release_mesh_vao(self) -> None:
        if self.mesh_vao is not None:
            self.mesh_vao.release()
            self.mesh_vao = None
        for buffer in self._mesh_buffers:
            buffer.release()
        self._mesh_buffers = []

    def _ensure_textures(self, resolution: int) -> None:
        if self._texture_resolution == resolution:
            return
        for texture in (self.tex_curvature, self.tex_position, self.tex_output):
            if texture is not None:
                texture.release()
        if self.output_fbo is not None:
            self.output_fbo.release()

        size = (resolution, resolution)
        self.tex_curvature = self.ctx.texture(size, 1, dtype="f4")
        self.tex_position = self.ctx.texture(size, 3, dtype="f4")
        self.tex_output = self.ctx.texture(size, 1, dtype="f4")
        for texture in (self.tex_curvature, self.tex_position, self.tex_output):
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = False
            texture.repeat_y = False

        self.output_fbo = self.ctx.framebuffer(color_attachments=[self.tex_output])
        self._texture_resolution = resolution

    # -- per-frame sync ---------------------------------------------------

    def _sync_bake_outputs(self) -> None:
        controller = self.controller

        result = controller.unwrap_result
        if result is not None:
            token = ("unwrapped", controller.mesh_token, id(result))
            if token != self._vao_token:
                self._build_mesh_vao(
                    result.vertices.astype("f4"),
                    result.normals.astype("f4"),
                    result.uvs.astype("f4"),
                    result.faces,
                )
                self._vao_token = token

        if controller.maps_version != self._uploaded_maps_version:
            if controller.curvature_map is not None:
                resolution = controller.curvature_map.shape[0]
                self._ensure_textures(resolution)
                assert self.tex_curvature is not None and self.tex_position is not None
                self.tex_curvature.write(
                    np.ascontiguousarray(controller.curvature_map, dtype="f4").tobytes()
                )
                if controller.position_map is not None:
                    self.tex_position.write(
                        np.ascontiguousarray(controller.position_map, dtype="f4").tobytes()
                    )
                self._uploaded_maps_version = controller.maps_version
                self._output_dirty = True

                if PREVIEW_MODES[self.preview_index].texture is None:
                    self.preview_index = 0
                timings = ", ".join(
                    f"{stage} {seconds:.2f}s" for stage, seconds in controller.timings.items()
                )
                self.set_status(f"Bake complete - {timings}")

    def mark_output_dirty(self) -> None:
        self._output_dirty = True

    def _run_shaping(self) -> None:
        """Re-derive the output map from the baked fields. Runs every frame a
        slider moves; it is one full-screen pass over two textures."""
        if self.output_fbo is None or self.tex_curvature is None or self.tex_position is None:
            return
        self.tex_curvature.use(0)
        self.tex_position.use(1)
        self.shape_program["u_showCurvature"].value = 0
        for name, value in self.wear_params.as_uniforms().items():
            self.shape_program[name].value = value

        self.output_fbo.use()
        self.ctx.viewport = (0, 0, self._texture_resolution, self._texture_resolution)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        self.shape_vao.render(moderngl.TRIANGLES, vertices=3)
        self._output_dirty = False
        self._thumbnail_dirty = True

    def _current_texture(self) -> Optional[moderngl.Texture]:
        name = PREVIEW_MODES[self.preview_index].texture
        return {
            "curvature": self.tex_curvature,
            "output": self.tex_output,
        }.get(name or "")

    def _update_thumbnail(self) -> None:
        source = self._current_texture()
        if source is None:
            return
        source.use(0)
        self.thumbnail_fbo.use()
        self.ctx.viewport = (0, 0, THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.blit_vao.render(moderngl.TRIANGLES, vertices=3)
        self._thumbnail_dirty = False

    # -- drawing ----------------------------------------------------------

    def _draw_scene(self) -> None:
        # wnd.use() rather than ctx.screen.use(): the headless backend renders
        # into its own framebuffer, and the self-test relies on that.
        self.wnd.use()
        self.ctx.viewport = (0, 0, *self.wnd.buffer_size)
        # Clear colour and depth together: moderngl's clear() always touches
        # both, so this has to happen before the background is drawn.
        self.ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

        # Depth testing off also disables depth writes, so the backdrop cannot
        # occlude the mesh drawn after it.
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        self.background_program["u_top"].value = (0.16, 0.17, 0.21)
        self.background_program["u_bottom"].value = (0.05, 0.05, 0.07)
        self.background_vao.render(moderngl.TRIANGLES, vertices=3)

        if self.mesh_vao is None:
            return

        self.ctx.enable(moderngl.DEPTH_TEST)

        mode = PREVIEW_MODES[self.preview_index]
        texture = self._current_texture()
        if mode.texture is not None and texture is None:
            mode = PREVIEW_MODES[SHADED_INDEX]  # nothing baked yet
        (texture or self._blank_texture).use(0)

        view = self.camera.matrix
        mvp = self.camera.projection.matrix * view

        self.preview_program["u_mvp"].write(_mat_bytes(mvp))
        self.preview_program["u_mode"].value = mode.shader_mode
        self.preview_program["u_lighting"].value = 1.0 if self.lighting else 0.0
        self.preview_program["u_checkerScale"].value = self.checker_scale

        eye = self.camera.eye
        light = glm.normalize(eye - self.camera.target + glm.vec3(0.35, 0.6, 0.25) * self.camera.radius)
        self.preview_program["u_lightDir"].value = (light.x, light.y, light.z)

        self.ctx.wireframe = self.wireframe
        self.mesh_vao.render(moderngl.TRIANGLES)
        self.ctx.wireframe = False

    # -- moderngl-window hooks --------------------------------------------

    def on_render(self, time: float, frame_time: float) -> None:
        self.controller.pump()
        self._sync_bake_outputs()

        if self._output_dirty:
            self._run_shaping()
        if self._thumbnail_dirty and self.show_map_inspector:
            self._update_thumbnail()

        self._draw_scene()

        imgui.new_frame()
        draw_panel(self)
        imgui.render()
        self.gui.render(imgui.get_draw_data())

    def on_resize(self, width: int, height: int) -> None:
        self.gui.resize(width, height)
        # Dragging the window to a display with a different pixel ratio changes
        # the scale, so re-apply rather than assuming it is fixed.
        self.apply_ui_scale()
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)

    def on_files_dropped_event(self, x: int, y: int, paths: list[str]) -> None:
        for raw in paths:
            candidate = Path(raw)
            if candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                self.open_mesh(candidate)
                return
        self.set_status(f"Unsupported file type: {Path(paths[0]).suffix}", error=True)

    def on_key_event(self, key, action, modifiers) -> None:
        self.gui.key_event(key, action, modifiers)
        if imgui.get_io().want_capture_keyboard:
            return

        keys = self.wnd.keys
        if action != keys.ACTION_PRESS:
            return

        shortcuts = {
            keys.NUMBER_1: 0, keys.NUMBER_2: 1, keys.NUMBER_3: 2,
            keys.NUMBER_4: 3, keys.NUMBER_5: 4, keys.NUMBER_6: 5,
        }
        if key in shortcuts:
            self.preview_index = shortcuts[key]
            self._thumbnail_dirty = True
        elif key == keys.B:
            self.request_bake()
        elif key == keys.E:
            self.export()
        elif key == keys.F and self.mesh is not None:
            self.camera.frame(self.mesh.bounds.mean(axis=0), self.mesh_info.scale)
        elif key == keys.W:
            self.wireframe = not self.wireframe
        elif key == keys.L:
            self.lighting = not self.lighting
        elif key in (getattr(keys, "EQUAL", None), getattr(keys, "PLUS", None)):
            self.set_ui_scale(self.ui_scale + 0.1)
        elif key == getattr(keys, "MINUS", None):
            self.set_ui_scale(self.ui_scale - 0.1)

    def on_unicode_char_entered(self, char: str) -> None:
        self.gui.unicode_char_entered(char)

    def on_mouse_position_event(self, x: int, y: int, dx: int, dy: int) -> None:
        self.gui.mouse_position_event(x, y, dx, dy)

    def on_mouse_press_event(self, x: int, y: int, button: int) -> None:
        self.gui.mouse_press_event(x, y, button)

    def on_mouse_release_event(self, x: int, y: int, button: int) -> None:
        self.gui.mouse_release_event(x, y, button)

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
        self.gui.mouse_drag_event(x, y, dx, dy)
        if imgui.get_io().want_capture_mouse:
            return

        states = self.wnd.mouse_states
        panning = states.middle or (states.left and self.wnd.modifiers.shift)
        if panning:
            self.camera.pan(dx, dy)
        elif states.left:
            self.camera.rot_state(-dx, -dy)
        elif states.right:
            self.camera.zoom_state(-dy * 0.25)

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
        self.gui.mouse_scroll_event(x_offset, y_offset)
        if imgui.get_io().want_capture_mouse:
            return
        self.camera.zoom_state(y_offset)

    def on_close(self) -> None:
        self.controller.release()

    # -- actions used by the panel ----------------------------------------

    def request_bake(self) -> None:
        if self.mesh is None:
            self.set_status("Load a mesh first", error=True)
            return
        if self.controller.running:
            return
        if not self.controller.request_bake():
            self.set_status("Everything is already up to date")
            return
        self.set_status("Baking...")

    def export(self) -> None:
        self._export(include_obj=False)

    def export_obj(self) -> None:
        self._export(include_obj=True)

    def _export(self, *, include_obj: bool) -> None:
        controller = self.controller
        if controller.curvature_map is None:
            self.set_status("Nothing baked yet - hit Bake first", error=True)
            return
        if include_obj and controller.unwrap_result is None:
            self.set_status("No unwrapped mesh available - hit Bake first", error=True)
            return
        if self.output_fbo is None:
            self.set_status("No render target for the edge wear map", error=True)
            return

        if self._output_dirty:
            self._run_shaping()

        resolution = self._texture_resolution
        # Read the shaped map back off the GPU rather than recomputing it on the
        # CPU, so the PNG is exactly the pixels shown in the viewport.
        shaped = np.frombuffer(
            self.output_fbo.read(components=1, dtype="f4"), dtype="f4"
        ).reshape(resolution, resolution)

        stem = Path(self.mesh_info.path).stem if self.mesh_info else "mesh"
        target = Path(self.export_dir) / stem
        try:
            if include_obj:
                written = export_textured_obj(
                    target,
                    stem,
                    controller.unwrap_result,
                    shaped,
                    controller.curvature_map,
                    bits=self.export_bits,
                )
            else:
                written = export_maps(
                    target,
                    shaped,
                    controller.curvature_map,
                    bits=self.export_bits,
                )
        except Exception:
            self.set_status(traceback.format_exc(limit=3), error=True)
            return

        kind = "textured OBJ package" if include_obj else f"{len(written)} maps"
        self.set_status(f"Wrote {kind} to {target}")
