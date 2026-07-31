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

from core.decal import DecalImage, DecalLoadError, load_decal
from core.export import export_maps
from core.layers import Slot, describe, mask_at, new_texture, slot_at
from core.mesh_io import SUPPORTED_SUFFIXES, MeshLoadError, load_mesh
from core.params import BevelParams, DecalParams, EdgeWearParams, MeshInfo
from core.picking import pick_uv, screen_ray
from core.pipeline import BakeController
from core.uv_unwrap import source_uvs
from render.composite import LayerCompositor
from render.imgui_renderer import ImGuiRenderer
from render.shaders import load_shader
from render.trackpad import install_pinch_zoom
from ui import gizmo, panel
from ui.panel import draw_panel

#: The Shaded view, which the app falls back to when a mode's map goes away.
#: It draws the mask tree, and plain grey until there is one -- see FLAT_MODE.
SHADED_INDEX = 0


@dataclass(frozen=True)
class PreviewMode:
    label: str
    shader_mode: int
    texture: Optional[str]  # which bake target to sample, if any
    needs_bake: bool = True
    """False for a map the app can produce without any bake -- the decal normals
    are placed in UV space and need no geometry pass behind them."""


#: The two products, and so the two things to look at. Shaded is the mask tree:
#: whatever the masks resolve to, lit -- a fresh tree is plain black and white,
#: so it shows the mask itself until colours are put under it. Normals is the
#: decal normal map, which is the other thing the app exports.
PREVIEW_MODES = (
    PreviewMode("Shaded", 4, "composite"),
    PreviewMode("Normals", 4, "normal", needs_bake=False),
)

#: Drawn in place of any mode whose map does not exist yet -- an unbaked mesh
#: under the Shaded view, mostly. Plain grey, so the geometry still reads.
FLAT_MODE = PreviewMode("Shaded", 3, None)


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


def _load_prefs() -> dict:
    """Read the saved preferences. Anything unreadable falls back to defaults."""
    try:
        stored = json.loads((_settings_dir() / "prefs.json").read_text())
        return stored if isinstance(stored, dict) else {}
    except Exception:
        return {}


def _save_prefs(values: dict) -> None:
    try:
        (_settings_dir() / "prefs.json").write_text(json.dumps(values, indent=2))
    except OSError:
        pass  # a read-only home directory is not worth failing over


@dataclass
class NavigationPrefs:
    """How fast the viewport responds, and which way round.

    Every speed is a plain multiplier on a base rate, so 1.0 is the reference
    behaviour and the numbers stay meaningful: orbit's base is Blender's own
    0.4 degrees per pixel.
    """

    orbit_speed: float = 1.0
    pan_speed: float = 1.0
    zoom_speed: float = 1.0
    scroll_speed: float = 1.0
    """Applies on top of orbit/pan/zoom, but only for scroll and trackpad
    gestures -- a trackpad's deltas are nothing like a wheel's, and one is
    usually far too fast when the other feels right."""

    smoothing: float = 1.0
    """How closely scroll output follows the *rate* of the input, 0 to 1.

    This exists because of the resolution macOS reports trackpad scrolling at.
    A scroll event carries its delta in ``NSEvent``'s fixed-point field --
    pyglet's ``deltaY`` (``pyglet/window/cocoa/pyglet_view.py``,
    ``getMouseDelta``) -- in units of a tenth of a point, one step per pixel the
    fingers travel. Move slowly and a pixel takes several frames to cross, so
    what arrives is a run of zeros punctuated by a step. Nothing downstream can
    invent the motion in between; all it can do is not present a step as though
    it happened in one frame.

    At 0 each step is applied whole, the moment it lands. At 1
    :meth:`MeshMapApp._drain_scroll` moves at exactly the speed the steps are
    arriving, so a sparse run of them leaves as steady motion; in between it
    runs proportionally ahead of that speed.

    Rate-following, not damping. A fast gesture has a fast rate and drains the
    frame it arrives, the delay at a crawl is about one step -- and it shrinks
    to nothing as the gesture speeds up, because the steps come closer together
    -- and the total travel is identical at every setting.

    Never applied to a mouse drag, which reports genuine per-pixel floats.
    """

    invert_orbit_x: bool = False
    invert_orbit_y: bool = False

    FIELDS = ("orbit_speed", "pan_speed", "zoom_speed", "scroll_speed",
              "smoothing", "invert_orbit_x", "invert_orbit_y")

    def as_dict(self) -> dict:
        return {
            name: (round(value, 4) if isinstance(value, float) else value)
            for name, value in ((field, getattr(self, field)) for field in self.FIELDS)
        }

    @classmethod
    def from_dict(cls, stored: dict) -> "NavigationPrefs":
        prefs = cls()
        for field in cls.FIELDS:
            if field not in stored:
                continue
            try:
                current = getattr(prefs, field)
                if isinstance(current, bool):
                    setattr(prefs, field, bool(stored[field]))
                elif field == "smoothing":
                    setattr(prefs, field, float(np.clip(float(stored[field]), 0.0, 1.0)))
                else:
                    setattr(prefs, field, float(np.clip(float(stored[field]), 0.05, 5.0)))
            except (TypeError, ValueError):
                pass  # keep the default for anything malformed
        return prefs


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


#: Degrees of orbit per pixel of drag. Blender's own default, from
#: ``DNA_userdef_types.h``: ``view_rotate_sensitivity_turntable = DEG2RAD(0.4)``.
_ORBIT_PER_PIXEL = 0.4
#: Pixels of equivalent drag per unit of scroll. A trackpad reports a gesture as
#: many small deltas, and macOS adds a momentum tail, so this converts to the
#: same units the drag path uses rather than being a second sensitivity.
_PIXELS_PER_SCROLL = 12.0
#: Largest single event honoured, as a teleport guard only -- see
#: ``_TELEPORT_NOTE``. Both sit far beyond anything a hand produces, because a
#: threshold tight enough to shape normal input is felt as a speed ceiling.
_MAX_SCROLL_STEP = 120.0
_MAX_DRAG_STEP = 2000.0
#: Scroll units per unit of pinch magnification. NSEvent reports a pinch as an
#: incremental change in scale, so a whole-hand spread totals around 1.0; at 8
#: units that is roughly a 2.5x change in viewing distance, which puts the
#: gesture in the same place it lands in Blender.
_SCROLL_PER_MAGNIFICATION = 8.0
#: How quickly the arrival-rate estimate follows a change in gesture speed,
#: per event. Rising is followed harder than falling: lagging an acceleration
#: is felt as the view dragging behind the hand, while chasing every wobble
#: downward would put back the unevenness this is here to remove.
_RATE_BLEND_UP = 0.3
_RATE_BLEND_DOWN = 0.2
#: Longest the buffer may hold input, in frames. The backstop for a rate
#: estimate that has drifted low; at 60 Hz this is about an eighth of a second.
_DRAIN_LIMIT_FRAMES = 8.0
#: Frames of silence that end a gesture, after which the rate estimate is
#: forgotten. Comfortably longer than the gap between steps in even a crawl, so
#: it cannot fire in the middle of one.
_GESTURE_END_FRAMES = 15
#: Teleport guard for a single pinch event, in magnification -- see
#: ``_TELEPORT_NOTE``. Real events are hundredths; this only catches a gesture
#: interrupted by the window losing and regaining the pointer.
_MAX_PINCH_STEP = 1.0

_TELEPORT_NOTE = """
These guards exist for one case: the pointer leaving the window and re-entering
while a button is held, which arrives as a single delta spanning the whole gap.

They are deliberately huge, and they *clamp* rather than discard. Earlier values
of 6.0 and 250 px were small enough to catch real gestures -- a quick trackpad
flick exceeds 6.0, and a fast drag on a frame where rendering hitched exceeds
250 px. Capping the scroll put a ceiling on flick speed, and discarding the drag
froze the view for that frame. Both read as the view sticking or snapping.
"""
#: World up. Z, matching Blender -- imports are rotated into this convention by
#: ``_axis_fix`` in :mod:`core.mesh_io`, so the axis names here mean what they
#: mean in Blender.
_WORLD_UP = (0.0, 0.0, 1.0)
#: The default view, as the direction the camera *looks* along. Negated, it puts
#: the eye front-right-above at (+X, -Y, +Z) -- Blender's startup view, 30
#: degrees above the horizon.
_HOME_FORWARD = (-0.6124, 0.6124, -0.5)


class PanOrbitCamera(OrbitCamera):
    """A turntable camera, matching Blender's orbit, plus pan and ortho views.

    Three departures from the stock ``OrbitCamera``:

    **Turntable orbit.** Orientation is a quaternion, and :meth:`orbit` is a
    port of the non-trackball branch of Blender's ``viewrotate_apply``
    (``view3d_navigate_view_rotate.cc``): horizontal motion spins about world
    up, vertical motion pitches about the screen horizon. Because yaw is always
    about the *same world axis*, the horizon stays level no matter how you get
    there -- which is what a trackball cannot promise, and what makes a
    trackball feel like it is fighting you. The stock ``rot_state`` clamps its
    polar angle to ``[-175, -5]`` degrees instead, so it can never look straight
    down; nothing here clamps, and Blender's horizon blend (below) is what keeps
    the poles well-behaved without one.

    **Orthographic views.** Aligning to an axis switches to ortho, mirroring
    Blender's Auto Perspective; orbiting away switches back.

    **Scale-aware zoom.** Multiplicative and clamped relative to the subject.
    The stock one is additive with a floor of 1.0, useless for a 5 cm bolt or a
    40 m building.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subject_scale = 1.0
        self.pan_sensitivity = 1.0
        self.orthographic = False
        self.orientation = self._home_orientation()

    @staticmethod
    def _home_orientation() -> glm.quat:
        return glm.quatLookAt(
            glm.normalize(glm.vec3(*_HOME_FORWARD)), glm.vec3(*_WORLD_UP)
        )

    def _axes(self) -> tuple[glm.vec3, glm.vec3, glm.vec3]:
        """The camera's own right / up / forward, in world space."""
        rotation = glm.mat3_cast(self.orientation)
        return (
            rotation * glm.vec3(1.0, 0.0, 0.0),
            rotation * glm.vec3(0.0, 1.0, 0.0),
            rotation * glm.vec3(0.0, 0.0, -1.0),
        )

    @property
    def eye(self) -> glm.vec3:
        return self.target - self._axes()[2] * self.radius

    @property
    def matrix(self) -> glm.mat4:
        _, up, forward = self._axes()
        position = self.target - forward * self.radius
        # The base class tracks this for anything that asks the camera where it
        # is; keep it in step even though the view matrix is built here.
        self.set_position(*position)
        return glm.lookAt(position, self.target, up)

    @property
    def projection_matrix(self) -> glm.mat4:
        """Perspective, or an ortho box framing the same subject."""
        if not self.orthographic:
            return self.projection.matrix
        # Match the height the perspective view covers at the target, so
        # toggling projection does not appear to change the zoom level.
        half_height = self.radius * float(
            np.tan(np.radians(self.projection.fov * 0.5))
        )
        half_width = half_height * self.projection.aspect_ratio
        depth = self.radius + self.subject_scale * 1.5
        # An ortho box distributes depth linearly, so it can start behind the
        # target without costing any precision.
        return glm.ortho(-half_width, half_width, -half_height, half_height,
                         -depth, depth)

    def _pitch_axis(self) -> glm.vec3:
        """The screen horizon to pitch about, per Blender's gimbal-lock blend.

        Straight from ``viewrotate_apply``. Two candidate axes:

        ``cross(world_up, back)`` is the true turntable horizon -- horizontal in
        world terms -- but it collapses to nothing at the poles, where ``back``
        and ``world_up`` are parallel.

        The camera's own right axis never collapses, but on its own it lets the
        turntable gimbal-lock: with the view rolled 90 degrees, pitching and
        yawing become the same motion and there is no way out.

        Blending between them by how close the view is to a pole gets both
        properties. ``fac`` reaches 1 at either pole -- where the camera's right
        axis is the only usable answer -- and 0 at the equator, where the world
        horizon is exactly right. Squaring biases the blend toward the world
        horizon, so ordinary orbiting is a true turntable and the correction
        only appears near the poles.
        """
        right, _, back = self._axes()
        world_up = glm.vec3(*_WORLD_UP)

        horizon = glm.cross(world_up, back)
        if glm.length(horizon) < 1e-6:
            return right
        if glm.dot(horizon, right) < 0.0:
            horizon = -horizon

        fraction = float(np.arccos(np.clip(glm.dot(world_up, back), -1.0, 1.0)) / np.pi)
        blend = abs(fraction - 0.5) * 2.0
        blend *= blend
        return glm.normalize(glm.mix(glm.normalize(horizon), right, blend))

    def orbit(self, dx: float, dy: float) -> None:
        """Turntable-orbit by ``dx``/``dy`` degrees.

        ``dx`` spins about world up and ``dy`` pitches about the screen horizon.
        Both are world-space axes, so they pre-multiply the camera's
        orientation.
        """
        _, up, _ = self._axes()
        # Blender's `vod->reverse`: once the view is upside down, dragging right
        # should still swing the subject the same way it looks like it should.
        reverse = -1.0 if glm.dot(up, glm.vec3(*_WORLD_UP)) < 0.0 else 1.0

        yaw = glm.angleAxis(glm.radians(float(dx) * reverse), glm.vec3(*_WORLD_UP))
        pitch = glm.angleAxis(glm.radians(float(dy)), self._pitch_axis())
        self.orientation = glm.normalize(yaw * pitch * self.orientation)

        # Any orbit means the view is no longer axis-aligned, so Blender's Auto
        # Perspective hands perspective back.
        self.orthographic = False

    def rot_state(self, dx: float, dy: float) -> None:
        """Orbit from a mouse drag, in pixels."""
        scale = self.mouse_sensitivity * _ORBIT_PER_PIXEL
        self.orbit(dx * scale, dy * scale)

    def align_to_axis(self, axis: tuple[float, float, float]) -> None:
        """Look down ``axis`` in orthographic projection, as Blender's gizmo does.

        ``axis`` points from the target toward the new eye, so ``(0, 0, 1)`` is
        the top view.
        """
        direction = glm.normalize(glm.vec3(*axis))
        world_up = glm.vec3(*_WORLD_UP)
        if abs(glm.dot(direction, world_up)) > 0.999:
            # Straight up or down leaves roll unconstrained. Blender's top view
            # puts +Y up the screen, and the bottom view mirrors it.
            along = 1.0 if glm.dot(direction, world_up) > 0.0 else -1.0
            hint = glm.vec3(0.0, 1.0, 0.0) * along
        else:
            hint = world_up
        self.orientation = glm.quatLookAt(-direction, hint)
        self.orthographic = True

    def pan(self, dx: float, dy: float) -> None:
        right, up, _ = self._axes()
        step = self.radius * 0.0018 * self.pan_sensitivity
        self.target = self.target + (-right * float(dx) + up * float(dy)) * step

    def zoom_state(self, y_offset: float) -> None:
        self.radius *= float(np.exp(-y_offset * 0.12 * self._zoom_sensitivity))
        self.radius = float(np.clip(self.radius, self.subject_scale * 0.02, self.subject_scale * 60.0))
        self._update_clip()

    def _update_clip(self) -> None:
        """Fit the depth range to where the subject actually is.

        A perspective depth buffer spends its precision near the near plane, so
        a near plane much closer than it needs to be starves the subject of
        resolution and coplanar surfaces start flickering against each other as
        the view moves. Tracking the eye distance keeps ``far / near`` in the
        low hundreds at every zoom level instead of the 40,000 a fixed pair gave.
        """
        reach = self.subject_scale * 0.75
        near = max(self.radius - reach, self.radius * 0.01, 1e-6)
        self.projection.update(near=near, far=near + reach * 4.0 + self.radius)

    def frame(self, center, scale: float) -> None:
        """Point the camera at a subject of the given size, upright again."""
        self.subject_scale = max(float(scale), 1e-6)
        self.target = glm.vec3(*[float(v) for v in center])
        self.radius = self.subject_scale * 1.5
        self.orientation = self._home_orientation()
        self.orthographic = False
        self._update_clip()


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
    #: False bakes into the mesh's own UVs; True lets xatlas invent an atlas.
    initial_auto_unwrap: bool = False
    initial_bevel: Optional[BevelParams] = None
    initial_ui_scale: float = 1.35
    #: True when --ui-scale was passed, so it overrides the saved preference.
    initial_ui_scale_explicit: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # moderngl-window quits on its exit_key, which defaults to Escape and is
        # checked before the key ever reaches on_key_event. Escape means "cancel
        # what I am doing" here -- cancelling a decal placement -- and an app
        # holding a bake and a set of parameters has no business closing on a
        # stray keypress. Quitting stays where every other app puts it: the
        # window's close button, or Cmd-Q.
        self.wnd.exit_key = None

        imgui.create_context()
        imgui.get_io().set_ini_filename(str(_settings_dir() / "layout.ini"))
        self.gui = ImGuiRenderer(self.wnd)

        self.controller = BakeController(self.ctx)
        self.controller.bake_params.resolution = int(self.initial_resolution)
        self.controller.unwrap_params.use_source_uvs = not self.initial_auto_unwrap
        if self.initial_bevel is not None:
            self.controller.bevel_params = self.initial_bevel
        self.wear_params = EdgeWearParams()
        #: Every texture made this session. A texture is a colour, or a mask
        #: deciding between two more of them, recursively. One is active at a
        #: time -- that is the one shown, edited and exported -- and the rest
        #: are kept so a variant can be gone back to.
        self.textures: list[Slot] = []
        self.texture_index: int = -1
        #: The slot the Texture tab is editing, within the active texture: a
        #: colour or a mask, either way.
        self.texture_path: tuple[str, ...] = ()
        self._texture_dirty = True
        #: Counts textures made this session, so each gets its own name.
        self._texture_serial = 0
        #: Which row the tree is renaming in place, if any.
        self.renaming_path: Optional[tuple[str, ...]] = None
        #: What has been typed into the texture picker's search box.
        self.texture_filter: str = ""
        self.decal_params = DecalParams()
        self.decal_image: Optional[DecalImage] = None
        #: True while the decal is following the cursor, waiting to be dropped.
        self.decal_placing = False
        #: Where it was before that started, to put back if the user cancels.
        self._decal_anchor: Optional[tuple[float, float]] = None
        self.mesh_info: Optional[MeshInfo] = None
        self.mesh: Optional[trimesh.Trimesh] = None

        self.preview_index = 0
        self.lighting = True
        self.checker_scale = 24.0
        self.wireframe = False
        self.source_z_up = bool(self.initial_z_up)
        self.export_dir = str(Path.cwd() / "output")
        self.export_bits = 8
        self.show_gizmo = True
        self.status = "Drop an FBX onto the window, or click 'Open mesh...'."
        self.status_is_error = False
        self.file_dialog = None
        self.folder_dialog = None
        self.decal_dialog = None

        self._mouse: tuple[float, float] = (0.0, 0.0)
        #: What the in-flight drag belongs to: "camera", "gizmo", "ui" or None.
        #: Latched on press so a gesture cannot change hands halfway through.
        self._drag_owner: Optional[str] = None

        #: Scroll input waiting to be applied, drained a share per frame by
        #: :meth:`_drain_scroll`.
        self._pending_orbit = [0.0, 0.0]
        self._pending_pan = [0.0, 0.0]
        self._pending_zoom = 0.0
        #: Input that landed since the last frame, and the running average of
        #: it, which is what :meth:`_drain_scroll` paces the camera by.
        self._scroll_arrived = 0.0
        self._scroll_rate = 0.0
        self._scroll_idle = 0
        self._scroll_gaps = 0
        #: Last known keyboard modifiers. pyglet clears them on scroll events,
        #: so scroll reads this rather than ``wnd.modifiers``.
        self._modifiers = self.wnd.modifiers

        self._prefs = _load_prefs()
        self.navigation = NavigationPrefs.from_dict(self._prefs.get("navigation", {}))

        stored_scale = self._prefs.get("ui_scale")
        self.ui_scale = self.initial_ui_scale
        if not self.initial_ui_scale_explicit and stored_scale is not None:
            try:
                self.ui_scale = float(np.clip(float(stored_scale), 0.6, 3.0))
            except (TypeError, ValueError):
                pass
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
        self.decal_program = self.ctx.program(
            vertex_shader=load_shader("fullscreen.vert"),
            fragment_shader=load_shader("decal.frag"),
        )

        self.shape_program["u_curvature"].value = 0
        self.shape_program["u_position"].value = 1
        self.preview_program["u_map"].value = 0
        self.preview_program["u_normalMap"].value = 1
        self.decal_program["u_decal"].value = 0

        self.compositor = LayerCompositor(self.ctx)
        self._registered_thumbnails: set = set()

        self.shape_vao = self.ctx.vertex_array(self.shape_program, [])
        self.background_vao = self.ctx.vertex_array(self.background_program, [])
        self.decal_vao = self.ctx.vertex_array(self.decal_program, [])

        self.mesh_vao: Optional[moderngl.VertexArray] = None
        self._mesh_buffers: list = []
        self._vao_token: tuple = ()
        #: The geometry the VAO was built from, kept for cursor picking: the
        #: ray has to hit exactly what is on screen, seams and all.
        self._pick_geometry: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None

        self.tex_curvature: Optional[moderngl.Texture] = None
        self.tex_position: Optional[moderngl.Texture] = None
        self.tex_output: Optional[moderngl.Texture] = None
        self.output_fbo: Optional[moderngl.Framebuffer] = None
        self._texture_resolution = 0
        self._uploaded_maps_version = -1
        self._output_dirty = True

        # The decal chain: the imported image, and the atlas-sized normal map it
        # is composited into. Independent of the bake -- a decal is placed in UV
        # space, so it needs no geometry pass behind it.
        self.tex_decal: Optional[moderngl.Texture] = None
        self.tex_normal: Optional[moderngl.Texture] = None
        self.normal_fbo: Optional[moderngl.Framebuffer] = None
        self._normal_resolution = 0
        self._normal_dirty = True

        # Bound whenever the active preview mode samples no map, so unit 0 is
        # never left pointing at nothing.
        self._blank_texture = self.ctx.texture((1, 1), 1, data=np.ones(1, dtype="f4").tobytes(), dtype="f4")
        # Flat tangent-space normal, for when there is no decal: (0.5, 0.5, 1)
        # decodes to straight out of the surface, so the preview shader can bind
        # this unconditionally and light the mesh unchanged.
        self._flat_normal = self.ctx.texture(
            (1, 1), 3, data=np.array([0.5, 0.5, 1.0], dtype="f4").tobytes(), dtype="f4"
        )

        self.camera = PanOrbitCamera(
            target=(0.0, 0.0, 0.0), radius=3.0, aspect_ratio=self.wnd.aspect_ratio,
            fov=50.0, near=0.01, far=1000.0,
        )
        # 1.0 leaves _ORBIT_PER_PIXEL alone, which is Blender's own rate. This
        # used to be 3.0, left over from the stock OrbitCamera's rot_state, which
        # divided by 10 internally; once that was replaced the 3.0 was no longer
        # being cancelled and the viewport orbited at three times Blender's speed.
        self.camera.mouse_sensitivity = 1.0
        self._apply_navigation()

        #: True once the macOS pinch gesture is wired up; False everywhere else,
        #: where a pinch never reaches the process at all.
        self.pinch_zoom = install_pinch_zoom(self.wnd, self.on_pinch_zoom)

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
        self.save_prefs()

    def apply_navigation(self) -> None:
        """Push the navigation preferences onto the camera, and remember them."""
        self._apply_navigation()
        self.save_prefs()

    def _apply_navigation(self) -> None:
        self.camera.mouse_sensitivity = self.navigation.orbit_speed
        self.camera.pan_sensitivity = self.navigation.pan_speed
        self.camera.zoom_sensitivity = self.navigation.zoom_speed

    def save_prefs(self) -> None:
        """Persist the interface and navigation settings."""
        self._prefs["ui_scale"] = round(self.ui_scale, 3)
        self._prefs["navigation"] = self.navigation.as_dict()
        _save_prefs(self._prefs)

    # -- mesh loading -----------------------------------------------------

    def open_mesh(self, path: str | Path) -> None:
        try:
            mesh, info = load_mesh(path, source_z_up=self.source_z_up)
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
        # with the seam-split version once it finishes. The mesh's own UVs come
        # along if it has any, so a decal placed on the mesh reads correctly
        # before any bake -- it does not wait on the geometry pass.
        uvs = source_uvs(mesh)
        self._build_mesh_vao(
            np.asarray(mesh.vertices, dtype="f4"),
            np.asarray(mesh.vertex_normals, dtype="f4"),
            np.zeros((len(mesh.vertices), 2), dtype="f4") if uvs is None
            else uvs.astype("f4"),
            np.asarray(mesh.faces, dtype="u4"),
        )
        self._vao_token = ("raw", self.controller.mesh_token)

        self.camera.frame(mesh.bounds.mean(axis=0), info.scale)
        mode = PREVIEW_MODES[self.preview_index]
        if mode.texture is not None and mode.needs_bake:
            # Nothing baked for the new mesh yet; Shaded copes on its own,
            # drawing plain grey until the tree has something to stand on.
            self.preview_index = SHADED_INDEX

        note = f" ({'; '.join(info.notes)})" if info.notes else ""
        self.set_status(
            f"Loaded {Path(info.path).name} via {info.backend}: "
            f"{info.vertices:,} verts / {info.faces:,} tris{note}"
        )

    def set_status(self, message: str, *, error: bool = False) -> None:
        self.status = message
        self.status_is_error = error

    # -- decals -----------------------------------------------------------

    def open_decal(self, path: str | Path) -> None:
        """Import a normal map to stamp into the atlas.

        A grayscale image is read as a height map and converted, since that is
        the only reading of it that describes a surface -- see
        :func:`core.decal.load_decal`.
        """
        try:
            image = load_decal(path)
        except DecalLoadError as exc:
            self.set_status(str(exc), error=True)
            return
        except Exception:
            self.set_status(traceback.format_exc(limit=3), error=True)
            return

        self.decal_image = image
        self.decal_params.path = image.path
        self.decal_params.enabled = True
        self._upload_decal()
        self.mark_normal_dirty()

        width, height = image.size
        source = "height map, converted" if image.from_height else "normal map"
        self.set_status(
            f"Decal: {Path(image.path).name} ({width}x{height} {source}). "
            f"Exports as normal.png."
        )

    def clear_decal(self) -> None:
        """Drop the decal and put the normal map back to flat."""
        self.decal_placing = False
        self._decal_anchor = None
        self.decal_image = None
        self.decal_params = DecalParams(enabled=False)
        self._release_decal_texture()
        self.mark_normal_dirty()
        if PREVIEW_MODES[self.preview_index].texture == "normal":
            self.preview_index = SHADED_INDEX
        self.set_status("Decal cleared")

    def mark_normal_dirty(self) -> None:
        self._normal_dirty = True

    # -- placing a decal by pointing at the mesh --------------------------

    def begin_decal_placement(self) -> None:
        """Pick the decal up: it now follows the cursor until a click drops it.

        The current placement is remembered, so cancelling puts it back rather
        than leaving it wherever the cursor last wandered.
        """
        if not self.decal_params.loaded():
            self.set_status("Import a normal map first", error=True)
            return
        if self.mesh is None:
            self.set_status("Load a mesh to place a decal on", error=True)
            return

        self.decal_params.enabled = True
        self.decal_placing = True
        self._decal_anchor = (self.decal_params.center_u, self.decal_params.center_v)
        self.set_status("Placing decal - click on the mesh to drop it, Esc to cancel")
        self.follow_cursor_with_decal()

    def end_decal_placement(self, *, keep: bool) -> None:
        """Drop the decal where it is, or put it back where it came from."""
        if not self.decal_placing:
            return
        self.decal_placing = False

        if not keep and self._decal_anchor is not None:
            self.decal_params.center_u, self.decal_params.center_v = self._decal_anchor
            self.mark_normal_dirty()
            self.set_status("Placement cancelled")
        else:
            self.set_status(
                f"Decal placed at UV "
                f"{self.decal_params.center_u:.3f}, {self.decal_params.center_v:.3f}"
            )
        self._decal_anchor = None

    def follow_cursor_with_decal(self) -> bool:
        """Move the decal to the surface under the cursor. False if it missed.

        A miss leaves the decal where it was rather than snapping it somewhere
        arbitrary -- dragging off the silhouette and back should not lose the
        placement you were lining up.
        """
        uv = self.surface_uv_at(self._mouse)
        if uv is None:
            return False
        self.decal_params.center_u, self.decal_params.center_v = uv
        self.mark_normal_dirty()
        return True

    def surface_uv_at(self, mouse: tuple[float, float]) -> Optional[tuple[float, float]]:
        """The mesh's UV under a cursor position, or None if it points at sky."""
        if self._pick_geometry is None:
            return None

        # Mouse events arrive in the window's logical units; the framebuffer may
        # be larger on a HiDPI display. Scale into buffer pixels the same way
        # the ImGui renderer does, then into the 3D view's own rect -- the
        # projection belongs to that rect, so a ray built off the whole window
        # would point somewhere the model was never drawn.
        ratio = float(self.wnd.pixel_ratio)
        rect_x, _, rect_width, rect_height = self.viewport_rect
        navbar = int(panel.NAVBAR_HEIGHT * self.ui_pixel_scale)

        # Normalised device coordinates: y runs up the screen, the cursor's down.
        ndc_x = ((mouse[0] * ratio) - rect_x) / rect_width * 2.0 - 1.0
        ndc_y = 1.0 - ((mouse[1] * ratio) - navbar) / rect_height * 2.0

        mvp = self.camera.projection_matrix * self.camera.matrix
        inverse = np.array(glm.inverse(mvp).to_list(), dtype=np.float64).T
        origin, direction = screen_ray(inverse, ndc_x, ndc_y)

        vertices, faces, uvs = self._pick_geometry
        return pick_uv(origin, direction, vertices, faces, uvs)

    def _release_decal_texture(self) -> None:
        if self.tex_decal is None:
            return
        # The panel draws it, so ImGui knows about it too and has to be told.
        try:
            self.gui.remove_texture(self.tex_decal)
        except KeyError:
            pass
        self.tex_decal.release()
        self.tex_decal = None

    def _upload_decal(self) -> None:
        self._release_decal_texture()
        if self.decal_image is None:
            return

        image = self.decal_image
        self.tex_decal = self.ctx.texture(
            image.size, 4, data=np.ascontiguousarray(image.rgba(), dtype="f4").tobytes(),
            dtype="f4",
        )
        # Mipmaps because the decal is usually larger than the patch of atlas it
        # lands in, so minification without them aliases the fine detail a vent
        # is made of. Clamped rather than repeating: outside its rectangle the
        # decal contributes nothing, and the shader already tests for that.
        self.tex_decal.build_mipmaps()
        self.tex_decal.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.tex_decal.repeat_x = False
        self.tex_decal.repeat_y = False
        # The Decal tab shows it, and clicking it starts a placement.
        self.gui.register_texture(self.tex_decal)

    def _sync_decal(self) -> None:
        """Keep the normal map in step with the decal and the export resolution.

        Nothing is allocated until a decal actually exists -- an atlas-sized
        float target is real memory, and most sessions never place one.
        """
        if self.tex_normal is None and not self.decal_params.active():
            return
        if self._normal_resolution != int(self.controller.bake_params.resolution):
            self._normal_dirty = True
        if self._normal_dirty:
            self._run_decal()

    def _run_decal(self) -> None:
        """Re-composite the normal map. One full-screen pass over the atlas."""
        resolution = int(self.controller.bake_params.resolution)
        self._ensure_normal_texture(resolution)
        assert self.normal_fbo is not None

        params = self.decal_params
        active = params.active() and self.tex_decal is not None
        if active:
            assert self.decal_image is not None
            self.tex_decal.use(0)
            for name, value in params.as_uniforms(self.decal_image.aspect).items():
                self.decal_program[name].value = value
        else:
            # Nothing to stamp: place the decal nowhere, so the pass writes the
            # flat normal everywhere rather than being skipped and leaving
            # whatever the last decal put there.
            self._blank_texture.use(0)
            self.decal_program["u_size"].value = (0.0, 0.0)
            self.decal_program["u_intensity"].value = 0.0

        self.normal_fbo.use()
        self.ctx.viewport = (0, 0, resolution, resolution)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        self.decal_vao.render(moderngl.TRIANGLES, vertices=3)
        self._normal_dirty = False

    def read_color_map(self) -> Optional[np.ndarray]:
        """The mask tree resolved to colour, or None if it has never rendered.

        Read back off the GPU rather than recomputed, so the PNG is exactly the
        pixels the viewport is showing.
        """
        self._sync_texture()
        return self.compositor.read()

    def read_normal_map(self) -> Optional[np.ndarray]:
        """The composited normal map as an (n, n, 3) array, or None if flat.

        Read back off the GPU rather than recomputed on the CPU, so the PNG is
        exactly the pixels the viewport is lighting with.
        """
        if not self.decal_params.active() or self.normal_fbo is None:
            return None
        if self._normal_dirty:
            self._run_decal()
        resolution = self._normal_resolution
        return np.frombuffer(
            self.normal_fbo.read(components=3, dtype="f4"), dtype="f4"
        ).reshape(resolution, resolution, 3)

    # -- GL resources -----------------------------------------------------

    def _build_mesh_vao(
        self, vertices: np.ndarray, normals: np.ndarray, uvs: np.ndarray, faces: np.ndarray
    ) -> None:
        self._release_mesh_vao()
        self._pick_geometry = (
            np.ascontiguousarray(vertices, dtype=np.float64),
            np.ascontiguousarray(faces, dtype=np.int64),
            np.ascontiguousarray(uvs, dtype=np.float64),
        )
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

    def _ensure_normal_texture(self, resolution: int) -> None:
        """The decal normal map's own target, sized by the bake resolution.

        Separate from :meth:`_ensure_textures` because the decal does not wait
        for a bake: this exists as soon as an image is imported, at whatever
        resolution the export is set to.
        """
        if self._normal_resolution == resolution and self.tex_normal is not None:
            return
        if self.tex_normal is not None:
            self.tex_normal.release()
        if self.normal_fbo is not None:
            self.normal_fbo.release()

        self.tex_normal = self.ctx.texture((resolution, resolution), 3, dtype="f4")
        self.tex_normal.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_normal.repeat_x = False
        self.tex_normal.repeat_y = False
        self.normal_fbo = self.ctx.framebuffer(color_attachments=[self.tex_normal])
        self._normal_resolution = resolution

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
                self._texture_dirty = True  # the tree reads the new bake

                if PREVIEW_MODES[self.preview_index].texture is None:
                    self.preview_index = 0
                timings = ", ".join(
                    f"{stage} {seconds:.2f}s" for stage, seconds in controller.timings.items()
                )
                self.set_status(f"Bake complete - {timings}")

    def mark_output_dirty(self) -> None:
        self._output_dirty = True

    # -- the texture ------------------------------------------------------

    def mark_texture_dirty(self) -> None:
        self._texture_dirty = True

    @property
    def texture(self) -> Optional[Slot]:
        """The active texture: the one shown, edited and exported."""
        if 0 <= self.texture_index < len(self.textures):
            return self.textures[self.texture_index]
        return None

    def create_texture(self) -> None:
        """Add a texture: one flat colour, named, and made active.

        Added rather than swapped in, so an earlier one is still there to go
        back to through the picker.
        """
        self._texture_serial += 1
        name = f"Texture {self._texture_serial:02d}"
        self.textures.append(new_texture(name))
        self.select_texture(len(self.textures) - 1)
        self.set_status(f"{name} - pick a colour, or turn it into a mask")

    def select_texture(self, index: int) -> None:
        """Make one of the textures the active one."""
        if not 0 <= index < len(self.textures):
            return
        self.texture_index = index
        self.texture_path = ()
        self.renaming_path = None
        self.mark_texture_dirty()

    def remove_texture(self) -> None:
        """Drop the active texture, and fall back to its neighbour."""
        if self.texture is None:
            return
        name = describe(self.textures.pop(self.texture_index))
        self.texture_index = min(self.texture_index, len(self.textures) - 1)
        self.texture_path = ()
        self.renaming_path = None
        if not self.textures:
            self.compositor.clear()
            self._sync_texture_thumbnails()
        self.mark_texture_dirty()
        self.set_status(f"Removed {name}")

    @property
    def selected_slot(self) -> Optional[Slot]:
        """What the Texture tab is editing: a colour or a mask, or None with no
        texture at all. The path is kept valid by :meth:`set_texture`."""
        if self.texture is None:
            return None
        return slot_at(self.texture, self.texture_path)

    def select_slot(self, path: tuple[str, ...]) -> None:
        """Point the Texture tab at a slot, ignoring one that does not exist."""
        if self.texture is None:
            return
        try:
            slot_at(self.texture, tuple(path))
        except KeyError:
            return
        self.texture_path = tuple(path)

    def set_texture(self, texture: Slot) -> None:
        """Swap in an edited version of the active texture and re-composite.

        Edits rebuild the tree rather than mutating it, so this is also where
        the selection is re-validated: collapsing a mask to a colour can pull
        the ground out from under wherever the panel was standing.
        """
        if self.texture is None:
            self.textures.append(texture)
            self.texture_index = len(self.textures) - 1
        else:
            self.textures[self.texture_index] = texture
        self.renaming_path = None
        while self.texture_path:
            try:
                slot_at(self.texture, self.texture_path)
                break
            except KeyError:
                self.texture_path = self.texture_path[:-1]
        self.mark_texture_dirty()

    def _sync_texture(self) -> None:
        """Re-composite when the texture or the bake behind it has changed."""
        if not self._texture_dirty or self.texture is None:
            return
        if self.tex_curvature is None or self.tex_position is None:
            return  # the masks read the bake; nothing to stand on yet

        self.compositor.render(
            self.texture, self._texture_resolution,
            self.tex_curvature, self.tex_position,
        )
        self._texture_dirty = False
        self._sync_texture_thumbnails()

    def _sync_texture_thumbnails(self) -> None:
        """Keep ImGui's texture registry in step with the tree's previews."""
        for texture in self.compositor.reclaim():
            try:
                self.gui.remove_texture(texture)
            except KeyError:
                pass
            texture.release()

        live = set(self.compositor.thumbnails.values())
        for texture in live - self._registered_thumbnails:
            self.gui.register_texture(texture)
        self._registered_thumbnails = live

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

    def _current_texture(self) -> Optional[moderngl.Texture]:
        name = PREVIEW_MODES[self.preview_index].texture
        return {
            "curvature": self.tex_curvature,
            "output": self.tex_output,
            "normal": self.tex_normal,
            "composite": self.compositor.texture,
        }.get(name or "")

    # -- layout -----------------------------------------------------------

    @property
    def viewport_rect(self) -> tuple[int, int, int, int]:
        """Where the 3D view sits, in GL terms: ``(x, y, width, height)``.

        The window is a navigation bar across the top, a sidebar down the left
        and a status bar along the bottom; what is left over is the model's.
        GL measures y from the bottom, so the status bar is the offset.

        The sidebar is capped at half the window: a panel wider than the thing
        it is describing is not a layout worth honouring.
        """
        width, height = self.wnd.buffer_size
        scale = self.ui_pixel_scale
        sidebar = int(min(panel.PANEL_WIDTH * scale, width * 0.5))
        navbar = int(panel.NAVBAR_HEIGHT * scale)
        status = int(panel.STATUS_BAR_HEIGHT * scale)
        return (
            sidebar,
            status,
            max(1, width - sidebar),
            max(1, height - navbar - status),
        )

    def _sync_projection(self) -> None:
        """Keep the camera's aspect matching the rect it actually draws into.

        Re-checked every frame rather than only on resize: the UI scale changes
        the chrome's size, and so the shape of what is left.
        """
        _, _, width, height = self.viewport_rect
        aspect = width / max(height, 1)
        if abs(self.camera.projection.aspect_ratio - aspect) > 1e-6:
            self.camera.projection.update(aspect_ratio=aspect)

    # -- drawing ----------------------------------------------------------

    def _draw_scene(self) -> None:
        # wnd.use() rather than ctx.screen.use(): the headless backend renders
        # into its own framebuffer, and the self-test relies on that.
        self.wnd.use()
        self.ctx.viewport = (0, 0, *self.wnd.buffer_size)
        # Clear colour and depth together: moderngl's clear() always touches
        # both, so this has to happen before the background is drawn.
        self.ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

        # The model lives in the space the chrome leaves it, not behind the
        # sidebar. Everything that turns a screen position into a direction --
        # the projection, the cursor ray, the gizmo -- works off this same rect.
        self.ctx.viewport = self.viewport_rect

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
            mode = FLAT_MODE  # that map does not exist yet
        (texture or self._blank_texture).use(0)

        # The decal lights the mesh in every mode, the way a normal map in a
        # material does; with no decal this is the flat 1x1, which changes
        # nothing and costs one texture unit.
        decal_lit = self.decal_params.active() and self.tex_normal is not None
        (self.tex_normal if decal_lit else self._flat_normal).use(1)

        view = self.camera.matrix
        mvp = self.camera.projection_matrix * view

        self.preview_program["u_mvp"].write(_mat_bytes(mvp))
        self.preview_program["u_mode"].value = mode.shader_mode
        self.preview_program["u_lighting"].value = 1.0 if self.lighting else 0.0
        self.preview_program["u_useNormalMap"].value = 1.0 if decal_lit else 0.0
        self.preview_program["u_checkerScale"].value = self.checker_scale

        eye = self.camera.eye
        light = glm.normalize(eye - self.camera.target + glm.vec3(0.35, 0.6, 0.25) * self.camera.radius)
        self.preview_program["u_lightDir"].value = (light.x, light.y, light.z)

        self.ctx.wireframe = self.wireframe
        self.mesh_vao.render(moderngl.TRIANGLES)
        self.ctx.wireframe = False

    # -- moderngl-window hooks --------------------------------------------

    def on_render(self, time: float, frame_time: float) -> None:
        self._drain_scroll()
        self.controller.pump()
        self._sync_bake_outputs()

        if self._output_dirty:
            self._run_shaping()
        self._sync_texture()
        self._sync_decal()
        self._sync_projection()
        self._draw_scene()

        # The scene draws into its own rect; the interface is drawn over the
        # whole window, and ImGui renders through whatever viewport is bound.
        self.ctx.viewport = (0, 0, *self.wnd.buffer_size)

        imgui.new_frame()
        draw_panel(self)
        if self.show_gizmo and self.mesh is not None:
            gizmo.draw(self.camera, self._gizmo_center(), self.ui_pixel_scale,
                       self._mouse)
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
        self._modifiers = modifiers
        if imgui.get_io().want_capture_keyboard:
            return

        keys = self.wnd.keys
        if action != keys.ACTION_PRESS:
            return

        if key == keys.ESCAPE and self.decal_placing:
            self.end_decal_placement(keep=False)
            return

        # One number key per preview mode, and no more: a shortcut for a mode
        # that does not exist would index straight off the end of the tuple.
        number_keys = (keys.NUMBER_1, keys.NUMBER_2, keys.NUMBER_3, keys.NUMBER_4,
                       keys.NUMBER_5, keys.NUMBER_6)
        shortcuts = {
            key_code: index
            for index, key_code in enumerate(number_keys[:len(PREVIEW_MODES)])
        }
        if key in shortcuts:
            self.preview_index = shortcuts[key]
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
        self._mouse = (float(x), float(y))
        self._modifiers = self.wnd.modifiers

        # A decal being placed rides the cursor. Not over the panel, though:
        # crossing it on the way to the mesh must not fling the decal about.
        if self.decal_placing and not imgui.get_io().want_capture_mouse:
            self.follow_cursor_with_decal()

    def on_mouse_press_event(self, x: int, y: int, button: int) -> None:
        self.gui.mouse_press_event(x, y, button)
        self._mouse = (float(x), float(y))
        self._modifiers = self.wnd.modifiers

        # Decide once, here, what this gesture is for. Deciding per-event instead
        # lets a drag that began on a slider grab the camera the moment the
        # pointer leaves the panel, which is a jarring jump mid-drag.
        if imgui.get_io().want_capture_mouse:
            self._drag_owner = "ui"
            return

        if self.decal_placing:
            # The click that ends placement belongs to the decal and nothing
            # else -- it must not also start an orbit.
            self._drag_owner = "decal"
            if button == self.wnd.mouse.left:
                # What you see is what you get: the decal is already sitting
                # where it will land, so a click keeps it there. A click that
                # misses the mesh keeps the last spot it did land on.
                self.follow_cursor_with_decal()
                self.end_decal_placement(keep=True)
            else:
                self.end_decal_placement(keep=False)
            return

        if button == self.wnd.mouse.left and self.mesh is not None:
            target = gizmo.pick(self.camera, self._gizmo_center(), self.ui_pixel_scale,
                                self._mouse)
            if target is not None:
                self.camera.align_to_axis(target.axis)
                self._drag_owner = "gizmo"
                self.set_status(
                    f"{'+' if target.positive else '-'}{gizmo._LABELS[target.index]} "
                    f"orthographic view"
                )
                return
            if gizmo.hit_test(self._gizmo_center(), self.ui_pixel_scale, self._mouse):
                self._drag_owner = "gizmo"
                return

        self._drag_owner = "camera"

    def on_mouse_release_event(self, x: int, y: int, button: int) -> None:
        self.gui.mouse_release_event(x, y, button)
        self._drag_owner = None

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
        self.gui.mouse_drag_event(x, y, dx, dy)
        self._mouse = (float(x), float(y))
        if self._drag_owner != "camera":
            return

        # Clamp, never discard: dropping the event stalls the view for a frame,
        # which is exactly what a fast flick feels like it should not do.
        dx = float(np.clip(dx, -_MAX_DRAG_STEP, _MAX_DRAG_STEP))
        dy = float(np.clip(dy, -_MAX_DRAG_STEP, _MAX_DRAG_STEP))

        states = self.wnd.mouse_states
        panning = states.middle or (states.left and self.wnd.modifiers.shift)
        if panning:
            self.camera.pan(dx, dy)
        elif states.left:
            self.camera.rot_state(*self._orbit_deltas(-dx, -dy))
        elif states.right:
            self.camera.zoom_state(-dy * 0.25)

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
        self.gui.mouse_scroll_event(x_offset, y_offset)
        if imgui.get_io().want_capture_mouse:
            return

        # Blender's own trackpad bindings for the 3D viewport, from
        # blender_default.py: TRACKPADPAN orbits, +shift pans, +ctrl zooms, and a
        # pinch (TRACKPADZOOM) zooms. macOS delivers a pinch as ctrl+scroll, so
        # the ctrl branch covers both.
        x_offset = float(np.clip(x_offset, -_MAX_SCROLL_STEP, _MAX_SCROLL_STEP))
        y_offset = float(np.clip(y_offset, -_MAX_SCROLL_STEP, _MAX_SCROLL_STEP))

        # Scroll gets its own speed on top of the per-action ones: a trackpad's
        # deltas are nothing like a wheel's, so one is usually far too fast when
        # the other is right.
        step = _PIXELS_PER_SCROLL * self.navigation.scroll_speed

        # pyglet wipes the modifier state on every scroll event
        # (``_handle_modifiers(0)`` in its moderngl-window backend), so read the
        # copy kept from the last key or button event instead.
        modifiers = self._modifiers
        if modifiers.ctrl or modifiers.alt:
            self._queue_scroll(zoom=y_offset * self.navigation.scroll_speed)
        elif modifiers.shift:
            self._queue_scroll(pan=(x_offset * step, y_offset * step))
        else:
            # Same units and sign as a left-drag, so both gestures feel alike.
            self._queue_scroll(
                orbit=self._orbit_deltas(-x_offset * step, y_offset * step)
            )

    def on_pinch_zoom(self, magnification: float) -> None:
        """A trackpad pinch, delivered by :mod:`render.trackpad`.

        Fingers spreading apart is a positive magnification and zooms in, which
        is the direction every other macOS app uses. It joins the same buffer as
        scroll zoom, so a gesture that lands several events in one frame is
        applied once, at the sum.
        """
        if imgui.get_io().want_capture_mouse:
            return

        magnification = float(np.clip(magnification, -_MAX_PINCH_STEP, _MAX_PINCH_STEP))
        # scroll_speed applies here for the same reason it applies to scroll:
        # this is a trackpad gesture, and the wheel's speed rarely suits it.
        self._queue_scroll(
            zoom=magnification * _SCROLL_PER_MAGNIFICATION * self.navigation.scroll_speed
        )

    def _queue_scroll(
        self,
        *,
        orbit: tuple[float, float] = (0.0, 0.0),
        pan: tuple[float, float] = (0.0, 0.0),
        zoom: float = 0.0,
    ) -> None:
        """Buffer one gesture event for :meth:`_drain_scroll` to pace out."""
        self._pending_orbit[0] += orbit[0]
        self._pending_orbit[1] += orbit[1]
        self._pending_pan[0] += pan[0]
        self._pending_pan[1] += pan[1]
        self._pending_zoom += zoom
        # How much motion this event asked for, in the same mixed magnitude the
        # drain measures the buffer in. Only its size matters, not its axis.
        self._scroll_arrived += (
            abs(orbit[0]) + abs(orbit[1]) + abs(pan[0]) + abs(pan[1]) + abs(zoom)
        )

    def _track_arrival_rate(self) -> None:
        """Update :attr:`_scroll_rate`: input per frame, averaged over arrivals.

        Measured as *one event divided by the frames since the last one*, not as
        an average over every frame. That distinction is the whole trick. An
        average taken every frame is pulled down by the empty ones and jerked
        back up by each arrival, so it peaks the instant a step lands and sags
        until the next -- pacing by it reproduces the very burst it is meant to
        even out. A per-gap measurement holds flat between steps, which is what
        makes the output flat.
        """
        arrived, self._scroll_arrived = self._scroll_arrived, 0.0

        if arrived <= 0.0:
            self._scroll_idle += 1
            if self._scroll_idle > _GESTURE_END_FRAMES:
                self._scroll_rate = 0.0  # gesture over; the next starts fresh
                self._scroll_gaps = 0
            return

        gap, self._scroll_idle = self._scroll_idle + 1, 0
        if self._scroll_gaps == 0:
            # Nothing measured yet. Emit this one whole rather than pace it by
            # a guess: at the start of a gesture that is felt as the view
            # refusing to move, and one step is too small to read as a jump.
            self._scroll_rate = arrived
        elif self._scroll_gaps == 1:
            self._scroll_rate = arrived / gap  # first real measurement, trusted
        else:
            sample = arrived / gap
            # Rise faster than it falls. Speeding up is felt directly, as the
            # view failing to keep up with the hand, so follow that quickly;
            # the wobble in the gaps a hand produces is what the average is
            # here to absorb, so give way to it slowly.
            blend = _RATE_BLEND_UP if sample > self._scroll_rate else _RATE_BLEND_DOWN
            self._scroll_rate += (sample - self._scroll_rate) * blend
        self._scroll_gaps += 1

    def _drain_scroll(self) -> None:
        """Hand the accumulated scroll input to the camera, once per frame.

        Scroll arrives a pixel at a time (see :attr:`NavigationPrefs.smoothing`),
        so at low speed several frames pass with nothing and then one lands a
        whole step. Applying each step as it arrives is what the chop is: the
        view moves in bursts separated by stillness.

        So don't emit what arrived -- emit how fast it is arriving.
        :meth:`_track_arrival_rate` keeps that figure, and that much leaves the
        buffer each frame, which turns a sparse run of steps into motion at the
        speed of the fingers. The buffer holds about one
        step's worth in hand to do it, so the view trails the gesture by roughly
        the time between steps: a frame or two at slow speed, and less as the
        gesture speeds up, since the steps arrive closer together.

        Three things keep it honest. The first event of a gesture is emitted
        whole, because there is no history to average yet and holding it back is
        felt as the view refusing to start -- by the second or third step the
        average has learned the speed and pacing takes over. A floor drains
        whatever is waiting within :data:`_DRAIN_LIMIT_FRAMES` no matter how far
        the estimate has drifted, so a low estimate costs a fraction of a step of
        lag rather than a growing backlog. And nothing is ever added or dropped,
        only moved between frames, so the view lands exactly where the gesture
        asked.
        """
        smoothing = float(np.clip(self.navigation.smoothing, 0.0, 1.0))
        pending = (
            abs(self._pending_orbit[0]) + abs(self._pending_orbit[1])
            + abs(self._pending_pan[0]) + abs(self._pending_pan[1])
            + abs(self._pending_zoom)
        )

        self._track_arrival_rate()

        if pending <= 1e-6:
            return

        if smoothing <= 0.0:
            share = 1.0
        else:
            # One frame's worth of travel at the speed input is arriving. The
            # preference divides in, so 1.0 paces exactly at the gesture's speed
            # and lower values run proportionally ahead of it.
            budget = self._scroll_rate / smoothing
            # The floor: never sit on the buffer longer than
            # _DRAIN_LIMIT_FRAMES, however far the estimate has drifted.
            budget = max(budget, pending / _DRAIN_LIMIT_FRAMES)
            share = float(np.clip(budget / pending, 0.0, 1.0))
        # Never leave a tail dribbling for ever.
        if pending < 0.05:
            share = 1.0

        if self._pending_orbit[0] or self._pending_orbit[1]:
            self.camera.rot_state(
                self._pending_orbit[0] * share, self._pending_orbit[1] * share
            )
            self._pending_orbit[0] *= 1.0 - share
            self._pending_orbit[1] *= 1.0 - share
        if self._pending_pan[0] or self._pending_pan[1]:
            self.camera.pan(self._pending_pan[0] * share, self._pending_pan[1] * share)
            self._pending_pan[0] *= 1.0 - share
            self._pending_pan[1] *= 1.0 - share
        if self._pending_zoom:
            self.camera.zoom_state(self._pending_zoom * share)
            self._pending_zoom *= 1.0 - share

    def _orbit_deltas(self, dx: float, dy: float) -> tuple[float, float]:
        """Apply the invert-axis preferences to an orbit delta."""
        return (
            -dx if self.navigation.invert_orbit_x else dx,
            -dy if self.navigation.invert_orbit_y else dy,
        )

    def _gizmo_center(self) -> tuple[float, float]:
        """The gizmo's home: the top-right corner of the 3D view.

        Of the view, not of the window -- the navigation bar owns the strip
        above it, and nothing floats over the view any more to dodge.
        """
        rect_x, _, rect_width, _ = self.viewport_rect
        # ImGui measures y down from the top of the window, so the top of the
        # 3D view is the navigation bar's height.
        top = float(panel.NAVBAR_HEIGHT * self.ui_pixel_scale)
        reach = gizmo.radius(self.ui_pixel_scale)
        margin = gizmo.MARGIN * self.ui_pixel_scale
        return float(rect_x + rect_width) - reach - margin, top + margin + reach

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
        """Write every map there is, in one go.

        One button, whatever exists: the mask tree's colours, the decals'
        normals, and the bake's own two maps. They come from independent halves
        of the app -- a decal is placed in UV space and owes the bake nothing --
        so each is written if it is there and skipped if it is not.
        """
        controller = self.controller
        baked = controller.curvature_map is not None and self.output_fbo is not None
        normal = self.read_normal_map()
        color = self.read_color_map()

        if not baked and normal is None and color is None:
            self.set_status(
                "Nothing to export - bake a mesh, or import a decal", error=True
            )
            return

        shaped = None
        if baked:
            if self._output_dirty:
                self._run_shaping()
            resolution = self._texture_resolution
            # Read the shaped map back off the GPU rather than recomputing it on
            # the CPU, so the PNG is exactly the pixels shown in the viewport.
            shaped = np.frombuffer(
                self.output_fbo.read(components=1, dtype="f4"), dtype="f4"
            ).reshape(resolution, resolution)

        stem = Path(self.mesh_info.path).stem if self.mesh_info else "mesh"
        target = Path(self.export_dir) / stem
        try:
            written = export_maps(
                target,
                shaped,
                controller.curvature_map if baked else None,
                normal=normal,
                color=color,
                bits=self.export_bits,
            )
        except Exception:
            self.set_status(traceback.format_exc(limit=3), error=True)
            return

        names = ", ".join(path.name for path in written)
        self.set_status(f"Wrote {names} to {target}")
