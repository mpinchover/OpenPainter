"""The application window: 3D viewport, live EdgeWear001 pass, event routing."""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
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

from core.decal import (
    DecalImage,
    DecalLoadError,
    contains as decal_contains,
    library as decal_library,
    load_decal,
    outline_uvs,
    uv_aspect,
)
from core.decal_wrap import wrapped_vertices
from core.decal_thumbnail import DecalThumbnail, load_thumbnail
from core.export import (
    COLOR_NAME,
    MAP_NAMES,
    METALLIC_NAME,
    NORMAL_NAME,
    OCCLUSION_NAME,
    ROUGHNESS_NAME,
    export_maps,
)
from core.layers import (
    EDGE_WEAR_KIND,
    MAX_EMISSION,
    MaskLayer,
    Slot,
    describe,
    mask_at,
    new_texture,
    slot_at,
    texture_key,
    walk,
)
from core.mesh_io import SUPPORTED_SUFFIXES, MeshLoadError, default_mesh, load_mesh
from core.metadata import METADATA
from core.params import BevelParams, DecalParams, MeshInfo
from core.picking import face_at_uv, ray_mesh_hit, screen_ray, surface_at_uv
from core.pipeline import BakeController
from core.uv_unwrap import source_uvs
from render.composite import LayerCompositor
from render.imgui_renderer import ImGuiRenderer
from render.shaders import load_shader
from render.trackpad import install_pinch_zoom
from ui import decal_gizmo, gizmo, light_gizmo, panel
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

# Full export resolution is unnecessary while a decal follows the pointer.
# Keeping the live target bounded prevents 4K/8K float-buffer work per frame.
DECAL_INTERACTIVE_RESOLUTION = 2048
MAX_VIEW_PROJECTORS = 8

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


def _stored_maps(stored) -> set[str]:
    """Which maps an export writes, as read back from the preferences file.

    Anything unrecognised is dropped and anything malformed falls back to the
    whole set, because a hand-edited typo should cost a tick box rather than
    every map the session would have written.
    """
    if not isinstance(stored, (list, tuple, set)):
        return set(MAP_NAMES)
    chosen = {name for name in stored if name in MAP_NAMES}
    return chosen if chosen else set(MAP_NAMES)


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
class WorldPrefs:
    """The lighting: a key light you can walk around the model, and the world
    it stands in.

    A single lamp leaves a metal with nothing to reflect and a rough surface
    with nothing to scatter -- both render black however their sliders are set.
    This is the rest of the room: a colour, and how much of it.
    """

    rotation: float = 0.0
    """Where the key light stands, in degrees around the model: 0 is the +X
    axis, 90 is +Y, counting the way the gizmo's axes do. It used to hang off
    the camera, which meant it could not be moved -- orbiting took it along."""

    strength: float = 1.5
    """Enough for a metal to read as metal on opening. A metal shows only what
    it reflects, so a dimmer world than this leaves one looking muddy and the
    slider looking broken -- which is the complaint this was built for."""

    color: tuple[float, float, float] = (0.55, 0.60, 0.72)
    """Slightly blue, like an overcast sky, so a neutral surface does not read
    as tinted the moment the world is turned up."""

    FIELDS = ("rotation", "strength", "color")

    def light_direction(self) -> tuple[float, float, float]:
        """Which way the surface has to look to see the key light.

        A unit vector pointing *at* the light, which is what the shader wants:
        ``dot(normal, light)`` is how much of it a surface catches.
        """
        azimuth = math.radians(self.rotation)
        elevation = math.radians(LIGHT_ELEVATION)
        flat = math.cos(elevation)
        return (
            flat * math.cos(azimuth),
            flat * math.sin(azimuth),
            math.sin(elevation),
        )

    def as_dict(self) -> dict:
        return {
            "rotation": round(self.rotation, 2),
            "strength": round(self.strength, 4),
            "color": [round(float(c), 4) for c in self.color],
        }

    @classmethod
    def from_dict(cls, stored: dict) -> "WorldPrefs":
        prefs = cls()
        try:
            if "rotation" in stored:
                prefs.rotation = float(stored["rotation"]) % 360.0
            if "strength" in stored:
                prefs.strength = float(
                    np.clip(float(stored["strength"]), 0.0, MAX_WORLD_STRENGTH)
                )
            if "color" in stored:
                red, green, blue = (float(c) for c in stored["color"][:3])
                prefs.color = tuple(float(np.clip(c, 0.0, 1.0))
                                    for c in (red, green, blue))
        except (TypeError, ValueError):
            pass  # keep the defaults for anything malformed
        return prefs


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


#: How far above white the scene target can hold a colour. The target stores
#: everything divided by this, which keeps it inside [0, 1] whatever the driver
#: decides about clamping float buffers -- the same dodge the material map's
#: emission channel uses. An emissive surface brighter than this stops getting
#: brighter, which is why it sits comfortably above MAX_EMISSION.
HDR_SCALE = 32.0

#: How much of the blurred over-white light is added back over the scene. Enough
#: for an emissive surface to read as a light source, not so much that it fogs
#: the model behind it.
BLOOM_STRENGTH = 1.1

#: The glow is blurred at a fraction of the window's resolution. A wide, soft
#: blur is most of the cost of a bloom, and none of that width survives being
#: seen at full size anyway.
BLOOM_DIVISOR = 4

#: How long the light arrow stays in the viewport after the rotation slider is
#: let go, and how much of that is spent fading. Long enough to look at what you
#: just set, short enough that it is not another thing permanently in the way.
LIGHT_HINT_SECONDS = 1.8
LIGHT_HINT_FADE = 0.7

#: How high the key light sits above the horizon, in degrees. Not adjustable:
#: it is where the light already sat before it could be turned, and the useful
#: control is which side of the model it comes from rather than how steeply.
LIGHT_ELEVATION = 38.0

#: The eight directions the light's bearing is reported as, starting at +X and
#: turning the way the rotation does. Named after the gizmo's axes, because that
#: is what is on screen to compare against.
COMPASS = ("+X", "+X +Y", "+Y", "-X +Y", "-X", "-X -Y", "-Y", "+X -Y")


def bearing(rotation: float) -> tuple[str, str]:
    """Where the light stands and which way it therefore shines.

    Two labels from :data:`COMPASS`, the second always the opposite of the
    first. Rounded to the nearest eighth: the exact angle is on the slider, and
    what a reader wants from a line of text is which side of the model is lit.
    """
    sector = int(round((rotation % 360.0) / 45.0)) % len(COMPASS)
    return COMPASS[sector], COMPASS[(sector + len(COMPASS) // 2) % len(COMPASS)]


#: Brightest the world light may be set to. Beyond this the highlight rolloff
#: is doing all the work and the slider stops meaning anything.
MAX_WORLD_STRENGTH = 4.0


#: How narrow and how wide the sidebar may be dragged, in unscaled pixels.
#: The floor is about what the widest label needs; the ceiling stops the panel
#: from becoming the application.
SIDEBAR_MIN = 260.0
SIDEBAR_MAX = 900.0


def _clamp_sidebar(value: float) -> float:
    return float(np.clip(value, SIDEBAR_MIN, SIDEBAR_MAX))


#: Neither pane of the Texture tab may be dragged below this share of the tab.
#: A pane too short to show a row of its own is a pane that has been closed by
#: accident, and the splitter has no other way back.
MIN_SPLIT = 0.15


def _clamp_split(value: float) -> float:
    return float(np.clip(value, MIN_SPLIT, 1.0 - MIN_SPLIT))


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
    #: Read from metadata.json, which moderngl-window puts in the title bar.
    title = METADATA.title
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

        self._apply_icon()

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
        self._full_texture_render_requested = False
        #: Counts textures made this session, so each gets its own name.
        self._texture_serial = 0
        #: Which row the tree is renaming in place, if any, and whether its
        #: field still has to be given focus.
        self.renaming_path: Optional[tuple[str, ...]] = None
        self.renaming_opened = False
        #: What has been typed into the texture picker's search box.
        self.texture_filter: str = ""
        #: True when the composited texture is a single colour -- see
        #: :meth:`_check_texture_variation`.
        self.texture_is_flat = False
        #: What the compositor last drew, so an edit anywhere in the tree is
        #: noticed without every control having to say so.
        self._composited_key: Optional[tuple] = None
        #: Every decal stamped on the mesh, in the order they were placed --
        #: which is the order they are drawn, so the last one is on top.
        self.decals: list[DecalParams] = []
        #: Which one the Decal tab is inspecting, or -1 for none.
        self.decal_index: int = -1
        #: Images and their GL textures, cached by path. Several placements of
        #: one picture share both.
        self.decal_images: dict[str, DecalImage] = {}
        self.decal_textures: dict = {}
        #: The shelf offered in the Decal tab, from metadata.json.
        self.decal_library: list[Path] = decal_library(METADATA.decals)
        #: Library images are decoded away from the render thread.  Opening
        #: the Decal tab must not synchronously decode every large PNG before
        #: the window can draw its next frame.
        self._decal_library_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="decal-library"
        )
        self._decal_library_future: Optional[tuple[Path, Future[DecalThumbnail]]] = None
        self._decal_library_attempted: set[str] = set()
        self.decal_thumbnail_sizes: dict[str, tuple[int, int]] = {}
        self.decal_thumbnail_textures: dict = {}
        self._decal_image_futures: dict[str, Future[DecalImage]] = {}
        self._pending_decal_drop: Optional[
            tuple[Path, tuple[float, float], int]
        ] = None
        self._live_decal_projector: Optional[dict] = None
        self._full_decal_render_requested = False
        #: A decal being dragged out of the shelf, waiting to be dropped.
        self.dragging_decal: Optional[Path] = None
        #: Temporary placement rendered under the cursor during a shelf drag.
        #: It never enters ``self.decals`` (and therefore cannot be exported)
        #: until the mouse is released over the mesh.
        self.dragging_decal_preview: Optional[DecalParams] = None
        #: True while the decal is following the cursor, waiting to be dropped.
        self.decal_placing = False
        #: Where it was before that started, to put back if the user cancels.
        self._decal_anchor: Optional[tuple[float, float]] = None
        #: Blender-style keyboard transforms: G moves, S scales and R rotates
        #: around the surface normal. Pointer events supply the motion.
        self._decal_transform_mode: Optional[str] = None
        self._decal_transform_axis: Optional[str] = None
        self._decal_transform_anchor: Optional[tuple] = None
        self._decal_transform_last_hit: Optional[tuple[tuple[float, float], int]] = None
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
        #: When the light arrow was last asked for. Far in the past, so nothing
        #: is drawn until the rotation is actually touched.
        self._light_hint_at = float("-inf")
        self.status = "Drop an FBX onto the window, or use File > Import mesh."
        self.status_is_error = False
        self.file_dialog = None
        self.folder_dialog = None
        #: Set while the File menu's export is waiting on a folder to be picked.
        self.export_dialog = None
        #: Set by File > Export textures or the E key; the panel picks it up and
        #: opens the folder chooser.
        self.export_pending = False
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
        self._shift_down = False

        self._prefs = _load_prefs()
        self.navigation = NavigationPrefs.from_dict(self._prefs.get("navigation", {}))
        self.world = WorldPrefs.from_dict(self._prefs.get("world", {}))
        #: Which maps an export writes. Everything, until told otherwise --
        #: they describe one surface between them, and a set with holes in it is
        #: a choice worth having to make. Occlusion is the one that costs
        #: anything to leave out: see :meth:`set_map_enabled`.
        self.enabled_maps: set[str] = _stored_maps(self._prefs.get("maps"))
        self.controller.bake_occlusion = OCCLUSION_NAME in self.enabled_maps

        stored_scale = self._prefs.get("ui_scale")
        self.ui_scale = self.initial_ui_scale
        if not self.initial_ui_scale_explicit and stored_scale is not None:
            try:
                self.ui_scale = float(np.clip(float(stored_scale), 0.6, 3.0))
            except (TypeError, ValueError):
                pass
        #: Sidebar width, in the same unscaled units as panel.PANEL_WIDTH so it
        #: means the same thing at any UI scale. Dragged by its right edge.
        self.sidebar_width = float(panel.PANEL_WIDTH)
        try:
            stored_width = self._prefs.get("sidebar_width")
            if stored_width is not None:
                self.sidebar_width = _clamp_sidebar(float(stored_width))
        except (TypeError, ValueError):
            pass

        #: How much of the Texture tab the inspector gets, the tree taking the
        #: rest. Dragged with the splitter between them, and remembered.
        self.texture_split = 0.5
        #: Where the Decal tab's inspector ends and its shelf begins.
        self.decal_split = 0.5
        try:
            stored_decal_split = self._prefs.get("decal_split")
            if stored_decal_split is not None:
                self.decal_split = _clamp_split(float(stored_decal_split))
        except (TypeError, ValueError):
            pass

        try:
            stored_split = self._prefs.get("texture_split")
            if stored_split is not None:
                self.texture_split = _clamp_split(float(stored_split))
        except (TypeError, ValueError):
            pass

        # Snapshot before the first scale is applied, so the baseline is pristine.
        self._style_defaults = snapshot_style()
        self.apply_ui_scale()

        self.preview_program = self.ctx.program(
            vertex_shader=load_shader("preview.vert"),
            fragment_shader=load_shader("preview.frag"),
        )
        self.background_program = self.ctx.program(
            vertex_shader=load_shader("fullscreen.vert"),
            fragment_shader=load_shader("background.frag"),
        )
        self.decal_program = self.ctx.program(
            vertex_shader=load_shader("fullscreen.vert"),
            fragment_shader=load_shader("decal.frag"),
        )
        self.decal_wrap_program = self.ctx.program(
            vertex_shader=load_shader("decal_wrap.vert"),
            fragment_shader=load_shader("decal_wrap.frag"),
        )
        self.decal_project_program = self.ctx.program(
            vertex_shader=load_shader("decal_project.vert"),
            fragment_shader=load_shader("decal_project.frag"),
        )
        self.encode_program = self.ctx.program(
            vertex_shader=load_shader("fullscreen.vert"),
            fragment_shader=load_shader("normal_encode.frag"),
        )
        self.bright_program = self.ctx.program(
            vertex_shader=load_shader("fullscreen.vert"),
            fragment_shader=load_shader("bright.frag"),
        )
        self.blur_program = self.ctx.program(
            vertex_shader=load_shader("fullscreen.vert"),
            fragment_shader=load_shader("blur.frag"),
        )
        self.tonemap_program = self.ctx.program(
            vertex_shader=load_shader("fullscreen.vert"),
            fragment_shader=load_shader("tonemap.frag"),
        )

        self.preview_program["u_map"].value = 0
        self.preview_program["u_normalMap"].value = 1
        self.preview_program["u_material"].value = 2
        self.preview_program["u_liveDecal"].value = 3
        self.preview_program["u_viewProjectorTextures"].value = tuple(
            range(4, 4 + MAX_VIEW_PROJECTORS)
        )
        self.preview_program["u_emissionScale"].value = MAX_EMISSION
        self.decal_program["u_decal"].value = 0
        self.decal_wrap_program["u_decal"].value = 0
        self.decal_project_program["u_decal"].value = 0
        self.encode_program["u_slope"].value = 0
        for program in (self.preview_program, self.background_program,
                        self.bright_program, self.tonemap_program):
            program["u_hdrScale"].value = HDR_SCALE
        self.bright_program["u_scene"].value = 0
        self.blur_program["u_tex"].value = 0
        self.tonemap_program["u_scene"].value = 0
        self.tonemap_program["u_bloom"].value = 1
        self.tonemap_program["u_bloomStrength"].value = BLOOM_STRENGTH

        self.compositor = LayerCompositor(self.ctx)
        self._registered_thumbnails: set = set()

        self.bright_vao = self.ctx.vertex_array(self.bright_program, [])
        self.blur_vao = self.ctx.vertex_array(self.blur_program, [])
        self.tonemap_vao = self.ctx.vertex_array(self.tonemap_program, [])

        # The scene is drawn into a target with room above white, then brought
        # down to the window. Built on first use and rebuilt when the window
        # changes size.
        self._scene_fbo: Optional[moderngl.Framebuffer] = None
        self._scene_tex: Optional[moderngl.Texture] = None
        self._scene_depth: Optional[moderngl.Renderbuffer] = None
        self._bloom_fbos: list[moderngl.Framebuffer] = []
        self._bloom_texs: list[moderngl.Texture] = []
        self._target_size: tuple[int, int] = (0, 0)
        self.background_vao = self.ctx.vertex_array(self.background_program, [])
        self.decal_vao = self.ctx.vertex_array(self.decal_program, [])
        self._decal_wrap_cache: dict[
            int, tuple[Optional[moderngl.Buffer], Optional[moderngl.VertexArray], dict[int, np.ndarray]]
        ] = {}
        self._decal_wrap_futures: dict[int, tuple[tuple, Future]] = {}
        self.encode_vao = self.ctx.vertex_array(self.encode_program, [])

        self.mesh_vao: Optional[moderngl.VertexArray] = None
        self.decal_project_vao: Optional[moderngl.VertexArray] = None
        self._mesh_buffers: list = []
        self._vao_token: tuple = ()
        #: The geometry the VAO was built from, kept for cursor picking: the
        #: ray has to hit exactly what is on screen, seams and all.
        self._pick_geometry: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        #: Where the last press landed, for telling a click from a drag.
        self._press_at: Optional[tuple[float, float]] = None
        #: The selected decal's border on the mesh, and what it was built for.
        self._outline: Optional[np.ndarray] = None
        self._outline_key: tuple = ()
        #: What :meth:`refresh_decal_aspect` last measured, and for which mesh.
        self._aspect_centres: tuple = ()
        self._aspect_token: tuple = ()

        self.tex_curvature: Optional[moderngl.Texture] = None
        self.tex_position: Optional[moderngl.Texture] = None
        self._texture_resolution = 0
        self._uploaded_maps_version = -1

        # The decal chain: the imported image, and the atlas-sized normal map it
        # is composited into. Independent of the bake -- a decal is placed in UV
        # space, so it needs no geometry pass behind it.
        self.tex_normal: Optional[moderngl.Texture] = None
        self.normal_fbo: Optional[moderngl.Framebuffer] = None
        #: Where the decals' gradients are added up before being encoded.
        self._slope_texture: Optional[moderngl.Texture] = None
        self._slope_fbo: Optional[moderngl.Framebuffer] = None
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
        # Stands in for the texture's surface channels when there is no texture:
        # not metal, half rough, opaque, unlit.
        self._flat_material = self.ctx.texture(
            (1, 1), 4, data=np.array([0.0, 0.5, 1.0, 0.0], dtype="f4").tobytes(),
            dtype="f4",
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

        # Something to work on the moment the window opens, at both ends of the
        # pipeline: a mesh to bake, and a texture to put on it.
        self.create_texture()

        if self.initial_mesh:
            self.open_mesh(self.initial_mesh)
        else:
            self.open_default_mesh()

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

        io = imgui.get_io()
        # ImGui's own defaults are 0.30s and 6 pixels, and both are too strict
        # here. The window is driven in *physical* pixels, so 6 of them is a
        # third of the distance a hand actually holds still over on a HiDPI
        # display, and 0.30s is quicker than a comfortable double-click --
        # macOS ships that slider at half a second. A rename that will not
        # trigger reads as a broken control, so both are given room.
        io.mouse_double_click_time = 0.55
        io.mouse_double_click_max_dist = 6.0 * self.ui_pixel_scale

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

    def set_decal_split(self, value: float) -> None:
        """Where the Decal tab's divider sits, as a share of the room."""
        self.decal_split = _clamp_split(value)

    def set_texture_split(self, value: float) -> None:
        """Move the divider between the Texture tab's two panes."""
        self.texture_split = _clamp_split(value)

    def set_sidebar_width(self, value: float) -> None:
        """Widen or narrow the sidebar, in unscaled reference pixels."""
        self.sidebar_width = _clamp_sidebar(value)

    def over_sidebar_edge(self, mouse: tuple[float, float]) -> bool:
        """Whether a cursor position is on the sidebar's right edge.

        Plain arithmetic against the one number that decides the width, rather
        than an ImGui window sitting there to be hovered: a window of its own
        brought its own padding and border, so the band it offered was beside
        the edge rather than across it -- and it had to be drawn, which is a
        bar nobody asked for.
        """
        scale = self.ui_pixel_scale
        x = mouse[0] * float(self.wnd.pixel_ratio)
        y = mouse[1] * float(self.wnd.pixel_ratio)
        edge = self.sidebar_pixels

        within_height = (
            panel.NAVBAR_HEIGHT * scale
            <= y
            <= self.wnd.buffer_size[1] - panel.STATUS_BAR_HEIGHT * scale
        )
        return within_height and (
            edge - panel.SIDEBAR_GRAB_INSIDE * scale
            <= x
            <= edge + panel.SIDEBAR_GRAB_OUTSIDE * scale
        )

    @property
    def sidebar_pixels(self) -> int:
        """The sidebar's width on screen, capped at half the window.

        The one place that decides it: the panel draws itself this wide and the
        3D view starts where it ends, so the two cannot disagree.
        """
        return int(min(self.sidebar_width * self.ui_pixel_scale,
                       self.wnd.buffer_size[0] * 0.5))

    def save_prefs(self) -> None:
        """Persist the interface and navigation settings."""
        self._prefs["ui_scale"] = round(self.ui_scale, 3)
        self._prefs["sidebar_width"] = round(self.sidebar_width, 1)
        self._prefs["texture_split"] = round(self.texture_split, 3)
        self._prefs["decal_split"] = round(self.decal_split, 3)
        self._prefs["navigation"] = self.navigation.as_dict()
        self._prefs["world"] = self.world.as_dict()
        self._prefs["maps"] = [name for name in MAP_NAMES if name in self.enabled_maps]
        _save_prefs(self._prefs)

    # -- mesh loading -----------------------------------------------------

    def exportable(self) -> list[str]:
        """Which maps an export would write right now, in the order it writes
        them: the ones switched on, of the ones that exist.

        The panel shows this, the File menu greys itself out on it, and
        :meth:`export` refuses on it -- one answer, so none of the three can
        promise something the others do not.
        """
        textured = self.texture is not None and self.controller.curvature_map is not None
        exists = {
            COLOR_NAME: textured,
            NORMAL_NAME: self.any_decal_active(),
            METALLIC_NAME: textured,
            ROUGHNESS_NAME: textured,
            OCCLUSION_NAME: self.controller.occlusion_map is not None,
        }
        return [
            name for name in MAP_NAMES
            if exists.get(name) and name in self.enabled_maps
        ]

    def map_enabled(self, name: str) -> bool:
        return name in self.enabled_maps

    def set_map_enabled(self, name: str, enabled: bool) -> None:
        """Turn one map of the export on or off, and remember it.

        Four of the five cost nothing to produce -- colour, metallic and
        roughness are one composite pass over the texture, normals one pass over
        the decal -- so switching those off only stops the file being written.

        Occlusion is different: it is the one stage that traces rays, and it
        costs more than the whole rest of the bake. Switching it off takes it
        out of the bake as well, which is the point of being able to.
        """
        if enabled:
            self.enabled_maps.add(name)
        else:
            self.enabled_maps.discard(name)

        # One place where the two meanings meet, so they cannot disagree.
        self.controller.bake_occlusion = self.map_enabled(OCCLUSION_NAME)
        self.save_prefs()

    def open_mesh(self, path: str | Path) -> None:
        try:
            mesh, info = load_mesh(path, source_z_up=self.source_z_up)
        except MeshLoadError as exc:
            self.set_status(str(exc), error=True)
            return
        except Exception:
            self.set_status(traceback.format_exc(limit=3), error=True)
            return

        self._show_mesh(mesh, info)
        note = f" ({'; '.join(info.notes)})" if info.notes else ""
        self.set_status(
            f"Loaded {Path(info.path).name} via {info.backend}: "
            f"{info.vertices:,} verts / {info.faces:,} tris{note}"
        )

    def open_default_mesh(self) -> None:
        """Start on a plain cube rather than an empty viewport.

        Something to bake and paint on the moment the window opens, without
        importing anything -- and box-unwrapped, so it takes the same path a
        mesh out of Blender does rather than a special case.
        """
        mesh, info = default_mesh()
        self._show_mesh(mesh, info)
        # Sharp edges have no width for the curvature bake to put a gradient
        # in, so edge wear finds nothing on this cube until the Bevel panel
        # gives it something -- worth saying, since the alternative is a mask
        # that silently does nothing.
        self.set_status(
            f"Starter cube ({info.extents[0]:g} m, sharp). Bake to work on it; "
            f"turn on Bevel for edge wear to have an edge to find. "
            f"Drop a mesh on the window to swap it out."
        )

    def _show_mesh(self, mesh: trimesh.Trimesh, info: MeshInfo) -> None:
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

    def set_status(self, message: str, *, error: bool = False) -> None:
        self.status = message
        self.status_is_error = error

    # -- decals -----------------------------------------------------------

    @property
    def selected_decal(self) -> Optional[DecalParams]:
        if 0 <= self.decal_index < len(self.decals):
            return self.decals[self.decal_index]
        return None

    def select_decal(self, index: Optional[int]) -> None:
        """Point the inspector at one decal, or at none."""
        if self._decal_transform_mode is not None:
            self.end_decal_transform(keep=False)
        self.decal_index = -1 if index is None else int(index)
        if not (0 <= self.decal_index < len(self.decals)):
            self.decal_index = -1

    def decal_image_for(self, params: DecalParams) -> Optional[DecalImage]:
        return self.decal_images.get(params.path)

    def any_decal_active(self) -> bool:
        return any(params.active() for params in self.decals)

    def any_visible_decal_active(self) -> bool:
        """Whether the viewport has a committed decal or a drag preview."""
        preview = self.dragging_decal_preview
        return self.any_decal_active() or (preview is not None and preview.active())

    def decals_key(self) -> tuple:
        """Everything about the decals that changes the normal map."""
        return tuple(params.key() for params in self.decals)

    def add_decal(
        self, path: str | Path, center: Optional[tuple[float, float]] = None,
        face: Optional[int] = None,
    ) -> Optional[int]:
        """Place a copy of an image on the mesh, and select it.

        The image is loaded once and shared: stamping the same vent in six
        places costs six placements and one picture.
        """
        image = self.load_decal_image(path)
        if image is None:
            return None

        params = DecalParams(path=image.path, image_aspect=image.aspect)
        if center is not None:
            params.center_u, params.center_v = center
        if face is not None:
            params.surface_face = int(face)
        self.decals.append(params)
        self.decal_index = len(self.decals) - 1
        self.refresh_decal_aspect()
        self.mark_normal_dirty()
        self.set_status(f"Placed {Path(image.path).name}")
        return self.decal_index

    def load_decal_image(self, path: str | Path) -> Optional[DecalImage]:
        """Read an image, or hand back the one already read for that path."""
        path = Path(path)
        cached = self.decal_images.get(str(path))
        if cached is not None:
            return cached
        try:
            image = load_decal(path)
        except DecalLoadError as error:
            self.set_status(str(error), error=True)
            return None

        self.decal_images[image.path] = image
        self._upload_decal(image)
        return image

    def pump_decal_library(self) -> None:
        """Advance lazy library loading without blocking the UI thread.

        PNG decoding and normal-map preparation happen on a worker.  The GL
        upload stays on the render thread, one completed image per frame.
        """
        pending = self._decal_library_future
        if pending is not None:
            path, future = pending
            if not future.done():
                return
            self._decal_library_future = None
            try:
                image = future.result()
            except Exception as error:  # keep one bad library item non-fatal
                self.set_status(f"Could not load {path.name}: {error}", error=True)
            else:
                self._upload_decal_thumbnail(image)

        if self._decal_library_future is None:
            for path in self.decal_library:
                key = str(path)
                if key in self.decal_thumbnail_textures or key in self._decal_library_attempted:
                    continue
                self._decal_library_attempted.add(key)
                cache = _settings_dir() / "cache" / "decal-thumbnails"
                future = self._decal_library_executor.submit(load_thumbnail, path, cache)
                self._decal_library_future = (path, future)
                break

    def request_decal_image(self, path: str | Path) -> None:
        """Begin preparing a full-resolution decal without stalling a drag."""
        key = str(Path(path).expanduser())
        if key not in self.decal_images and key not in self._decal_image_futures:
            self._decal_image_futures[key] = self._decal_library_executor.submit(
                load_decal, Path(key)
            )

    def _pump_decal_images(self) -> None:
        """Move completed full decals onto GL, which must stay on this thread."""
        for key, future in tuple(self._decal_image_futures.items()):
            if not future.done():
                continue
            del self._decal_image_futures[key]
            try:
                image = future.result()
            except DecalLoadError as error:
                self.set_status(str(error), error=True)
                if self._pending_decal_drop and str(self._pending_decal_drop[0]) == key:
                    self._pending_decal_drop = None
            except Exception as error:
                self.set_status(f"Could not load {Path(key).name}: {error}", error=True)
                if self._pending_decal_drop and str(self._pending_decal_drop[0]) == key:
                    self._pending_decal_drop = None
            else:
                if image.path not in self.decal_images:
                    self.decal_images[image.path] = image
                    self._upload_decal(image)
                if self.dragging_decal is not None and str(self.dragging_decal) == key:
                    self._update_dragged_decal_preview()
                pending = self._pending_decal_drop
                if pending is not None and str(pending[0]) == key:
                    self._pending_decal_drop = None
                    self.add_decal(pending[0], center=pending[1], face=pending[2])

    def _pump_decal_wraps(self) -> None:
        """Upload completed surface charts without unfolding during a frame."""
        for face, (token, future) in tuple(self._decal_wrap_futures.items()):
            if not future.done():
                continue
            del self._decal_wrap_futures[face]
            if token != self._vao_token:
                continue
            try:
                data, layout = future.result()
            except Exception as error:
                self.set_status(f"Could not project decal: {error}", error=True)
                continue
            self._store_wrapped_decal(face, data, layout)
            self.mark_normal_dirty()

    def remove_decal(self, index: Optional[int] = None) -> None:
        """Take one decal off the mesh. The neighbour takes the selection."""
        if self._decal_transform_mode is not None:
            self.end_decal_transform(keep=False)
        index = self.decal_index if index is None else index
        if not (0 <= index < len(self.decals)):
            return
        name = Path(self.decals[index].path).name
        del self.decals[index]
        self.decal_index = min(index, len(self.decals) - 1)
        self.mark_normal_dirty()
        self.set_status(f"Removed {name}")

    def open_decal(self, path: str | Path) -> None:
        """Import an image and stamp it on the mesh, in the middle of the atlas.

        A grayscale image is read as a height map and converted, since that is
        the only reading of it that describes a surface -- see
        :func:`core.decal.load_decal`.
        """
        index = self.add_decal(path)
        if index is None:
            return

        image = self.decal_images[self.decals[index].path]
        width, height = image.size
        source = "height map, converted" if image.from_height else "normal map"
        self.set_status(
            f"Decal: {Path(image.path).name} ({width}x{height} {source}). "
            f"Exports as normal.png."
        )

    def clear_decals(self) -> None:
        """Take every decal off the mesh."""
        if self._decal_transform_mode is not None:
            self.end_decal_transform(keep=False)
        self.decals.clear()
        self.decal_index = -1
        self.decal_placing = False
        self.mark_normal_dirty()
        self.set_status("Decals cleared")

    def mark_normal_dirty(self) -> None:
        self._normal_dirty = True

    # -- keyboard decal transforms --------------------------------------

    def begin_decal_transform(self, mode: str) -> bool:
        """Start a pointer-driven G (move) or S (scale) transform."""
        params = self.selected_decal
        if params is None:
            self.set_status(f"Select a decal before pressing {mode.upper()}", error=True)
            return False
        if mode not in ("move", "scale", "rotate"):
            return False
        if self._decal_transform_mode is not None:
            if self._decal_transform_mode == mode:
                return True
            self.end_decal_transform(keep=False)
        self._decal_transform_mode = mode
        self._decal_transform_axis = None
        self._decal_transform_anchor = (
            params.center_u, params.center_v, params.surface_face, params.scale,
            params.scale_x, params.scale_y, params.rotation,
            params.projector_center, params.projector_right, params.projector_up,
            params.projector_forward, params.projector_size,
        )
        self._decal_transform_last_hit = None
        if params.projector_center is not None:
            self._live_decal_projector = {
                "path": params.path,
                "center": params.projector_center,
                "right": params.projector_right,
                "up": params.projector_up,
                "forward": params.projector_forward,
                "size": params.projector_size,
                "intensity": float(params.intensity),
                "flip_green": 1.0 if params.flip_green else 0.0,
            }
        elif self._pick_geometry is not None:
            vertices, faces, uvs = self._pick_geometry
            point = surface_at_uv(
                vertices, faces, uvs, params.center_u, params.center_v
            )
            image = self.decal_images.get(params.path)
            if point is not None and image is not None:
                self._set_live_decal_projector(
                    image, point, params,
                    params.surface_face if params.surface_face >= 0 else None,
                )
        # Rebuild once without the decal being transformed. From here until
        # confirmation the shader projector is the only moving contribution.
        self.mark_normal_dirty()
        verb = {"move": "Move", "scale": "Scale", "rotate": "Rotate"}[mode]
        self.set_status(
            f"{verb} decal with one-finger pointer motion; X/Y constrains, "
            "click or Enter confirms, Esc cancels"
        )
        return True

    def constrain_decal_transform(self, axis: str) -> bool:
        if self._decal_transform_mode not in ("move", "scale") or axis not in ("x", "y"):
            return False
        self._decal_transform_axis = axis
        self.set_status(
            f"{self._decal_transform_mode.title()} decal on {axis.upper()} only; "
            "move one finger to adjust, click or Enter confirms, Esc cancels"
        )
        return True

    def end_decal_transform(self, *, keep: bool) -> None:
        if self._decal_transform_mode is None:
            return
        params = self.selected_decal
        if not keep and params is not None and self._decal_transform_anchor is not None:
            (params.center_u, params.center_v, params.surface_face, params.scale,
             params.scale_x, params.scale_y, params.rotation,
             params.projector_center, params.projector_right, params.projector_up,
             params.projector_forward, params.projector_size) = self._decal_transform_anchor
            self.mark_normal_dirty()
        elif keep and params is not None and self._decal_transform_last_hit is not None:
            uv, face = self._decal_transform_last_hit
            params.center_u, params.center_v = uv
            params.surface_face = int(face)
            params.surface_aspect = self.measure_uv_aspect(face)
            self.mark_normal_dirty()
        if keep and params is not None and self._live_decal_projector is not None:
            self._commit_live_projector(params, self._live_decal_projector)
        self.set_status("Decal transform applied" if keep else "Decal transform cancelled")
        self._decal_transform_mode = None
        self._decal_transform_axis = None
        self._decal_transform_anchor = None
        self._decal_transform_last_hit = None
        self._live_decal_projector = None
        self.mark_normal_dirty()

    def transform_decal_with_pointer(self, dx: float, dy: float) -> bool:
        """Apply ordinary cursor motion to the active decal transform.

        Unconstrained movement follows real surface hits rather than adding UV
        deltas. That is what lets a decal travel over an edge onto another face
        even when the two faces are separated in the atlas.
        """
        params = self.selected_decal
        mode = self._decal_transform_mode
        if params is None or mode is None:
            return False

        dx = float(np.clip(dx, -_MAX_DRAG_STEP, _MAX_DRAG_STEP))
        dy = float(np.clip(dy, -_MAX_DRAG_STEP, _MAX_DRAG_STEP))
        axis = self._decal_transform_axis
        if mode == "move":
            if axis is None:
                hit = self.surface_hit_at(self._mouse)
                world_hit = self.world_surface_hit_at(self._mouse)
                if hit is None or world_hit is None:
                    return False
                uv, face = hit
                self._decal_transform_last_hit = (uv, int(face))
                image = self.decal_images.get(params.path)
                if image is not None:
                    self._set_live_decal_projector(image, world_hit[0], params, int(face))
            else:
                _, _, width, height = self.viewport_rect
                if axis == "x":
                    params.center_u = float(np.clip(
                        params.center_u + dx * float(self.wnd.pixel_ratio) / max(width, 1),
                        0.0, 1.0,
                    ))
                else:
                    params.center_v = float(np.clip(
                        params.center_v - dy * float(self.wnd.pixel_ratio) / max(height, 1),
                        0.0, 1.0,
                    ))
                if self._pick_geometry is not None:
                    vertices, faces, uvs = self._pick_geometry
                    point = surface_at_uv(
                        vertices, faces, uvs, params.center_u, params.center_v
                    )
                    image = self.decal_images.get(params.path)
                    if point is not None and image is not None:
                        self._set_live_decal_projector(image, point, params)
        elif mode == "scale":
            # Up/right grows, down/left shrinks. Prefer vertical travel because
            # that is the natural one-finger gesture, but a horizontal stroke
            # still works when it is clearly the larger movement.
            primary = -dy if abs(dy) >= abs(dx) else dx
            factor = math.exp(primary * 0.012)
            if axis == "x":
                params.scale_x = float(np.clip(params.scale_x * factor, 0.02, 50.0))
            elif axis == "y":
                params.scale_y = float(np.clip(params.scale_y * factor, 0.02, 50.0))
            else:
                params.scale = float(np.clip(params.scale * factor, 0.002, 4.0))
            projector = self._live_decal_projector
            image = self.decal_images.get(params.path)
            if projector is not None and image is not None:
                self._set_live_decal_projector(image, projector["center"], params)
        else:
            # Horizontal motion turns naturally like a dial; vertical motion
            # is accepted too for a comfortable one-finger trackpad gesture.
            primary = dx if abs(dx) >= abs(dy) else -dy
            degrees = -primary * 0.5
            params.rotation = (float(params.rotation) + degrees) % 360.0
            projector = self._live_decal_projector
            if projector is not None:
                angle = math.radians(degrees)
                cosine, sine = math.cos(angle), math.sin(angle)
                right = np.asarray(projector["right"], dtype=np.float64)
                up = np.asarray(projector["up"], dtype=np.float64)
                projector["right"] = tuple(right * cosine + up * sine)
                projector["up"] = tuple(up * cosine - right * sine)
        return True

    # -- placing a decal by pointing at the mesh --------------------------

    def begin_decal_placement(self) -> None:
        """Pick the decal up: it now follows the cursor until a click drops it.

        The current placement is remembered, so cancelling puts it back rather
        than leaving it wherever the cursor last wandered.
        """
        params = self.selected_decal
        if params is None or not params.loaded():
            self.set_status("Import a normal map first", error=True)
            return
        if self.mesh is None:
            self.set_status("Load a mesh to place a decal on", error=True)
            return

        params.enabled = True
        self.decal_placing = True
        self._decal_anchor = (params.center_u, params.center_v)
        self.set_status("Placing decal - click on the mesh to drop it, Esc to cancel")
        self.follow_cursor_with_decal()

    def end_decal_placement(self, *, keep: bool) -> None:
        """Drop the decal where it is, or put it back where it came from."""
        if not self.decal_placing:
            return
        self.decal_placing = False

        params = self.selected_decal
        if not keep and self._decal_anchor is not None:
            if params is not None:
                params.center_u, params.center_v = self._decal_anchor
            self.mark_normal_dirty()
            self.set_status("Placement cancelled")
        elif params is not None:
            self.set_status(
                f"Decal placed at UV "
                f"{params.center_u:.3f}, {params.center_v:.3f}"
            )
        self._decal_anchor = None

    def follow_cursor_with_decal(self) -> bool:
        """Move the decal to the surface under the cursor. False if it missed.

        A miss leaves the decal where it was rather than snapping it somewhere
        arbitrary -- dragging off the silhouette and back should not lose the
        placement you were lining up.
        """
        hit = self.surface_hit_at(self._mouse)
        if hit is None:
            return False
        uv, _ = hit
        params = self.selected_decal
        if params is None:
            return False
        params.center_u, params.center_v = uv
        params.surface_face = int(hit[1])
        self.mark_normal_dirty()
        return True

    def surface_uv_at(self, mouse: tuple[float, float]) -> Optional[tuple[float, float]]:
        """The mesh's UV under a cursor position, or None if it points at sky."""
        hit = self.surface_hit_at(mouse)
        return None if hit is None else hit[0]

    def surface_hit_at(
        self, mouse: tuple[float, float]
    ) -> Optional[tuple[tuple[float, float], int]]:
        """The mesh's UV under a cursor position, and the triangle it hit."""
        ray = self._cursor_ray(mouse)
        if ray is None:
            return None
        vertices, faces, uvs = self._pick_geometry
        hit = ray_mesh_hit(*ray, vertices, faces)
        if hit is None:
            return None

        face, u, v, _ = hit
        corners = uvs[faces[face]]
        # Barycentric weights: the first corner takes what the other two leave.
        texel = corners[0] * (1.0 - u - v) + corners[1] * u + corners[2] * v
        return (float(texel[0]), float(texel[1])), int(face)

    def world_surface_hit_at(
        self, mouse: tuple[float, float]
    ) -> Optional[tuple[np.ndarray, int]]:
        """World-space cursor hit used by the live GPU decal projector."""
        ray = self._cursor_ray(mouse)
        if ray is None or self._pick_geometry is None:
            return None
        vertices, faces, _ = self._pick_geometry
        hit = ray_mesh_hit(*ray, vertices, faces)
        if hit is None:
            return None
        face, _, _, distance = hit
        point = np.asarray(ray[0]) + np.asarray(ray[1]) * float(distance)
        return point, int(face)

    def _cursor_ray(self, mouse: tuple[float, float]):
        """The ray a cursor position casts into the scene, in world space."""
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
        return screen_ray(inverse, ndc_x, ndc_y)

    def measure_uv_aspect(self, face: Optional[int] = None) -> float:
        """How far from square a UV rectangle lands on this mesh. 1.0 is square.

        See :func:`core.decal.uv_aspect`. Without a face, the whole mesh's own
        average -- which is what a decal sitting in the gutter has to go on.
        """
        if self._pick_geometry is None:
            return 1.0
        vertices, faces, uvs = self._pick_geometry
        return uv_aspect(vertices, faces, uvs, face)

    def refresh_decal_aspect(self) -> None:
        """Re-measure how stretched the surface is under each decal's centre.

        Derived rather than remembered. A layout can be dense in one island and
        stretched in the next, so the ratio that keeps a round decal round
        belongs to wherever each decal is *now* -- and storing the one measured
        when it was placed would go quietly wrong the moment it was moved onto
        another island, or the mesh was re-baked into a different atlas.
        """
        if self._pick_geometry is None:
            return
        centres = tuple(
            (params.center_u, params.center_v) for params in self.decals
        )
        if centres == self._aspect_centres and self._vao_token == self._aspect_token:
            return  # nothing has moved; the search is not free on a dense mesh

        _, faces, uvs = self._pick_geometry
        for params in self.decals:
            face = params.surface_face
            if not (0 <= face < len(faces)):
                face = face_at_uv(faces, uvs, params.center_u, params.center_v)
                params.surface_face = -1 if face is None else int(face)
            params.surface_aspect = self.measure_uv_aspect(face)
        self._aspect_centres = centres
        self._aspect_token = self._vao_token

    def _release_decal_textures(self) -> None:
        for texture in (*self.decal_textures.values(), *self.decal_thumbnail_textures.values()):
            try:
                self.gui.remove_texture(texture)
            except KeyError:
                pass
            texture.release()
        self.decal_textures.clear()
        self.decal_thumbnail_textures.clear()

    def _upload_decal_thumbnail(self, thumbnail: DecalThumbnail) -> None:
        """Upload only the small browser preview, never the source texture."""
        if thumbnail.path in self.decal_thumbnail_textures:
            return
        width, height = thumbnail.size
        # Match the row-0-at-bottom convention used by full decal textures.
        pixels = np.frombuffer(thumbnail.rgba, dtype=np.uint8).reshape(height, width, 4)
        texture = self.ctx.texture(
            thumbnail.size, 4, data=np.ascontiguousarray(np.flipud(pixels)).tobytes()
        )
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
        self.gui.register_texture(texture)
        self.decal_thumbnail_sizes[thumbnail.path] = thumbnail.size
        self.decal_thumbnail_textures[thumbnail.path] = texture

    def _upload_decal(self, image: DecalImage) -> None:
        """Put one decal image on the GPU, once, and keep it there.

        Cached by path and shared by every placement of it: the shelf shows the
        same textures the mesh is stamped with, and dragging one out six times
        uploads nothing further.
        """
        if image.path in self.decal_textures:
            return

        # Source images have already gone through Pillow's 8-bit RGBA decode,
        # so a 32-bit float GPU texture adds no source detail. Normalized u8
        # samples identically in the shader while using one quarter of the
        # upload bandwidth and memory (including its mip chain).
        pixels = np.clip(image.rgba() * 255.0 + 0.5, 0.0, 255.0).astype("u1")
        texture = self.ctx.texture(
            image.size, 4, data=np.ascontiguousarray(pixels).tobytes(), dtype="f1"
        )
        # Mipmaps because the decal is usually larger than the patch of atlas it
        # lands in, so minification without them aliases the fine detail a vent
        # is made of. Clamped rather than repeating: outside its rectangle the
        # decal contributes nothing, and the shader already tests for that.
        texture.build_mipmaps()
        texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
        # The Decal tab shows the shelf, and dragging one out places it.
        self.gui.register_texture(texture)
        self.decal_textures[image.path] = texture

    def _sync_decal(self) -> None:
        """Keep the normal map in step with the decal and the export resolution.

        Nothing is allocated until a decal actually exists -- an atlas-sized
        float target is real memory, and most sessions never place one.
        """
        self.refresh_decal_aspect()
        if self.tex_normal is None and not self.any_visible_decal_active():
            return
        if self._normal_resolution != self._decal_render_resolution():
            self._normal_dirty = True
        if self._normal_dirty:
            self._run_decal()

    def _decal_render_resolution(self) -> int:
        """Keep viewport compositing cheap; full size is reserved for output."""
        requested = int(self.controller.bake_params.resolution)
        if self._full_decal_render_requested:
            return requested
        return min(requested, DECAL_INTERACTIVE_RESOLUTION)

    def _run_decal(self) -> None:
        """Re-composite the normal map: one full-screen pass per decal.

        Each pass adds its decal's surface *gradient* into a float target with
        additive blending, and one last pass turns the total back into a normal
        map. Slopes rather than normals because they add up the way the surfaces
        do: a vent stamped across a panel line should read as both, where adding
        the encoded normals would average them towards flat and let the second
        decal rub out the first.
        """
        resolution = self._decal_render_resolution()
        self._ensure_normal_texture(resolution)
        assert self.normal_fbo is not None and self._slope_fbo is not None

        self.ctx.viewport = (0, 0, resolution, resolution)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self._slope_fbo.use()
        self._slope_fbo.clear(0.0, 0.0, 0.0, 0.0)

        # Additive, so the passes accumulate instead of overwriting.
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        placements = list(self.decals)
        if self._decal_transform_mode is not None and 0 <= self.decal_index < len(placements):
            del placements[self.decal_index]
        if not self._full_decal_render_requested:
            # The newest projectors are sampled from their source images in
            # the mesh shader, preserving detail at any on-screen size. Keep
            # only overflow/legacy decals in the working UV normal target.
            direct = [
                params for params in placements
                if params.projector_center is not None
                and params.active()
                and params.path in self.decal_textures
            ][-MAX_VIEW_PROJECTORS:]
            direct_ids = {id(params) for params in direct}
            placements = [params for params in placements if id(params) not in direct_ids]
        if self.dragging_decal_preview is not None:
            placements.append(self.dragging_decal_preview)
        for params in placements:
            texture = self.decal_textures.get(params.path)
            if not params.active() or texture is None:
                continue
            texture.use(0)
            if params.projector_center is not None:
                # Never send a world-projector placement through the legacy UV
                # rectangle path. During the first mesh/atlas handoff its VAO
                # can be absent for a frame; drawing nothing for that frame is
                # preferable to stamping a differently oriented decal that can
                # remain in the accumulated normal target.
                if self.decal_project_vao is not None:
                    program = self.decal_project_program
                    program["u_projectorCenter"].value = params.projector_center
                    program["u_projectorRight"].value = params.projector_right
                    program["u_projectorUp"].value = params.projector_up
                    program["u_projectorForward"].value = params.projector_forward
                    program["u_projectorSize"].value = params.projector_size
                    program["u_intensity"].value = float(params.intensity)
                    program["u_flipGreen"].value = 1.0 if params.flip_green else 0.0
                    program["u_falloff"].value = float(params.falloff)
                    self.decal_project_vao.render(moderngl.TRIANGLES)
                continue
            wrapped = self._wrapped_decal_vao(params.surface_face)
            for name, value in params.as_uniforms().items():
                self.decal_program[name].value = value
            # The authored UV island is already continuous across its internal
            # triangle edges, and the full-atlas pass handles it without any
            # unfolding ambiguity. Wrapped geometry adds only faces whose
            # authored UVs differ from the continuous surface chart.
            self.decal_vao.render(moderngl.TRIANGLES, vertices=3)
            if wrapped is not None:
                for name, value in params.as_uniforms().items():
                    self.decal_wrap_program[name].value = value
                wrapped.render(moderngl.TRIANGLES)
        self.ctx.disable(moderngl.BLEND)

        self.normal_fbo.use()
        self._slope_texture.use(0)
        self.encode_vao.render(moderngl.TRIANGLES, vertices=3)
        self._normal_dirty = False

    def _wrapped_decal_vao(self, face: int) -> Optional[moderngl.VertexArray]:
        """Triangle atlas geometry carrying continuous coordinates from face."""
        if self._pick_geometry is None or face < 0:
            return None
        cached = self._decal_wrap_cache.get(int(face))
        if cached is not None:
            return cached[1]
        if int(face) in self._decal_wrap_futures:
            return None
        vertices, faces, uvs = self._pick_geometry
        future = self._decal_library_executor.submit(
            wrapped_vertices, vertices, faces, uvs, int(face), return_layout=True
        )
        self._decal_wrap_futures[int(face)] = (self._vao_token, future)
        return None

    def _store_wrapped_decal(self, face: int, data, layout) -> None:
        """Create render-thread GL resources from a prepared surface chart."""
        if len(data) == 0:
            self._decal_wrap_cache[int(face)] = (None, None, layout)
            return
        blocks = data.reshape(-1, 3, data.shape[1])
        seam_faces = np.max(np.abs(blocks[:, :, :2] - blocks[:, :, 2:4]), axis=(1, 2)) > 1e-5
        data = np.ascontiguousarray(blocks[seam_faces].reshape(-1, data.shape[1]))
        if len(data) == 0:
            self._decal_wrap_cache[int(face)] = (None, None, layout)
            return
        vbo = self.ctx.buffer(data.tobytes())
        vao = self.ctx.vertex_array(
            self.decal_wrap_program,
            [(vbo, "2f 2f 4f", "in_atlas_uv", "in_surface_uv", "in_slope_transform")],
        )
        self._decal_wrap_cache[int(face)] = (vbo, vao, layout)

    def _surface_uv_on_wrap(
        self, anchor_face: int, target_face: int, uv: tuple[float, float]
    ) -> Optional[tuple[float, float]]:
        """Convert a hit on any connected face into the anchor's stable chart."""
        vao = self._wrapped_decal_vao(anchor_face)
        cached = self._decal_wrap_cache.get(int(anchor_face))
        if vao is None or cached is None or self._pick_geometry is None:
            return None
        surface = cached[2].get(int(target_face))
        if surface is None:
            return None  # disconnected or deliberately clipped backfold
        _, faces, uvs = self._pick_geometry
        atlas = uvs[faces[int(target_face)]]
        matrix = np.column_stack([atlas[1] - atlas[0], atlas[2] - atlas[0]])
        try:
            bary = np.linalg.solve(matrix, np.asarray(uv) - atlas[0])
        except np.linalg.LinAlgError:
            return None
        point = surface[0] + bary[0] * (surface[1] - surface[0]) \
            + bary[1] * (surface[2] - surface[0])
        return float(point[0]), float(point[1])

    def read_color_map(self) -> Optional[np.ndarray]:
        """The mask tree resolved to colour, or None if it has never rendered.

        Read back off the GPU rather than recomputed, so the PNG is exactly the
        pixels the viewport is showing.
        """
        self._full_texture_render_requested = True
        try:
            self._texture_dirty = True
            self._sync_texture()
            return self.compositor.read()
        finally:
            self._full_texture_render_requested = False
            self._texture_dirty = True

    def read_normal_map(self) -> Optional[np.ndarray]:
        """The composited normal map as an (n, n, 3) array, or None if flat.

        Read back off the GPU rather than recomputed on the CPU, so the PNG is
        exactly the pixels the viewport is lighting with.
        """
        if not self.any_decal_active() or self.normal_fbo is None:
            return None
        self._full_decal_render_requested = True
        try:
            if self._normal_resolution != self._decal_render_resolution():
                self._normal_dirty = True
            if self._normal_dirty:
                self._run_decal()
            resolution = self._normal_resolution
            return np.frombuffer(
                self.normal_fbo.read(components=3, dtype="f4"), dtype="f4"
            ).reshape(resolution, resolution, 3)
        finally:
            self._full_decal_render_requested = False

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
        self.mark_normal_dirty()  # a new atlas is a new set of UV aspects
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
        self.decal_project_vao = self.ctx.vertex_array(
            self.decal_project_program,
            [(vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv")],
            index_buffer=ibo,
            index_element_size=4,
        )

    def _release_mesh_vao(self) -> None:
        if self.mesh_vao is not None:
            self.mesh_vao.release()
            self.mesh_vao = None
        if self.decal_project_vao is not None:
            self.decal_project_vao.release()
            self.decal_project_vao = None
        for buffer in self._mesh_buffers:
            buffer.release()
        self._mesh_buffers = []
        for buffer, vao, _ in self._decal_wrap_cache.values():
            if vao is not None:
                vao.release()
            if buffer is not None:
                buffer.release()
        self._decal_wrap_cache.clear()
        for _, future in self._decal_wrap_futures.values():
            future.cancel()
        self._decal_wrap_futures.clear()

    def _ensure_textures(self, resolution: int) -> None:
        if self._texture_resolution == resolution:
            return
        for texture in (self.tex_curvature, self.tex_position):
            if texture is not None:
                texture.release()

        size = (resolution, resolution)
        self.tex_curvature = self.ctx.texture(size, 1, dtype="f4")
        self.tex_position = self.ctx.texture(size, 3, dtype="f4")
        for texture in (self.tex_curvature, self.tex_position):
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = False
            texture.repeat_y = False

        self._texture_resolution = resolution

    def _ensure_normal_texture(self, resolution: int) -> None:
        """The decal normal map's own target, sized by the bake resolution.

        Separate from :meth:`_ensure_textures` because the decal does not wait
        for a bake: this exists as soon as an image is imported, at whatever
        resolution the export is set to.
        """
        if self._normal_resolution == resolution and self.tex_normal is not None:
            return
        for owned in (self.tex_normal, self.normal_fbo,
                      self._slope_texture, self._slope_fbo):
            if owned is not None:
                owned.release()

        # Two channels and full float: a slope is unbounded, and several steep
        # decals overlapping can add up well past anything 0..1 could hold.
        self._slope_texture = self.ctx.texture((resolution, resolution), 2, dtype="f4")
        self._slope_fbo = self.ctx.framebuffer(color_attachments=[self._slope_texture])

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
                self._texture_dirty = True  # the tree reads the new bake

                if PREVIEW_MODES[self.preview_index].texture is None:
                    self.preview_index = 0
                timings = ", ".join(
                    f"{stage} {seconds:.2f}s" for stage, seconds in controller.timings.items()
                )
                self.set_status(f"Bake complete - {timings}")

    # -- the texture ------------------------------------------------------

    def _apply_icon(self) -> None:
        """Put the icon from metadata.json on the window, if there is one.

        Never fatal. The headless backend raises for want of a window to put it
        on, a platform can refuse it, and an icon is not worth failing to start
        over -- so anything that goes wrong here leaves the default icon and
        says so in the status bar rather than in a traceback.
        """
        if METADATA.icon is None:
            return
        try:
            self.wnd.set_icon(str(METADATA.icon))
        except NotImplementedError:
            pass  # headless, and there is no dock to show it in
        except Exception as exc:  # pragma: no cover - platform dependent
            self.set_status(f"Could not load the application icon: {exc}", error=True)

    def decal_outline(self):
        """The selected decal's border, as world-space points to draw through.

        Rebuilt only when the selection, its placement or the mesh changes.
        Each point of the border is a search through every triangle for the one
        covering it, which is cheap enough once and much too dear every frame.

        Nothing while a decal is being moved: it is following the cursor, the
        border would be rebuilt on each of those frames, and an outline around
        something already glued to the pointer says nothing anyway.
        """
        params = self.selected_decal
        if (params is None or self._pick_geometry is None or self.decal_placing
                or self._decal_transform_mode is not None):
            return None

        if params.projector_center is not None:
            center = np.asarray(params.projector_center, dtype=np.float64)
            right = np.asarray(params.projector_right, dtype=np.float64)
            up = np.asarray(params.projector_up, dtype=np.float64)
            width, height = params.projector_size
            corners = [
                center - right * width * 0.5 - up * height * 0.5,
                center + right * width * 0.5 - up * height * 0.5,
                center + right * width * 0.5 + up * height * 0.5,
                center - right * width * 0.5 + up * height * 0.5,
            ]
            return np.asarray(corners + [corners[0]])

        key = (params.key(), self._vao_token)
        if key != self._outline_key:
            vertices, faces, uvs = self._pick_geometry
            points = [
                surface_at_uv(vertices, faces, uvs, u, v)
                for u, v in outline_uvs(params)
            ]
            found = [point for point in points if point is not None]
            self._outline = np.asarray(found) if len(found) > 1 else None
            self._outline_key = key
        return self._outline

    def flash_light_gizmo(self) -> None:
        """Show the light arrow in the viewport, and keep it up for a moment.

        Called while the rotation is being moved. Held on a timestamp rather
        than a flag so that letting go of the slider fades it out instead of
        snatching it away mid-look.
        """
        self._light_hint_at = time.monotonic()

    def light_hint_alpha(self, now: float | None = None) -> float:
        """How solid the light arrow should be drawn right now, 0 to 1."""
        elapsed = (time.monotonic() if now is None else now) - self._light_hint_at
        if elapsed < 0.0 or elapsed >= LIGHT_HINT_SECONDS:
            return 0.0
        remaining = LIGHT_HINT_SECONDS - elapsed
        return float(min(1.0, remaining / LIGHT_HINT_FADE))

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
        self.end_rename()
        self.mark_texture_dirty()

    def remove_texture(self) -> None:
        """Drop the active texture, and fall back to its neighbour."""
        if self.texture is None:
            return
        name = describe(self.textures.pop(self.texture_index))
        self.texture_index = min(self.texture_index, len(self.textures) - 1)
        self.texture_path = ()
        self.end_rename()
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

    def begin_rename(self, path: tuple[str, ...]) -> None:
        """Put a row of the tree into rename mode."""
        self.renaming_path = tuple(path)
        self.renaming_opened = True

    def end_rename(self) -> None:
        self.renaming_path = None
        self.renaming_opened = False

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
        self.end_rename()
        while self.texture_path:
            try:
                slot_at(self.texture, self.texture_path)
                break
            except KeyError:
                self.texture_path = self.texture_path[:-1]
        self.mark_texture_dirty()

    def _sync_texture(self) -> None:
        """Re-composite when the texture or the bake behind it has changed.

        Driven by comparing the tree with what was last drawn rather than by a
        flag alone. Every control in the Texture tab edits the tree in place,
        and one that forgot to announce it would be a control that silently did
        nothing -- so the composite follows the data, and the flag is only there
        to force a redraw when something outside the tree moves.
        """
        if self.texture is None:
            self._composited_key = None
            return
        if self.tex_curvature is None or self.tex_position is None:
            return  # the masks read the bake; nothing to stand on yet

        render_resolution = int(self._texture_resolution)
        if not self._full_texture_render_requested:
            render_resolution = min(render_resolution, DECAL_INTERACTIVE_RESOLUTION)
        key = (
            texture_key(self.texture),
            render_resolution,
            self._uploaded_maps_version,
        )
        if not self._texture_dirty and key == self._composited_key:
            return

        self.compositor.render(
            self.texture, render_resolution,
            self.tex_curvature, self.tex_position,
        )
        self._composited_key = key
        self._texture_dirty = False
        self._sync_texture_thumbnails()
        self._check_texture_variation()

    def _check_texture_variation(self) -> None:
        """Notice a texture that has come out one flat colour.

        A mask whose every texel lands on the same side -- edge wear on a mesh
        with nothing for it to grip, a threshold past the end of the field --
        renders the whole model in one colour, which reads as broken rather
        than as empty. The panel says which it is; this is how it knows.

        Measured on the root's 96px preview rather than the map itself: it is
        already on the GPU, it only happens when the tree changes, and a flat
        map is flat at any size.
        """
        preview = self.compositor.thumbnail(())
        if preview is None:
            self.texture_is_flat = False
            return
        pixels = np.frombuffer(preview.read(), dtype=np.uint8).reshape(-1, 4)[:, :3]
        self.texture_is_flat = bool(np.ptp(pixels, axis=0).max() <= 1)

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

    def _current_texture(self) -> Optional[moderngl.Texture]:
        name = PREVIEW_MODES[self.preview_index].texture
        return {
            "curvature": self.tex_curvature,
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
        sidebar = self.sidebar_pixels
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

    def _ensure_scene_targets(self) -> None:
        """Build the over-white scene target and the glow chain to fit the window.

        Half floats rather than full: the scene is stored as a fraction of
        :data:`HDR_SCALE`, so nothing here ever leaves [0, 1] and half precision
        has depth to spare over a display's eight bits. It also halves the
        bandwidth of a pass that touches every pixel four times.
        """
        size = tuple(int(max(1, n)) for n in self.wnd.buffer_size)
        if size == self._target_size and self._scene_fbo is not None:
            return

        for owned in (self._scene_fbo, self._scene_tex, self._scene_depth,
                      *self._bloom_fbos, *self._bloom_texs):
            if owned is not None:
                owned.release()

        self._scene_tex = self.ctx.texture(size, 4, dtype="f2")
        self._scene_tex.repeat_x = self._scene_tex.repeat_y = False
        self._scene_depth = self.ctx.depth_renderbuffer(size)
        self._scene_fbo = self.ctx.framebuffer([self._scene_tex], self._scene_depth)

        # Two targets at a fraction of the size: bright-pass into the first,
        # blur across into the second, blur down and back into the first.
        small = tuple(max(1, n // BLOOM_DIVISOR) for n in size)
        self._bloom_texs = [self.ctx.texture(small, 4, dtype="f2") for _ in range(2)]
        for texture in self._bloom_texs:
            texture.repeat_x = texture.repeat_y = False
        self._bloom_fbos = [self.ctx.framebuffer([t]) for t in self._bloom_texs]
        self._target_size = size

    def _present(self) -> None:
        """Glow the over-white parts of the scene, then bring it to the window."""
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)

        width, height = self._target_size
        small = self._bloom_fbos[0].size

        self._bloom_fbos[0].use()
        self.ctx.viewport = (0, 0, *small)
        self._scene_tex.use(0)
        self.bright_program["u_texel"].value = (1.0 / width, 1.0 / height)
        self.bright_vao.render(moderngl.TRIANGLES, vertices=3)

        for source, target, direction in (
            (0, 1, (1.0 / small[0], 0.0)),
            (1, 0, (0.0, 1.0 / small[1])),
        ):
            self._bloom_fbos[target].use()
            self._bloom_texs[source].use(0)
            self.blur_program["u_direction"].value = direction
            self.blur_vao.render(moderngl.TRIANGLES, vertices=3)

        # wnd.use() rather than ctx.screen.use(): the headless backend renders
        # into its own framebuffer, and the self-test relies on that.
        self.wnd.use()
        self.ctx.viewport = (0, 0, width, height)
        self._scene_tex.use(0)
        self._bloom_texs[0].use(1)
        self.tonemap_vao.render(moderngl.TRIANGLES, vertices=3)

    def _draw_scene(self) -> None:
        self._ensure_scene_targets()
        self._scene_fbo.use()
        self.ctx.viewport = (0, 0, *self._target_size)
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
            self._present()  # the backdrop still has to reach the window
            return

        # Blending, so a texture with alpha below 1 reads as see-through. Depth
        # is still written, so a concave model can hide its own far side -- a
        # proper transparent pass would sort by depth, which is more machinery
        # than a preview of an alpha channel is worth.
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        mode = PREVIEW_MODES[self.preview_index]
        texture = self._current_texture()
        if mode.texture is not None and texture is None:
            mode = FLAT_MODE  # that map does not exist yet
        (texture or self._blank_texture).use(0)

        # The decal lights the mesh in every mode, the way a normal map in a
        # material does; with no decal this is the flat 1x1, which changes
        # nothing and costs one texture unit.
        decal_lit = self.any_visible_decal_active() and self.tex_normal is not None
        (self.tex_normal if decal_lit else self._flat_normal).use(1)

        # The surface channels the texture carries: metal, rough, alpha, glow.
        material = self.compositor.material_texture
        (material or self._flat_material).use(2)

        view = self.camera.matrix
        mvp = self.camera.projection_matrix * view

        self.preview_program["u_mvp"].write(_mat_bytes(mvp))
        self.preview_program["u_mode"].value = mode.shader_mode
        self.preview_program["u_lighting"].value = 1.0 if self.lighting else 0.0
        self.preview_program["u_useNormalMap"].value = 1.0 if decal_lit else 0.0
        self.preview_program["u_useMaterial"].value = 1.0 if material else 0.0
        self.preview_program["u_worldColor"].value = tuple(
            float(c) for c in self.world.color
        )
        self.preview_program["u_worldStrength"].value = float(self.world.strength)
        eye = self.camera.eye
        self.preview_program["u_eye"].value = (eye.x, eye.y, eye.z)
        self.preview_program["u_checkerScale"].value = self.checker_scale

        projector = self._live_decal_projector
        live_texture = None if projector is None else self.decal_textures.get(projector["path"])
        live = projector is not None and live_texture is not None
        self.preview_program["u_useLiveDecal"].value = 1.0 if live else 0.0
        if live:
            live_texture.use(3)
            self.preview_program["u_projectorCenter"].value = projector["center"]
            self.preview_program["u_projectorRight"].value = projector["right"]
            self.preview_program["u_projectorUp"].value = projector["up"]
            self.preview_program["u_projectorForward"].value = projector["forward"]
            self.preview_program["u_projectorSize"].value = projector["size"]
            self.preview_program["u_projectorIntensity"].value = projector["intensity"]
            self.preview_program["u_projectorFlipGreen"].value = projector["flip_green"]

        view_projectors = []
        for index, params in enumerate(self.decals):
            if params.projector_center is None or not params.active():
                continue
            if self._decal_transform_mode is not None and index == self.decal_index:
                continue
            decal_texture = self.decal_textures.get(params.path)
            if decal_texture is not None:
                view_projectors.append((params, decal_texture))
        view_projectors = view_projectors[-MAX_VIEW_PROJECTORS:]
        self.preview_program["u_viewProjectorCount"].value = len(view_projectors)
        if view_projectors:
            centers, rights, ups, forwards, sizes, intensities, flips = [], [], [], [], [], [], []
            for unit, (params, decal_texture) in enumerate(view_projectors, start=4):
                decal_texture.use(unit)
                centers.append(params.projector_center)
                rights.append(params.projector_right)
                ups.append(params.projector_up)
                forwards.append(params.projector_forward)
                sizes.append(params.projector_size)
                intensities.append(float(params.intensity))
                flips.append(1.0 if params.flip_green else 0.0)
            program = self.preview_program
            def padded(values, width=1):
                shape = (MAX_VIEW_PROJECTORS,) if width == 1 else (MAX_VIEW_PROJECTORS, width)
                result = np.zeros(shape, dtype="f4")
                result[:len(values)] = np.asarray(values, dtype="f4")
                return result.tobytes()

            # ModernGL validates writes against the declared GLSL array size,
            # not u_viewProjectorCount, so every upload must contain all slots.
            program["u_viewProjectorCenters"].write(padded(centers, 3))
            program["u_viewProjectorRights"].write(padded(rights, 3))
            program["u_viewProjectorUps"].write(padded(ups, 3))
            program["u_viewProjectorForwards"].write(padded(forwards, 3))
            program["u_viewProjectorSizes"].write(padded(sizes, 2))
            program["u_viewProjectorIntensities"].write(padded(intensities))
            program["u_viewProjectorFlipGreens"].write(padded(flips))

        # Anchored to the model, not to the camera: orbiting moves your view of
        # the lighting rather than the lighting itself, which is the only way
        # "put the light over there" can mean anything.
        self.preview_program["u_lightDir"].value = self.world.light_direction()

        self.ctx.wireframe = self.wireframe
        self.mesh_vao.render(moderngl.TRIANGLES)
        self.ctx.wireframe = False
        self.ctx.disable(moderngl.BLEND)
        self._present()

    # -- moderngl-window hooks --------------------------------------------

    def on_render(self, time: float, frame_time: float) -> None:
        self._drain_scroll()
        self.controller.pump()
        self._sync_bake_outputs()
        self._pump_decal_images()
        self._pump_decal_wraps()

        self._sync_texture()
        self._sync_decal()
        self._sync_projection()
        self._draw_scene()

        # The scene draws into its own rect; the interface is drawn over the
        # whole window, and ImGui renders through whatever viewport is bound.
        self.ctx.viewport = (0, 0, *self.wnd.buffer_size)

        self.gui.sync_mouse_buttons()
        imgui.new_frame()
        draw_panel(self)
        self.draw_dragged_decal_cursor()
        if self.show_gizmo and self.mesh is not None:
            gizmo.draw(self.camera, self._gizmo_center(), self.ui_pixel_scale,
                       self._mouse)
        outline = self.decal_outline()
        if outline is not None:
            decal_gizmo.draw(
                self.camera, outline, self.viewport_rect,
                self.wnd.buffer_size[1], self.ui_pixel_scale,
            )

        alpha = self.light_hint_alpha()
        if alpha > 0.0 and self.mesh is not None:
            light_gizmo.draw(
                self.camera,
                self.world.light_direction(),
                tuple(float(v) for v in self.mesh.bounds.mean(axis=0)),
                self.mesh_info.scale if self.mesh_info else 1.0,
                self.viewport_rect,
                self.wnd.buffer_size[1],
                self.ui_pixel_scale,
                alpha,
            )
        imgui.render()
        self.gui.render(imgui.get_draw_data())
        # After the frame, so it reflects what this frame's widgets asked for --
        # plus the sidebar's edge, which is the app's own to hit-test.
        # Not while another gesture owns the mouse: an orbit that happens to
        # pass over the edge should not flicker the cursor at it.
        edge = self._drag_owner == "sidebar" or (
            self._drag_owner is None and self.over_sidebar_edge(self._mouse)
        )
        self.gui.sync_mouse_cursor(
            imgui.MouseCursor_.resize_ew if edge else None
        )

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
        keys = self.wnd.keys
        if key in (getattr(keys, "LEFT_SHIFT", None), getattr(keys, "RIGHT_SHIFT", None)):
            self._shift_down = action != keys.ACTION_RELEASE
        if imgui.get_io().want_capture_keyboard:
            return

        if action != keys.ACTION_PRESS:
            return

        if key == keys.ESCAPE:
            if self._decal_transform_mode is not None:
                self.end_decal_transform(keep=False)
                return
            if self.decal_placing:
                self.end_decal_placement(keep=False)
                return

        if self._decal_transform_mode is not None:
            if key in (getattr(keys, "ENTER", None), getattr(keys, "NUMPAD_ENTER", None)):
                self.end_decal_transform(keep=True)
            elif key == getattr(keys, "X", None):
                self.constrain_decal_transform("x")
            elif key == getattr(keys, "Y", None):
                self.constrain_decal_transform("y")
            elif key == getattr(keys, "G", None):
                self.begin_decal_transform("move")
            elif key == getattr(keys, "S", None):
                self.begin_decal_transform("scale")
            elif key == getattr(keys, "R", None):
                self.begin_decal_transform("rotate")
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
            self.request_export()
        elif key == keys.F and self.mesh is not None:
            self.camera.frame(self.mesh.bounds.mean(axis=0), self.mesh_info.scale)
        elif key == keys.W:
            self.wireframe = not self.wireframe
        elif key == keys.L:
            self.lighting = not self.lighting
        elif key == getattr(keys, "G", None):
            self.begin_decal_transform("move")
        elif key == getattr(keys, "S", None):
            self.begin_decal_transform("scale")
        elif key == getattr(keys, "R", None):
            self.begin_decal_transform("rotate")
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

        if self.dragging_decal is not None:
            self._update_dragged_decal_preview()

        if self._decal_transform_mode is not None:
            self.transform_decal_with_pointer(dx, dy)

        # A decal being placed rides the cursor. Not over the panel, though:
        # crossing it on the way to the mesh must not fling the decal about.
        if self.decal_placing and not imgui.get_io().want_capture_mouse:
            self.follow_cursor_with_decal()

    def on_mouse_press_event(self, x: int, y: int, button: int) -> None:
        self._mouse = (float(x), float(y))
        self._modifiers = self.wnd.modifiers

        # The sidebar's edge is claimed before ImGui is even told about the
        # press. Half the band lies over the panel's last few pixels -- which
        # is where a hand aims for an edge -- and ImGui captures anything
        # inside its window, so asking it first means the edge only ever works
        # from the outside. Not forwarding the press also keeps whatever
        # control sits near the border from reacting to a grab meant for this.
        if button == self.wnd.mouse.left and self.over_sidebar_edge(self._mouse):
            self._drag_owner = "sidebar"
            return

        self.gui.mouse_press_event(x, y, button)
        #: Where the press landed, so a release can tell a click from a drag.
        self._press_at = self._mouse

        # Decide once, here, what this gesture is for. Deciding per-event instead
        # lets a drag that began on a slider grab the camera the moment the
        # pointer leaves the panel, which is a jarring jump mid-drag.
        if imgui.get_io().want_capture_mouse:
            self._drag_owner = "ui"
            return

        if self._decal_transform_mode is not None:
            self._drag_owner = "decal_transform"
            self.end_decal_transform(keep=button == self.wnd.mouse.left)
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
        if self._drag_owner == "sidebar":
            # ImGui never saw the press, so it must not see the release either.
            self._drag_owner = None
            self.save_prefs()  # on release, not on every pixel of the drag
            return

        self._mouse = (float(x), float(y))
        dropped = self._drop_dragged_decal()
        self.gui.mouse_release_event(x, y, button)

        # A click on the model selects the decal under it, or deselects. On
        # release rather than on press, and only if the pointer stayed put:
        # otherwise every orbit that happened to start over a decal would
        # select it, and every one that started beside it would clear it.
        if (not dropped and button == self.wnd.mouse.left
                and self._drag_owner == "camera" and self._was_a_click()):
            self.select_decal_at(self._mouse)

        self._drag_owner = None

    def _was_a_click(self, slack: float = 3.0) -> bool:
        """Whether the pointer stayed where it was pressed, near enough."""
        if self._press_at is None:
            return False
        return (abs(self._mouse[0] - self._press_at[0]) <= slack
                and abs(self._mouse[1] - self._press_at[1]) <= slack)

    def select_decal_at(self, mouse: tuple[float, float]) -> Optional[int]:
        """Select whichever decal the cursor is over, or none if it is over
        bare surface. Returns the index chosen."""
        hit = self.surface_hit_at(mouse)
        index = None
        if hit is not None:
            uv, face = hit
            world_hit = self.world_surface_hit_at(mouse)
            for candidate in range(len(self.decals) - 1, -1, -1):
                params = self.decals[candidate]
                if params.projector_center is not None and world_hit is not None:
                    offset = world_hit[0] - np.asarray(params.projector_center)
                    local_x = float(np.dot(offset, params.projector_right))
                    local_y = float(np.dot(offset, params.projector_up))
                    width, height = params.projector_size
                    if abs(local_x) <= width * 0.5 and abs(local_y) <= height * 0.5:
                        index = candidate
                        break
                    continue
                point = uv
                if params.surface_face >= 0:
                    point = self._surface_uv_on_wrap(params.surface_face, face, uv)
                    if point is None:
                        continue
                if decal_contains(params, *point):
                    index = candidate
                    break
        self.select_decal(index)
        if index is None:
            self.set_status("No decal there")
        else:
            self.set_status(f"Selected {Path(self.decals[index].path).name}")
        return index

    def begin_decal_drag(self, path: str | Path) -> None:
        """Start dragging one off the shelf. It lands where it is let go."""
        path = Path(path)
        if self.dragging_decal == path:
            return
        self.dragging_decal = path
        self.request_decal_image(path)
        self._clear_dragged_decal_preview()

    def _set_live_decal_projector(
        self, image: DecalImage, point, params: DecalParams,
        face: Optional[int] = None,
    ) -> None:
        """Describe a projector tangent to the surface under its centre."""
        existing = self._live_decal_projector
        preserve_axes = (
            face is None
            and
            self._decal_transform_mode is not None
            and existing is not None
            and existing["path"] == image.path
        )
        if face is not None and self._pick_geometry is not None:
            vertices, faces, _ = self._pick_geometry
            triangle = vertices[faces[int(face)]]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            continuing = existing is not None and existing["path"] == image.path
            if continuing:
                old_normal = np.asarray(existing["forward"], dtype=np.float64)
                old_normal /= max(float(np.linalg.norm(old_normal)), 1e-12)
                old_right = np.asarray(existing["right"], dtype=np.float64)
                old_up = np.asarray(existing["up"], dtype=np.float64)
                rotated_right = self._transport_surface_axis(
                    old_right, old_normal, normal, old_up
                )
                # Rebuild up exactly orthogonal after transport so thousands
                # of pointer updates cannot accumulate numerical skew.
                rotated_up = np.cross(normal, rotated_right)
                if np.dot(rotated_up, self._transport_surface_axis(
                    old_up, old_normal, normal, old_right
                )) < 0.0:
                    rotated_right = -rotated_right
                    rotated_up = -rotated_up
            else:
                camera_right, camera_up, _ = self.camera._axes()
                right_hint = np.asarray(
                    (float(camera_right.x), float(camera_right.y), float(camera_right.z))
                )
                tangent = right_hint - normal * float(np.dot(right_hint, normal))
                if np.linalg.norm(tangent) < 1e-8:
                    up_hint = np.asarray(
                        (float(camera_up.x), float(camera_up.y), float(camera_up.z))
                    )
                    tangent = up_hint - normal * float(np.dot(up_hint, normal))
                tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
                bitangent = np.cross(normal, tangent)
                angle = math.radians(float(params.rotation))
                cosine, sine = math.cos(angle), math.sin(angle)
                rotated_right = tangent * cosine + bitangent * sine
                rotated_up = bitangent * cosine - tangent * sine
            projected_right = tuple(float(value) for value in rotated_right)
            projected_up = tuple(float(value) for value in rotated_up)
            projected_forward = tuple(float(value) for value in normal)
        elif preserve_axes:
            projected_right = tuple(existing["right"])
            projected_up = tuple(existing["up"])
            projected_forward = tuple(existing["forward"])
        else:
            right, up, forward = self.camera._axes()
            angle = math.radians(float(params.rotation))
            cosine, sine = math.cos(angle), math.sin(angle)
            rotated_right = right * cosine + up * sine
            rotated_up = up * cosine - right * sine
            projected_right = (
                float(rotated_right.x), float(rotated_right.y), float(rotated_right.z)
            )
            projected_up = (
                float(rotated_up.x), float(rotated_up.y), float(rotated_up.z)
            )
            projected_forward = (float(forward.x), float(forward.y), float(forward.z))
        base = max((self.mesh_info.scale if self.mesh_info else 1.0) * params.scale, 1e-6)
        self._live_decal_projector = {
            "path": image.path,
            "center": tuple(float(value) for value in point),
            "right": projected_right,
            "up": projected_up,
            "forward": projected_forward,
            "size": (
                base * max(params.scale_x, 1e-6),
                base * max(params.scale_y, 1e-6) / max(image.aspect, 1e-6),
            ),
            "intensity": float(params.intensity),
            "flip_green": 1.0 if params.flip_green else 0.0,
        }

    @staticmethod
    def _transport_surface_axis(axis, old_normal, new_normal, fallback_axis):
        """Carry an in-plane axis across a bend with no extra twist."""
        cosine = float(np.clip(np.dot(old_normal, new_normal), -1.0, 1.0))
        cross = np.cross(old_normal, new_normal)
        sine = float(np.linalg.norm(cross))
        if sine > 1e-8:
            rotation_axis = cross / sine
            transported = (
                axis * cosine
                + np.cross(rotation_axis, axis) * sine
                + rotation_axis * np.dot(rotation_axis, axis) * (1.0 - cosine)
            )
        elif cosine >= 0.0:
            transported = np.asarray(axis, dtype=np.float64)
        else:
            # Opposite normals have no unique shortest rotation. The other
            # decal axis supplies a stable hinge instead of choosing at random.
            rotation_axis = np.asarray(fallback_axis, dtype=np.float64)
            rotation_axis /= max(float(np.linalg.norm(rotation_axis)), 1e-12)
            transported = 2.0 * rotation_axis * np.dot(rotation_axis, axis) - axis
        transported -= new_normal * float(np.dot(transported, new_normal))
        length = float(np.linalg.norm(transported))
        if length < 1e-10:
            transported = np.cross(new_normal, fallback_axis)
            length = float(np.linalg.norm(transported))
        return transported / max(length, 1e-12)

    @staticmethod
    def _commit_live_projector(params: DecalParams, projector: dict) -> None:
        """Persist exactly the transform that the interactive shader displayed."""
        params.projector_center = tuple(projector["center"])
        params.projector_right = tuple(projector["right"])
        params.projector_up = tuple(projector["up"])
        params.projector_forward = tuple(projector["forward"])
        params.projector_size = tuple(projector["size"])

    def _update_dragged_decal_preview(self) -> bool:
        """Move the view-aligned GPU projector to the cursor's surface hit."""
        if self.dragging_decal is None:
            return False
        hit = self.world_surface_hit_at(self._mouse)
        if hit is None:
            self._clear_dragged_decal_preview()
            return False

        image = self.decal_images.get(str(self.dragging_decal))
        if image is None:
            self._clear_dragged_decal_preview()
            return False
        point, face = hit
        self._set_live_decal_projector(
            image, point, DecalParams(path=image.path, image_aspect=image.aspect), face
        )
        return True

    def _clear_dragged_decal_preview(self) -> None:
        self._live_decal_projector = None
        if self.dragging_decal_preview is not None:
            self.dragging_decal_preview = None
            self.mark_normal_dirty()

    def dragged_decal_cursor_rect(self) -> Optional[tuple[float, float, float, float]]:
        """Screen rectangle for an off-surface drag thumbnail, or None.

        The thumbnail belongs only to empty space inside the 3D viewport. Over
        the mesh the projected normal-map preview is more useful; over the
        sidebar the library thumbnail already shows what the hand is holding.
        """
        if self.dragging_decal is None or self._live_decal_projector is not None:
            return None

        ratio = float(self.wnd.pixel_ratio)
        rect_x, _, rect_width, rect_height = self.viewport_rect
        navbar = float(panel.NAVBAR_HEIGHT * self.ui_pixel_scale)
        left = rect_x / ratio
        right = (rect_x + rect_width) / ratio
        top = navbar / ratio
        bottom = (navbar + rect_height) / ratio
        mouse_x, mouse_y = self._mouse
        if not (left <= mouse_x <= right and top <= mouse_y <= bottom):
            return None

        size = self.decal_thumbnail_sizes.get(str(self.dragging_decal))
        if size is None:
            return None
        width, height = size
        longest = 72.0 * self.ui_pixel_scale / ratio
        if width >= height:
            thumb_width = longest
            thumb_height = longest * height / max(width, 1)
        else:
            thumb_height = longest
            thumb_width = longest * width / max(height, 1)

        gap = 14.0 * self.ui_pixel_scale / ratio
        x1 = min(mouse_x + gap, right - thumb_width - 4.0)
        y1 = min(mouse_y + gap, bottom - thumb_height - 4.0)
        return x1, y1, x1 + thumb_width, y1 + thumb_height

    def draw_dragged_decal_cursor(self) -> None:
        """Show what is being carried while a shelf drag is over empty space."""
        rect = self.dragged_decal_cursor_rect()
        if rect is None or self.dragging_decal is None:
            return
        texture = self.decal_thumbnail_textures.get(str(self.dragging_decal))
        if texture is None:
            return

        x1, y1, x2, y2 = rect
        draw_list = imgui.get_foreground_draw_list()
        padding = 3.0
        draw_list.add_rect_filled(
            imgui.ImVec2(x1 - padding, y1 - padding),
            imgui.ImVec2(x2 + padding, y2 + padding),
            imgui.get_color_u32(imgui.ImVec4(0.08, 0.09, 0.11, 0.86)),
            5.0,
        )
        draw_list.add_image(
            imgui.ImTextureRef(texture.glo),
            imgui.ImVec2(x1, y1), imgui.ImVec2(x2, y2),
            col=imgui.get_color_u32(imgui.ImVec4(1.0, 1.0, 1.0, 0.82)),
        )
        draw_list.add_rect(
            imgui.ImVec2(x1 - padding, y1 - padding),
            imgui.ImVec2(x2 + padding, y2 + padding),
            imgui.get_color_u32(imgui.ImVec4(0.30, 0.90, 1.00, 0.95)),
            5.0, thickness=1.5,
        )

    def _drop_dragged_decal(self) -> bool:
        """Finish a drag out of the shelf, if there is one. True if it landed.

        A drop that misses the model is a drag abandoned, not a decal at the
        origin of the atlas: the shelf is where it came from and where it stays.
        """
        path, self.dragging_decal = self.dragging_decal, None
        if path is None:
            return False

        hit = self.surface_hit_at(self._mouse)
        projector = self._live_decal_projector
        self._clear_dragged_decal_preview()
        if hit is None:
            self.set_status("Dropped nowhere - drag a decal onto the model")
            return False
        uv, face = hit
        if str(path) not in self.decal_images:
            self.request_decal_image(path)
            self._pending_decal_drop = (path, uv, face)
            self.set_status(f"Loading {path.name}...")
        else:
            index = self.add_decal(path, center=uv, face=face)
            if index is not None and projector is not None:
                self._commit_live_projector(self.decals[index], projector)
                self.mark_normal_dirty()
        return True

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
        self._mouse = (float(x), float(y))

        if self.dragging_decal is not None:
            self._update_dragged_decal_preview()

        if self._drag_owner == "sidebar":
            # The width is in unscaled units, so the travel has to come back
            # through the scale to mean the same thing at any of them.
            self.set_sidebar_width(
                self.sidebar_width + dx * float(self.wnd.pixel_ratio) / self.ui_pixel_scale
            )
            return

        self.gui.mouse_drag_event(x, y, dx, dy)
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
        if self._decal_transform_mode is not None:
            # Modal decal transforms use ordinary one-finger pointer motion.
            # Do not orbit the camera underneath one if a second finger happens
            # to touch the pad before the transform is confirmed.
            return
        self.gui.mouse_scroll_event(x_offset, y_offset)
        if imgui.get_io().want_capture_mouse:
            return

        # Blender-style trackpad navigation: translating two fingers orbits,
        # Shift translates the view, and the distinct native magnification
        # gesture handled by :meth:`on_pinch_zoom` changes zoom.
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
        elif self._shift_down or modifiers.shift:
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
        self._decal_library_executor.shutdown(wait=False, cancel_futures=True)
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

    def request_export(self) -> None:
        """Ask to export. Where to put them is the only question, so the panel
        answers it with a folder chooser and the export follows.

        A flag rather than the dialog itself: native dialogs belong to the
        interface layer, and this is reached from the keyboard as well as from
        the File menu -- both have to mean exactly the same thing.
        """
        self.export_pending = True

    def export(self) -> None:
        """Write every map that is switched on and exists, in one go.

        A standard PBR set: colour and its metallic and roughness from the mask
        tree, normals from the decals, occlusion from the bake. They come from
        independent halves of the app -- a decal is placed in UV space and owes
        the bake nothing -- so each is written if it is there and skipped if it
        is not.
        """
        controller = self.controller
        normal = self.read_normal_map()
        color = self.read_color_map()
        material = self.compositor.read_material() if color is not None else None
        occlusion = controller.occlusion_map

        if not self.exportable():
            self.set_status(
                "Nothing to export - bake a mesh, import a decal, or switch a "
                "map back on under Settings",
                error=True,
            )
            return

        stem = Path(self.mesh_info.path).stem if self.mesh_info else "mesh"
        target = Path(self.export_dir) / stem
        try:
            written = export_maps(
                target,
                color=color,
                normal=normal,
                material=material,
                occlusion=occlusion,
                maps=self.enabled_maps,
                bits=self.export_bits,
            )
        except Exception:
            self.set_status(traceback.format_exc(limit=3), error=True)
            return

        names = ", ".join(path.name for path in written)
        self.set_status(f"Wrote {names} to {target}")
