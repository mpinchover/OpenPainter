"""How the viewport lights a surface: metallic, roughness, emission, world.

These are the things a colour slot promises and the preview has to deliver. The
checks are deliberately about behaviour rather than pixel values -- a metal is
darker than the same colour as a dielectric, a rough metal is still a metal, a
bright emission stays its own colour -- because the exact numbers belong to the
shading model and are free to move.

Everything renders through the real window config, headless, so what is measured
is the frame the user would be looking at.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh
from imgui_bundle import imgui

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.layers import MAX_EMISSION, ColorSlot  # noqa: E402
from render.viewport import (  # noqa: E402
    COMPASS,
    LIGHT_ELEVATION,
    LIGHT_HINT_FADE,
    LIGHT_HINT_SECONDS,
    MAX_WORLD_STRENGTH,
    WorldPrefs,
    bearing,
)
from ui import light_gizmo  # noqa: E402

#: Strongly coloured, so a colour that survives shading is obvious and a colour
#: that has been washed out is equally obvious.
RED = (0.85, 0.12, 0.05)


@pytest.fixture
def ball(tmp_path):
    """A sphere, UV-mapped. Curvature is irrelevant here; shape is not -- a
    sphere shows a highlight, and a flat quad facing the camera does not."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.35)
    longitude = np.arctan2(mesh.vertices[:, 1], mesh.vertices[:, 0]) / (2 * np.pi)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.stack([longitude + 0.5, mesh.vertices[:, 2] / 0.7 + 0.5], axis=1)
    )
    path = tmp_path / "ball.obj"
    mesh.export(path)
    return path


@pytest.fixture
def app(ball):
    import moderngl_window as mglw

    from render.viewport import MeshMapApp

    MeshMapApp.initial_mesh = str(ball)
    MeshMapApp.initial_resolution = 128
    try:
        instance = mglw.create_window_config_instance(MeshMapApp, args=["-wnd", "headless"])
    except Exception as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"no headless window available: {exc}")
    instance.show_gizmo = False  # nothing to confuse the silhouette with

    # The composite stands on the bake -- a mask reads curvature, so nothing is
    # drawn into the texture until the bake has landed, flat colour included.
    # The bake runs off the main thread and its result is picked up by a frame,
    # so this waits by rendering rather than by sleeping.
    instance.request_bake()
    deadline = time.monotonic() + 30.0
    while instance.tex_curvature is None and time.monotonic() < deadline:
        instance.on_render(0.0, 1 / 60.0)
        time.sleep(0.002)
    assert instance.tex_curvature is not None, instance.status

    yield instance
    instance.controller.release()
    imgui.destroy_context()
    MeshMapApp.initial_mesh = None


def frame(app, **surface) -> np.ndarray:
    """Render the sphere as one colour with the given surface, and read it back.

    Cropped to the 3D view. The panel and the bars are drawn into the same
    framebuffer, and they are full of warm pixels -- a highlight measured over
    the whole window can turn out to be a button.
    """
    app.set_texture(ColorSlot(RED, **surface))
    for _ in range(3):  # compositing lands a frame before the preview samples it
        app.on_render(0.0, 1 / 60.0)

    width, height = app.wnd.buffer_size
    pixels = np.frombuffer(app.wnd.fbo.read(components=3), np.uint8)
    image = pixels.reshape(height, width, 3)[::-1].astype(np.int32)

    left, bottom, view_width, view_height = app.viewport_rect
    top = height - (bottom + view_height)
    return image[top:top + view_height, left:left + view_width]


def silhouette(image: np.ndarray) -> np.ndarray:
    """Where the sphere is: warm pixels, against a cool grey backdrop."""
    return (image[:, :, 0] - image[:, :, 2]) > 4


def brightness(image: np.ndarray, mask: np.ndarray) -> float:
    return float(image[mask].mean())


def saturation(image: np.ndarray, mask: np.ndarray) -> float:
    """0 for grey or white, 1 for a pure hue."""
    channels = image[mask].mean(axis=0)
    return float((channels.max() - channels.min()) / max(channels.max(), 1.0))


def _grow(mask: np.ndarray, steps: int) -> np.ndarray:
    """Spread a mask outwards, to clear the sphere's antialiased edge."""
    grown = mask.copy()
    for _ in range(steps):
        grown[1:] |= grown[:-1]
        grown[:-1] |= grown[1:]
        grown[:, 1:] |= grown[:, :-1]
        grown[:, :-1] |= grown[:, 1:]
    return grown


def halo(app):
    """A measure of how far light spills past the sphere, in pixels.

    The backdrop is a vertical gradient, so every row of it is one colour --
    which makes a column far from the model a free reference for what that row
    would look like with nothing glowing on it. Anything brighter than its own
    row's backdrop, outside the model, is spill.

    The model's outline is taken once, from a frame that is not emitting.
    Reading it per-frame would defeat the whole measurement: a red glow is red,
    so a colour test would count the halo as part of the model.
    """
    outline = _grow(silhouette(frame(app, emission=0.0)), 8)

    def measure(image: np.ndarray) -> int:
        lift = image - image[:, -4:-3, :]
        return int((~outline & (lift.max(axis=2) > 3)).sum())

    return measure


# --------------------------------------------------------------------------
# metallic and roughness
# --------------------------------------------------------------------------

def test_a_metal_is_darker_than_the_same_colour_as_a_dielectric(app):
    """A metal has no diffuse of its own -- its colour is what it reflects, and
    a scene this sparse gives it little to reflect. This is the difference the
    slider is meant to make, and it has to be visible."""
    dielectric = frame(app, metallic=0.0, roughness=0.3)
    metal = frame(app, metallic=1.0, roughness=0.3)
    mask = silhouette(dielectric)

    assert brightness(metal, mask) < brightness(dielectric, mask) * 0.75


def test_a_rough_metal_is_still_lit(app):
    """Rough metal was black: the reflection was faded out as roughness climbed
    instead of being blurred, so a fully rough metal reflected nothing at all.
    A rough surface returns as much light as a polished one, just scattered."""
    polished = frame(app, metallic=1.0, roughness=0.05)
    rough = frame(app, metallic=1.0, roughness=1.0)
    mask = silhouette(polished)

    assert brightness(rough, mask) > brightness(polished, mask) * 0.5
    assert brightness(rough, mask) > 12, "a rough metal is not a black hole"


def test_polishing_a_surface_concentrates_its_highlight(app):
    """Low roughness is a small bright highlight, high roughness a broad dim
    one -- the peak is where that shows, not the total light.

    Read off the blue channel: the base colour is red, the lamp is white, so
    blue is almost entirely the highlight. Red is no use here because a lit red
    surface pins at 255 whatever the roughness does.
    """
    polished = frame(app, metallic=0.0, roughness=0.05)
    chalky = frame(app, metallic=0.0, roughness=1.0)
    mask = silhouette(polished)

    assert polished[:, :, 2][mask].max() > chalky[:, :, 2][mask].max() * 1.5


# --------------------------------------------------------------------------
# emission
# --------------------------------------------------------------------------

def test_emission_keeps_the_colour_it_was_given(app):
    """Cranking emission used to drive the surface to white: each channel
    clipped at 1 in turn and the hue climbed to the corner of the cube. An
    emissive red surface has to stay red however bright it is."""
    for strength in (1.0, MAX_EMISSION * 0.5, MAX_EMISSION):
        image = frame(app, emission=strength)
        mask = silhouette(image)
        assert saturation(image, mask) > 0.8, f"washed out at emission {strength}"


def test_a_brighter_emission_glows_further(app):
    """Past white a surface cannot get any brighter on screen, so what carries
    the difference is the glow spilling past its edge. Without that, every
    emission above about 1 looks identical -- which is what 'it should be more
    powerful' was about."""
    measure = halo(app)
    floor = measure(frame(app, emission=0.0))

    spread = [measure(frame(app, emission=strength))
              for strength in (1.0, 4.0, MAX_EMISSION)]

    assert spread[0] > floor * 2, "an emissive surface glows"
    assert spread[1] > spread[0] * 1.5, "and a brighter one glows further"
    assert spread[2] > spread[1]


def test_a_surface_that_is_not_emissive_does_not_glow(app):
    """The glow is emission's alone: it is what is left after subtracting white,
    and nothing but emission gets there. A bright highlight is not a light
    source, and a plain colour must not fog the backdrop around it."""
    measure = halo(app)
    floor = measure(frame(app, emission=0.0))

    for surface in ({"roughness": 0.02, "metallic": 1.0},  # a mirror
                    {"roughness": 0.9},                    # chalk
                    {"roughness": 0.3, "alpha": 0.5}):     # see-through
        assert measure(frame(app, **surface)) <= floor, surface


# --------------------------------------------------------------------------
# world lighting
# --------------------------------------------------------------------------

def test_the_world_strength_changes_how_much_light_arrives(app):
    dark = app.world
    dark.strength = 0.0
    dim = frame(app, metallic=0.0, roughness=0.6)
    mask = silhouette(dim)

    dark.strength = 3.0
    bright = frame(app, metallic=0.0, roughness=0.6)

    assert brightness(bright, mask) > brightness(dim, mask) + 8


def test_the_world_colour_tints_what_it_falls_on(app):
    app.world.strength = 3.0
    app.world.color = (0.1, 0.2, 1.0)
    blue = frame(app, metallic=0.0, roughness=0.6)

    app.world.color = (1.0, 0.2, 0.1)
    warm = frame(app, metallic=0.0, roughness=0.6)

    mask = silhouette(warm)
    assert blue[mask][:, 2].mean() > warm[mask][:, 2].mean() + 5


def test_a_metal_needs_the_world_to_be_a_metal(app):
    """The one lamp gives a metal a highlight and nothing else. Turning the
    world off is what the sliders are measured against elsewhere here, so it
    has to actually take the light away."""
    app.world.strength = 0.0
    starved = frame(app, metallic=1.0, roughness=0.5)
    mask = silhouette(frame(app, metallic=0.0, roughness=0.5))

    app.world.strength = 2.0
    lit = frame(app, metallic=1.0, roughness=0.5)
    assert brightness(lit, mask) > brightness(starved, mask) * 1.5


# --------------------------------------------------------------------------
# walking the key light around the model
# --------------------------------------------------------------------------

def test_the_rotation_walks_the_light_around_the_axes():
    """0 is +X and it turns towards +Y, counting the way the gizmo does -- so
    the number on the slider means something you can check against the screen."""
    for degrees, expected in ((0, (1, 0)), (90, (0, 1)), (180, (-1, 0)), (270, (0, -1))):
        x, y, z = WorldPrefs(rotation=degrees).light_direction()
        flat = math.hypot(x, y)
        assert (x / flat, y / flat) == pytest.approx(expected, abs=1e-6), degrees
        assert z > 0, "the light is above the model, not below it"


def test_the_light_direction_is_a_unit_vector_at_a_fixed_height():
    for degrees in range(0, 360, 17):
        direction = WorldPrefs(rotation=degrees).light_direction()
        assert math.hypot(*direction) == pytest.approx(1.0)
        elevation = math.degrees(math.asin(direction[2]))
        assert elevation == pytest.approx(LIGHT_ELEVATION)


def test_rotating_the_light_moves_the_lit_side(app):
    """What the control is for. The camera does not move, so anything that
    changes in the frame is the lighting."""
    def lit_centre(degrees: float) -> float:
        """Where the brightest part of the sphere is, in pixels across."""
        app.world.rotation = degrees
        image = frame(app, metallic=0.0, roughness=0.4)
        mask = silhouette(image)
        luminance = image.mean(axis=2)
        bright = mask & (luminance > luminance[mask].max() * 0.75)
        return float(np.nonzero(bright)[1].mean())

    # A light on the +X side lights the +X side; turn it to -X and the bright
    # part of the sphere has to cross to the other side of it.
    assert lit_centre(0.0) > lit_centre(180.0) + 40, "the lit side crosses over"


def test_the_light_is_anchored_to_the_model_not_the_camera(app):
    """It used to hang off the camera, which meant it could not be moved:
    orbiting took it along, and every view was lit identically."""
    app.world.rotation = 0.0
    before = frame(app, metallic=0.0, roughness=0.4)

    app.camera.orbit(140.0, 0.0)
    after = frame(app, metallic=0.0, roughness=0.4)

    mask_before = silhouette(before)
    mask_after = silhouette(after)
    assert mask_before.sum() > 0 and mask_after.sum() > 0
    assert brightness(after, mask_after) != pytest.approx(
        brightness(before, mask_before), rel=0.02
    ), "orbiting has to show a differently lit side, not carry the lamp round"


def test_the_bearing_names_where_the_light_is_and_where_it_shines():
    assert bearing(0.0) == ("+X", "-X")
    assert bearing(90.0) == ("+Y", "-Y")
    assert bearing(45.0) == ("+X +Y", "-X -Y")
    assert bearing(225.0) == ("-X -Y", "+X +Y")


def test_the_bearing_is_the_nearest_eighth_and_wraps():
    assert bearing(359.0) == bearing(0.0) == bearing(360.0)
    assert bearing(1.0) == bearing(0.0), "the slider carries the exact angle"
    assert bearing(-90.0) == bearing(270.0)
    for degrees in range(0, 720, 13):
        stands, shines = bearing(degrees)
        assert stands in COMPASS and shines in COMPASS
        assert stands != shines, "a light never shines where it stands"


# --------------------------------------------------------------------------
# the arrow in the viewport
# --------------------------------------------------------------------------

def warm_pixels(image: np.ndarray) -> int:
    """How much of the frame is the arrow's amber, which nothing else here is:
    the backdrop is cold grey and the navigation gizmo owns red, green and blue.
    """
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return int(((red > 150) & (green > 110) & (blue < green - 40)).sum())


def test_the_arrow_appears_when_the_light_is_moved(app):
    """A line of text says the light is at +X; the arrow says which side of what
    you are looking at that is, which is the question actually being asked."""
    app.world.rotation = 45.0
    before = warm_pixels(frame(app, roughness=0.4))

    app.flash_light_gizmo()
    during = warm_pixels(frame(app, roughness=0.4))

    assert before == 0, "nothing is drawn until the rotation is touched"
    assert during > 100, "and the arrow is unmissable once it is"


def test_the_arrow_stands_where_the_light_does(app):
    """It is the light's own position, so turning the rotation has to carry it
    round -- an arrow that pointed the same way at every angle would be a
    decoration rather than a readout."""
    def arrow_centre(degrees: float) -> float:
        app.world.rotation = degrees
        app.flash_light_gizmo()
        image = frame(app, roughness=0.4)
        red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        warm = (red > 150) & (green > 110) & (blue < green - 40)
        return float(np.nonzero(warm)[1].mean())

    assert arrow_centre(0.0) > arrow_centre(180.0) + 40


def test_the_arrow_goes_away_on_its_own(app):
    app.flash_light_gizmo()
    app._light_hint_at -= LIGHT_HINT_SECONDS + 1.0  # a while ago

    assert warm_pixels(frame(app, roughness=0.4)) == 0


def test_the_arrow_fades_rather_than_being_snatched_away(app):
    """Letting go of the slider while still looking at the model should not
    blink the answer out of existence."""
    app.flash_light_gizmo()
    now = app._light_hint_at

    assert app.light_hint_alpha(now) == pytest.approx(1.0)
    assert app.light_hint_alpha(now + LIGHT_HINT_SECONDS - LIGHT_HINT_FADE) == (
        pytest.approx(1.0, abs=0.05)
    ), "solid until the fade begins"

    half_faded = app.light_hint_alpha(now + LIGHT_HINT_SECONDS - LIGHT_HINT_FADE / 2)
    assert 0.2 < half_faded < 0.8

    assert app.light_hint_alpha(now + LIGHT_HINT_SECONDS) == 0.0
    assert app.light_hint_alpha(now + 999.0) == 0.0


def test_nothing_is_drawn_before_the_light_is_ever_touched(app):
    assert app.light_hint_alpha() == 0.0


def test_the_ring_is_the_path_the_light_travels():
    """The circle is drawn at the light's own elevation, so it is the path the
    lamp actually follows as the slider turns rather than a line under it."""
    from pyglm import glm

    direction = glm.normalize(glm.vec3(*WorldPrefs(rotation=30.0).light_direction()))
    centre = glm.vec3(1.0, -2.0, 0.5)
    reach = 3.0

    points = light_gizmo.ring_points(direction, centre, reach)
    heights = {round(point.z, 6) for point in points}
    assert len(heights) == 1, "one elevation, all the way round"
    assert heights.pop() == pytest.approx(centre.z + direction.z * reach)

    radii = [math.hypot(point.x - centre.x, point.y - centre.y) for point in points]
    assert radii == pytest.approx([math.hypot(direction.x, direction.y) * reach] * len(radii))
    assert points[0].x == pytest.approx(points[-1].x), "the ring closes"


def test_a_point_behind_the_camera_projects_to_nothing(app):
    """There is no sensible place on screen for it, and dividing by a negative
    w would put it somewhere confidently wrong -- mirrored through the middle of
    the view, which is worse than not drawing it."""
    from pyglm import glm

    camera = app.camera
    mvp = camera.projection_matrix * camera.matrix
    behind = camera.eye + glm.normalize(camera.eye - camera.target) * 10.0
    rect = app.viewport_rect
    height = app.wnd.buffer_size[1]

    assert light_gizmo.project(mvp, behind, rect, height) is None
    assert light_gizmo.project(mvp, camera.target, rect, height) is not None


def test_the_arrow_survives_a_camera_inside_the_model(app):
    """Zooming right in leaves the lamp behind the eye and the shaft a point."""
    app.flash_light_gizmo()
    app.camera.frame((0.0, 0.0, 0.0), 0.001)
    frame(app, roughness=0.4)  # must not raise


# --------------------------------------------------------------------------
# the preference itself
# --------------------------------------------------------------------------

def test_world_settings_survive_a_round_trip():
    prefs = WorldPrefs(rotation=127.5, strength=2.5, color=(0.2, 0.4, 0.6))
    restored = WorldPrefs.from_dict(prefs.as_dict())
    assert restored.rotation == pytest.approx(127.5)
    assert restored.strength == pytest.approx(2.5)
    assert restored.color == pytest.approx((0.2, 0.4, 0.6))


def test_a_stored_world_out_of_range_is_brought_back_in():
    restored = WorldPrefs.from_dict({"strength": 500.0, "color": [2.0, -1.0, 0.5]})
    assert restored.strength == MAX_WORLD_STRENGTH
    assert restored.color == (1.0, 0.0, 0.5)


def test_a_stored_rotation_comes_back_as_an_angle():
    """It is a bearing, so it wraps rather than clamping: 450 is 90."""
    assert WorldPrefs.from_dict({"rotation": 450.0}).rotation == pytest.approx(90.0)
    assert WorldPrefs.from_dict({"rotation": -90.0}).rotation == pytest.approx(270.0)
    assert WorldPrefs.from_dict({"rotation": "north"}).rotation == WorldPrefs().rotation


def test_a_malformed_world_falls_back_to_the_default():
    """A hand-edited or truncated prefs file must not stop the app opening."""
    default = WorldPrefs()
    for broken in ({"strength": "bright"}, {"color": "blue"}, {"color": [1.0]}):
        restored = WorldPrefs.from_dict(broken)
        assert restored.strength == default.strength
        assert restored.color == default.color


def test_the_settings_tab_draws_the_world_controls(app):
    """The sliders exist and the tab draws with them. Nothing here is worth a
    pixel comparison; what would break is an import or a stale attribute name,
    and drawing the tab is what catches that."""
    from ui import panel

    imgui.new_frame()
    imgui.begin("Parameters")
    panel._draw_settings_tab(app)  # must not raise
    imgui.end()
    imgui.end_frame()


def test_the_world_is_remembered_between_launches(app, monkeypatch):
    app.world.rotation = 210.0
    app.world.strength = 2.25
    app.world.color = (0.3, 0.9, 0.4)
    app.save_prefs()

    from render.viewport import _load_prefs

    stored = WorldPrefs.from_dict(_load_prefs()["world"])
    assert stored.rotation == pytest.approx(210.0)
    assert stored.strength == pytest.approx(2.25)
    assert stored.color == pytest.approx((0.3, 0.9, 0.4), abs=1e-3)
