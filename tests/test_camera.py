"""Tests for the viewport camera and the axis gizmo.

The camera is a turntable, matching Blender: yaw about world up, pitch about the
screen horizon. Two properties matter and are tested here. It must reach every
orientation -- including straight down and underneath -- without the view matrix
degenerating, which is what lets us drop the stock OrbitCamera's polar clamp.
And the horizon must stay level through any sequence of orbits, which is the
whole difference between a turntable and a trackball.
"""

from __future__ import annotations

import sys
from pathlib import Path

import glm
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from render.viewport import _WORLD_UP, PanOrbitCamera  # noqa: E402
from ui import gizmo  # noqa: E402

WORLD_UP = glm.vec3(*_WORLD_UP)


def up_of(vector: glm.vec3) -> float:
    """The component of ``vector`` along world up.

    Written this way so the suite states what it means -- "how far up" -- rather
    than naming an axis, and so a change of up-axis convention cannot quietly
    turn these assertions into tests of something else.
    """
    return float(glm.dot(vector, WORLD_UP))


@pytest.fixture
def camera() -> PanOrbitCamera:
    cam = PanOrbitCamera(target=(0.0, 0.0, 0.0), radius=2.0, aspect_ratio=1.0)
    cam.frame((0.0, 0.0, 0.0), 2.0)
    return cam


def view_matrix(camera: PanOrbitCamera) -> np.ndarray:
    return np.array(camera.matrix.to_list(), dtype=np.float64)


def test_orbit_reaches_every_orientation(camera):
    """No clamping anywhere: sweep both axes and check the view stays valid.

    A view matrix that is finite and has determinant 1 is a rigid rotation --
    exactly what breaks when a camera's up vector becomes parallel to its
    forward at a pole.
    """
    for yaw in range(0, 360, 15):
        for pitch in range(0, 360, 15):
            cam = PanOrbitCamera(target=(0.0, 0.0, 0.0), radius=2.0, aspect_ratio=1.0)
            cam.frame((0.0, 0.0, 0.0), 2.0)
            cam.orbit(float(yaw), float(pitch))

            matrix = view_matrix(cam)
            assert np.isfinite(matrix).all(), f"NaN view at yaw={yaw} pitch={pitch}"
            assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-6)
            assert glm.length(cam.eye - cam.target) == pytest.approx(cam.radius, rel=1e-5)


def test_orbit_passes_over_the_poles(camera):
    """Straight down and straight up -- the orientations the Euler camera banned.

    Negative pitch raises the camera, matching Blender: dragging down tips the
    subject's top toward you.
    """
    camera.orbit(0.0, -60.0)  # home sits 60 degrees short of the top
    assert up_of(camera.eye) == pytest.approx(camera.radius, rel=1e-4), "directly above"
    assert np.isfinite(view_matrix(camera)).all()

    camera.orbit(0.0, 180.0)
    assert up_of(camera.eye) == pytest.approx(-camera.radius, rel=1e-4), "directly below"
    assert np.isfinite(view_matrix(camera)).all()


def test_a_full_turn_returns_home(camera):
    before = glm.vec3(camera.eye)
    for _ in range(12):
        camera.orbit(0.0, 30.0)
    assert glm.length(camera.eye - before) < 1e-4


def test_orbit_keeps_the_radius_and_target(camera):
    """Orbiting only rotates -- it must not creep in, out, or sideways."""
    radius, target = camera.radius, glm.vec3(camera.target)
    for step in range(50):
        camera.orbit(13.0, -7.0)  # deliberately not axis-aligned
        assert camera.radius == pytest.approx(radius)
        assert glm.length(camera.target - target) == 0.0
        assert glm.length(camera.eye - camera.target) == pytest.approx(radius, rel=1e-5)


def test_frame_returns_to_the_home_view(camera):
    camera.orbit(37.0, 121.0)
    camera.frame((0.0, 0.0, 0.0), 2.0)
    right, up, _ = camera._axes()
    assert up_of(up) > 0.8, "back to roughly world-up"
    assert up_of(right) == pytest.approx(0.0, abs=1e-6), "and level"


def test_zoom_is_multiplicative_and_bounded(camera):
    """Additive zoom cannot serve a 5cm bolt and a 40m building at once."""
    camera.frame((0.0, 0.0, 0.0), 4.0)
    start = camera.radius

    camera.zoom_state(1.0)
    zoomed = camera.radius
    assert zoomed < start
    camera.zoom_state(-1.0)
    assert camera.radius == pytest.approx(start, rel=1e-9), "zoom must be reversible"

    for _ in range(500):
        camera.zoom_state(1.0)
    assert camera.radius == pytest.approx(4.0 * 0.02)
    for _ in range(1000):
        camera.zoom_state(-1.0)
    assert camera.radius == pytest.approx(4.0 * 60.0)


def test_pan_moves_the_target_across_the_view(camera):
    """Panning slides the subject, so it must move the target, not the radius."""
    radius = camera.radius
    _, up, forward = camera._axes()

    camera.pan(0.0, 10.0)
    moved = camera.target
    assert camera.radius == pytest.approx(radius)
    assert glm.length(moved) > 0.0
    # Straight up on screen: no component along the view direction.
    assert glm.dot(glm.normalize(moved), forward) == pytest.approx(0.0, abs=1e-6)
    assert glm.dot(glm.normalize(moved), up) == pytest.approx(1.0, abs=1e-6)


def test_drag_and_scroll_agree_on_direction(camera):
    """Scroll routes through rot_state too, so both gestures must land alike."""
    from render.viewport import _ORBIT_PER_PIXEL

    by_drag = PanOrbitCamera(target=(0.0, 0.0, 0.0), radius=2.0, aspect_ratio=1.0)
    by_drag.frame((0.0, 0.0, 0.0), 2.0)
    by_drag.mouse_sensitivity = 1.0
    by_drag.rot_state(10.0, 0.0)

    by_scroll = PanOrbitCamera(target=(0.0, 0.0, 0.0), radius=2.0, aspect_ratio=1.0)
    by_scroll.frame((0.0, 0.0, 0.0), 2.0)
    by_scroll.orbit(10.0 * _ORBIT_PER_PIXEL, 0.0)

    assert glm.length(by_drag.eye - by_scroll.eye) < 1e-5


# --------------------------------------------------------------------------
# turntable behaviour
# --------------------------------------------------------------------------

def test_yaw_keeps_the_horizon_level(camera):
    """The turntable guarantee: spinning never tilts the horizon.

    A trackball cannot promise this -- its yaw is about a camera-local axis, so
    the horizon drifts -- and that drift is what makes a trackball feel like it
    is fighting the user.
    """
    worst = 0.0
    for _ in range(24):
        camera.orbit(15.0, 0.0)
        worst = max(worst, abs(up_of(camera._axes()[0])))
    assert worst < 1e-5, "yaw must not roll the view"


def test_yaw_does_not_change_elevation(camera):
    """Yaw is about world up, so the eye stays at the same height."""
    height = up_of(camera.eye)
    for _ in range(12):
        camera.orbit(30.0, 0.0)
        assert up_of(camera.eye) == pytest.approx(height, abs=1e-5)


def test_a_circular_stroke_accumulates_no_roll(camera):
    """The trackball failure this camera replaces."""
    for _ in range(4):
        for dx, dy in ((10.0, 0.0), (0.0, 10.0), (-10.0, 0.0), (0.0, -10.0)):
            camera.orbit(dx, dy)
    assert abs(up_of(camera._axes()[0])) < 1e-5


def swing_toward_own_right(camera, dx: float = 20.0) -> float:
    """How far a yaw of ``dx`` carries the view toward the camera's own right.

    Measured in the camera's frame rather than the world's, which is what makes
    it a statement about how the gesture *feels* regardless of where the camera
    has ended up.
    """
    right, _, before = camera._axes()
    camera.orbit(dx, 0.0)
    _, _, after = camera._axes()
    return glm.dot(after - before, right)


def test_yaw_reverses_when_the_view_is_upside_down(camera):
    """Blender's `vod->reverse`.

    Once the view is upside down, the same world-space yaw would carry the
    subject across the screen the *other* way and the gesture would feel
    inverted. Blender flips the yaw direction to compensate, so the motion in
    the camera's own frame stays identical.
    """
    upright = PanOrbitCamera(target=(0.0, 0.0, 0.0), radius=2.0, aspect_ratio=1.0)
    upright.frame((0.0, 0.0, 0.0), 2.0)
    expected = swing_toward_own_right(upright)
    assert abs(expected) > 0.1, "the probe has to actually move"

    flipped = PanOrbitCamera(target=(0.0, 0.0, 0.0), radius=2.0, aspect_ratio=1.0)
    flipped.frame((0.0, 0.0, 0.0), 2.0)
    flipped.orbit(0.0, 180.0)  # over the pole; now upside down
    assert up_of(flipped._axes()[1]) < 0.0, "the view really is inverted"

    assert swing_toward_own_right(flipped) == pytest.approx(expected, rel=1e-5)


def test_without_the_reverse_an_inverted_view_would_feel_backwards(camera):
    """Pins down what the reverse is worth: skipping it inverts the gesture."""
    from render.viewport import _WORLD_UP

    reference = PanOrbitCamera(target=(0.0, 0.0, 0.0), radius=2.0, aspect_ratio=1.0)
    reference.frame((0.0, 0.0, 0.0), 2.0)
    expected = swing_toward_own_right(reference)

    camera.orbit(0.0, 180.0)
    right, _, before = camera._axes()
    # The raw world-axis yaw, with no reverse applied.
    camera.orientation = glm.normalize(
        glm.angleAxis(glm.radians(20.0), glm.vec3(*_WORLD_UP)) * camera.orientation
    )
    _, _, after = camera._axes()
    assert glm.dot(after - before, right) == pytest.approx(-expected, rel=1e-5)


def test_clip_range_stays_precise_across_zoom(camera):
    """Depth precision. A fixed near/far gave far/near around 40,000, which is
    where coplanar surfaces start flickering as the view moves."""
    camera.frame((0.0, 0.0, 0.0), 4.0)
    for _ in range(6):
        ratio = camera.projection.far / camera.projection.near
        assert camera.projection.near > 0.0
        assert camera.projection.far > camera.projection.near
        assert ratio < 1000.0, f"far/near {ratio:.0f} is too wide for the depth buffer"
        camera.zoom_state(1.0)


# --------------------------------------------------------------------------
# axis views
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "axis",
    [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
)
def test_align_to_axis_looks_down_it_in_ortho(camera, axis):
    camera.align_to_axis(axis)

    direction = glm.normalize(camera.eye - camera.target)
    assert glm.length(direction - glm.normalize(glm.vec3(*axis))) < 1e-5
    assert camera.orthographic
    matrix = view_matrix(camera)
    assert np.isfinite(matrix).all()
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-6)


def test_orbiting_leaves_ortho(camera):
    """Blender's Auto Perspective: rotating off an axis restores perspective."""
    camera.align_to_axis(_WORLD_UP)
    assert camera.orthographic
    camera.orbit(5.0, 0.0)
    assert not camera.orthographic


def test_ortho_frames_the_same_height_as_perspective(camera):
    """Switching projection must not appear to change the zoom level."""
    half_height = camera.radius * np.tan(np.radians(camera.projection.fov * 0.5))

    camera.align_to_axis((0.0, -1.0, 0.0))
    matrix = camera.projection_matrix
    # In an ortho matrix, m[1][1] is 2 / (top - bottom).
    assert matrix[1][1] == pytest.approx(1.0 / half_height, rel=1e-5)


def test_top_view_has_a_defined_roll(camera):
    """Looking along world up leaves roll unconstrained; it must still be sane."""
    camera.align_to_axis(_WORLD_UP)
    right, up, _ = camera._axes()
    assert np.isfinite([right.x, right.y, right.z]).all()
    assert glm.length(right) == pytest.approx(1.0, abs=1e-5)
    assert glm.length(up) == pytest.approx(1.0, abs=1e-5)
    assert abs(glm.dot(right, up)) < 1e-5


# --------------------------------------------------------------------------
# the axis gizmo
# --------------------------------------------------------------------------

GIZMO_CENTER = (1600.0, 60.0)


def test_gizmo_projects_axes_where_they_appear(camera):
    """+Y is up on screen at the home view, and the near axes read as nearer."""
    camera.align_to_axis((0.0, -1.0, 0.0))  # Blender's front view: -Y at us, Z up
    found = {
        (ball.index, ball.positive): ball
        for ball in gizmo.balls(camera, GIZMO_CENTER, 1.0)
    }

    # ImGui's y grows downward, so "up on screen" is a smaller y. Z is up.
    assert found[(2, True)].y < GIZMO_CENTER[1] - 10.0
    assert found[(2, False)].y > GIZMO_CENTER[1] + 10.0
    assert found[(0, True)].x > GIZMO_CENTER[0] + 10.0
    assert found[(0, False)].x < GIZMO_CENTER[0] - 10.0
    # Looking down -Y puts it at the centre, pointing at the viewer.
    assert found[(1, False)].depth == pytest.approx(1.0, abs=1e-5)
    assert abs(found[(1, False)].x - GIZMO_CENTER[0]) < 1e-3


def test_gizmo_draws_far_balls_first(camera):
    depths = [ball.depth for ball in gizmo.balls(camera, GIZMO_CENTER, 1.0)]
    assert depths == sorted(depths), "painter's order, so nearer balls land on top"


def test_gizmo_picks_the_ball_under_the_cursor(camera):
    for ball in gizmo.balls(camera, GIZMO_CENTER, 1.0):
        picked = gizmo.pick(camera, GIZMO_CENTER, 1.0, (ball.x, ball.y))
        assert picked is not None
        assert (picked.index, picked.positive) == (ball.index, ball.positive)


def test_gizmo_ignores_clicks_outside_it(camera):
    assert gizmo.pick(camera, GIZMO_CENTER, 1.0, (400.0, 500.0)) is None
    assert not gizmo.hit_test(GIZMO_CENTER, 1.0, (400.0, 500.0))
    assert gizmo.hit_test(GIZMO_CENTER, 1.0, GIZMO_CENTER)


def test_gizmo_scales_with_the_ui(camera):
    """Retina and the UI-scale slider both feed through one factor."""
    small = gizmo.balls(camera, GIZMO_CENTER, 1.0)
    large = gizmo.balls(camera, GIZMO_CENTER, 2.0)
    for a, b in zip(small, large):
        assert abs(b.x - GIZMO_CENTER[0]) == pytest.approx(
            2.0 * abs(a.x - GIZMO_CENTER[0]), abs=1e-6
        )


# --------------------------------------------------------------------------
# navigation preferences
# --------------------------------------------------------------------------

def test_orbit_speed_scales_the_rotation(camera):
    """The Settings slider has to actually reach the orbit."""
    from render.viewport import _ORBIT_PER_PIXEL

    camera.mouse_sensitivity = 1.0
    camera.rot_state(50.0, 0.0)
    full = camera.orientation

    half = PanOrbitCamera(target=(0.0, 0.0, 0.0), radius=2.0, aspect_ratio=1.0)
    half.frame((0.0, 0.0, 0.0), 2.0)
    half.mouse_sensitivity = 0.5
    half.rot_state(50.0, 0.0)

    # Half the speed over the same drag is half the angle.
    reference = PanOrbitCamera(target=(0.0, 0.0, 0.0), radius=2.0, aspect_ratio=1.0)
    reference.frame((0.0, 0.0, 0.0), 2.0)
    reference.orbit(50.0 * _ORBIT_PER_PIXEL * 0.5, 0.0)
    assert glm.length(half.eye - reference.eye) < 1e-4
    assert glm.length(half.eye - camera.eye) > 1e-3, "the two speeds must differ"


def test_default_orbit_speed_matches_blender(camera):
    """Regression: mouse_sensitivity was 3.0, tripling Blender's rate.

    The 3.0 was calibrated for the stock OrbitCamera's rot_state, which divided
    by 10 internally. Replacing that method left the 3.0 uncancelled.
    """
    from render.viewport import NavigationPrefs, _ORBIT_PER_PIXEL

    assert NavigationPrefs().orbit_speed == 1.0
    assert _ORBIT_PER_PIXEL == pytest.approx(0.4), "Blender's turntable default"

    camera.mouse_sensitivity = NavigationPrefs().orbit_speed
    # Yaw is an azimuth about world up, so measure it there. The eye sits on a
    # 60-degree cone, so its 3D sweep is smaller than the azimuth it turned.
    # Azimuth in the plane perpendicular to world up.
    flat = glm.vec3(1.0, 0.0, 0.0) - WORLD_UP * glm.dot(glm.vec3(1.0, 0.0, 0.0), WORLD_UP)
    flat = glm.normalize(flat)
    side = glm.cross(WORLD_UP, flat)

    def azimuth(camera):
        return np.degrees(np.arctan2(glm.dot(camera.eye, side), glm.dot(camera.eye, flat)))

    before = azimuth(camera)
    camera.rot_state(100.0, 0.0)
    after = azimuth(camera)
    turned = abs((after - before + 180.0) % 360.0 - 180.0)
    assert turned == pytest.approx(40.0, abs=0.1), "100 px should turn 40 degrees"


def test_navigation_prefs_round_trip():
    from render.viewport import NavigationPrefs

    prefs = NavigationPrefs(
        orbit_speed=0.35, pan_speed=1.4, zoom_speed=0.8,
        scroll_speed=0.5, invert_orbit_y=True,
    )
    restored = NavigationPrefs.from_dict(prefs.as_dict())
    assert restored.as_dict() == prefs.as_dict()


def test_navigation_prefs_survive_junk():
    """A hand-edited or truncated prefs file must not stop the app starting."""
    from render.viewport import NavigationPrefs

    defaults = NavigationPrefs()
    for junk in ({}, {"orbit_speed": "fast"}, {"orbit_speed": None},
                 {"invert_orbit_x": "yes"}, {"unknown": 1.0}):
        restored = NavigationPrefs.from_dict(junk)
        assert restored.orbit_speed > 0.0
        if "orbit_speed" in junk and not isinstance(junk["orbit_speed"], (int, float)):
            assert restored.orbit_speed == defaults.orbit_speed

    # Out-of-range values are clamped rather than rejected.
    assert NavigationPrefs.from_dict({"orbit_speed": 1e9}).orbit_speed <= 5.0
    assert NavigationPrefs.from_dict({"orbit_speed": -3.0}).orbit_speed >= 0.05


# --------------------------------------------------------------------------
# scroll smoothing
# --------------------------------------------------------------------------

class FakeApp:
    """Just the scroll buffer and drain, lifted off MeshMapApp.

    Bound methods let the real logic be exercised without a GL context or a
    window, which is the only part of the app this behaviour actually needs.
    """

    def __init__(self, camera, smoothing: float):
        from render.viewport import MeshMapApp, NavigationPrefs

        self.camera = camera
        self.navigation = NavigationPrefs(smoothing=smoothing)
        self._pending_orbit = [0.0, 0.0]
        self._pending_pan = [0.0, 0.0]
        self._pending_zoom = 0.0
        self._scroll_arrived = 0.0
        self._scroll_rate = 0.0
        self._scroll_idle = 0
        self._scroll_gaps = 0
        self._queue = MeshMapApp._queue_scroll.__get__(self)
        self._track_arrival_rate = MeshMapApp._track_arrival_rate.__get__(self)
        self._drain = MeshMapApp._drain_scroll.__get__(self)

    def push_orbit(self, pixels: float) -> None:
        """One scroll event's worth of orbit, in pixels of equivalent drag."""
        self._queue(orbit=(pixels, 0.0))

    def frame(self) -> None:
        self._drain()


def azimuth(camera) -> float:
    flat = glm.vec3(1.0, 0.0, 0.0) - WORLD_UP * glm.dot(glm.vec3(1.0, 0.0, 0.0), WORLD_UP)
    flat = glm.normalize(flat)
    side = glm.cross(WORLD_UP, flat)
    return float(np.degrees(np.arctan2(
        glm.dot(camera.eye, side), glm.dot(camera.eye, flat)
    )))


def replay(smoothing: float, deltas: list[float]) -> np.ndarray:
    """Per-frame azimuth change for a scroll pattern at a given smoothing."""
    camera = PanOrbitCamera(target=(0.0, 0.0, 0.0), radius=2.0, aspect_ratio=1.0)
    camera.frame((0.0, 0.0, 0.0), 2.0)
    camera.mouse_sensitivity = 1.0
    app = FakeApp(camera, smoothing)

    steps, previous = [], azimuth(camera)
    for delta in deltas:
        if delta:
            app.push_orbit(delta)
        app.frame()
        current = azimuth(camera)
        steps.append(current - previous)
        previous = current
    return np.array(steps)


#: One pixel of finger travel, in pixels of equivalent drag: a scroll delta of
#: 0.1 (macOS reports a tenth of a point per pixel) times _PIXELS_PER_SCROLL.
QUANTUM = 1.2

#: What a slow two-finger swipe hands us: a pixel of travel every few frames,
#: nothing in between. The steps are all macOS reports at this speed -- the
#: motion between them is not in the event stream to be recovered.
QUANTISED_SLOW = [0, 0, 0, QUANTUM] * 8 + [0] * 12

#: Long enough for the transient at the start of a gesture -- the opening step
#: goes out whole, before there is any rate to pace by -- to be over.
SETTLED = 16


def moving(steps: np.ndarray) -> np.ndarray:
    """The frames of a replay that are inside the gesture, after it settles."""
    return steps[SETTLED:-12]


def test_smoothing_evens_out_quantised_scroll():
    raw = replay(0.0, QUANTISED_SLOW)
    smoothed = replay(1.0, QUANTISED_SLOW)

    # Raw moves on one frame in four and sits still on the rest. Paced by the
    # arrival rate, every frame gets its share -- the same speed, spread out.
    assert np.count_nonzero(np.abs(moving(raw)) < 1e-9) > len(moving(raw)) * 0.5
    assert np.count_nonzero(np.abs(moving(smoothed)) < 1e-9) == 0
    assert moving(smoothed).std() < moving(raw).std() * 0.1


def test_smoothing_holds_a_steady_speed_through_a_steady_gesture():
    """The point of the whole exercise: constant input, constant motion."""
    steps = moving(replay(1.0, QUANTISED_SLOW))
    assert steps.std() / steps.mean() < 0.01


def test_partial_smoothing_lands_between_raw_and_paced():
    """The preference has to mean something across its range, not just at 1."""
    spread = [
        moving(replay(smoothing, QUANTISED_SLOW)).std()
        for smoothing in (0.0, 0.5, 1.0)
    ]
    assert spread[0] > spread[1] > spread[2]


def test_smoothing_never_changes_how_far_you_travel():
    """It redistributes motion across frames; it must not add or lose any.

    Run well past the end of the gesture so even the heaviest smoothing has
    fully drained before the totals are compared. The tolerance is loose enough
    for float noise only: pacing composes the same rotation out of several times
    as many, much smaller quaternion steps.
    """
    settled = QUANTISED_SLOW + [0] * 200
    expected = replay(0.0, settled).sum()
    for smoothing in (0.0, 0.3, 0.6, 1.0):
        assert replay(smoothing, settled).sum() == pytest.approx(expected, rel=1e-4)


def test_smoothing_leaves_fast_gestures_alone():
    """Nothing to spread when a step lands every frame: it must pass straight
    through, or the pacing would be felt as lag exactly when it is least wanted.
    """
    fast = [12.0] * 10 + [0] * 10
    raw = replay(0.0, fast)
    smoothed = replay(1.0, fast)
    assert np.abs(smoothed[:10] - raw[:10]).max() < 1e-9


def test_the_opening_step_of_a_gesture_is_not_held_back():
    """There is no rate to pace by yet, and a view that will not start moving
    reads as a dropped gesture rather than as smoothing."""
    first = replay(1.0, [0, 0, QUANTUM] + [0] * 10)
    assert first[2] == pytest.approx(replay(0.0, [0, 0, QUANTUM])[2], rel=1e-6)


def test_smoothing_off_applies_every_step_the_frame_it_lands():
    raw = replay(0.0, QUANTISED_SLOW)
    assert np.count_nonzero(np.abs(raw) < 1e-9) == QUANTISED_SLOW.count(0)


def test_the_pending_buffer_always_drains():
    """No dribbling tail: the buffer has to reach zero and stay there."""
    camera = PanOrbitCamera(target=(0.0, 0.0, 0.0), radius=2.0, aspect_ratio=1.0)
    camera.frame((0.0, 0.0, 0.0), 2.0)
    app = FakeApp(camera, 1.0)
    app.push_orbit(5.0)

    for _ in range(200):
        app.frame()
    assert abs(app._pending_orbit[0]) == 0.0
    assert abs(app._pending_orbit[1]) == 0.0


def test_a_stalled_gesture_cannot_leave_the_buffer_sitting():
    """If the rate estimate ever reads low, the floor still empties the buffer.

    Input that stops dead mid-gesture is the case that exposes it: the estimate
    still says 'moving', and nothing more is coming to correct it.
    """
    camera = PanOrbitCamera(target=(0.0, 0.0, 0.0), radius=2.0, aspect_ratio=1.0)
    camera.frame((0.0, 0.0, 0.0), 2.0)
    app = FakeApp(camera, 1.0)
    for _ in range(6):  # a slow gesture, teaching it a slow rate
        app.push_orbit(QUANTUM)
        for _ in range(4):
            app.frame()

    app.push_orbit(40.0)  # then one large step, and silence
    from render.viewport import _DRAIN_LIMIT_FRAMES

    for _ in range(int(_DRAIN_LIMIT_FRAMES) * 6):
        app.frame()
    assert abs(app._pending_orbit[0]) == 0.0
