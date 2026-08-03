"""Trackpad pinch-to-zoom.

macOS sends a pinch to the view as ``magnifyWithEvent:``, and pyglet's Cocoa
view implements no gesture handlers at all, so the event walked the responder
chain and was dropped: pinching moved nothing, while the controls table
promised it zoomed. It is also not ctrl+scroll -- that translation is a browser
convention, not an AppKit one -- so the existing ctrl branch could never have
covered it.

:mod:`render.trackpad` adds the missing selector to pyglet's view class at
runtime. These tests cover both halves: that the selector really is installed
and delivers, and that what arrives moves the camera the way a pinch should.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from imgui_bundle import imgui

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from render import trackpad  # noqa: E402
from render.viewport import MeshMapApp, NavigationPrefs, PanOrbitCamera  # noqa: E402

macos_only = pytest.mark.skipif(
    sys.platform != "darwin", reason="the gesture only exists on macOS"
)


@pytest.fixture
def pyglet_view():
    """pyglet's Cocoa view class, or a skip if it cannot be reached.

    Importing ``pyglet.window`` builds a hidden shadow window on the way in,
    and that needs a display to exist: ``CGGetActiveDisplayList`` reports none
    while the screen is asleep or the session is detached from the console, and
    the import then dies on an empty screen list. Nothing here needs a window --
    only the class object to add a selector to -- so turn the shadow window off
    for the import and put the setting back afterwards.
    """
    import pyglet

    previous = pyglet.options.shadow_window
    pyglet.options.shadow_window = False
    try:
        from pyglet.window.cocoa.pyglet_view import PygletView_Implementation
    except Exception as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"pyglet's Cocoa window is unavailable here: {exc}")
    finally:
        pyglet.options.shadow_window = previous
    return PygletView_Implementation


class FakeApp:
    """Just the pinch handler and the scroll drain, lifted off MeshMapApp.

    Bound methods run the real logic without a GL context or a window, which is
    all this behaviour touches.
    """

    def __init__(self, scroll_speed: float = 1.0):
        self.camera = PanOrbitCamera(target=(0.0, 0.0, 0.0), radius=2.0, aspect_ratio=1.0)
        self.camera.frame((0.0, 0.0, 0.0), 2.0)
        self.navigation = NavigationPrefs(scroll_speed=scroll_speed, smoothing=0.0)
        self._pending_orbit = [0.0, 0.0]
        self._pending_pan = [0.0, 0.0]
        self._pending_zoom = 0.0
        self._scroll_arrived = 0.0
        self._scroll_rate = 0.0
        self._scroll_idle = 0
        self._scroll_gaps = 0
        self.on_pinch_zoom = MeshMapApp.on_pinch_zoom.__get__(self)
        self._queue_scroll = MeshMapApp._queue_scroll.__get__(self)
        self._track_arrival_rate = MeshMapApp._track_arrival_rate.__get__(self)
        self._drain = MeshMapApp._drain_scroll.__get__(self)

    def pinch(self, *magnifications: float) -> float:
        """Run a gesture to completion and report the resulting radius."""
        for magnification in magnifications:
            self.on_pinch_zoom(magnification)
        for _ in range(200):  # long enough for any smoothing tail to drain
            self._drain()
        return self.camera.radius


@pytest.fixture(autouse=True)
def imgui_context():
    """The handler asks ImGui whether the panel wants the mouse."""
    context = imgui.create_context()
    imgui.set_current_context(context)
    yield
    imgui.destroy_context(context)


# --------------------------------------------------------------------------
# what a pinch does to the camera
# --------------------------------------------------------------------------

def test_spreading_fingers_zooms_in():
    app = FakeApp()
    before = app.camera.radius
    assert app.pinch(0.1, 0.1, 0.1) < before, "positive magnification moves closer"


def test_pinching_together_zooms_out():
    app = FakeApp()
    before = app.camera.radius
    assert app.pinch(-0.1, -0.1, -0.1) > before


def test_a_whole_hand_gesture_is_a_useful_amount_of_zoom():
    """A pinch across the trackpad totals roughly 1.0 of magnification.

    That has to be worth a substantial move -- a few percent would read as the
    gesture being ignored -- without flying past the subject either.
    """
    app = FakeApp()
    ratio = app.camera.radius / app.pinch(*[0.05] * 20)
    assert 1.5 < ratio < 5.0, f"1.0 of magnification changed distance {ratio:.2f}x"


def test_pinch_follows_the_trackpad_speed_preference():
    slow = FakeApp(scroll_speed=0.25).pinch(0.3)
    fast = FakeApp(scroll_speed=2.0).pinch(0.3)
    assert fast < slow, "a higher trackpad speed must zoom further in"


def test_stale_panel_mouse_capture_cannot_disable_native_pinch():
    app = FakeApp()
    imgui.get_io().want_capture_mouse = True
    before = app.camera.radius

    assert app.pinch(0.5) < before
    assert app._pending_zoom == pytest.approx(0.0, abs=1e-6)


def test_a_teleport_sized_event_is_clamped_not_obeyed():
    """A gesture interrupted by the pointer leaving can arrive as one huge event.

    Clamped like the scroll and drag paths: the view still moves, it just does
    not jump to the far end of the zoom range.
    """
    sane = FakeApp().pinch(1.0)
    absurd = FakeApp().pinch(500.0)
    assert absurd == pytest.approx(sane, rel=1e-6)


# --------------------------------------------------------------------------
# the missing Objective-C selector
# --------------------------------------------------------------------------

def test_install_declines_without_a_cocoa_window():
    """Headless runs and every other backend must fall through quietly."""

    class NotACocoaWindow:
        pass

    assert trackpad.install_pinch_zoom(NotACocoaWindow(), lambda _: None) is False


def test_native_view_pointer_survives_wrapper_replacement():
    """A new Python wrapper for the same NSView must retain its callback."""

    class Pointer:
        value = 123456

    class FirstWrapper:
        ptr = Pointer()

    class ReplacementWrapper:
        ptr = Pointer()

    delivered: list[float] = []
    trackpad._handlers[123456] = delivered.append
    try:
        assert trackpad._dispatch_pinch(ReplacementWrapper(), 0.25)
    finally:
        del trackpad._handlers[123456]

    assert delivered == [0.25]


@macos_only
def test_pyglet_view_gains_the_gesture_selector(pyglet_view):
    from pyglet.libs.darwin import cocoapy

    assert trackpad._install_selectors()
    view_class = cocoapy.ObjCClass("PygletView")
    assert view_class.instancesRespondToSelector_(
        cocoapy.get_selector("magnifyWithEvent:")
    ), "AppKit only sends the gesture to a view that answers for it"
    assert view_class.instancesRespondToSelector_(
        cocoapy.get_selector("meshMapMagnify:")
    ), "the explicit native recognizer needs a target action"


@macos_only
def test_a_gesture_event_reaches_the_registered_handler(pyglet_view):
    """The whole path, through the real Objective-C runtime.

    The event is a stand-in class answering ``magnification``, which is the only
    thing the handler asks of it -- a genuine ``NSEventTypeMagnify`` cannot be
    manufactured without a trackpad under a finger.
    """
    from pyglet.libs.darwin import cocoapy

    assert trackpad._install_selectors()

    fake_event_class = cocoapy.ObjCSubclass("NSObject", "MeshMapTestMagnifyEvent")

    @fake_event_class.method(b"d")
    def magnification(self) -> float:
        return 0.25

    event = cocoapy.ObjCClass("MeshMapTestMagnifyEvent").alloc().init()

    view = cocoapy.ObjCClass("PygletView").alloc().init()
    # Retain it for good: this view never went through pyglet's own
    # initialiser, so its dealloc would fail looking for the text view and
    # tracking area it never got.
    view.retain()

    delivered: list[float] = []
    view_key = trackpad._pointer_key(view)
    trackpad._handlers[view_key] = delivered.append
    try:
        view.magnifyWithEvent_(event)
    finally:
        del trackpad._handlers[view_key]

    assert delivered == [0.25]


@macos_only
def test_native_gesture_recognizer_reaches_the_registered_handler(pyglet_view):
    from pyglet.libs.darwin import cocoapy

    assert trackpad._install_selectors()
    view = cocoapy.ObjCClass("PygletView").alloc().init()
    view.retain()
    key = trackpad._pointer_key(view)
    delivered: list[float] = []
    trackpad._handlers[key] = delivered.append
    try:
        assert trackpad._install_recognizer(view)
        recognizer = trackpad._recognizers[key]
        recognizer.setMagnification_(0.25)
        view.meshMapMagnify_(recognizer)
        assert float(recognizer.magnification()) == pytest.approx(0.0)
    finally:
        trackpad._handlers.pop(key, None)
        trackpad._recognizers.pop(key, None)

    assert delivered == [0.25]
