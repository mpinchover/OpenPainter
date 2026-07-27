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


def test_pinch_is_ignored_while_the_panel_has_the_mouse():
    app = FakeApp()
    imgui.get_io().want_capture_mouse = True
    assert app.pinch(0.5) == pytest.approx(app.camera.radius)
    assert app._pending_zoom == 0.0


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


@macos_only
def test_pyglet_view_gains_the_gesture_selector():
    from pyglet.libs.darwin import cocoapy

    assert trackpad._install_selector()
    view_class = cocoapy.ObjCClass("PygletView")
    assert view_class.instancesRespondToSelector_(
        cocoapy.get_selector("magnifyWithEvent:")
    ), "AppKit only sends the gesture to a view that answers for it"


@macos_only
def test_a_gesture_event_reaches_the_registered_handler():
    """The whole path, through the real Objective-C runtime.

    The event is a stand-in class answering ``magnification``, which is the only
    thing the handler asks of it -- a genuine ``NSEventTypeMagnify`` cannot be
    manufactured without a trackpad under a finger.
    """
    from pyglet.libs.darwin import cocoapy

    assert trackpad._install_selector()

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

    class FakeWindow:
        pass

    window = FakeWindow()
    view._window = window  # what pyglet's initialiser leaves on the view

    delivered: list[float] = []
    trackpad._handlers[id(window)] = delivered.append
    try:
        view.magnifyWithEvent_(event)
    finally:
        del trackpad._handlers[id(window)]

    assert delivered == [0.25]
