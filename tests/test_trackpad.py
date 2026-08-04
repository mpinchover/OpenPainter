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
import time
import types
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


def test_native_focus_state_is_empty_without_a_cocoa_window():
    """Headless runs and every other backend must fall through quietly."""

    class NotACocoaWindow:
        pass

    assert trackpad.native_focus_state(NotACocoaWindow()) == {}


def test_native_focus_state_reports_appkits_own_answer(monkeypatch):
    """This must reflect AppKit directly, not pyglet's activate/deactivate
    belief -- that is the whole point of cross-checking against it."""

    class FakeNSWindow:
        def isKeyWindow(self):
            return True

    class FakeNSApp:
        def sharedApplication(self):
            return self

        def isActive(self):
            return False

    fake_native = types.SimpleNamespace(_nswindow=FakeNSWindow())
    monkeypatch.setattr(trackpad, "_cocoa_window", lambda window: fake_native)

    cocoapy = types.ModuleType("pyglet.libs.darwin.cocoapy")
    cocoapy.ObjCClass = lambda _name: FakeNSApp()
    monkeypatch.setitem(sys.modules, "pyglet.libs.darwin.cocoapy", cocoapy)

    assert trackpad.native_focus_state(object()) == {
        "is_key_window": True,
        "is_app_active": False,
    }


def _fake_reclaim_cocoapy(dock_apps: list):
    """A minimal stand-in for ``pyglet.libs.darwin.cocoapy`` covering both
    calls ``_reclaim_window_focus`` makes: the Dock-first activation and the
    direct one. ``dock_apps`` is what ``runningApplicationsWithBundleIdentifier_``
    returns -- empty to exercise the "no Dock process found" fallback."""

    class FakeNSApp:
        def __init__(self):
            self.activated = False

        def sharedApplication(self):
            return self

        def activateIgnoringOtherApps_(self, _value):
            self.activated = True

    class FakeRunningApps:
        def count(self):
            return len(dock_apps)

        def objectAtIndex_(self, index):
            return dock_apps[index]

    class FakeNSRunningApplicationClass:
        def runningApplicationsWithBundleIdentifier_(self, _bundle_id):
            return FakeRunningApps()

    fake_app = FakeNSApp()
    classes = {
        "NSApplication": fake_app,
        "NSRunningApplication": FakeNSRunningApplicationClass(),
    }
    cocoapy = types.ModuleType("pyglet.libs.darwin.cocoapy")
    cocoapy.ObjCClass = lambda name: classes[name]
    cocoapy.get_NSString = lambda value: value
    cocoapy.NSApplicationActivateIgnoringOtherApps = 1
    return cocoapy, fake_app


def test_reclaim_window_focus_activates_and_orders_front(monkeypatch):
    """The self-heal for pyglet's unbundled-launch focus-stealing bug."""

    class FakeNSWindow:
        def __init__(self):
            self.ordered_front = False

        def makeKeyAndOrderFront_(self, _sender):
            self.ordered_front = True

    nswindow = FakeNSWindow()
    cocoapy, fake_app = _fake_reclaim_cocoapy(dock_apps=[])
    monkeypatch.setitem(sys.modules, "pyglet.libs.darwin.cocoapy", cocoapy)

    assert trackpad._reclaim_window_focus(types.SimpleNamespace(_nswindow=nswindow)) is True
    assert fake_app.activated is True
    assert nswindow.ordered_front is True


def test_reclaim_window_focus_activates_the_dock_process_first(monkeypatch):
    """The two-step pyglet's own startup code uses to win the same race --
    see the docstring on ``_reclaim_window_focus`` for why a plain
    ``activateIgnoringOtherApps_`` alone was not enough."""

    class FakeNSWindow:
        def makeKeyAndOrderFront_(self, _sender):
            pass

    class FakeDockApp:
        def __init__(self):
            self.activated_with = None

        def activateWithOptions_(self, options):
            self.activated_with = options

    dock_app = FakeDockApp()
    cocoapy, fake_app = _fake_reclaim_cocoapy(dock_apps=[dock_app])
    monkeypatch.setitem(sys.modules, "pyglet.libs.darwin.cocoapy", cocoapy)
    monkeypatch.setattr(trackpad.time, "sleep", lambda _seconds: None)

    assert trackpad._reclaim_window_focus(
        types.SimpleNamespace(_nswindow=FakeNSWindow())
    ) is True
    assert dock_app.activated_with == cocoapy.NSApplicationActivateIgnoringOtherApps
    assert fake_app.activated is True


def test_reclaim_window_focus_declines_without_a_native_window():
    assert trackpad._reclaim_window_focus(types.SimpleNamespace(_nswindow=None)) is False


def test_install_reclaims_focus_when_appkit_reports_not_key(monkeypatch):
    """A pinch never reaches a window AppKit does not consider key, so arming
    the bridge must not stop at "the recognizer is attached" -- it has to
    notice and fix a window that is not really focused yet."""

    class FakeView:
        value = 987654

    fake_native = types.SimpleNamespace(_nsview=FakeView(), _nswindow=object())
    monkeypatch.setattr(trackpad, "_cocoa_window", lambda window: fake_native)
    monkeypatch.setattr(trackpad, "_install_selectors", lambda: True)
    monkeypatch.setattr(trackpad, "_install_recognizer", lambda view: True)

    focus_states = iter([
        {"is_key_window": False, "is_app_active": False},
        {"is_key_window": True, "is_app_active": True},
    ])
    monkeypatch.setattr(trackpad, "native_focus_state", lambda window: next(focus_states))
    reclaimed = []
    monkeypatch.setattr(
        trackpad, "_reclaim_window_focus",
        lambda native: reclaimed.append(native) or True,
    )

    try:
        assert trackpad.install_pinch_zoom(object(), lambda _: None) is True
        assert reclaimed == [fake_native]
    finally:
        trackpad._handlers.pop(987654, None)


def test_install_does_not_reclaim_focus_when_already_key(monkeypatch):
    class FakeView:
        value = 987655

    fake_native = types.SimpleNamespace(_nsview=FakeView(), _nswindow=object())
    monkeypatch.setattr(trackpad, "_cocoa_window", lambda window: fake_native)
    monkeypatch.setattr(trackpad, "_install_selectors", lambda: True)
    monkeypatch.setattr(trackpad, "_install_recognizer", lambda view: True)
    monkeypatch.setattr(
        trackpad, "native_focus_state",
        lambda window: {"is_key_window": True, "is_app_active": True},
    )
    reclaimed = []
    monkeypatch.setattr(
        trackpad, "_reclaim_window_focus",
        lambda native: reclaimed.append(native) or True,
    )

    try:
        assert trackpad.install_pinch_zoom(object(), lambda _: None) is True
        assert reclaimed == []
    finally:
        trackpad._handlers.pop(987655, None)


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


def test_duplicate_native_paths_are_collapsed(monkeypatch):
    """A direct event and a recognizer callback for the same native event count once."""

    class Pointer:
        value = 654321

    class Wrapper:
        ptr = Pointer()

    delivered: list[float] = []
    times = iter((100.0, 100.001, 100.02))
    monkeypatch.setattr(trackpad.time, "monotonic", lambda: next(times))
    trackpad._handlers[654321] = delivered.append
    try:
        assert trackpad._dispatch_pinch(Wrapper(), 0.25, source="event")
        assert trackpad._dispatch_pinch(Wrapper(), 0.25, source="recognizer")
        assert trackpad._dispatch_pinch(Wrapper(), 0.25, source="recognizer")
    finally:
        trackpad._handlers.pop(654321, None)
        trackpad._last_dispatch.pop(654321, None)

    assert delivered == [0.25, 0.25]


def _fake_retry_app(focus_state: dict, focus_retry_at: float = float("-inf")):
    app = type("FakeApp", (), {})()
    app.wnd = object()
    app.pinch_zoom = False
    app.on_pinch_zoom = lambda magnitude: None
    app._focus_retry_at = focus_retry_at
    app._WINDOW_FOCUS_RETRY_INTERVAL = MeshMapApp._WINDOW_FOCUS_RETRY_INTERVAL
    app._native_focus_state = lambda: focus_state
    return app


def test_retry_window_focus_reclaims_when_not_key(monkeypatch):
    """install_pinch_zoom's own re-arm triggers can be minutes apart if the
    user is only orbiting the camera, so on_render must keep trying too --
    but only for the broken state this exists to recover from: the app still
    active (a bold menu bar) while its window is never actually key."""
    app = _fake_retry_app({"is_key_window": False, "is_app_active": True})
    calls = []
    monkeypatch.setattr(
        "render.viewport.install_pinch_zoom",
        lambda window, handler: calls.append((window, handler)) or True,
    )

    MeshMapApp._retry_window_focus_if_needed(app)

    assert calls == [(app.wnd, app.on_pinch_zoom)]


def test_retry_window_focus_does_nothing_when_app_is_not_active(monkeypatch):
    """The user deliberately switching to another application also makes the
    window not-key, but must never be fought -- reclaiming here would steal
    focus back from whatever app they switched to, every render frame."""
    app = _fake_retry_app({"is_key_window": False, "is_app_active": False})
    calls = []
    monkeypatch.setattr(
        "render.viewport.install_pinch_zoom",
        lambda window, handler: calls.append(1) or True,
    )

    MeshMapApp._retry_window_focus_if_needed(app)

    assert calls == []


def test_retry_window_focus_is_throttled(monkeypatch):
    """Must not hammer AppKit every single frame while stuck not-key."""
    app = _fake_retry_app(
        {"is_key_window": False, "is_app_active": True},
        focus_retry_at=time.monotonic(),
    )
    calls = []
    monkeypatch.setattr(
        "render.viewport.install_pinch_zoom",
        lambda window, handler: calls.append(1) or True,
    )

    MeshMapApp._retry_window_focus_if_needed(app)

    assert calls == []


def test_retry_window_focus_does_nothing_once_key(monkeypatch):
    app = _fake_retry_app({"is_key_window": True, "is_app_active": True})
    calls = []
    monkeypatch.setattr(
        "render.viewport.install_pinch_zoom",
        lambda window, handler: calls.append(1) or True,
    )

    MeshMapApp._retry_window_focus_if_needed(app)

    assert calls == []


def test_window_activation_rearms_pinch_bridge(monkeypatch):
    """Focus changes can detach native gesture plumbing, so activation re-installs it."""

    app = type("FakeApp", (), {})()
    app._window_active = False
    app._window_activated_at = float("-inf")
    app.wnd = object()
    app.pinch_zoom = False
    app.on_pinch_zoom = lambda magnitude: None
    app._trace_action = lambda *args, **kwargs: None
    calls = []

    def install(window, handler):
        calls.append((window, handler))
        return True

    monkeypatch.setattr("render.viewport.install_pinch_zoom", install)

    MeshMapApp._on_window_activate(app)

    assert calls == [(app.wnd, app.on_pinch_zoom)]
    assert app.pinch_zoom is True


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
def test_direct_gesture_event_survives_stale_recognizer_cache(pyglet_view):
    """A cached recognizer must not suppress the direct AppKit magnify fallback."""
    from pyglet.libs.darwin import cocoapy

    assert trackpad._install_selectors()

    fake_event_class = cocoapy.ObjCSubclass(
        "NSObject", "MeshMapTestStaleRecognizerMagnifyEvent"
    )

    @fake_event_class.method(b"d")
    def magnification(self) -> float:
        return 0.25

    event = cocoapy.ObjCClass("MeshMapTestStaleRecognizerMagnifyEvent").alloc().init()
    view = cocoapy.ObjCClass("PygletView").alloc().init()
    view.retain()

    delivered: list[float] = []
    view_key = trackpad._pointer_key(view)
    trackpad._handlers[view_key] = delivered.append
    trackpad._recognizers[view_key] = object()
    try:
        view.magnifyWithEvent_(event)
    finally:
        trackpad._handlers.pop(view_key, None)
        trackpad._recognizers.pop(view_key, None)
        trackpad._last_dispatch.pop(view_key, None)

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
