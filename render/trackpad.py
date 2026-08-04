"""Trackpad pinch gestures on macOS, which pyglet does not deliver on its own.

AppKit sends a pinch to the view under the pointer as ``magnifyWithEvent:``
(``NSEventTypeMagnify``). pyglet's Cocoa view implements ``scrollWheel:`` and
the mouse and key handlers but nothing for the gesture events
(``pyglet/window/cocoa/pyglet_view.py``), so a pinch walks the responder chain,
finds nobody who wants it, and is dropped. Nothing arrives, and the viewport
sits still.

The other half of the confusion is that a pinch is *not* ctrl+scroll. Browsers
synthesise that translation; AppKit does not. A handler keyed on ctrl therefore
only ever fires for a real ctrl held over a two-finger scroll, never for
fingers spreading on the glass.

The repair is to add the missing selector to pyglet's view class at runtime.
The Objective-C runtime allows adding methods to an already-registered class --
it only forbids new ivars -- and pyglet's own ``ObjCSubclass`` wrapper both does
the adding and keeps a reference to the trampoline. That second part matters:
a bare ``class_addMethod`` with a locally-built ctypes callback would leave
AppKit calling freed memory the next time a finger touched the trackpad.
"""

from __future__ import annotations

import sys
import time
import traceback
from typing import Callable, Optional

from core.action_log import log_action

#: Registered callbacks by the native ``NSView *`` address.  Do not key these
#: by the Python ``CocoaWindow`` wrapper: pyglet can replace/reconstruct its
#: ObjCInstance wrappers while the actual AppKit view remains the same.
_handlers: dict[int, Callable[[float], None]] = {}
_recognizers: dict[int, object] = {}
_last_dispatch: dict[int, tuple[float, float]] = {}
_installed = False
_DUPLICATE_WINDOW_SECONDS = 0.006
_DUPLICATE_MAGNIFICATION_EPS = 1e-5


def _pointer_key(value: object) -> int:
    """Return the stable native address behind a ctypes/ObjC wrapper."""
    pointer = getattr(value, "ptr", value)
    raw = getattr(pointer, "value", pointer)
    return int(raw)


def _dispatch_pinch(
    view: object, magnification: float, *, source: str = "unknown"
) -> bool:
    """Deliver one native event without depending on pyglet wrapper identity."""
    key = _pointer_key(view)
    handler = _handlers.get(key)
    magnification = float(magnification)
    now = time.monotonic()
    last = _last_dispatch.get(key)
    duplicate = bool(
        handler is not None
        and last is not None
        and now - last[0] <= _DUPLICATE_WINDOW_SECONDS
        and abs(magnification - last[1]) <= _DUPLICATE_MAGNIFICATION_EPS
    )
    log_action(
        "native_magnify",
        view_pointer=key,
        handler_found=handler is not None,
        magnification=magnification,
        source=source,
        duplicate=duplicate,
    )
    if handler is None:
        return False
    if duplicate:
        return True
    _last_dispatch[key] = (now, magnification)
    handler(magnification)
    return True


def native_focus_state(window) -> dict:
    """AppKit's own opinion of whether this window is focused right now.

    ``on_activate``/``on_deactivate`` (pyglet's belief, driven by
    ``windowDidBecomeKey:``/``windowDidResignKey:``) tell the app when focus
    changed, but there is no callback for "a gesture was attempted and
    dropped" -- nothing fires, so there is nothing to log at the moment it
    matters. Querying AppKit directly and folding this into the ordinary
    input trace (see ``MeshMapApp._trace_action``) means every logged mouse
    or keyboard event carries the ground truth alongside it, so a pinch that
    silently went nowhere can be checked against what the window was actually
    doing at that exact timestamp -- including whether pyglet's own belief
    ever disagreed with AppKit's.
    """
    native = _cocoa_window(window)
    if native is None:
        return {}
    nswindow = getattr(native, "_nswindow", None)
    if nswindow is None:
        return {}
    try:
        from pyglet.libs.darwin.cocoapy import ObjCClass

        app = ObjCClass("NSApplication").sharedApplication()
        return {
            "is_key_window": bool(nswindow.isKeyWindow()),
            "is_app_active": bool(app.isActive()),
        }
    except Exception:
        return {}


def _reclaim_window_focus(native) -> bool:
    """Force this window to the front and key when AppKit disagrees it is.

    pyglet's own ``applicationDidFinishLaunching_`` (``pyglet/app/cocoa.py``)
    documents why this is needed: launched unbundled (``python main.py``,
    no ``.app``), moving the mouse during startup can steal focus to
    whatever is under the cursor, leaving the app looking active -- a bold
    menu bar -- while its window is never actually key, "until reactivating
    it." A trackpad gesture only ever reaches the key window, so this state
    reads as pinch being silently broken.

    A plain ``activateIgnoringOtherApps_`` call was measured doing nothing at
    all here -- confirmed via ``native_focus_state`` immediately after calling
    it -- for well over a minute, on the same macOS build pyglet's own
    comment names, until something else (an actual click into the window)
    finally made it key. This uses the same two-step pyglet's own startup
    code already relies on to win that race: hand activation through the
    Dock process first, briefly, before asking directly. Called repeatedly
    (see ``MeshMapApp._retry_window_focus_if_needed``) rather than once,
    since even this is not guaranteed to land on the first attempt.
    """
    nswindow = getattr(native, "_nswindow", None)
    if nswindow is None:
        return False
    try:
        from pyglet.libs.darwin.cocoapy import (
            NSApplicationActivateIgnoringOtherApps,
            ObjCClass,
            get_NSString,
        )

        dock = ObjCClass("NSRunningApplication").runningApplicationsWithBundleIdentifier_(
            get_NSString("com.apple.dock")
        )
        if dock is not None and int(dock.count()) > 0:
            dock.objectAtIndex_(0).activateWithOptions_(
                NSApplicationActivateIgnoringOtherApps
            )
            # Matches pyglet's own startup code exactly: the follow-up
            # activation below was observed not to take without this pause.
            time.sleep(0.01)

        app = ObjCClass("NSApplication").sharedApplication()
        app.activateIgnoringOtherApps_(True)
        nswindow.makeKeyAndOrderFront_(None)
        return True
    except Exception:
        return False


def install_pinch_zoom(window, on_pinch: Callable[[float], None]) -> bool:
    """Route trackpad pinches on ``window`` to ``on_pinch``.

    ``on_pinch`` is handed the magnification of a single event: NSEvent's own
    incremental change in scale, positive when the fingers spread apart. A
    whole-hand gesture arrives as a stream of these and totals roughly 1.0.

    Returns whether the gesture is now available. False means there was nothing
    to install -- any platform or windowing backend other than macOS on pyglet
    has no ``magnifyWithEvent:`` to be missing -- and is not an error.
    """
    native = _cocoa_window(window)
    if native is None:
        log_action("pinch_install_failed", reason="no_cocoa_window")
        return False
    if not _install_selectors():
        log_action("pinch_install_failed", reason="selectors_unavailable")
        return False
    view = getattr(native, "_nsview", None)
    if view is None:
        log_action("pinch_install_failed", reason="no_nsview")
        return False
    key = _pointer_key(view)
    _handlers[key] = on_pinch
    recognizer_attached = _install_recognizer(view)
    focus = native_focus_state(window)
    reclaimed = None
    if focus.get("is_key_window") is False:
        reclaimed = _reclaim_window_focus(native)
        focus = native_focus_state(window)
    log_action(
        "pinch_handler_armed",
        view_pointer=key,
        recognizer_attached=recognizer_attached,
        reclaimed_focus=reclaimed,
        **focus,
    )
    return True


def _cocoa_window(window) -> Optional[object]:
    """The pyglet ``CocoaWindow`` behind a moderngl-window window, if any."""
    if sys.platform != "darwin":
        return None
    native = getattr(window, "_window", None)
    if native is None:
        # headless, or a backend that keeps its window elsewhere -- expected
        # often enough (tests, CI) that logging it would be noise there, but
        # worth knowing about if it happens on a real desktop session.
        log_action("pinch_cocoa_window_missing", reason="no_window_attr")
        return None
    try:
        from pyglet.window.cocoa import CocoaWindow
    except Exception as exc:
        log_action("pinch_cocoa_window_missing", reason="import_failed", error=str(exc))
        return None
    if not isinstance(native, CocoaWindow):
        log_action(
            "pinch_cocoa_window_missing", reason="wrong_type",
            found_type=type(native).__name__,
        )
        return None
    return native


def _install_selectors() -> bool:
    """Add direct-event and gesture-recognizer callbacks to PygletView."""
    global _installed
    try:
        from pyglet.libs.darwin import cocoapy
        from pyglet.window.cocoa.pyglet_view import PygletView_Implementation
    except Exception as exc:
        log_action("pinch_selectors_unavailable", error=str(exc))
        return False

    event_selector = cocoapy.get_selector("magnifyWithEvent:")
    action_selector = cocoapy.get_selector("meshMapMagnify:")
    view_class = cocoapy.ObjCClass("PygletView")
    if _installed:
        # Do not trust the Python flag alone.  The Cocoa class can be rebuilt
        # during backend/window lifecycle changes, which previously left the
        # application believing a selector was installed when it was not.
        if (
            view_class.instancesRespondToSelector_(event_selector)
            and view_class.instancesRespondToSelector_(action_selector)
        ):
            return True
        log_action("pinch_selectors_reinstalling")
        _installed = False

    def magnify(objc_self, objc_cmd, nsevent) -> None:
        """``- (void)magnifyWithEvent:(NSEvent *)event``, encoded ``v@:@``."""
        # Logged before anything that can raise: this is the one line that
        # proves AppKit actually called us at all. If a pinch never reaches
        # the app, this line's absence -- not an exception below it -- is the
        # answer, since everything past here is naturally silent if we are
        # simply never invoked.
        log_action("pinch_callback_entered", path="event", view_pointer=_pointer_key(objc_self))
        try:
            # The explicit recognizer is normally the path that fires, but macOS
            # can detach or silence recognizers across focus/view lifecycle
            # transitions. Let the view's native event path remain a fallback;
            # _dispatch_pinch collapses the duplicate if both paths fire.
            magnitude = float(cocoapy.ObjCInstance(nsevent).magnification())
            _dispatch_pinch(objc_self, magnitude, source="event")
        except Exception as exc:
            # An exception must never unwind into Objective-C: there is no
            # Python frame to catch it and the process dies mid-gesture.
            log_action("pinch_callback_error", path="event", error=str(exc))
            traceback.print_exc()

    def mesh_map_magnify(objc_self, objc_cmd, sender) -> None:
        """Action invoked by the explicit NSMagnificationGestureRecognizer."""
        log_action(
            "pinch_callback_entered", path="recognizer",
            view_pointer=_pointer_key(objc_self),
        )
        try:
            recognizer = cocoapy.ObjCInstance(sender)
            magnitude = float(recognizer.magnification())
            # NSMagnificationGestureRecognizer reports the total since the
            # gesture began. Reset it after every action so the application
            # receives the incremental values its zoom queue expects.
            recognizer.setMagnification_(0.0)
            _dispatch_pinch(objc_self, magnitude, source="recognizer")
        except Exception as exc:
            log_action("pinch_callback_error", path="recognizer", error=str(exc))
            traceback.print_exc()

    PygletView_Implementation.PygletView.add_method(
        magnify, b"magnifyWithEvent:", b"v@:@"
    )
    PygletView_Implementation.PygletView.add_method(
        mesh_map_magnify, b"meshMapMagnify:", b"v@:@"
    )
    _installed = True
    return True


# Backward-compatible private name used by older tests and local tooling.
_install_selector = _install_selectors


def _install_recognizer(view: object) -> bool:
    """Attach a native pinch recognizer to one concrete Pyglet NSView."""
    key = _pointer_key(view)
    if key in _recognizers:
        recognizer = _recognizers[key]
        _enable_recognizer(recognizer)
        if _recognizer_attached(view, recognizer):
            return True
        try:
            view.addGestureRecognizer_(recognizer)
            log_action("pinch_recognizer_reattached", view_pointer=key)
            return True
        except Exception:
            _recognizers.pop(key, None)
            log_action("pinch_recognizer_reattach_failed", view_pointer=key)
    try:
        from pyglet.libs.darwin import cocoapy

        recognizer = (
            cocoapy.ObjCClass("NSMagnificationGestureRecognizer")
            .alloc()
            .initWithTarget_action_(
                view, cocoapy.get_selector("meshMapMagnify:")
            )
        )
        _enable_recognizer(recognizer)
        view.addGestureRecognizer_(recognizer)
        # NSView retains it; keeping the wrapper as well prevents cocoapy from
        # losing the Python-side object while the native recognizer is active.
        _recognizers[key] = recognizer
        log_action("pinch_recognizer_created", view_pointer=key)
        return True
    except Exception as exc:
        log_action("pinch_recognizer_create_failed", view_pointer=key, error=str(exc))
        traceback.print_exc()
        return False


def _enable_recognizer(recognizer: object) -> None:
    try:
        recognizer.setEnabled_(True)
    except Exception:
        pass


def _recognizer_attached(view: object, recognizer: object) -> bool:
    key = _pointer_key(view)
    try:
        recognizers = view.gestureRecognizers()
        if recognizers is None:
            log_action("pinch_recognizer_check", view_pointer=key, basis="empty_list")
            return False
        recognizer_key = _pointer_key(recognizer)
        for index in range(int(recognizers.count())):
            if _pointer_key(recognizers.objectAtIndex_(index)) == recognizer_key:
                log_action("pinch_recognizer_check", view_pointer=key, basis="found")
                return True
        log_action("pinch_recognizer_check", view_pointer=key, basis="not_found")
        return False
    except Exception as exc:
        # Older pyglet/cocoapy combinations can fail to expose the array cleanly.
        # In that case, do not risk adding duplicate native recognizers -- but
        # log it, since this is exactly the kind of silent false-positive that
        # would otherwise make a genuinely detached recognizer look fine.
        log_action(
            "pinch_recognizer_check", view_pointer=key,
            basis="check_failed_assumed_attached", error=str(exc),
        )
        return True
