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
import traceback
from typing import Callable, Optional

#: Registered callbacks, by ``id`` of the pyglet window owning the view. The
#: selector is added to the view *class*, so one handler serves every window and
#: has to look up which one it was actually called for.
_handlers: dict[int, Callable[[float], None]] = {}
_installed = False


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
    if native is None or not _install_selector():
        return False
    _handlers[id(native)] = on_pinch
    return True


def _cocoa_window(window) -> Optional[object]:
    """The pyglet ``CocoaWindow`` behind a moderngl-window window, if any."""
    if sys.platform != "darwin":
        return None
    native = getattr(window, "_window", None)
    if native is None:
        return None  # headless, or a backend that keeps its window elsewhere
    try:
        from pyglet.window.cocoa import CocoaWindow
    except Exception:
        return None
    return native if isinstance(native, CocoaWindow) else None


def _install_selector() -> bool:
    """Add ``magnifyWithEvent:`` to pyglet's view class. Idempotent."""
    global _installed
    if _installed:
        return True
    try:
        from pyglet.libs.darwin import cocoapy
        from pyglet.window.cocoa.pyglet_view import PygletView_Implementation
    except Exception:
        return False

    def magnify(objc_self, objc_cmd, nsevent) -> None:
        """``- (void)magnifyWithEvent:(NSEvent *)event``, encoded ``v@:@``."""
        try:
            view = cocoapy.ObjCInstance(objc_self)
            # The view keeps the CocoaWindow it belongs to on itself, and
            # ObjCInstance is cached per pointer, so this is the same wrapper
            # object pyglet set the attribute on.
            handler = _handlers.get(id(view._window))
            if handler is not None:
                handler(float(cocoapy.ObjCInstance(nsevent).magnification()))
        except Exception:
            # An exception must never unwind into Objective-C: there is no
            # Python frame to catch it and the process dies mid-gesture.
            traceback.print_exc()

    PygletView_Implementation.PygletView.add_method(
        magnify, b"magnifyWithEvent:", b"v@:@"
    )
    _installed = True
    return True
