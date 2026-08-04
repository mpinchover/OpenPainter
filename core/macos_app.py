"""Small macOS integration helpers.

The development build is launched with ``python main.py``.  Cocoa therefore
inherits ``Python`` as the application name unless we replace the native
process identity before Pyglet creates its application menu.
"""

from __future__ import annotations

import sys

from core.action_log import log_action

_REOPEN_IMP = None


def enable_dock_reactivation(window) -> bool:
    """Bring the window forward when the Dock icon is clicked.

    ``python main.py`` runs unbundled -- there is no Info.plist telling the
    Dock this is a normal windowed app -- and pyglet's own
    ``NSApplicationDelegate`` (``pyglet.app.cocoa._AppDelegate``) implements
    no ``applicationShouldHandleReopen:hasVisibleWindows:``. Recent macOS no
    longer falls back to useful default handling for that combination: a
    click on the Dock icon activates the process (the menu bar goes bold)
    but never re-shows or re-focuses the window itself.

    Same technique as :mod:`render.trackpad`'s pinch fix: add the missing
    delegate method to the already-registered class at runtime. Best-effort,
    like the rest of this module -- this must never prevent the app from
    starting.
    """
    global _REOPEN_IMP
    if sys.platform != "darwin":
        return False
    try:
        from pyglet.libs.darwin.cocoapy import ObjCClass
        from pyglet.libs.darwin.cocoapy.runtime import add_method, get_class

        native = getattr(window, "_window", None)
        nswindow = getattr(native, "_nswindow", None)
        if nswindow is None:
            return False

        if _REOPEN_IMP is None:
            def application_should_handle_reopen(
                objc_self, objc_cmd, application, has_visible_windows
            ) -> bool:
                log_action(
                    "dock_reactivated",
                    has_visible_windows=bool(has_visible_windows),
                )
                app = ObjCClass("NSApplication").sharedApplication()
                app.activateIgnoringOtherApps_(True)
                nswindow.makeKeyAndOrderFront_(None)
                return True

            _REOPEN_IMP = add_method(
                get_class("_AppDelegate"),
                "applicationShouldHandleReopen:hasVisibleWindows:",
                application_should_handle_reopen, b"B@:@B",
            )
        return True
    except Exception:
        return False


def apply_macos_application_name(title: str) -> bool:
    """Expose *title* to Cocoa instead of the Python interpreter's name.

    This is intentionally best-effort: branding must never prevent the app
    from starting on a different Pyglet version or on a headless test runner.
    A packaged ``.app`` should still put the same title in its Info.plist.
    """
    if sys.platform != "darwin" or not title.strip():
        return False

    try:
        from pyglet.libs.darwin.cocoapy import ObjCClass, get_NSString

        native_title = get_NSString(title.strip())

        # NSProcessInfo supplies the name shown by Cocoa for an unbundled
        # executable.  Do this before Pyglet constructs its main menu.
        process_info = ObjCClass("NSProcessInfo").processInfo()
        process_info.setProcessName_(native_title)

        # Python.org builds run inside Python.app, whose bundle metadata still
        # says Python.  Override the in-memory values Cocoa consults without
        # modifying that shared interpreter installation.
        bundle_info = ObjCClass("NSBundle").mainBundle().infoDictionary()
        if bundle_info is not None:
            bundle_info.setObject_forKey_(native_title, get_NSString("CFBundleName"))
            bundle_info.setObject_forKey_(native_title, get_NSString("CFBundleDisplayName"))

        return True
    except Exception:
        return False
