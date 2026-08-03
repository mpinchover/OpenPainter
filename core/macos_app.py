"""Small macOS integration helpers.

The development build is launched with ``python main.py``.  Cocoa therefore
inherits ``Python`` as the application name unless we replace the native
process identity before Pyglet creates its application menu.
"""

from __future__ import annotations

import sys


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
