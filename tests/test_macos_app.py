from __future__ import annotations

import sys
import types

import core.macos_app as macos_app
from core.macos_app import apply_macos_application_name, enable_dock_reactivation


class _Recorder:
    def __init__(self) -> None:
        self.process_name = None
        self.bundle_values = {}

    def setProcessName_(self, value):
        self.process_name = value

    def processInfo(self):
        return self

    def mainBundle(self):
        return self

    def infoDictionary(self):
        return self

    def setObject_forKey_(self, value, key):
        self.bundle_values[key] = value


def test_application_name_is_noop_off_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert apply_macos_application_name("OpenPainter") is False


def test_application_name_updates_cocoa_process_and_bundle(monkeypatch):
    recorder = _Recorder()
    cocoapy = types.ModuleType("pyglet.libs.darwin.cocoapy")
    cocoapy.ObjCClass = lambda _name: recorder
    cocoapy.get_NSString = lambda value: value

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "pyglet.libs.darwin.cocoapy", cocoapy)

    assert apply_macos_application_name(" OpenPainter ") is True
    assert recorder.process_name == "OpenPainter"
    assert recorder.bundle_values == {
        "CFBundleName": "OpenPainter",
        "CFBundleDisplayName": "OpenPainter",
    }


class _NSWindowRecorder:
    def __init__(self) -> None:
        self.ordered_front = False

    def makeKeyAndOrderFront_(self, _sender):
        self.ordered_front = True


class _NSApplicationRecorder:
    def __init__(self) -> None:
        self.activated = False

    def sharedApplication(self):
        return self

    def activateIgnoringOtherApps_(self, _value):
        self.activated = True


def _fake_window(nswindow):
    native = types.SimpleNamespace(_nswindow=nswindow)
    return types.SimpleNamespace(_window=native)


def test_dock_reactivation_is_noop_off_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert enable_dock_reactivation(_fake_window(_NSWindowRecorder())) is False


def test_dock_reactivation_is_noop_without_a_native_window(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    empty = types.SimpleNamespace(_window=types.SimpleNamespace(_nswindow=None))
    assert enable_dock_reactivation(empty) is False


def test_dock_reactivation_installs_the_reopen_handler(monkeypatch):
    monkeypatch.setattr(macos_app, "_REOPEN_IMP", None)
    monkeypatch.setattr(sys, "platform", "darwin")

    nsapp = _NSApplicationRecorder()
    cocoapy = types.ModuleType("pyglet.libs.darwin.cocoapy")
    cocoapy.ObjCClass = lambda _name: nsapp

    installed = {}

    def add_method(cls, selector, func, encoding):
        installed["cls"] = cls
        installed["selector"] = selector
        installed["func"] = func
        installed["encoding"] = encoding
        return object()

    runtime = types.ModuleType("pyglet.libs.darwin.cocoapy.runtime")
    runtime.add_method = add_method
    runtime.get_class = lambda name: name

    monkeypatch.setitem(sys.modules, "pyglet.libs.darwin.cocoapy", cocoapy)
    monkeypatch.setitem(sys.modules, "pyglet.libs.darwin.cocoapy.runtime", runtime)

    nswindow = _NSWindowRecorder()
    assert enable_dock_reactivation(_fake_window(nswindow)) is True
    assert installed["cls"] == "_AppDelegate"
    assert installed["selector"] == "applicationShouldHandleReopen:hasVisibleWindows:"

    # The installed callback must activate the app and re-show the window --
    # this is what a Dock icon click needs to actually do something.
    result = installed["func"](None, None, None, False)
    assert result is True
    assert nsapp.activated is True
    assert nswindow.ordered_front is True
