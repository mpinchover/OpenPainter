from __future__ import annotations

import sys
import types

from core.macos_app import apply_macos_application_name


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
