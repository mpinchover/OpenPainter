from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from render.viewport import _primary_shortcut_down


def test_primary_shortcut_uses_ctrl_outside_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert _primary_shortcut_down(SimpleNamespace(ctrl=True))
    assert not _primary_shortcut_down(SimpleNamespace(ctrl=False))


def test_primary_shortcut_reads_command_from_cocoa_on_macos(monkeypatch):
    command_mask = 1 << 6
    event_class = SimpleNamespace(modifierFlags=lambda: command_mask)
    cocoa = types.ModuleType("pyglet.libs.darwin.cocoapy")
    cocoa.NSCommandKeyMask = command_mask
    cocoa.ObjCClass = lambda name: event_class

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "pyglet.libs.darwin.cocoapy", cocoa)

    # The portable object says Ctrl is up because moderngl-window discarded
    # Pyglet's Command bit; Cocoa remains the source of truth on macOS.
    assert _primary_shortcut_down(SimpleNamespace(ctrl=False))
