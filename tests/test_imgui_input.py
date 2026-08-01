"""The event bridge between the window and Dear ImGui."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from render.imgui_renderer import ImGuiRenderer  # noqa: E402


class _Mouse:
    left = 1
    right = 2
    middle = 3


class _Window:
    mouse = _Mouse()


class _IO:
    def __init__(self):
        self.events = []

    def add_mouse_pos_event(self, x, y):
        self.events.append(("position", x, y))

    def add_mouse_button_event(self, button, down):
        self.events.append(("button", button, down))

    def add_mouse_wheel_event(self, x, y):
        self.events.append(("wheel", x, y))


class _Bridge:
    """Use the real bridge methods without constructing an OpenGL renderer."""

    _queue_mouse_position = ImGuiRenderer._queue_mouse_position
    _mouse_button_index = ImGuiRenderer._mouse_button_index
    mouse_press_event = ImGuiRenderer.mouse_press_event
    mouse_release_event = ImGuiRenderer.mouse_release_event
    mouse_drag_event = ImGuiRenderer.mouse_drag_event
    mouse_scroll_event = ImGuiRenderer.mouse_scroll_event

    def __init__(self):
        self.wnd = _Window()
        self.io = _IO()

    def _mouse_pos_viewport(self, x, y):
        return x * 2, y * 2


def test_a_quick_click_queues_both_transitions():
    """Press and release may arrive between frames on a Mac trackpad."""
    bridge = _Bridge()
    bridge.mouse_press_event(10, 20, bridge.wnd.mouse.left)
    bridge.mouse_release_event(10, 20, bridge.wnd.mouse.left)

    assert bridge.io.events == [
        ("position", 20.0, 40.0),
        ("button", 0, True),
        ("position", 20.0, 40.0),
        ("button", 0, False),
    ]


def test_drag_motion_does_not_invent_more_button_transitions():
    bridge = _Bridge()
    bridge.mouse_press_event(10, 20, bridge.wnd.mouse.left)
    bridge.mouse_drag_event(14, 25, 4, 5)
    bridge.mouse_release_event(14, 25, bridge.wnd.mouse.left)

    buttons = [event for event in bridge.io.events if event[0] == "button"]
    assert buttons == [("button", 0, True), ("button", 0, False)]

