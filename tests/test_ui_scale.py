"""Regression tests for UI scaling.

ImGui's own ``ScaleAllSizes`` truncates to integers, so it is lossy and not
invertible: scaling up and back down erodes the style, and repeated shrinking
drives border sizes to zero, which trips
``IM_ASSERT(WindowBorderHoverPadding > 0)`` inside ``new_frame()`` and takes the
whole app down. Dragging the UI scale slider leftward was enough to do it.

The fix re-derives every metric from a pristine snapshot instead of applying
deltas, so these tests pin down that it stays idempotent and never degrades.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from imgui_bundle import imgui

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from render.viewport import apply_style_scale, snapshot_style  # noqa: E402


@pytest.fixture
def defaults():
    """A fresh ImGui context, snapshotted before anything scales it."""
    context = imgui.create_context()
    imgui.set_current_context(context)
    snapshot = snapshot_style()
    yield snapshot
    imgui.destroy_context(context)


def _metrics() -> dict:
    style = imgui.get_style()
    return {
        "border_hover": style.window_border_hover_padding,
        "border": style.window_border_size,
        "padding": (style.window_padding.x, style.window_padding.y),
        "spacing": (style.item_spacing.x, style.item_spacing.y),
        "scrollbar": style.scrollbar_size,
        "font": style.font_scale_main,
    }


def test_snapshot_captures_size_metrics(defaults):
    assert defaults["window_border_hover_padding"] == pytest.approx(4.0)
    assert defaults["window_padding"] == (8.0, 8.0)
    # Opacity, timings and alignment ratios are not sizes and must be excluded.
    for excluded in ("alpha", "disabled_alpha", "hover_delay_normal", "window_title_align"):
        assert excluded not in defaults


def test_scaling_is_idempotent(defaults):
    apply_style_scale(defaults, 2.7)
    once = _metrics()
    for _ in range(10):
        apply_style_scale(defaults, 2.7)
    assert _metrics() == once


def test_round_trip_restores_exactly(defaults):
    apply_style_scale(defaults, 1.0)
    baseline = _metrics()

    apply_style_scale(defaults, 2.7)
    apply_style_scale(defaults, 1.0)
    assert _metrics() == baseline


def test_repeated_shrinking_never_degrades_the_style(defaults):
    """The exact sequence that used to crash: dragging the slider down."""
    scale = 3.0
    while scale > 0.6:
        apply_style_scale(defaults, scale)
        assert imgui.get_style().window_border_hover_padding > 0.0
        scale -= 0.05

    # Back up again, and the style must be identical to a direct application.
    apply_style_scale(defaults, 2.0)
    direct = _metrics()
    apply_style_scale(defaults, 0.6)
    apply_style_scale(defaults, 2.0)
    assert _metrics() == direct


def test_border_padding_stays_positive_at_the_smallest_scale(defaults):
    apply_style_scale(defaults, 0.6)
    assert imgui.get_style().window_border_hover_padding >= 1.0
    apply_style_scale(defaults, 0.1)
    assert imgui.get_style().window_border_hover_padding >= 1.0


def test_metrics_track_the_scale(defaults):
    apply_style_scale(defaults, 1.0)
    small = _metrics()
    apply_style_scale(defaults, 2.0)
    large = _metrics()

    assert large["padding"][0] == pytest.approx(small["padding"][0] * 2.0)
    assert large["scrollbar"] == pytest.approx(small["scrollbar"] * 2.0)
    assert large["font"] == pytest.approx(2.0)


def test_new_frame_survives_a_scale_sweep(defaults):
    """End to end: the assert that crashed fires inside new_frame()."""
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 720)
    io.delta_time = 1.0 / 60.0
    # No renderer here, so tell ImGui the backend supplies its own font texture.
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.fonts.add_font_default()

    for scale in (1.0, 2.7, 1.35, 0.6, 3.0, 0.6, 1.0):
        apply_style_scale(defaults, scale)
        imgui.new_frame()
        imgui.begin("probe")
        imgui.text("hello")
        imgui.end()
        imgui.render()
