"""Blender's navigation gizmo: six axis balls you can click to align the view.

Modelled on ``view3d_gizmo_navigate_type.cc``. Six balls sit on a circle in the
top-right corner, one per signed axis, positioned by projecting that axis into
the current view. Positive axes are filled and lettered, negative ones hollow,
and whichever is nearest the viewer draws last so the depth ordering reads
correctly. Clicking one looks down that axis in orthographic projection.

The axis colours are Blender's own theme defaults, from
``release/datafiles/userdef/userdef_default_theme.c``.

The viewport is Z-up like Blender -- imports are rotated into that convention by
``_axis_fix`` in :mod:`core.mesh_io` -- so the letters mean here what they mean
there, and +Z is the ball pointing up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from imgui_bundle import imgui

#: Blender's theme: xaxis 0xff3352, yaxis 0x8bdc00, zaxis 0x2890ff.
_AXIS_COLORS = {
    0: (1.000, 0.200, 0.322),
    1: (0.545, 0.863, 0.000),
    2: (0.157, 0.565, 1.000),
}
_LABELS = ("X", "Y", "Z")

#: Gizmo radius in unscaled pixels, and the ball radius within it. Blender's
#: ``gizmo_size_navigate_v3d`` defaults to 80 for the full widget.
GIZMO_RADIUS = 40.0
BALL_RADIUS = 8.5
#: Inset from the top-right corner of the viewport.
MARGIN = 18.0


@dataclass(frozen=True)
class AxisBall:
    """One of the six clickable balls, already projected to screen space."""

    axis: tuple[float, float, float]
    """World-space direction from the target toward the eye, for this view."""
    index: int          # 0=X, 1=Y, 2=Z
    positive: bool
    x: float
    y: float
    depth: float        # +1 toward the viewer, -1 away

    @property
    def color(self) -> tuple[float, float, float]:
        return _AXIS_COLORS[self.index]

    @property
    def label(self) -> str:
        return _LABELS[self.index] if self.positive else ""


def radius(scale: float) -> float:
    """Outer reach of the widget, for a caller placing it."""
    return (GIZMO_RADIUS + BALL_RADIUS) * scale


def balls(camera, center: tuple[float, float], scale: float) -> list[AxisBall]:
    """Project the six signed axes into screen space, far ones first.

    ``center`` is in ImGui's coordinates, which put the origin at the top left
    with y increasing downward -- hence the negated ``up`` component below.
    """
    right, up, forward = camera._axes()
    center_x, center_y = center
    reach = (GIZMO_RADIUS - BALL_RADIUS) * scale

    found: list[AxisBall] = []
    for index in range(3):
        for positive in (True, False):
            axis = np.zeros(3)
            axis[index] = 1.0 if positive else -1.0
            vector = (float(axis[0]), float(axis[1]), float(axis[2]))

            screen_x = axis[0] * right.x + axis[1] * right.y + axis[2] * right.z
            screen_y = axis[0] * up.x + axis[1] * up.y + axis[2] * up.z
            # forward points away from the viewer, so negate for a depth where
            # +1 means "toward me".
            depth = -(axis[0] * forward.x + axis[1] * forward.y + axis[2] * forward.z)

            found.append(AxisBall(
                axis=vector,
                index=index,
                positive=positive,
                x=center_x + screen_x * reach,
                y=center_y - screen_y * reach,
                depth=float(depth),
            ))

    return sorted(found, key=lambda ball: ball.depth)


def pick(camera, center: tuple[float, float], scale: float,
         mouse: tuple[float, float]) -> AxisBall | None:
    """The ball under ``mouse``, or None. Nearest the viewer wins an overlap."""
    if not hit_test(center, scale, mouse):
        return None
    grab = BALL_RADIUS * scale * 1.35
    candidates = [
        ball for ball in balls(camera, center, scale)
        if (ball.x - mouse[0]) ** 2 + (ball.y - mouse[1]) ** 2 <= grab * grab
    ]
    return candidates[-1] if candidates else None


def hit_test(center: tuple[float, float], scale: float,
             mouse: tuple[float, float]) -> bool:
    """Whether ``mouse`` is inside the gizmo's disc at all.

    The camera drag handler asks this so a stroke that begins on the gizmo never
    also orbits the view.
    """
    reach = radius(scale)
    return (mouse[0] - center[0]) ** 2 + (mouse[1] - center[1]) ** 2 <= reach * reach


def draw(camera, center: tuple[float, float], scale: float,
         mouse: tuple[float, float]) -> None:
    """Paint the gizmo.

    The background draw list puts it above the 3D scene but *below* every ImGui
    window, so a panel dragged over the gizmo covers it -- which is both the
    correct layering and consistent with the press handler, where ImGui gets
    first refusal on the click.
    """
    draw_list = imgui.get_background_draw_list()
    center_x, center_y = center
    hovered = pick(camera, center, scale, mouse)

    if hit_test(center, scale, mouse):
        draw_list.add_circle_filled(
            imgui.ImVec2(center_x, center_y),
            (GIZMO_RADIUS + BALL_RADIUS * 0.5) * scale,
            imgui.get_color_u32(imgui.ImVec4(1.0, 1.0, 1.0, 0.06)),
        )

    ordered = balls(camera, center, scale)
    ball_radius = BALL_RADIUS * scale

    # Spokes first, so no line crosses a ball drawn in front of it. Only the
    # positive axes get one, which is what makes the orientation readable.
    for ball in ordered:
        if not ball.positive:
            continue
        red, green, blue = ball.color
        fade = 0.45 + 0.55 * (ball.depth * 0.5 + 0.5)
        draw_list.add_line(
            imgui.ImVec2(center_x, center_y),
            imgui.ImVec2(ball.x, ball.y),
            imgui.get_color_u32(imgui.ImVec4(red, green, blue, fade)),
            2.0 * scale,
        )

    for ball in ordered:
        red, green, blue = ball.color
        # Blender fades a ball toward the background as it goes behind.
        fade = 0.4 + 0.6 * (ball.depth * 0.5 + 0.5)
        is_hovered = hovered is not None and (
            hovered.index == ball.index and hovered.positive == ball.positive
        )
        outline = imgui.get_color_u32(imgui.ImVec4(red, green, blue, min(1.0, fade + 0.25)))

        if ball.positive or is_hovered:
            filled = imgui.ImVec4(red, green, blue, fade)
            draw_list.add_circle_filled(
                imgui.ImVec2(ball.x, ball.y), ball_radius,
                imgui.get_color_u32(filled), 20,
            )
        else:
            # Hollow, over the viewport background, exactly as Blender draws the
            # negative axes.
            draw_list.add_circle_filled(
                imgui.ImVec2(ball.x, ball.y), ball_radius,
                imgui.get_color_u32(imgui.ImVec4(0.09, 0.09, 0.11, 0.85)), 20,
            )
            draw_list.add_circle(
                imgui.ImVec2(ball.x, ball.y), ball_radius, outline, 20, 1.6 * scale,
            )

        if is_hovered:
            draw_list.add_circle(
                imgui.ImVec2(ball.x, ball.y), ball_radius + 2.0 * scale,
                imgui.get_color_u32(imgui.ImVec4(1.0, 1.0, 1.0, 0.9)), 20, 1.6 * scale,
            )

        if ball.label:
            # Letters only on the front-facing balls; behind, they would read
            # mirrored and add nothing.
            text_size = imgui.calc_text_size(ball.label)
            draw_list.add_text(
                imgui.ImVec2(ball.x - text_size.x * 0.5, ball.y - text_size.y * 0.5),
                imgui.get_color_u32(imgui.ImVec4(0.06, 0.06, 0.08, min(1.0, fade + 0.3))),
                ball.label,
            )
