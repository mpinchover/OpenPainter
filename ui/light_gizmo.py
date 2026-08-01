"""The arrow that shows where the key light is while you move it.

A line of text can say the light is at +X, but a model is a 3D thing seen from
an angle, and the useful question -- *which side of what I am looking at is
lit* -- is answered fastest by pointing at it. So while the Rotation slider is
moving, an arrow stands off the model at the light's own position and points at
it, along the path the light travels as the slider turns.

It is drawn in ImGui's background draw list, above the 3D view and below every
panel, the same way the navigation gizmo is: no shader, no depth buffer, and no
GL state to put back. Unlike the gizmo, it is projected properly through the
camera's matrices rather than by its axes alone, because it is a thing standing
in the scene at a place rather than a widget in a corner.
"""

from __future__ import annotations

import math

from imgui_bundle import imgui
from pyglm import glm

#: How far the light stands off the model, and where the arrow stops short of
#: it, both as multiples of the model's radius. The gap is what makes the arrow
#: read as pointing *at* the model rather than skewering it.
STANDOFF = 1.45
GAP = 0.62

#: Arrowhead length and half-width, in unscaled pixels. Screen-space, so the
#: head stays legible whether the model fills the view or sits far off.
HEAD_LENGTH = 15.0
HEAD_WIDTH = 6.5
#: The lamp itself, at the tail of the arrow.
LAMP_RADIUS = 5.0
#: Segments in the orbit ring. Enough that a circle looks like one.
RING_SEGMENTS = 72

#: Warm, like a lamp, and unlike anything else on screen: the navigation gizmo
#: owns red, green and blue, and the backdrop is cold grey.
COLOR = (1.0, 0.83, 0.38)


def project(mvp, point, rect, buffer_height: int):
    """World point to ImGui pixels, or None if it is behind the camera.

    ImGui measures y down from the top of the *window*; the viewport rect
    measures y up from the bottom, GL's way. Hence the flip.
    """
    clip = mvp * glm.vec4(point.x, point.y, point.z, 1.0)
    if clip.w <= 1e-6:
        return None

    ndc_x = clip.x / clip.w
    ndc_y = clip.y / clip.w
    rect_x, rect_y, rect_width, rect_height = rect
    screen_x = rect_x + (ndc_x * 0.5 + 0.5) * rect_width
    screen_y = rect_y + (ndc_y * 0.5 + 0.5) * rect_height
    return imgui.ImVec2(float(screen_x), float(buffer_height - screen_y))


def ring_points(direction, center, radius: float) -> list:
    """The circle the light travels as the rotation turns, in world space.

    Drawn at the light's own elevation, so it is the path the arrow's tail
    actually follows rather than a horizon line under it.
    """
    height = center.z + direction.z * radius
    flat = math.hypot(direction.x, direction.y) * radius

    points = []
    for step in range(RING_SEGMENTS + 1):
        angle = (step / RING_SEGMENTS) * math.tau
        points.append(glm.vec3(
            center.x + math.cos(angle) * flat,
            center.y + math.sin(angle) * flat,
            height,
        ))
    return points


def draw(
    camera,
    direction: tuple[float, float, float],
    center: tuple[float, float, float],
    size: float,
    rect: tuple[int, int, int, int],
    buffer_height: int,
    scale: float,
    alpha: float,
) -> None:
    """Paint the arrow. ``direction`` points from the model toward the light.

    ``size`` is the model's diagonal, so the arrow is sized by the thing it is
    pointing at and stays clear of it whether that is a bolt or a bridge.
    ``alpha`` fades the whole thing out once the slider has been let go.
    """
    if alpha <= 0.0:
        return

    light = glm.normalize(glm.vec3(*direction))
    middle = glm.vec3(*center)
    radius = max(size, 1e-6) * 0.5

    tail = middle + light * (radius * STANDOFF)
    head = middle + light * (radius * GAP)

    mvp = camera.projection_matrix * camera.matrix
    start = project(mvp, tail, rect, buffer_height)
    end = project(mvp, head, rect, buffer_height)
    if start is None or end is None:
        return  # the light is behind the camera; nothing useful to draw

    draw_list = imgui.get_background_draw_list()
    red, green, blue = COLOR
    solid = imgui.get_color_u32(imgui.ImVec4(red, green, blue, alpha))

    # The path the light travels, faint, so a rotation reads as going round the
    # model rather than as an arrow jumping about.
    ring = [project(mvp, point, rect, buffer_height)
            for point in ring_points(light, middle, radius * STANDOFF)]
    faint = imgui.get_color_u32(imgui.ImVec4(red, green, blue, alpha * 0.22))
    for first, second in zip(ring, ring[1:]):
        if first is not None and second is not None:
            draw_list.add_line(first, second, faint, 1.0 * scale)

    shaft = imgui.ImVec2(end.x - start.x, end.y - start.y)
    length = math.hypot(shaft.x, shaft.y)
    if length < 1e-3:
        # Dead-on: the light is between the camera and the model, so the arrow
        # is a point. The lamp alone still says where it is.
        draw_list.add_circle_filled(start, LAMP_RADIUS * scale, solid)
        return

    along = imgui.ImVec2(shaft.x / length, shaft.y / length)
    across = imgui.ImVec2(-along.y, along.x)
    head_length = min(HEAD_LENGTH * scale, length * 0.6)
    head_width = HEAD_WIDTH * scale * (head_length / max(HEAD_LENGTH * scale, 1e-6))
    base = imgui.ImVec2(end.x - along.x * head_length, end.y - along.y * head_length)

    draw_list.add_line(start, base, solid, 2.5 * scale)
    draw_list.add_triangle_filled(
        end,
        imgui.ImVec2(base.x + across.x * head_width, base.y + across.y * head_width),
        imgui.ImVec2(base.x - across.x * head_width, base.y - across.y * head_width),
        solid,
    )
    draw_list.add_circle_filled(start, LAMP_RADIUS * scale, solid)
    # A ring around the lamp, so it reads as a source rather than a dot.
    draw_list.add_circle(
        start, LAMP_RADIUS * 2.0 * scale,
        imgui.get_color_u32(imgui.ImVec4(red, green, blue, alpha * 0.5)),
        0, 1.5 * scale,
    )
