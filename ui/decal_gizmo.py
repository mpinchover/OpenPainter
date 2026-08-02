"""The outline around the decal the Decal tab is inspecting.

A decal is a rectangle of the *atlas*, and the atlas is not what you are looking
at. On the model that rectangle is wherever the UV layout put it: bent across a
curved panel, split down a seam, mirrored onto the other side of the mesh. So
the outline is not drawn as a rectangle -- it is the decal's border walked in UV
space, every point of it looked up on the surface and projected like any other
point of the scene.

Drawn in ImGui's background draw list, above the 3D view and below every panel,
the same way the navigation gizmo and the light arrow are.
"""

from __future__ import annotations

from imgui_bundle import imgui
from pyglm import glm

from ui.light_gizmo import project

#: Bright, and unlike anything else on screen: the navigation gizmo owns red,
#: green and blue, the light arrow owns amber, and the model is lit in greys.
COLOR = (0.30, 0.90, 1.00)

#: How thick the line is, in unscaled pixels, and how wide the darker line laid
#: under it is. A selection outline has to read against a light surface and a
#: dark one alike, and two lines does that where any one colour will not.
WIDTH = 1.8
SHADOW_WIDTH = 4.0


def draw(camera, points, rect, buffer_height: int, scale: float) -> None:
    """Paint the outline. ``points`` are world-space, in order, closed.

    Points that fall behind the camera are dropped and the line is broken there
    rather than joined across the screen, which is what a decal wrapping around
    the far side of the model would otherwise look like.
    """
    if points is None or len(points) == 0 or (not isinstance(points, list) and len(points) < 2):
        return

    mvp = camera.projection_matrix * camera.matrix
    world_segments = (
        [(segment[0], segment[1]) for segment in points]
        if isinstance(points, list)
        else list(zip(points, points[1:]))
    )

    def screen(point):
        return project(
            mvp, glm.vec3(float(point[0]), float(point[1]), float(point[2])),
            rect, buffer_height,
        )

    draw_list = imgui.get_background_draw_list()
    red, green, blue = COLOR
    shadow = imgui.get_color_u32(imgui.ImVec4(0.0, 0.0, 0.0, 0.55))
    line = imgui.get_color_u32(imgui.ImVec4(red, green, blue, 0.95))

    segments = []
    for world_first, world_second in world_segments:
        first, second = screen(world_first), screen(world_second)
        if first is not None and second is not None:
            segments.append((first, second))
    for width, color in ((SHADOW_WIDTH * scale, shadow), (WIDTH * scale, line)):
        for first, second in segments:
            draw_list.add_line(first, second, color, width)
