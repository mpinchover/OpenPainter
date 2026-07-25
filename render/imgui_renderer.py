"""ImGui renderer for moderngl-window, updated for the Dear ImGui 1.92 texture API.

moderngl-window 3.1.1 still ships the pre-1.92 backend: it builds the font
atlas itself via ``ImFontAtlas.get_tex_data_as_rgba32()`` and reads
``ImDrawCmd.texture_id``. Dear ImGui 1.92 replaced both with a backend-driven
protocol -- the renderer advertises ``renderer_has_textures``, and ImGui then
hands it a list of :class:`ImTextureData` each frame with create / update /
destroy requests. imgui_bundle 1.92 dropped the old bindings entirely, so the
bundled integration raises AttributeError on construction.

Only Python 3.14 wheels of imgui_bundle 1.92.x exist, so pinning back to a
pre-1.92 release is not an option here. This subclass implements the new
protocol instead and leaves the rest of the integration (input handling, key
maps, shaders) to moderngl-window.

Drop this module once moderngl-window ships 1.92 support upstream.
"""

from __future__ import annotations

import ctypes

import moderngl
from imgui_bundle import imgui
from moderngl_window.integrations.imgui_bundle import ModernglWindowRenderer


class ImGuiRenderer(ModernglWindowRenderer):
    """ModernglWindowRenderer with ImGui 1.92 backend-managed textures."""

    def __init__(self, window):
        super().__init__(window)
        self.io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        self.resize(*window.buffer_size)

    # -- high-DPI ---------------------------------------------------------

    def resize(self, width: int, height: int) -> None:
        """Drive ImGui in physical pixels rather than logical points.

        moderngl-window's mixin hands ImGui the logical window size plus a
        framebuffer scale, which means glyphs get rasterised at 1x and then
        upsampled -- soft text on any Retina/HiDPI display. Working in physical
        pixels instead lets ImGui 1.92's dynamic font system rasterise at native
        resolution. The app compensates for the size difference by scaling the
        font and style (see MeshMapApp.apply_ui_scale).
        """
        self.io.display_size = self.wnd.buffer_size
        self.io.display_framebuffer_scale = (1.0, 1.0)

    def _mouse_pos_viewport(self, x: int, y: int) -> tuple[int, int]:
        """Convert the window's logical mouse coords into that pixel space."""
        logical_x, logical_y = super()._mouse_pos_viewport(x, y)
        ratio = self.wnd.pixel_ratio
        return int(logical_x * ratio), int(logical_y * ratio)

    # -- texture protocol -------------------------------------------------

    def refresh_font_texture(self) -> None:
        """No-op: from 1.92 onward ImGui owns the atlas and asks us to upload it."""

    def _update_texture(self, tex_data: imgui.ImTextureData) -> None:
        status = tex_data.status

        if status == imgui.ImTextureStatus.want_create:
            pixels = tex_data.get_pixels_array()
            texture = self.ctx.texture(
                (tex_data.width, tex_data.height), 4, data=pixels.tobytes()
            )
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.swizzle = "RGBA"
            self.register_texture(texture)
            tex_data.set_tex_id(texture.glo)
            tex_data.set_status(imgui.ImTextureStatus.ok)

        elif status == imgui.ImTextureStatus.want_updates:
            texture = self._textures.get(tex_data.get_tex_id())
            if texture is not None:
                # ImGui reports dirty rectangles, but a full re-upload is simpler
                # and only happens when new glyphs get rasterised.
                texture.write(tex_data.get_pixels_array().tobytes())
            tex_data.set_status(imgui.ImTextureStatus.ok)

        elif status == imgui.ImTextureStatus.want_destroy and tex_data.unused_frames > 0:
            texture = self._textures.pop(tex_data.get_tex_id(), None)
            if texture is not None:
                texture.release()
            tex_data.set_tex_id(0)
            tex_data.set_status(imgui.ImTextureStatus.destroyed)

    # -- draw -------------------------------------------------------------

    def render(self, draw_data: imgui.ImDrawData) -> None:
        io = self.io
        display_width, display_height = io.display_size
        fb_width = int(display_width * io.display_framebuffer_scale[0])
        fb_height = int(display_height * io.display_framebuffer_scale[1])

        if fb_width == 0 or fb_height == 0:
            return

        for tex_data in draw_data.textures or []:
            if tex_data.status != imgui.ImTextureStatus.ok:
                self._update_texture(tex_data)

        self.projMat.value = (
            2.0 / display_width, 0.0, 0.0, 0.0,
            0.0, 2.0 / -display_height, 0.0, 0.0,
            0.0, 0.0, -1.0, 0.0,
            -1.0, 1.0, 0.0, 1.0,
        )

        draw_data.scale_clip_rects(imgui.ImVec2(*io.display_framebuffer_scale))

        self.ctx.enable_only(moderngl.BLEND)
        self.ctx.blend_equation = moderngl.FUNC_ADD
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        for commands in draw_data.cmd_lists:
            # Write vertex and index data straight from ImGui's buffers.
            vtx_type = ctypes.c_byte * commands.vtx_buffer.size() * imgui.VERTEX_SIZE
            idx_type = ctypes.c_byte * commands.idx_buffer.size() * imgui.INDEX_SIZE
            vtx_arr = vtx_type.from_address(commands.vtx_buffer.data_address())
            idx_arr = idx_type.from_address(commands.idx_buffer.data_address())
            self._vertex_buffer.write(vtx_arr)
            self._index_buffer.write(idx_arr)

            idx_pos = 0
            for command in commands.cmd_buffer:
                texture_id = command.get_tex_id()
                texture = self._textures.get(texture_id)
                if texture is None:
                    raise ValueError(
                        f"Texture {texture_id} is not registered. Add it with "
                        f"register_texture(..). Current textures: {list(self._textures)}"
                    )
                texture.use(0)

                x, y, z, w = command.clip_rect
                self.ctx.scissor = int(x), int(fb_height - w), int(z - x), int(w - y)
                self._vao.render(moderngl.TRIANGLES, vertices=command.elem_count, first=idx_pos)
                idx_pos += command.elem_count

        self.ctx.scissor = None

    # -- teardown ---------------------------------------------------------

    def _invalidate_device_objects(self) -> None:
        for buffer in (self._vertex_buffer, self._index_buffer, self._vao, self._prog):
            if buffer:
                buffer.release()
        self._font_texture = None
