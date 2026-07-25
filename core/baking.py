"""The expensive half of the pipeline: the UV-space curvature bake.

This is a port of ArmorPaint's BAKE_TYPE_CURVATURE path. Two pieces matter:

1. The mesh is rasterised with its UV as clip-space position, so every fragment
   is a texel of the atlas and ``dFdx``/``dFdy`` measure how fast the
   interpolated normal changes *per texel* (``render/shaders/gbuffer.frag``).
2. The result is smoothed by bouncing it through a 95%-size render target and
   back, N times, exactly as ``render_path_paint.c`` does.

Nothing here knows about the EdgeWear001 node graph. That separation is what
keeps the sliders in :mod:`core.edge_wear` interactive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import moderngl
import numpy as np

from render.shaders import load_shader

from .params import BakeParams

#: ArmorPaint's blur target is 95% of the map's size; the round-trip through it
#: is what does the smoothing.
_BLUR_SCALE = 0.95


@dataclass
class GBuffer:
    """Per-texel surface data produced by the UV-space rasterisation pass."""

    position: np.ndarray   # (h, w, 3) float32 -- world space
    normal: np.ndarray     # (h, w, 3) float32 -- world space, unit length
    curvature: np.ndarray  # (h, w)    float32 -- 0..1, already clamped
    mask: np.ndarray       # (h, w)    bool    -- texel covered by a chart

    @property
    def resolution(self) -> int:
        return self.mask.shape[0]

    @property
    def coverage(self) -> float:
        return float(self.mask.mean())


class UVSpaceBaker:
    """Rasterises the curvature bake into UV space. GL-thread only."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.program = ctx.program(
            vertex_shader=load_shader("gbuffer.vert"),
            fragment_shader=load_shader("gbuffer.frag"),
        )
        self.copy_program = ctx.program(
            vertex_shader=load_shader("fullscreen.vert"),
            fragment_shader=load_shader("copy.frag"),
        )
        self.copy_program["u_tex"].value = 0
        self._copy_vao = ctx.vertex_array(self.copy_program, [])

        self._vao: Optional[moderngl.VertexArray] = None
        self._owned: list = []

        self._fbo: Optional[moderngl.Framebuffer] = None
        self._targets: list[moderngl.Texture] = []
        self._curv_fbo: Optional[moderngl.Framebuffer] = None
        self._blur_texture: Optional[moderngl.Texture] = None
        self._blur_fbo: Optional[moderngl.Framebuffer] = None
        self._fbo_resolution = 0

    # -- geometry ---------------------------------------------------------

    def upload(
        self,
        vertices: np.ndarray,
        normals: np.ndarray,
        uvs: np.ndarray,
        faces: np.ndarray,
    ) -> None:
        """Replace the geometry the next rasterisation will use."""
        self._release_geometry()

        interleaved = np.hstack(
            [
                np.ascontiguousarray(vertices, dtype="f4"),
                np.ascontiguousarray(normals, dtype="f4"),
                np.ascontiguousarray(uvs, dtype="f4"),
            ]
        ).astype("f4")

        vbo = self.ctx.buffer(interleaved.tobytes())
        ibo = self.ctx.buffer(np.ascontiguousarray(faces, dtype="u4").tobytes())
        self._owned = [vbo, ibo]
        self._vao = self.ctx.vertex_array(
            self.program,
            [(vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv")],
            index_buffer=ibo,
            index_element_size=4,
        )

    # -- render targets ---------------------------------------------------

    def _ensure_targets(self, resolution: int) -> None:
        if self._fbo is not None and self._fbo_resolution == resolution:
            return
        self._release_targets()

        size = (resolution, resolution)
        # position, normal, curvature
        self._targets = [self.ctx.texture(size, 4, dtype="f4") for _ in range(3)]
        for texture in self._targets:
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = False
            texture.repeat_y = False
        self._fbo = self.ctx.framebuffer(color_attachments=self._targets)

        # A framebuffer over the curvature attachment alone, so the smoothing
        # round-trip can write back into it without touching position/normal.
        self._curv_fbo = self.ctx.framebuffer(color_attachments=[self._targets[2]])

        blur_size = max(1, int(resolution * _BLUR_SCALE))
        self._blur_texture = self.ctx.texture((blur_size, blur_size), 4, dtype="f4")
        self._blur_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._blur_texture.repeat_x = False
        self._blur_texture.repeat_y = False
        self._blur_fbo = self.ctx.framebuffer(color_attachments=[self._blur_texture])

        self._fbo_resolution = resolution

    # -- the pass ---------------------------------------------------------

    def _smooth(self, iterations: int, resolution: int) -> None:
        """ArmorPaint's smoothing: down to 95% and back, `iterations` times.

        There is no kernel involved -- the blur is whatever the hardware's
        bilinear filter does over a 5% resample, applied repeatedly.
        """
        if iterations <= 0:
            return
        assert self._blur_fbo is not None and self._curv_fbo is not None
        assert self._blur_texture is not None

        blur_size = self._blur_texture.width
        for _ in range(int(iterations)):
            self._blur_fbo.use()
            self.ctx.viewport = (0, 0, blur_size, blur_size)
            self._targets[2].use(0)
            self._copy_vao.render(moderngl.TRIANGLES, vertices=3)

            self._curv_fbo.use()
            self.ctx.viewport = (0, 0, resolution, resolution)
            self._blur_texture.use(0)
            self._copy_vao.render(moderngl.TRIANGLES, vertices=3)

    def rasterize(self, resolution: int, params: BakeParams) -> GBuffer:
        """Render the current geometry into a fresh :class:`GBuffer`."""
        if self._vao is None:
            raise RuntimeError("No geometry uploaded")

        self._ensure_targets(resolution)
        assert self._fbo is not None

        for name, value in params.as_uniforms().items():
            self.program[name].value = value

        self._fbo.use()
        # No depth test: UV space has no overlaps in a valid atlas, and no
        # blending, so each texel keeps exactly the surface point it covers.
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.viewport = (0, 0, resolution, resolution)
        self._vao.render(moderngl.TRIANGLES)

        self._smooth(params.smooth, resolution)

        shape = (resolution, resolution, 4)
        packed_position = np.frombuffer(
            self._fbo.read(attachment=0, components=4, dtype="f4"), dtype="f4"
        ).reshape(shape)
        packed_normal = np.frombuffer(
            self._fbo.read(attachment=1, components=4, dtype="f4"), dtype="f4"
        ).reshape(shape)
        packed_curvature = np.frombuffer(
            self._fbo.read(attachment=2, components=4, dtype="f4"), dtype="f4"
        ).reshape(shape)

        mask = packed_position[..., 3] > 0.5
        normal = packed_normal[..., :3].copy()
        lengths = np.linalg.norm(normal, axis=-1, keepdims=True)
        np.divide(normal, np.clip(lengths, 1e-9, None), out=normal)

        return GBuffer(
            position=packed_position[..., :3].copy(),
            normal=normal,
            curvature=packed_curvature[..., 0].copy(),
            mask=mask,
        )

    # -- teardown ---------------------------------------------------------

    def _release_geometry(self) -> None:
        if self._vao is not None:
            self._vao.release()
            self._vao = None
        for resource in self._owned:
            resource.release()
        self._owned = []

    def _release_targets(self) -> None:
        for fbo in (self._fbo, self._curv_fbo, self._blur_fbo):
            if fbo is not None:
                fbo.release()
        self._fbo = self._curv_fbo = self._blur_fbo = None
        for texture in self._targets:
            texture.release()
        self._targets = []
        if self._blur_texture is not None:
            self._blur_texture.release()
            self._blur_texture = None
        self._fbo_resolution = 0

    def release(self) -> None:
        self._release_geometry()
        self._release_targets()
        self._copy_vao.release()
        self.copy_program.release()
        self.program.release()


# --------------------------------------------------------------------------
# Seam padding
# --------------------------------------------------------------------------

def _shift(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate by whole texels, filling the vacated border with zeros."""
    out = np.zeros_like(array)
    ys = slice(max(dy, 0), array.shape[0] + min(dy, 0))
    xs = slice(max(dx, 0), array.shape[1] + min(dx, 0))
    ys_src = slice(max(-dy, 0), array.shape[0] + min(-dy, 0))
    xs_src = slice(max(-dx, 0), array.shape[1] + min(-dx, 0))
    out[ys, xs] = array[ys_src, xs_src]
    return out


_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def dilate(image: np.ndarray, mask: np.ndarray, iterations: int) -> np.ndarray:
    """Bleed covered texels outward past the chart border.

    Single-sample rasterisation leaves a texel-wide hole around every chart, so
    bilinear filtering in a renderer would pull the empty gutter in and draw a
    dark fringe along each seam. Pushing the edge colour outward fixes it.
    """
    if iterations <= 0:
        return image

    filled = np.array(image, dtype=np.float32, copy=True)
    covered = np.array(mask, dtype=bool, copy=True)

    for _ in range(int(iterations)):
        total = np.zeros_like(filled)
        count = np.zeros(covered.shape, dtype=np.float32)
        for dy, dx in _NEIGHBOURS:
            shifted_mask = _shift(covered.astype(np.float32), dy, dx)
            count += shifted_mask
            weight = shifted_mask if filled.ndim == 2 else shifted_mask[..., None]
            total += _shift(filled, dy, dx) * weight

        grow = (~covered) & (count > 0)
        divisor = np.clip(count, 1.0, None)
        averaged = total / (divisor if filled.ndim == 2 else divisor[..., None])
        target = grow if filled.ndim == 2 else grow[..., None]
        filled = np.where(target, averaged, filled)
        covered |= grow

    return filled
