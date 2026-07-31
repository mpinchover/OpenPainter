"""Rendering the mask tree: one full-screen pass per node, bottom up.

Everything in this app already lives in UV space, which makes an arbitrarily
deep tree cheap in the only way that matters: each node is a pass into an
atlas-sized target, and its children are just textures by the time it runs. No
recursion in GLSL, no shader generated per tree, no depth limit in the pass
itself -- depth costs passes.

It also leaves every node holding a real texture, which is what lets the panel
show a thumbnail of what each branch actually produces. Drilling into a subtree
is then a navigation choice rather than the only way to find out what is in it.

Targets are recycled through :class:`_Pool`. A post-order walk holds at most one
result per level in flight, so a tree of depth *d* needs *d + 1* targets and no
more, however wide it is.
"""

from __future__ import annotations

from typing import Optional

import moderngl
import numpy as np

from core.layers import MAX_DEPTH, ColorSlot, MaskLayer, Path, Slot, walk
from render.shaders import load_shader

#: Side of the per-node preview the panel draws in its slot rows.
THUMBNAIL_SIZE = 96

#: Half floats: the intermediates are colours being blended, so 8 bits per
#: channel would band across a deep tree, and full floats cost four times the
#: memory for detail no monitor resolves.
_TARGET_DTYPE = "f2"

#: What the shader's u_kind means. 2 is a texture that is only a colour.
_KINDS = {"edge_wear": 0, "noise": 1}
_FLAT = 2


class _Pool:
    """Atlas-sized render targets, handed out and taken back."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._resolution = 0
        self._free: list[tuple[moderngl.Texture, moderngl.Framebuffer]] = []
        self._all: list[tuple[moderngl.Texture, moderngl.Framebuffer]] = []

    def resize(self, resolution: int) -> None:
        if resolution == self._resolution:
            return
        self.release()
        self._resolution = resolution

    def acquire(self) -> tuple[moderngl.Texture, moderngl.Framebuffer]:
        if self._free:
            return self._free.pop()
        texture = self.ctx.texture(
            (self._resolution, self._resolution), 3, dtype=_TARGET_DTYPE
        )
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
        pair = (texture, self.ctx.framebuffer(color_attachments=[texture]))
        self._all.append(pair)
        return pair

    def give_back(self, pair: Optional[tuple[moderngl.Texture, moderngl.Framebuffer]]) -> None:
        if pair is not None:
            self._free.append(pair)

    @property
    def size(self) -> int:
        """How many targets exist. The panel reports it; the tests pin it."""
        return len(self._all)

    def release(self) -> None:
        for texture, fbo in self._all:
            fbo.release()
            texture.release()
        self._all.clear()
        self._free.clear()


class LayerCompositor:
    """Evaluates a :class:`~core.layers.MaskLayer` tree into a colour map."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.program = ctx.program(
            vertex_shader=load_shader("fullscreen.vert"),
            fragment_shader=load_shader("layer.frag"),
        )
        self.program["u_curvature"].value = 0
        self.program["u_position"].value = 1
        self.program["u_white"].value = 2
        self.program["u_black"].value = 3
        self._vao = ctx.vertex_array(self.program, [])

        self.blit = ctx.program(
            vertex_shader=load_shader("fullscreen.vert"),
            fragment_shader=load_shader("blit.frag"),
        )
        self.blit["u_tex"].value = 0
        self.blit["u_rgb"].value = 1
        self._blit_vao = ctx.vertex_array(self.blit, [])

        # Bound to the child units whenever a slot is a flat colour. The shader
        # never samples them, but leaving a sampler pointing at nothing is
        # undefined behaviour and the driver says so.
        self._blank = ctx.texture((1, 1), 3, data=np.zeros(3, dtype="f2").tobytes(),
                                  dtype="f2")

        self._pool = _Pool(ctx)
        self._result: Optional[tuple[moderngl.Texture, moderngl.Framebuffer]] = None
        self._thumbnails: dict[Path, moderngl.Texture] = {}
        self._resolution = 0
        self._inputs: tuple[moderngl.Texture, moderngl.Texture] | None = None
        #: Thumbnails whose node has gone. Handed over rather than released
        #: here: the panel draws these through ImGui, which has to be told to
        #: forget a texture *before* it stops existing.
        self.retired: list[moderngl.Texture] = []

    # -- results ----------------------------------------------------------

    @property
    def texture(self) -> Optional[moderngl.Texture]:
        """The composited colour map, or None until something has been rendered."""
        return None if self._result is None else self._result[0]

    @property
    def resolution(self) -> int:
        return self._resolution

    def thumbnail(self, path: Path) -> Optional[moderngl.Texture]:
        """The preview of whatever the subtree at ``path`` produces."""
        return self._thumbnails.get(tuple(path))

    @property
    def thumbnails(self) -> dict[Path, moderngl.Texture]:
        return dict(self._thumbnails)

    def read(self) -> Optional[np.ndarray]:
        """The colour map as an (n, n, 3) float array, for export."""
        if self._result is None:
            return None
        _, fbo = self._result
        return np.frombuffer(
            fbo.read(components=3, dtype="f4"), dtype="f4"
        ).reshape(self._resolution, self._resolution, 3).copy()

    # -- rendering --------------------------------------------------------

    def render(
        self,
        root: Slot,
        resolution: int,
        curvature: moderngl.Texture,
        position: moderngl.Texture,
    ) -> moderngl.Texture:
        """Evaluate the whole tree. Returns the root's texture."""
        if resolution != self._resolution:
            self._pool.resize(resolution)
            self._drop_thumbnails()
            self._result = None
            self._resolution = resolution

        # Re-bound per node rather than once here: writing each node's
        # thumbnail borrows unit 0, so the next pass would otherwise read a
        # sibling's colours where it expects the curvature bake.
        self._inputs = (curvature, position)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)

        self._pool.give_back(self._result)
        self._result = None
        self._prune_thumbnails(root)

        self._result = self._render_node(root, ())
        return self._result[0]

    def _render_node(
        self, node: Slot, path: Path
    ) -> tuple[moderngl.Texture, moderngl.Framebuffer]:
        """Render one node, its children first, into a target from the pool.

        A colour reaching here is a whole texture that is only a colour -- the
        root before it has been made anything else -- so it fills the map.
        """
        if len(path) > MAX_DEPTH:
            raise ValueError(f"mask tree deeper than {MAX_DEPTH} at {path}")

        if isinstance(node, ColorSlot):
            white = black = None
        else:
            white = self._render_slot(node.white, path + ("white",))
            black = self._render_slot(node.black, path + ("black",))

        target = self._pool.acquire()
        self._set_uniforms(node, white, black)

        target[1].use()
        self.ctx.viewport = (0, 0, self._resolution, self._resolution)
        self._vao.render(moderngl.TRIANGLES, vertices=3)

        # The children have been consumed; their targets can serve the next
        # branch rather than growing the pool with the tree.
        self._pool.give_back(white)
        self._pool.give_back(black)

        self._store_thumbnail(path, target[0])
        return target

    def _render_slot(
        self, slot: Slot, path: Path
    ) -> Optional[tuple[moderngl.Texture, moderngl.Framebuffer]]:
        """A nested mask renders into a target; a colour needs none."""
        if isinstance(slot, ColorSlot):
            self._drop_thumbnail(path)
            return None
        return self._render_node(slot, path)

    def _set_uniforms(self, node: Slot, white, black) -> None:
        program = self.program
        curvature, position = self._inputs
        curvature.use(0)
        position.use(1)

        if isinstance(node, ColorSlot):
            program["u_kind"].value = _FLAT
            program["u_threshold"].value = 0.5
            program["u_softness"].value = 0.0
            program["u_whiteIsMap"].value = 0
            program["u_blackIsMap"].value = 0
            program["u_whiteColor"].value = tuple(float(c) for c in node.color)
            program["u_blackColor"].value = tuple(float(c) for c in node.color)
            self._blank.use(2)
            self._blank.use(3)
            return

        program["u_kind"].value = _KINDS.get(node.kind, 0)

        for name, value in node.boundary_uniforms().items():
            program[name].value = value
        for name, value in node.edge_wear.as_uniforms().items():
            program[name].value = value
        for name, value in node.noise.as_uniforms().items():
            program[name].value = value

        for slot_name, rendered, unit, colour_uniform, flag_uniform in (
            ("white", white, 2, "u_whiteColor", "u_whiteIsMap"),
            ("black", black, 3, "u_blackColor", "u_blackIsMap"),
        ):
            slot = node.slot(slot_name)
            if rendered is None:
                assert isinstance(slot, ColorSlot)
                program[flag_uniform].value = 0
                program[colour_uniform].value = tuple(float(c) for c in slot.color)
                self._blank.use(unit)
            else:
                program[flag_uniform].value = 1
                rendered[0].use(unit)

    # -- thumbnails -------------------------------------------------------

    def _store_thumbnail(self, path: Path, source: moderngl.Texture) -> None:
        texture = self._thumbnails.get(path)
        if texture is None:
            texture = self.ctx.texture((THUMBNAIL_SIZE, THUMBNAIL_SIZE), 4)
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._thumbnails[path] = texture

        fbo = self.ctx.framebuffer(color_attachments=[texture])
        source.use(0)
        fbo.use()
        self.ctx.viewport = (0, 0, THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        self._blit_vao.render(moderngl.TRIANGLES, vertices=3)
        fbo.release()

    def _prune_thumbnails(self, root: MaskLayer) -> None:
        """Forget previews for paths the tree no longer has."""
        # The root always keeps one, mask or not: it is the whole texture, and
        # the panel shows it whatever the texture happens to be.
        live = {path for path, slot in walk(root) if isinstance(slot, MaskLayer)} | {()}
        for path in list(self._thumbnails):
            if path not in live:
                self._drop_thumbnail(path)

    def _drop_thumbnail(self, path: Path) -> None:
        texture = self._thumbnails.pop(path, None)
        if texture is not None:
            self.retired.append(texture)

    def _drop_thumbnails(self) -> None:
        for path in list(self._thumbnails):
            self._drop_thumbnail(path)

    def clear(self) -> None:
        """Forget the composited map. For when there is no texture at all."""
        self._pool.give_back(self._result)
        self._result = None
        self._drop_thumbnails()

    def reclaim(self) -> list[moderngl.Texture]:
        """Hand back the retired thumbnails, for the caller to unregister."""
        retired, self.retired = self.retired, []
        return retired

    def release(self) -> None:
        self._drop_thumbnails()
        for texture in self.reclaim():
            texture.release()
        self._pool.release()
        self._blank.release()
        self._result = None
        self._vao.release()
        self._blit_vao.release()
        self.program.release()
        self.blit.release()
