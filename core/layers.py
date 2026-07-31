"""The mask tree: masks that choose between two things, recursively.

A mask produces black and white across the surface. What each of those *means*
is the tree: white can be a colour, or another mask that in turn decides between
two more things, and so does black. Edge wear picking between rust and bare
metal is one node; edge wear picking between rust and a noise that itself picks
between two greys is three.

The shape is a binary tree and the depth is unbounded in principle, which is why
the panel focuses one node at a time (see :func:`slot_at` and the breadcrumb in
``ui/panel.py``) rather than indenting the whole thing. :data:`MAX_DEPTH` caps it
somewhere sane -- the evaluation holds one render target per level, and a tree
deeper than this is a sign the model wants a different shape, not more nesting.

Nothing here evaluates anything. :mod:`render.composite` renders the tree as one
full-screen pass per node, bottom up, which is why depth costs passes rather
than shader complexity, and why every node has a real texture the panel can show
as a thumbnail.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterator, Union

from .params import EdgeWearParams

#: How deep the tree may go. Eight is already far past anything legible.
MAX_DEPTH = 8

#: The two slots under every mask, in the order the panel lists them.
SLOTS = ("white", "black")

#: What a slot is, as ``value: label``. A colour is a leaf; the other two are
#: masks, and choosing one grows the two slots underneath.
COLOR_KIND = "color"
MASK_KINDS = {
    "edge_wear": "Edge wear",
    "noise": "Noise",
}
SLOT_KINDS = {COLOR_KIND: "Colour", **MASK_KINDS}

#: A new mask starts as plain black and white, so what the Shaded view shows is
#: the mask itself. Colour is then something you put *under* it, and every step
#: away from the raw mask is one you asked for.
_DEFAULT_WHITE = (1.0, 1.0, 1.0)
_DEFAULT_BLACK = (0.0, 0.0, 0.0)
#: A new texture, before it is anything else: a mid grey, visibly a placeholder.
_DEFAULT_NEW = (0.55, 0.55, 0.57)


@dataclass
class NoiseMaskParams:
    """A noise mask: the same ArmorPaint noise the wear pass uses, thresholded.

    Sampled at the object-space position rather than the UV, so the pattern runs
    continuously across atlas seams -- the same reason the wear pass samples it
    there.
    """

    scale: float = 12.0
    """Noise frequency, in cycles across the model's bounding box."""

    detail: float = 4.0
    roughness: float = 0.55
    lacunarity: float = 2.5
    distortion: float = 0.0

    bias: float = 0.5
    """The noise value that lands on the midpoint. Lower gives more white."""

    contrast: float = 4.0
    """How hard the threshold is. 0 is flat grey, high values are a hard edge."""

    def as_uniforms(self) -> dict[str, float]:
        return {
            "u_noiseScale": self.scale,
            "u_noiseDetail": self.detail,
            "u_noiseRoughness": self.roughness,
            "u_noiseLacunarity": self.lacunarity,
            "u_noiseDistortion": self.distortion,
            "u_noiseBias": self.bias,
            "u_noiseContrast": self.contrast,
        }


@dataclass
class ColorSlot:
    """A leaf: this side of the mask is simply this colour."""

    color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    name: str = ""
    """What to call it in the tree. Empty falls back to the hex code, which is
    what most slots are better off showing anyway -- a name is for the ones
    worth naming."""

    @property
    def label(self) -> str:
        return self.name or self.auto_label

    @property
    def auto_label(self) -> str:
        red, green, blue = (int(round(c * 255.0)) for c in self.color)
        return f"#{red:02X}{green:02X}{blue:02X}"


@dataclass
class MaskLayer:
    """A branch: a mask, and what its white and black sides resolve to.

    Both parameter blocks are kept whichever kind is selected, so switching a
    node from wear to noise and back does not throw away what was dialled in.
    """

    kind: str = "edge_wear"
    name: str = ""
    """What to call it in the tree. Empty falls back to the kind."""

    threshold: float = 0.5
    """Where the boundary between the two sides falls, on the mask's 0..1.

    A mask is a continuous field, not a stencil: edge wear ramps up as the
    surface curves, noise wanders through every value between. Mixing the two
    sides by that directly gives a blend everywhere the field is mid-way --
    which reads as the colours bleeding into each other rather than as a
    boundary. This is the level that divides them: below it is black's, above
    it is white's."""

    softness: float = 0.0
    """How wide the crossing is, in the same 0..1. Zero is a clean division --
    every texel belongs to one side or the other. Raise it to feather."""
    edge_wear: EdgeWearParams = field(default_factory=EdgeWearParams)
    noise: NoiseMaskParams = field(default_factory=NoiseMaskParams)
    white: "Slot" = field(default_factory=lambda: ColorSlot(_DEFAULT_WHITE))
    black: "Slot" = field(default_factory=lambda: ColorSlot(_DEFAULT_BLACK))

    @property
    def label(self) -> str:
        return self.name or self.auto_label

    @property
    def auto_label(self) -> str:
        return MASK_KINDS.get(self.kind, self.kind)

    @property
    def needs_bake(self) -> bool:
        """Edge wear reads the curvature bake; noise only needs the positions."""
        return self.kind == "edge_wear"

    def boundary_uniforms(self) -> dict[str, float]:
        """Where the two sides divide, for the shader. See :attr:`threshold`."""
        return {"u_threshold": self.threshold, "u_softness": self.softness}

    def slot(self, name: str) -> "Slot":
        return getattr(self, _checked(name))

    def with_slot(self, name: str, value: "Slot") -> "MaskLayer":
        return replace(self, **{_checked(name): value})


Slot = Union[ColorSlot, MaskLayer]

#: Where a node sits in the tree: the slots followed from the root. The root
#: itself is the empty path.
Path = tuple[str, ...]


def _checked(name: str) -> str:
    if name not in SLOTS:
        raise KeyError(f"a mask has {SLOTS} slots, not {name!r}")
    return name


def new_texture(name: str = "") -> Slot:
    """A fresh texture: one flat colour, and nothing else.

    A texture starts as the simplest thing that is still a texture. Turning it
    into a mask -- :func:`convert_slot` -- is what grows the tree under it, so
    every branch that exists is one that was asked for.
    """
    return ColorSlot(_DEFAULT_NEW, name=name)


def convert_slot(slot: Slot, kind: str) -> Slot:
    """The same slot as a different kind, keeping whatever carries over.

    A colour becoming a mask grows plain white and black underneath it, so what
    shows on the model is the mask itself -- which is the thing just asked for,
    and the thing to judge before deciding what to put under it. A mask
    becoming a colour takes its white side, if that side is a colour, since
    that is the one being described; everything under it goes.

    A name given to the slot survives either way: it names the part of the
    texture, not the kind of thing it happens to be at the moment.
    """
    if kind == COLOR_KIND:
        if isinstance(slot, ColorSlot):
            return slot
        white = slot.slot("white")
        return ColorSlot(
            white.color if isinstance(white, ColorSlot) else _DEFAULT_WHITE,
            name=slot.name,
        )

    if kind not in MASK_KINDS:
        raise KeyError(f"no such kind: {kind!r}")

    if isinstance(slot, MaskLayer):
        return replace(slot, kind=kind)
    return MaskLayer(
        kind=kind,
        name=slot.name,
        white=ColorSlot(_DEFAULT_WHITE),
        black=ColorSlot(_DEFAULT_BLACK),
    )


def kind_of(slot: Slot) -> str:
    """Which entry of :data:`SLOT_KINDS` this slot is."""
    return COLOR_KIND if isinstance(slot, ColorSlot) else slot.kind


def slot_at(root: Slot, path: Path) -> Slot:
    """The slot at ``path``, the root being the empty path."""
    current: Slot = root
    for step in path:
        if not isinstance(current, MaskLayer):
            raise KeyError(f"{path} runs through a colour, which has no slots")
        current = current.slot(step)
    return current


def mask_at(root: Slot, path: Path) -> MaskLayer:
    """The mask at ``path``. Raises if the path lands on a colour."""
    found = slot_at(root, path)
    if not isinstance(found, MaskLayer):
        raise KeyError(f"{path} is a colour, not a mask")
    return found


def set_slot(root: Slot, path: Path, value: Slot) -> Slot:
    """A copy of the tree with ``path`` replaced. The root path replaces all.

    Rebuilding rather than mutating keeps the panel honest: every edit produces
    a new tree, so a stale reference cannot quietly write into the live one.
    """
    if not path:
        return value

    head, rest = path[0], path[1:]
    parent = mask_at(root, ())
    if rest:
        child = parent.slot(head)
        if not isinstance(child, MaskLayer):
            raise KeyError(f"{path} runs through a colour, which has no slots")
        value = set_slot(child, rest, value)
    return parent.with_slot(head, value)


def walk(root: Slot, path: Path = ()) -> Iterator[tuple[Path, Slot]]:
    """Every slot in the tree, parents before children, white before black."""
    yield path, root
    if isinstance(root, MaskLayer):
        for name in SLOTS:
            yield from walk(root.slot(name), path + (name,))


def depth(root: Slot) -> int:
    """Levels of mask nesting. A flat colour is 0; a mask over two colours, 1."""
    if not isinstance(root, MaskLayer):
        return 0
    return 1 + max(depth(root.slot(name)) for name in SLOTS)


def mask_count(root: Slot) -> int:
    return sum(1 for _, slot in walk(root) if isinstance(slot, MaskLayer))


def can_nest(root: Slot, path: Path) -> bool:
    """Whether a mask may still be put in the slot at ``path``.

    Depth is measured from the root rather than from the slot, because the cost
    that :data:`MAX_DEPTH` guards -- a render target per level -- is a property
    of the whole tree.
    """
    return len(path) < MAX_DEPTH


def describe(slot: Slot) -> str:
    """One-line label for a slot, for the panel's rows and breadcrumbs."""
    return slot.label


def path_labels(root: Slot, path: Path) -> list[str]:
    """Breadcrumb text: the root's label, then a slot name per step."""
    crumbs = [describe(root)]
    for index, step in enumerate(path):
        crumbs.append(step.capitalize())
        node = slot_at(root, path[: index + 1])
        if isinstance(node, MaskLayer):
            crumbs[-1] = f"{step.capitalize()}: {node.label}"
    return crumbs
