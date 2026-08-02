"""Parameter blocks, mirroring ArmorPaint's split.

:class:`BakeParams` holds what ArmorPaint bakes into its Curvature Texture --
Strength, Radius, Offset, Smooth and Axis. Changing any of them re-runs the
rasterisation, exactly as ArmorPaint re-bakes the BAKE_CURVATURE node.

:class:`EdgeWearParams` holds the EdgeWear001 node group's own values, which are
arithmetic on the already-baked curvature and so update every frame.

Defaults are the values shipped in EdgeWear001.arm, not library defaults: the
group's own sockets default to Strength 0.5 / Radius 1.0 / Value 2.0, but the
*instance* placed in the material overrides Radius to 2.0 and Value to 1.0, and
the Curvature Texture node inside pins Offset to -2.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

RESOLUTIONS = (512, 1024, 2048, 4096, 8192)

#: Most of a decal's half-width that its edge fade may eat. Past this the band
#: cut from each side would meet in the middle and there would be no decal left.
MAX_FALLOFF = 0.5

#: make_bake.c's bake_axis, which masks curvature by facing direction.
BAKE_AXES = ("XYZ", "X", "Y", "Z", "-X", "-Y", "-Z")
AXIS_VECTORS = (
    (0.0, 0.0, 0.0),  # XYZ -- no masking at all
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
)


@dataclass
class BevelParams:
    """Blender's Bevel modifier, as far as :mod:`core.bevel` reproduces it.

    Fixed to the configuration these three knobs belong to: Affect = Edges,
    Width Type = Offset, Limit Method = Angle, Miter Outer/Inner = Sharp, Clamp
    Overlap off, Loop Slide on. The bevel is geometry for the bake only -- it
    never reaches the exported texture, which still addresses the original
    unbeveled mesh through its own UVs.
    """

    enabled: bool = False

    amount: float = 0.001
    """Blender's Amount, in metres, under the Offset width type: the distance
    from the original edge to each new boundary edge. 1 mm is the subtle bevel
    that lets an edge catch a highlight without reading as rounded."""

    segments: int = 3
    """Subdivisions across the bevel. 1 is a flat chamfer, 2-3 lightly rounded,
    more is smoother at the cost of geometry."""

    angle: float = 30.0
    """Degrees. Edges whose two faces diverge by more than this get beveled, so
    box corners qualify while coplanar edges splitting a flat surface do not."""

    def key(self) -> tuple:
        return (
            self.enabled,
            round(self.amount, 9),
            self.segments,
            round(self.angle, 4),
        )


@dataclass
class UnwrapParams:
    """Where the bake's atlas comes from, plus xatlas' packing options."""

    use_source_uvs: bool = True
    """Bake into the mesh's own UV map instead of generating a new atlas.

    On by default, and the setting you want for a Blender round-trip: the PNG
    then applies directly to the mesh you exported. Every option below is dead
    while this is on -- they configure the xatlas fallback, whose atlas fits
    nothing but the triangulated OBJ this app writes.
    """

    padding: int = 4
    """Texels of gutter left between charts. Must exceed the dilation radius."""

    brute_force: bool = False
    """Slower, tighter chart packing."""

    max_chart_area: float = 0.0
    """0 disables the limit; larger values force more, smaller charts."""

    normal_deviation_weight: float = 2.0
    """How eagerly xatlas cuts a seam when the surface bends."""

    def key(self) -> tuple:
        return (
            self.use_source_uvs,
            self.padding,
            self.brute_force,
            self.max_chart_area,
            self.normal_deviation_weight,
        )


@dataclass
class BakeParams:
    """The Curvature Texture bake. Changing any of these forces a re-bake."""

    resolution: int = 1024

    strength: float = 0.5
    """Multiplies the lifted derivative. ArmorPaint slider range 0..2."""

    radius: float = 2.0
    """Not a distance -- an exponent, ``pow(curvature, (1/radius) * 0.25)``.

    Larger values mean a smaller exponent, which lifts the derivative harder and
    so spreads the band wider. ArmorPaint's node slider range is 0..2; the
    EdgeWear001 group exposes it as 0..4 and ships it at 2.0.
    """

    offset: float = -2.0
    """Added as ``offset / 10`` after the lift. Negative crushes flat areas to
    black, which is what the shipped -2.0 is for. Range -2..2."""

    smooth: int = 1
    """Blur iterations over the baked curvature (ArmorPaint's bake_curv_smooth)."""

    axis: int = 0
    """Index into :data:`BAKE_AXES`. Non-XYZ multiplies by ``dot(n, axis)``."""

    dilation: int = 4
    """Texels of edge padding pushed outward past each chart border."""

    def curvature_key(self) -> tuple:
        return (
            self.resolution,
            round(self.strength, 6),
            round(self.radius, 6),
            round(self.offset, 6),
            self.smooth,
            self.axis,
        )

    def axis_vector(self) -> tuple[float, float, float]:
        return AXIS_VECTORS[self.axis if 0 <= self.axis < len(AXIS_VECTORS) else 0]

    def as_uniforms(self) -> dict[str, Any]:
        return {
            "u_strength": self.strength,
            "u_radius": self.radius,
            "u_offset": self.offset,
            "u_axis": self.axis_vector(),
        }


@dataclass
class EdgeWearParams:
    """The EdgeWear001 node group. Every one of these updates live.

    ``mask = clamp((curvature - noise * wear_amount) * contrast, 0, 1)``
    """

    value: float = 1.0
    """Group Input.Value. Feeds a x10 Math node into the noise Scale, so the
    noise scale is ``value * 10``. Group socket range 0..5."""

    wear_amount: float = 0.6
    """The "Wear Amount" multiply node."""

    contrast: float = 3.0
    """The "Contrast" multiply node, clamped to 0..1 afterwards."""

    # TEX_NOISE ("Wear Noise") settings, as shipped.
    detail: float = 5.0
    roughness: float = 0.7
    lacunarity: float = 5.0
    distortion: float = 0.15

    def as_uniforms(self) -> dict[str, Any]:
        return {
            "u_value": self.value,
            "u_wearAmount": self.wear_amount,
            "u_contrast": self.contrast,
            "u_detail": self.detail,
            "u_roughness": self.roughness,
            "u_lacunarity": self.lacunarity,
            "u_distortion": self.distortion,
        }


@dataclass
class DecalParams:
    """A normal-map decal stamped into the mesh's UV layout.

    Interactive placement uses a view-aligned GPU projector, as in a painting
    viewport. Once committed, the placement is also stated in UV space so it
    can be composited into the exported normal map.

    The image is never modified. Everything below is applied when the decal is
    composited into the normal map, so all of it stays live.
    """

    enabled: bool = True
    path: str = ""

    center_u: float = 0.5
    center_v: float = 0.5
    """Where the middle of the decal sits in UV space."""

    surface_face: int = -1
    """Triangle anchoring the continuous surface-wrap coordinate system."""

    scale: float = 0.25
    """Fraction of the atlas width the decal spans. Its height follows from the
    image's own aspect ratio, so a wide vent stays wide."""

    scale_x: float = 1.0
    """Additional width multiplier used by axis-constrained scaling."""

    scale_y: float = 1.0
    """Additional height multiplier used by axis-constrained scaling."""

    image_aspect: float = 1.0
    """The image's own width over its height, remembered when it is placed.

    A placement outlives any one look at the image, and there can be several on
    a mesh at once, so each carries the shape of the picture it stamps rather
    than asking a loader for it every time it is drawn."""

    surface_aspect: float = 1.0
    """World distance per unit u over world distance per unit v, where the decal
    sits -- :func:`core.decal.uv_aspect`. A UV rectangle is only a rectangle of
    the same shape on the model when this is 1, so it is divided back out when
    the decal's height is worked out. Measured from the mesh, not set by hand."""

    rotation: float = 0.0
    """Degrees, counter-clockwise in UV space."""

    falloff: float = 0.06
    """How much of the decal's edge is dropped, as a fraction of its half-width.

    A decal is a rectangle of an image and the surface it is stamped onto does
    not stop there, so unless the image fades to a flat normal by its own border
    the rectangle shows up as a seam. Most do not: a great many normal maps are
    drawn on a white or a black canvas, and neither of those is flat -- white
    decodes to a normal tilted 45 degrees along both axes.

    The outer ``falloff`` is cut and the fade is spread over the same width
    again inside it, so what is left is the middle of the image. The default is
    enough to take a thin canvas border off; 0 turns it off for an image that
    fades out by itself.
    """

    intensity: float = 1.0
    """How deep the bump reads. Scales the decal's surface *slope* rather than
    the stored vector, which is what makes 2.0 twice as steep instead of
    pushing the normal past horizontal and folding it over. 0 is flat, 1 is the
    map exactly as authored."""

    flip_green: bool = False
    """Flip the green channel, for a map baked in DirectX (-Y) convention.
    Everything here is OpenGL (+Y up), which is what Blender expects."""

    projector_center: tuple[float, float, float] | None = None
    projector_right: tuple[float, float, float] | None = None
    projector_up: tuple[float, float, float] | None = None
    projector_forward: tuple[float, float, float] | None = None
    projector_size: tuple[float, float] | None = None
    texture_index: int = -1
    """Optional index of a Texture-tab material used to colour this decal."""

    def loaded(self) -> bool:
        return bool(self.path)

    def active(self) -> bool:
        """Whether this decal contributes anything to the normal map."""
        return self.enabled and self.loaded()

    def key(self) -> tuple:
        return (
            self.enabled,
            self.path,
            round(self.center_u, 6),
            round(self.center_v, 6),
            self.surface_face,
            round(self.scale, 6),
            round(self.scale_x, 6),
            round(self.scale_y, 6),
            round(self.image_aspect, 6),
            round(self.surface_aspect, 6),
            round(self.rotation, 4),
            round(self.falloff, 6),
            round(self.intensity, 6),
            self.flip_green,
            self.projector_center,
            self.projector_right,
            self.projector_up,
            self.projector_forward,
            self.projector_size,
            self.texture_index,
        )

    def size(self) -> tuple[float, float]:
        """Width and height in UV units.

        Two aspects to answer to. The image's own decides the shape wanted on
        the surface -- a vent twice as wide as it is tall should read that way.
        The surface's decides what UV rectangle produces that shape, since a UV
        unit covers ``surface_aspect`` times as much world across as it does up.
        """
        width = self.scale * max(float(self.scale_x), 1e-6)
        shape = max(float(self.image_aspect), 1e-6)
        surface = max(float(self.surface_aspect), 1e-6)
        height = self.scale * surface / shape * max(float(self.scale_y), 1e-6)
        return (width, height)

    def as_uniforms(self) -> dict[str, Any]:
        width, height = self.size()
        return {
            "u_center": (self.center_u, self.center_v),
            "u_size": (width, height),
            "u_rotation": float(np.radians(self.rotation)),
            "u_falloff": float(np.clip(self.falloff, 0.0, MAX_FALLOFF)),
            "u_intensity": self.intensity,
            "u_flipGreen": 1.0 if self.flip_green else 0.0,
        }


@dataclass
class MeshInfo:
    """Human-readable facts about the loaded mesh, surfaced in the UI."""

    path: str = ""
    backend: str = ""
    vertices: int = 0
    faces: int = 0
    extents: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0
    watertight: bool = False
    has_uvs: bool = False
    """Whether the source file carried a UV map we can bake into."""
    uv_density: float = 0.0
    """UV units per metre of surface. Times the resolution gives texels per
    metre, which is what decides whether a bevel is wide enough to bake."""
    to_world: Any = None
    """The 4x4 that took the source into the app's Z-up world, or the identity.
    Invert it to speak the source file's own coordinates again."""
    notes: list[str] = field(default_factory=list)
