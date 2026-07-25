"""Where the atlas the bake renders into comes from.

Two sources, both producing an :class:`UnwrapResult`:

:func:`source_uv_layout`  Use the UV map already on the mesh, as authored in
                          Blender and carried through the FBX. This is the
                          default, and the only option that lets the baked PNG
                          drop straight onto the original mesh -- see
                          :data:`UnwrapParams.use_source_uvs`.
:func:`unwrap`            Let xatlas chart and pack a brand new layout. Only
                          useful for a mesh that has no UVs at all; the result
                          is meaningless on any mesh but the triangulated one
                          we hand back from the OBJ export.

xatlas splits vertices along the seams it cuts, so it hands back a ``vmapping``
from its own vertex indices to the original ones. Everything downstream that
needs original-mesh topology -- notably the curvature estimator, whose
adjacency graph would be severed at every seam -- computes on the original mesh
and then gathers through ``vmapping``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
import xatlas

from .params import UnwrapParams


class SourceUVError(RuntimeError):
    """Raised when baking into source UVs was asked for but the mesh has none."""


@dataclass
class UnwrapResult:
    """A packed atlas plus the unwrapped vertex arrays that go with it."""

    vertices: np.ndarray   # (n, 3) float32 -- world-space positions
    normals: np.ndarray    # (n, 3) float32 -- original smooth vertex normals
    uvs: np.ndarray        # (n, 2) float32 -- 0..1 atlas coordinates
    faces: np.ndarray      # (m, 3) uint32
    vmapping: np.ndarray   # (n,)  uint32 -- index into the original vertices
    chart_count: int
    utilization: float
    atlas_size: tuple[int, int]
    source: str = "xatlas"
    """Which layout this is: ``"source"`` for the mesh's own UVs, else ``"xatlas"``."""

    def gather(self, per_original_vertex: np.ndarray) -> np.ndarray:
        """Pull a per-original-vertex attribute onto the unwrapped vertex set."""
        return np.ascontiguousarray(per_original_vertex[self.vmapping])


def source_uvs(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Return the mesh's own per-vertex UVs, or None if it has no usable set.

    trimesh keeps UVs as a per-vertex attribute, so a mesh imported with UV
    seams already has its vertices split along them -- ``merge_vertices``
    defaults to ``merge_tex=False`` and will not weld across differing UVs.
    """
    uv = getattr(getattr(mesh, "visual", None), "uv", None)
    if uv is None:
        return None
    uv = np.asarray(uv, dtype=np.float32)
    if uv.ndim != 2 or uv.shape[0] != len(mesh.vertices) or uv.shape[1] < 2:
        return None
    if not np.isfinite(uv[:, :2]).all():
        return None
    return np.ascontiguousarray(uv[:, :2])


def _welded_vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """Smooth vertex normals that ignore UV seams.

    The bake differentiates the *interpolated* normal per texel, so the normals
    it rasterises have to be continuous across a seam -- otherwise every seam
    reads as a crease and bakes a wear line down the middle of a flat face.
    Vertices split for UV reasons still sit at one position, so welding by
    position alone and scattering the result back is what keeps the seam
    invisible to the derivative.
    """
    unique, inverse = trimesh.grouping.unique_rows(mesh.vertices)
    if len(unique) == len(mesh.vertices):
        return np.ascontiguousarray(mesh.vertex_normals, dtype=np.float32)

    welded = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices)[unique],
        faces=inverse[np.asarray(mesh.faces)],
        process=False,
    )
    normals = np.asarray(welded.vertex_normals, dtype=np.float32)[inverse]
    return np.ascontiguousarray(normals, dtype=np.float32)


def source_uv_layout(mesh: trimesh.Trimesh, resolution: int) -> UnwrapResult:
    """Build an :class:`UnwrapResult` from the UV map already on ``mesh``.

    No charting, no packing and no reindexing: the vertex set is the mesh's own,
    so ``vmapping`` is the identity and the baked texture lines up with the
    source mesh in Blender exactly as authored.
    """
    uvs = source_uvs(mesh)
    if uvs is None:
        raise SourceUVError(
            "This mesh has no UV map, so there is nothing to bake into. Unwrap it "
            "in Blender (UV Editing workspace, or U > Smart UV Project) and "
            "re-export -- Blender's FBX exporter always includes UV layers. "
            "Alternatively turn off 'Bake into source UVs' to let xatlas "
            "generate a throwaway atlas, but that atlas only fits the OBJ this "
            "app exports, not your original mesh."
        )

    positions = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    normals = _welded_vertex_normals(mesh)

    # UV islands are separated by split vertices, so face adjacency -- which is
    # built from shared vertex indices -- is already severed at every seam.
    components = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces))
    )

    # Fraction of the 0..1 square the layout actually covers, from the summed
    # signed areas of the UV triangles.
    triangles = uvs[faces.astype(np.int64)]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    covered = float(np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]).sum()) / 2.0

    return UnwrapResult(
        vertices=positions,
        normals=normals,
        uvs=uvs,
        faces=faces,
        vmapping=np.arange(len(mesh.vertices), dtype=np.int64),
        chart_count=len(components),
        utilization=min(covered, 1.0),
        atlas_size=(int(resolution), int(resolution)),
        source="source",
    )


def unwrap(
    mesh: trimesh.Trimesh,
    params: UnwrapParams,
    resolution: int,
) -> UnwrapResult:
    """Run xatlas over ``mesh`` and pack into a ``resolution``-square atlas."""
    positions = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    indices = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    normals = np.ascontiguousarray(mesh.vertex_normals, dtype=np.float32)

    atlas = xatlas.Atlas()
    atlas.add_mesh(positions, indices, normals=normals)

    chart_options = xatlas.ChartOptions()
    chart_options.normal_deviation_weight = float(params.normal_deviation_weight)
    chart_options.max_chart_area = float(params.max_chart_area)
    chart_options.fix_winding = True

    pack_options = xatlas.PackOptions()
    pack_options.padding = int(params.padding)
    pack_options.resolution = int(resolution)
    pack_options.bruteForce = bool(params.brute_force)
    pack_options.blockAlign = True
    pack_options.rotate_charts = True
    pack_options.bilinear = True

    atlas.generate(chart_options=chart_options, pack_options=pack_options)

    vmapping, new_indices, uvs = atlas.get_mesh(0)

    vmapping = np.ascontiguousarray(vmapping, dtype=np.int64)
    faces = np.ascontiguousarray(new_indices, dtype=np.uint32)
    uvs = np.ascontiguousarray(uvs, dtype=np.float32)

    # xatlas reports UVs in atlas pixels, not normalised 0..1.
    width = float(atlas.width) or float(resolution)
    height = float(atlas.height) or float(resolution)
    if uvs.size and (uvs.max() > 1.5):
        uvs = uvs / np.array([width, height], dtype=np.float32)
    uvs = np.clip(uvs, 0.0, 1.0)

    return UnwrapResult(
        vertices=np.ascontiguousarray(positions[vmapping]),
        normals=np.ascontiguousarray(normals[vmapping]),
        uvs=uvs,
        faces=faces,
        vmapping=vmapping,
        chart_count=int(atlas.chart_count),
        utilization=float(atlas.utilization) if atlas.atlas_count else 0.0,
        atlas_size=(int(atlas.width), int(atlas.height)),
    )
