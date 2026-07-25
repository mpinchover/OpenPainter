"""Automatic UV unwrapping via xatlas.

xatlas charts and packs a brand new layout for a mesh that has no UVs. It
splits vertices along the seams it cuts, so it hands back a ``vmapping`` from
its own vertex indices to the original ones. Everything downstream that needs
original-mesh topology -- notably the curvature estimator, whose adjacency
graph would be severed at every seam -- computes on the original mesh and then
gathers through ``vmapping``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
import xatlas

from .params import UnwrapParams


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

    def gather(self, per_original_vertex: np.ndarray) -> np.ndarray:
        """Pull a per-original-vertex attribute onto the unwrapped vertex set."""
        return np.ascontiguousarray(per_original_vertex[self.vmapping])


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
