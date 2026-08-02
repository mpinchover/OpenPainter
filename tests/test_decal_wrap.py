"""Continuous decal coordinates across geometry edges and UV seams."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.decal_wrap import wrapped_outline_segments, wrapped_vertices  # noqa: E402


def folded_seam():
    """Two perpendicular triangles sharing positions but not vertex indices."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [1, 0, 0], [0, 0, 0], [0, 0, 1],
    ], dtype=float)
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    # Separate UV islands: a plain UV rectangle would stop at the first face.
    uvs = np.array([
        [0.1, 0.1], [0.4, 0.1], [0.1, 0.4],
        [0.8, 0.8], [0.6, 0.8], [0.8, 0.6],
    ], dtype=float)
    return vertices, faces, uvs


def test_wrap_reaches_the_face_across_a_disconnected_uv_seam():
    data = wrapped_vertices(*folded_seam(), anchor_face=0)
    assert len(data) == 6, "both triangles are emitted, not only the anchor island"


def test_unfolded_coordinates_are_continuous_on_the_shared_edge():
    data = wrapped_vertices(*folded_seam(), anchor_face=0)
    first_surface = data[:3, 2:4]
    second_surface = data[3:, 2:4]

    # Face 1 lists the same geometric edge in reverse. Its continuous decal
    # coordinates must meet face 0 even though its authored UVs are elsewhere.
    assert np.allclose(second_surface[0], first_surface[1])
    assert np.allclose(second_surface[1], first_surface[0])


def test_anchor_face_keeps_the_existing_decal_coordinate_system():
    _, _, uvs = folded_seam()
    data = wrapped_vertices(*folded_seam(), anchor_face=0)
    assert np.allclose(data[:3, 2:4], uvs[:3])


def test_nearly_coincident_split_vertices_are_still_one_geometric_edge():
    vertices, faces, uvs = folded_seam()
    vertices[3] += np.array([2e-5, 0.0, 0.0])
    vertices[4] += np.array([0.0, -2e-5, 0.0])

    data = wrapped_vertices(vertices, faces, uvs, anchor_face=0)
    assert len(data) == 6, "import precision must not cut the decal at the diagonal"


def test_inconsistent_triangle_winding_does_not_remove_half_the_decal():
    vertices, faces, uvs = folded_seam()
    faces[1] = faces[1][::-1]

    data = wrapped_vertices(vertices, faces, uvs, anchor_face=0)
    assert len(data) == 6


def test_selection_outline_uses_the_same_wrapped_faces_as_the_decal():
    vertices, faces, uvs = folded_seam()
    _, layout = wrapped_vertices(vertices, faces, uvs, 0, return_layout=True)
    segments = wrapped_outline_segments(
        vertices, faces, layout, center=(0.25, 0.1), size=(0.2, 0.2), rotation=0,
    )
    points = np.concatenate(segments)

    assert (points[:, 1] > 0).any(), "outline reaches the anchor face"
    assert (points[:, 2] > 0).any(), "outline continues across the folded face"
