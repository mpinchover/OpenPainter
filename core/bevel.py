"""An approximation of Blender's Bevel modifier, applied before the bake.

Why this exists: the curvature bake differentiates the interpolated vertex
normal per texel, so a perfectly sharp edge -- zero width, no geometry -- has
nowhere to put a gradient. Welding then averages the corner normals of the
faces meeting at that edge, and the resulting swing gets smeared across the
*whole* face, which bakes entire faces white instead of drawing a line along
the edge. Replacing sharp edges with a narrow strip of geometry confines that
gradient to the strip, which is what produces a thin wear line.

The parameters mirror the modifier's, and the fixed settings match the ones
described in Blender's docs for this configuration: Affect = Edges, Width Type
= Offset, Limit Method = Angle, Miter Outer/Inner = Sharp, Clamp Overlap off,
Loop Slide on.

**This is an approximation, not a port.** Blender's real implementation is
``source/blender/bmesh/tools/bmesh_bevel.cc`` -- 8,485 lines over BMesh, with
five distinct corner mesh kinds and an Eigen least-squares solve for matching
offsets around a vertex. What is faithful here:

* Edge selection by dihedral angle, and the Offset width convention -- the new
  boundary edge sits ``amount`` from the original edge, measured in the face
  plane.
* Loop slide, which falls out of solving each corner against both of its edges:
  a corner between a beveled and an unbeveled edge slides along the unbeveled
  one instead of pulling away from it.
* The circular profile, and ``segments = 1`` giving a flat chamfer.

Where it diverges: corners where three or more beveled edges meet are filled
with a fan lifted onto the corner sphere, not Blender's Grid Fill, so the
topology there is different and the surface is close but not identical. Miters
are whatever the corner solve produces rather than an explicit Sharp
construction. Since nothing here is exported -- the bevel exists only to feed
the bake, and the texture still lands on the original unbeveled mesh via its
own UVs -- the corner difference costs a little accuracy in the wear falloff at
corners, not correctness anywhere downstream.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .params import BevelParams
from .uv_unwrap import source_uvs

#: Below this the corner solve is singular (the two edges are collinear) and we
#: fall back to offsetting along a single edge normal.
_SINGULAR = 1e-9


def _unit(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(lengths, 1e-12, None)


class _Topology:
    """Position-welded triangles with per-corner UVs.

    UVs live on face corners rather than vertices for the duration of the
    bevel. Seam splits would otherwise fragment the topology we need to walk --
    the dihedral angle at an edge is a property of the geometry, not of the
    atlas.
    """

    def __init__(self, mesh: trimesh.Trimesh):
        unique, inverse = trimesh.grouping.unique_rows(mesh.vertices)
        self.positions = np.asarray(mesh.vertices, dtype=np.float64)[unique]
        self.faces = inverse[np.asarray(mesh.faces)]

        uv = source_uvs(mesh)
        self.corner_uvs = None if uv is None else uv[np.asarray(mesh.faces)].astype(np.float64)

        corners = self.positions[self.faces]
        cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        self.normals = _unit(cross)
        self.areas = np.linalg.norm(cross, axis=1) * 0.5

    def edge_faces(self) -> dict[tuple[int, int], list[tuple[int, int]]]:
        """Map each undirected edge to the (face, slot) pairs that use it.

        Slot ``i`` is the edge from corner ``i`` to corner ``i + 1``.
        """
        table: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for slot in range(3):
            starts = self.faces[:, slot]
            ends = self.faces[:, (slot + 1) % 3]
            for face, (a, b) in enumerate(zip(starts, ends)):
                table.setdefault((min(a, b), max(a, b)), []).append((face, slot))
        return table


def _select_edges(topo: _Topology, angle_degrees: float) -> tuple[np.ndarray, dict]:
    """Tag edges whose two faces diverge by more than the angle threshold.

    Blender does this in ``MOD_bevel.cc`` before ever calling the bevel
    operator: it compares ``dot(n1, n2)`` against ``cos(angle)`` and tags the
    edge. Only manifold edges qualify -- a boundary or non-manifold edge has no
    well-defined dihedral angle.
    """
    limit = float(np.cos(np.radians(angle_degrees)))
    beveled = np.zeros(topo.faces.shape, dtype=bool)
    pairs: dict[tuple[int, int], tuple[int, int, int, int]] = {}

    for key, uses in topo.edge_faces().items():
        if len(uses) != 2:
            continue
        (face_a, slot_a), (face_b, slot_b) = uses
        if float(np.dot(topo.normals[face_a], topo.normals[face_b])) >= limit:
            continue
        beveled[face_a, slot_a] = True
        beveled[face_b, slot_b] = True

        # Orient the pair so face_a is the one traversing the edge low -> high;
        # every strip is then wound consistently against face_a's inset edge.
        low, high = key
        if topo.faces[face_a, slot_a] != low:
            face_a, slot_a, face_b, slot_b = face_b, slot_b, face_a, slot_a
        pairs[key] = (face_a, slot_a, face_b, slot_b)

    return beveled, pairs


class _Sector:
    """One run of faces between two beveled edges, around a single vertex.

    Blender calls the equivalent a ``BoundVert``. It exists because the offset
    corner is a property of the *sector*, not of any one face: the faces inside
    a sector are joined by unbeveled edges, and if each solved its own corner
    they would disagree about where their shared edge now ends and tear the
    mesh open. Solving once per sector is what keeps the result watertight.
    """

    __slots__ = ("point", "corners", "exit_edge")

    def __init__(self, point, corners, exit_edge):
        self.point = point
        self.corners = corners          # [(face, corner), ...] in ring order
        self.exit_edge = exit_edge      # beveled edge leaving this sector, or None


def _edge_inward(topo: _Topology, face: int, corner: int, outgoing: bool) -> np.ndarray:
    """In-plane normal of one of a corner's two edges, pointing into the face."""
    positions, faces = topo.positions, topo.faces
    here = positions[faces[face, corner]]
    if outgoing:
        direction = positions[faces[face, (corner + 1) % 3]] - here
    else:
        direction = here - positions[faces[face, (corner + 2) % 3]]
    return _unit(np.cross(topo.normals[face], direction))


def _solve_corners(topo: _Topology, beveled: np.ndarray, amount: float,
                   incident: dict[int, list[tuple[int, int]]], touched: np.ndarray):
    """Offset every corner, one solve per sector, and return the sectors too.

    Blender's Offset width type measures perpendicular from the original edge to
    the new boundary edge, so a sector's corner is the point at ``amount`` from
    *both* of its bounding beveled edges. A sector bounded by only one beveled
    edge just steps off that one. Faces inside the sector inherit the answer,
    which is what produces Loop Slide: a corner whose other edges are unbeveled
    slides along them instead of lifting away.
    """
    inset = topo.positions[topo.faces].copy()
    sectors: dict[int, list[_Sector]] = {}

    for vertex in np.flatnonzero(touched):
        vertex = int(vertex)
        ring = _face_ring(topo, vertex, incident[vertex])
        if ring is None:
            continue

        # A cut sits on the outgoing edge of ring[i], shared with ring[i + 1].
        # The edge leaving corner `c` is slot `c` -- slot i spans corner i to i + 1.
        cuts = [
            index for index, (face, corner) in enumerate(ring)
            if beveled[face, corner]
        ]
        if not cuts:
            continue

        found: list[_Sector] = []
        for position, cut in enumerate(cuts):
            following = cuts[(position + 1) % len(cuts)]
            # Faces strictly after this cut, up to and including the next one.
            members, index = [], (cut + 1) % len(ring)
            while True:
                members.append(ring[index])
                if index == following:
                    break
                index = (index + 1) % len(ring)
                if len(members) > len(ring):
                    break

            first_face, first_corner = members[0]
            last_face, last_corner = members[-1]
            left = _edge_inward(topo, first_face, first_corner, outgoing=False)
            right = _edge_inward(topo, last_face, last_corner, outgoing=True)

            # Solve in the sector's own plane so a sector spanning several faces
            # -- two triangles of one quad, most often -- offsets from both of
            # its bounding edges rather than from whichever face owns each.
            plane = _unit(
                (topo.normals[[face for face, _ in members]]
                 * topo.areas[[face for face, _ in members]][:, None]).sum(axis=0)
            )
            point = _offset_point(
                topo.positions[vertex], plane, left, right, amount,
                single=(len(cuts) == 1),
            )

            exit_other = int(topo.faces[last_face, (last_corner + 1) % 3])
            found.append(_Sector(
                point, members,
                (min(vertex, exit_other), max(vertex, exit_other)),
            ))

        for sector in found:
            for face, corner in sector.corners:
                inset[face, corner] = sector.point
        sectors[vertex] = found

    return inset, sectors


def _offset_point(origin, plane, left, right, amount: float, single: bool):
    """The point ``amount`` inside both bounding edges, within ``plane``."""
    # Project the two edge normals into the sector plane; they were each
    # measured in their own face, which may tilt away from it.
    left = left - plane * float(np.dot(left, plane))
    right = right - plane * float(np.dot(right, plane))
    len_left, len_right = np.linalg.norm(left), np.linalg.norm(right)
    if len_left < _SINGULAR or len_right < _SINGULAR:
        return origin
    left, right = left / len_left, right / len_right

    if single:
        # One beveled edge terminating here: only that edge constrains us.
        return origin + amount * left

    basis_u = left
    basis_v = np.cross(plane, basis_u)
    matrix = np.array([
        [float(np.dot(left, basis_u)), float(np.dot(left, basis_v))],
        [float(np.dot(right, basis_u)), float(np.dot(right, basis_v))],
    ])
    if abs(float(np.linalg.det(matrix))) < 1e-7:
        # Collinear bounding edges (a flat sector); step off one of them.
        return origin + amount * left
    solution = np.linalg.solve(matrix, np.array([amount, amount]))
    return origin + solution[0] * basis_u + solution[1] * basis_v


def _arc(start: np.ndarray, end: np.ndarray,
         normal_a: np.ndarray, normal_b: np.ndarray, segments: int) -> np.ndarray:
    """Circular profile from ``start`` to ``end``, tangent to both faces.

    The centre is the point that sits at equal radius along each face normal,
    which is what makes the strip meet both faces tangentially instead of
    creasing. One segment degenerates to the straight chord, which is exactly
    Blender's flat chamfer.
    """
    if segments <= 1:
        return np.stack([start, end])

    delta_n = normal_a - normal_b
    denominator = float(np.dot(delta_n, delta_n))
    steps = np.linspace(0.0, 1.0, segments + 1)[:, None]
    if denominator < _SINGULAR:
        return start + (end - start) * steps

    radius = float(np.dot(start - end, delta_n)) / denominator
    centre = start - radius * normal_a

    spoke_a, spoke_b = start - centre, end - centre
    len_a, len_b = np.linalg.norm(spoke_a), np.linalg.norm(spoke_b)
    if len_a < _SINGULAR or len_b < _SINGULAR:
        return start + (end - start) * steps

    cosine = float(np.clip(np.dot(spoke_a, spoke_b) / (len_a * len_b), -1.0, 1.0))
    theta = float(np.arccos(cosine))
    if theta < 1e-6:
        return start + (end - start) * steps

    # Slerp the spoke, and lerp its length so an asymmetric pair still closes.
    sine = np.sin(theta)
    weight_a = np.sin((1.0 - steps) * theta) / sine
    weight_b = np.sin(steps * theta) / sine
    swept = weight_a * spoke_a + weight_b * spoke_b
    lengths = len_a + (len_b - len_a) * steps
    return centre + _unit(swept) * lengths


def _face_uvs(topo: _Topology, face: int, points: np.ndarray) -> np.ndarray:
    """Interpolate face ``face``'s corner UVs at arbitrary points on its plane.

    Every vertex the bevel creates lies within the band between a face's
    original outline and its inset outline, so its UV lands inside that face's
    UV island. That is what keeps the wear visible on the *original* mesh: the
    strip occupies the outer rim of the island, which on the unbeveled model is
    the texture right up against the edge.
    """
    triangle = topo.positions[topo.faces[face]]
    origin = triangle[0]
    edge_1, edge_2 = triangle[1] - origin, triangle[2] - origin

    d11 = float(np.dot(edge_1, edge_1))
    d12 = float(np.dot(edge_1, edge_2))
    d22 = float(np.dot(edge_2, edge_2))
    denominator = d11 * d22 - d12 * d12
    relative = np.atleast_2d(points) - origin
    if abs(denominator) < _SINGULAR:
        return np.repeat(topo.corner_uvs[face][:1], len(relative), axis=0)

    p1 = relative @ edge_1
    p2 = relative @ edge_2
    beta = (d22 * p1 - d12 * p2) / denominator
    gamma = (d11 * p2 - d12 * p1) / denominator
    weights = np.stack([1.0 - beta - gamma, beta, gamma], axis=-1)
    return weights @ topo.corner_uvs[face]


class _Soup:
    """Accumulates triangles with per-corner UVs, welded once at the end."""

    def __init__(self, topo: _Topology):
        self.topo = topo
        self._tris: list[np.ndarray] = []
        self._uvs: list[np.ndarray] = []

    def add(self, corners: np.ndarray, face: int) -> None:
        """Add one triangle, taking its UVs from original face ``face``."""
        corners = np.asarray(corners, dtype=np.float64)
        self._tris.append(corners)
        if self.topo.corner_uvs is not None:
            self._uvs.append(_face_uvs(self.topo, face, corners))

    def add_quad(self, a, b, c, d, face: int) -> None:
        self.add(np.stack([a, b, c]), face)
        self.add(np.stack([a, c, d]), face)

    def build(self) -> trimesh.Trimesh:
        vertices = np.concatenate(self._tris).reshape(-1, 3)
        faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
        visual = None
        if self._uvs:
            visual = trimesh.visual.TextureVisuals(
                uv=np.concatenate(self._uvs).reshape(-1, 2)
            )
        soup = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
        # Weld the soup back into a mesh, keeping UV seams split as always.
        soup.merge_vertices()
        soup.update_faces(soup.nondegenerate_faces())
        soup.remove_unreferenced_vertices()
        return soup


def _drop_repeats(loop: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    """Collapse consecutive coincident points, treating the loop as closed."""
    keep = [0]
    for index in range(1, len(loop)):
        if np.sum((loop[index] - loop[keep[-1]]) ** 2) > tolerance:
            keep.append(index)
    if len(keep) > 1 and np.sum((loop[keep[-1]] - loop[keep[0]]) ** 2) <= tolerance:
        keep.pop()
    return loop[keep]


def _face_ring(topo: _Topology, vertex: int,
               incident: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    """Order the faces around ``vertex`` by walking shared edges.

    Returns None for a fan that is not a closed loop -- a boundary vertex has
    no corner to fill, so the caller skips it rather than guessing.
    """
    start = incident[0]
    ring = [start]
    face, corner = start
    for _ in range(len(incident)):
        # Step across this face's outgoing edge into its neighbour.
        outgoing = topo.faces[face, (corner + 1) % 3]
        following = None
        for candidate, candidate_corner in incident:
            if (candidate, candidate_corner) == (face, corner):
                continue
            if topo.faces[candidate, (candidate_corner + 2) % 3] == outgoing:
                following = (candidate, candidate_corner)
                break
        if following is None:
            return None
        if following == start:
            return ring
        ring.append(following)
        face, corner = following
    return None


def bevel(mesh: trimesh.Trimesh, params: BevelParams) -> trimesh.Trimesh:
    """Return ``mesh`` with a bevel on every edge sharper than the threshold.

    Returns the mesh untouched when the bevel would be a no-op, so callers can
    hand everything through this without checking first.
    """
    if not params.enabled or params.amount <= 0.0 or params.segments < 1:
        return mesh

    topo = _Topology(mesh)
    beveled, pairs = _select_edges(topo, params.angle)
    if not pairs:
        return mesh

    segments = int(params.segments)

    incident_faces: dict[int, list[tuple[int, int]]] = {}
    for face, corners in enumerate(topo.faces):
        for corner, vertex in enumerate(corners):
            incident_faces.setdefault(int(vertex), []).append((face, corner))

    touched = np.zeros(len(topo.positions), dtype=bool)
    for face, slot in zip(*np.nonzero(beveled)):
        touched[topo.faces[face, slot]] = True
        touched[topo.faces[face, (slot + 1) % 3]] = True

    inset, sectors = _solve_corners(
        topo, beveled, float(params.amount), incident_faces, touched
    )
    soup = _Soup(topo)

    # 1. Every face, shrunk back from its beveled edges.
    for face in range(len(topo.faces)):
        soup.add(inset[face], face)

    # 2. A strip of `segments` quads along each beveled edge. Both ends of the
    #    edge get the same profile, so the strip is a ruled surface between them.
    arcs: dict[tuple[int, int], np.ndarray] = {}
    for (low, high), (face_a, slot_a, face_b, slot_b) in pairs.items():
        normal_a, normal_b = topo.normals[face_a], topo.normals[face_b]
        for vertex in (low, high):
            corner_a = int(np.flatnonzero(topo.faces[face_a] == vertex)[0])
            corner_b = int(np.flatnonzero(topo.faces[face_b] == vertex)[0])
            arcs[(low, high, vertex)] = _arc(
                inset[face_a, corner_a], inset[face_b, corner_b],
                normal_a, normal_b, segments,
            )

        arc_low, arc_high = arcs[(low, high, low)], arcs[(low, high, high)]
        # Quads nearer face_a take its UVs, the rest face_b's, so no quad ever
        # straddles two UV islands.
        midpoint = (segments + 1) // 2
        for step in range(segments):
            owner = face_a if step < midpoint else face_b
            soup.add_quad(
                arc_high[step], arc_low[step],
                arc_low[step + 1], arc_high[step + 1],
                owner,
            )

    # 3. Fill the hole each beveled corner leaves around its original vertex.
    for vertex, found in sectors.items():
        # One point per sector, joined by the arc of the beveled edge between
        # consecutive sectors -- that hole boundary is exactly what the inset
        # faces and the strips left open.
        loop: list[np.ndarray] = []
        for sector in found:
            loop.append(sector.point)
            arc = arcs.get(sector.exit_edge + (vertex,))
            if arc is None:
                continue
            # The arc runs face_a -> face_b; orient it to leave this sector.
            if np.sum((arc[0] - sector.point) ** 2) > np.sum((arc[-1] - sector.point) ** 2):
                arc = arc[::-1]
            loop.extend(arc[1:-1])

        boundary = _drop_repeats(np.asarray(loop))
        if len(boundary) < 3:
            continue

        ring = [pair for sector in found for pair in sector.corners]
        origin = topo.positions[vertex]
        outward = _unit(topo.normals[[face for face, _ in ring]].sum(axis=0))
        centres = topo.positions[topo.faces[[face for face, _ in ring]]].mean(axis=1)

        def owner_of(triangle: np.ndarray) -> int:
            """Whichever incident face the patch sits nearest.

            Keeps each corner triangle inside a single UV island; a triangle
            spanning two islands would sample garbage from between them.
            """
            index = int(np.argmin(np.linalg.norm(centres - triangle.mean(axis=0), axis=1)))
            return ring[index][0]

        # A three-sided hole is the flat chamfer's corner: emit it as one
        # planar triangle rather than fanning it, matching Blender.
        if len(boundary) == 3:
            if float(np.dot(np.cross(boundary[1] - boundary[0],
                                     boundary[2] - boundary[0]), outward)) < 0.0:
                boundary = boundary[::-1]
            soup.add(boundary, owner_of(boundary))
            continue

        # Otherwise fan from an apex lifted onto the corner sphere, so the
        # patch stays round instead of collapsing to a flat cap.
        spokes = boundary - origin
        radius = float(np.linalg.norm(spokes, axis=1).mean())
        centroid = spokes.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        apex = origin + (centroid / norm * radius if norm > _SINGULAR else centroid)

        winding = np.cross(boundary - apex, np.roll(boundary, -1, axis=0) - apex).sum(axis=0)
        if float(np.dot(winding, outward)) < 0.0:
            boundary = boundary[::-1]

        for index in range(len(boundary)):
            triangle = np.stack([apex, boundary[index], boundary[(index + 1) % len(boundary)]])
            soup.add(triangle, owner_of(triangle))

    return soup.build()
