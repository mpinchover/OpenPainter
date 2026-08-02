"""Unfold connected mesh faces into one continuous decal coordinate system."""

from __future__ import annotations

from collections import deque

import numpy as np


def wrapped_vertices(vertices, faces, uvs, anchor_face: int, *, return_layout=False):
    """Atlas UV, unfolded surface UV and slope transform for connected faces.

    Each triangle is duplicated, because its atlas mapping and unfolded mapping
    are both per-face. The returned rows are ``u, v, surface_u, surface_v`` plus
    the four entries of the matrix that rotates a slope from the unfolded
    surface into that face's tangent-space UV axes.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float64)
    if not (0 <= int(anchor_face) < len(faces)):
        empty = np.empty((0, 8), dtype=np.float32)
        return (empty, {}) if return_layout else empty

    anchor_face = int(anchor_face)
    anchor_ids = faces[anchor_face]
    # Importers commonly split a geometric edge into separate vertices for
    # normals, materials or UV seams, and their decoded positions can differ by
    # a few floating-point crumbs. Match at a millionth of the model diagonal:
    # wide enough to reconnect those corners, far too narrow to bridge real
    # gaps in ordinary geometry.
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    weld = max(diagonal * 1e-4, 1e-8)
    vertex_keys = _welded_keys(vertices, weld)

    def geometric_key(vertex):
        return int(vertex_keys[int(vertex)])

    corner_keys = [
        [geometric_key(vertex) for vertex in triangle]
        for triangle in faces
    ]
    anchor_2d = _triangle_2d(vertices[anchor_ids])
    # Affine map from the anchor's metric unfolding to its authored UVs. Every
    # neighboring face inherits it, so the old placement is exact on the
    # anchor while the coordinate system remains continuous over UV seams.
    transform = _affine(anchor_2d, uvs[anchor_ids])

    edges: dict[tuple, list[int]] = {}
    for face_index, triangle in enumerate(faces):
        for i in range(3):
            edge = tuple(sorted((corner_keys[face_index][i], corner_keys[face_index][(i + 1) % 3])))
            edges.setdefault(edge, []).append(face_index)

    laid: dict[int, dict[tuple, np.ndarray]] = {
        anchor_face: {corner_keys[anchor_face][i]: anchor_2d[i] for i in range(3)}
    }
    queue = deque([anchor_face])
    while queue:
        current = queue.popleft()
        triangle = faces[current]
        triangle_keys = corner_keys[current]
        current_points = laid[current]
        for i in range(3):
            a, b = triangle_keys[i], triangle_keys[(i + 1) % 3]
            shared = tuple(sorted((a, b)))
            for neighbor in edges.get(shared, ()):
                if neighbor == current or neighbor in laid:
                    continue
                neighbor_ids = faces[neighbor]
                neighbor_keys = corner_keys[neighbor]
                third = next(v for v in neighbor_keys if v not in shared)
                current_third = next(v for v in triangle_keys if v not in shared)
                qa, qb = current_points[a], current_points[b]
                qc = _third_point(
                    qa, qb,
                    np.linalg.norm(vertices[int(neighbor_ids[neighbor_keys.index(third)])]
                                   - vertices[int(neighbor_ids[neighbor_keys.index(a)])]),
                    np.linalg.norm(vertices[int(neighbor_ids[neighbor_keys.index(third)])]
                                   - vertices[int(neighbor_ids[neighbor_keys.index(b)])]),
                    current_points[current_third],
                )
                candidate = {a: qa, b: qb, third: qc}
                candidate_triangle = np.asarray([candidate[key] for key in neighbor_keys])
                # A closed mesh cannot be flattened globally without eventually
                # folding back over itself. Stop that branch when it would
                # overlap an already laid triangle; otherwise one decal appears
                # again on a distant face and its outline breaks into fragments.
                overlaps = False
                for existing_face, existing in laid.items():
                    existing_triangle = np.asarray([
                        existing[key] for key in corner_keys[existing_face]
                    ])
                    if _triangle_overlap_area(candidate_triangle, existing_triangle) > weld * weld:
                        overlaps = True
                        break
                if overlaps:
                    continue
                laid[neighbor] = candidate
                queue.append(neighbor)

    rows = []
    layout = {}
    for face_index, points in laid.items():
        ids = faces[face_index]
        metric = np.asarray([points[key] for key in corner_keys[face_index]])
        surface = _apply_affine(metric, transform)
        layout[int(face_index)] = surface
        atlas = uvs[ids]
        matrix = _linear_map(atlas, surface)
        packed = np.concatenate([atlas, surface, np.tile(matrix.reshape(1, 4), (3, 1))], axis=1)
        rows.append(packed)
    data = np.ascontiguousarray(np.concatenate(rows), dtype=np.float32)
    return (data, layout) if return_layout else data


def wrapped_outline_segments(
    vertices, faces, layout: dict[int, np.ndarray], center, size, rotation,
    samples_per_edge: int = 32,
) -> list[np.ndarray]:
    """World-space line segments for a decal rectangle on its unfolded mesh."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    width, height = size
    angle = np.radians(float(rotation))
    cosine, sine = np.cos(angle), np.sin(angle)

    def placed(local):
        x, y = local
        return np.array([
            center[0] + x * cosine - y * sine,
            center[1] + x * sine + y * cosine,
        ])

    corners = [
        placed((-width * 0.5, -height * 0.5)),
        placed(( width * 0.5, -height * 0.5)),
        placed(( width * 0.5,  height * 0.5)),
        placed((-width * 0.5,  height * 0.5)),
    ]
    points = []
    steps = max(int(samples_per_edge), 2)
    for edge, start in enumerate(corners):
        end = corners[(edge + 1) % 4]
        for step in range(steps):
            points.append(start + (end - start) * (step / steps))
    points.append(points[0])

    world = [_world_at_surface(point, vertices, faces, layout) for point in points]
    return [
        np.asarray([first, second], dtype=np.float64)
        for first, second in zip(world, world[1:])
        if first is not None and second is not None
    ]


def _world_at_surface(point, vertices, faces, layout):
    for face_index, surface in layout.items():
        matrix = np.column_stack([surface[1] - surface[0], surface[2] - surface[0]])
        try:
            bary = np.linalg.solve(matrix, np.asarray(point) - surface[0])
        except np.linalg.LinAlgError:
            continue
        first, second = float(bary[0]), float(bary[1])
        if first >= -1e-5 and second >= -1e-5 and first + second <= 1.0 + 1e-5:
            triangle = vertices[faces[int(face_index)]]
            return triangle[0] + first * (triangle[1] - triangle[0]) \
                + second * (triangle[2] - triangle[0])
    return None


def _triangle_2d(points):
    a, b, c = points
    ab = b - a
    length = max(float(np.linalg.norm(ab)), 1e-12)
    x = float(np.dot(c - a, ab / length))
    y = float(np.sqrt(max(np.dot(c - a, c - a) - x * x, 0.0)))
    return np.array([[0.0, 0.0], [length, 0.0], [x, y]])


def _affine(source, target):
    return np.linalg.lstsq(np.column_stack([source, np.ones(3)]), target, rcond=None)[0]


def _apply_affine(points, transform):
    return np.column_stack([points, np.ones(len(points))]) @ transform


def _linear_map(source, target):
    ds = np.stack([source[1] - source[0], source[2] - source[0]])
    dt = np.stack([target[1] - target[0], target[2] - target[0]])
    try:
        return np.linalg.solve(ds, dt)
    except np.linalg.LinAlgError:
        return np.eye(2)


def _third_point(a, b, distance_a, distance_b, current_third):
    edge = b - a
    length = max(float(np.linalg.norm(edge)), 1e-12)
    along = (distance_a * distance_a - distance_b * distance_b + length * length) / (2 * length)
    height = np.sqrt(max(distance_a * distance_a - along * along, 0.0))
    perpendicular = np.array([-edge[1], edge[0]]) / length
    base = a + edge * (along / length)
    first, second = base + perpendicular * height, base - perpendicular * height
    side = _cross2(edge, current_third - a)
    return first if _cross2(edge, first - a) * side <= 0 else second


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _welded_keys(vertices: np.ndarray, tolerance: float) -> np.ndarray:
    """Stable geometric vertex ids, including points near hash-cell borders."""
    origin = vertices.min(axis=0)
    cells: dict[tuple[int, int, int], list[int]] = {}
    representatives: list[np.ndarray] = []
    keys = np.empty(len(vertices), dtype=np.int64)
    for index, point in enumerate(vertices):
        cell = tuple(np.floor((point - origin) / tolerance).astype(np.int64))
        found = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    for candidate in cells.get(neighbor, ()):
                        if np.linalg.norm(point - representatives[candidate]) <= tolerance:
                            found = candidate
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            found = len(representatives)
            representatives.append(point)
            cells.setdefault(cell, []).append(found)
        keys[index] = found
    return keys


def _triangle_overlap_area(subject: np.ndarray, clip: np.ndarray) -> float:
    """Area shared by two 2D triangles; touching along an edge is zero."""
    polygon = [np.asarray(point, dtype=np.float64) for point in subject]
    orientation = np.sign(_cross2(clip[1] - clip[0], clip[2] - clip[0])) or 1.0
    for index in range(3):
        edge_start = clip[index]
        edge_end = clip[(index + 1) % 3]
        incoming = polygon
        polygon = []
        if not incoming:
            return 0.0

        def inside(point):
            return orientation * _cross2(edge_end - edge_start, point - edge_start) >= -1e-12

        previous = incoming[-1]
        previous_inside = inside(previous)
        for current in incoming:
            current_inside = inside(current)
            if current_inside != previous_inside:
                direction = current - previous
                denominator = _cross2(direction, edge_end - edge_start)
                if abs(denominator) > 1e-15:
                    amount = _cross2(edge_start - previous, edge_end - edge_start) / denominator
                    polygon.append(previous + direction * amount)
            if current_inside:
                polygon.append(current)
            previous, previous_inside = current, current_inside
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    for first, second in zip(polygon, polygon[1:] + polygon[:1]):
        area += _cross2(first, second)
    return abs(area) * 0.5
