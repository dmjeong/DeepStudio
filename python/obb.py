"""Normalized oriented-box labels and full-image geometry. Angles have no head/tail direction."""
import math
import numpy as np


def validate_corners(coordinates):
    """Accept four distinct, cyclic, convex corners in normalized image space."""
    points = np.asarray(coordinates, dtype=np.float64)
    if points.size != 8 or not np.isfinite(points).all():
        raise ValueError("OBB는 유한한 네 꼭짓점 좌표 8개가 필요합니다")
    points = points.reshape(4, 2)
    if np.any(points < 0) or np.any(points > 1):
        raise ValueError("OBB 꼭짓점 좌표는 0~1 범위여야 합니다")
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not (np.all(cross > 1e-12) or np.all(cross < -1e-12)):
        raise ValueError("OBB는 겹치지 않는 네 꼭짓점을 윤곽 순서로 지정해야 합니다")
    return points


def require_unrotated_image(path):
    from PIL import Image
    with Image.open(path) as image:
        if image.getexif().get(274, 1) not in (None, 1):
            raise ValueError(f"OBB 라벨과 이미지 방향 불일치 방지: EXIF 회전을 픽셀에 적용해 저장한 뒤 라벨링해 주세요: {path}")


def rectangle_from_three_points(points, image_size):
    """First two clicks define an edge, third click defines perpendicular depth."""
    width, height = image_size
    p = np.asarray(points, dtype=np.float64).reshape(3, 2) * [width, height]
    edge = p[1] - p[0]
    length = np.linalg.norm(edge)
    if length < 2:
        raise ValueError("OBB 첫 변은 2 px 이상이어야 합니다")
    normal = np.array([-edge[1], edge[0]]) / length
    depth = float(np.dot(p[2] - p[0], normal))
    if abs(depth) < 2:
        raise ValueError("OBB 높이는 2 px 이상이어야 합니다")
    corners = np.array([p[0], p[1], p[1] + depth * normal, p[0] + depth * normal]) / [width, height]
    return validate_corners(corners).ravel().tolist()


def crop_obb_row(row, image_size, crop_box):
    """Keep complete boxes; reject cut objects rather than create false background."""
    points = validate_corners(row[1:]) * image_size
    left, top, right, bottom = crop_box
    inside = ((points[:, 0] >= left) & (points[:, 0] <= right)
              & (points[:, 1] >= top) & (points[:, 1] <= bottom))
    if inside.all():
        points = (points - [left, top]) / [right-left, bottom-top]
        return [int(row[0]), *points.ravel().tolist()]
    # Separating axis test handles convex quadrilaterals, including diagonals
    # whose axis-aligned envelope overlaps the crop while the object does not.
    roi = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
    edges = np.roll(points, -1, axis=0) - points
    axes = [np.array([1., 0.]), np.array([0., 1.])]
    axes.extend(np.array([-edge[1], edge[0]]) / np.linalg.norm(edge) for edge in edges)
    for axis in axes:
        a, b = points @ axis, roi @ axis
        if min(a.max(), b.max()) - max(a.min(), b.min()) <= 1e-9:
            return None
    raise ValueError("중앙 크롭 경계가 OBB 객체를 자릅니다. 객체 전체가 포함되도록 크롭 크기를 늘려 주세요")
