// Work in image pixels so rectangles remain perpendicular on non-square images.
export function obbFromThreePoints(points: number[], width: number, height: number): number[] {
  if (points.length !== 6 || ![...points, width, height].every(Number.isFinite) || width <= 0 || height <= 0)
    throw new Error("OBB 좌표 또는 이미지 크기가 올바르지 않습니다.");
  const [x, y, bx, by, cx, cy] = points.map((v, i) => v * (i % 2 ? height : width));
  const dx = bx - x, dy = by - y, length = Math.hypot(dx, dy);
  if (length < 2) throw new Error("OBB 첫 변은 2 px 이상이어야 합니다.");
  const nx = -dy / length, ny = dx / length, depth = (cx - x) * nx + (cy - y) * ny;
  if (Math.abs(depth) < 2) throw new Error("OBB 높이는 2 px 이상이어야 합니다.");
  const result = [x, y, bx, by, bx + depth * nx, by + depth * ny, x + depth * nx, y + depth * ny]
    .map((v, i) => v / (i % 2 ? height : width));
  if (result.some(v => v < 0 || v > 1)) throw new Error("OBB 꼭짓점이 이미지 밖으로 나갑니다. 다시 그려 주세요.");
  return result;
}
