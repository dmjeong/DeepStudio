"""검출 패치 학습의 타일 데이터셋 (python/detection_tiles.py).

    큰 영상 + YOLO 라벨 ─▶ 겹치는 타일 + 타일 좌표 라벨
        · 타일 격자가 영상을 빠짐없이 덮고 밖으로 나가지 않는다
        · 상자는 타일 좌표로 옮겨지고, 조금만 걸친 상자는 버린다
        · 물체가 있는 타일은 전부, 배경 타일은 일부만 쓴다 · 원본은 바뀌지 않는다
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from detection_tiles import prepare_tiled_detection_dataset, tile_origins, tile_rows


def test_tile_origins_cover_the_axis_and_stay_inside():
    assert tile_origins(1600, 640, 128) == [0, 512, 960]        # 마지막 타일은 끝(1600 - 640)에 맞춘다
    assert tile_origins(640, 640, 128) == [0]
    assert tile_origins(300, 640, 128) == [0]                    # 영상이 타일보다 짧으면 하나
    for length in (641, 1000, 3580):
        origins = tile_origins(length, 640, 128)
        assert origins[0] == 0 and origins[-1] + 640 == length
        assert all(b - a <= 512 for a, b in zip(origins, origins[1:]))   # 겹침 128px 이상 유지


def test_tile_rows_move_boxes_into_tile_coordinates_and_drop_slivers():
    # 1000×1000 영상, 타일 (왼쪽 500, 위 0, 500×500)
    inside = [1, 0.75, 0.25, 0.1, 0.1]         # 중심 (750, 250), 100×100 → 타일 안 (250, 250)
    sliver = [0, 0.49, 0.25, 0.1, 0.1]         # 440~540 중 타일에 40px(40%)만 걸침 → 유지, 잘림
    barely = [0, 0.47, 0.25, 0.1, 0.1]         # 420~520 중 20px(20%)만 → 버림
    rows = tile_rows([inside, sliver, barely], (1000, 1000), (500, 0), (500, 500))
    assert len(rows) == 2
    assert rows[0][0] == 1 and rows[0][1:] == pytest.approx([0.5, 0.5, 0.2, 0.2])
    assert rows[1][1:] == pytest.approx([0.04, 0.5, 0.08, 0.2])            # 500~540 → 타일 0~40


def _dataset(root, size=(1600, 960)):
    rng = np.random.default_rng(0)
    for split, count in (("train", 2), ("val", 1)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for index in range(count):
            pixels = rng.integers(100, 140, (size[1], size[0], 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / "images" / split / f"image{index}.png")
            # 물체 하나: 왼쪽 위 타일에만 온전히 들어간다
            (root / "labels" / split / f"image{index}.txt").write_text("1 0.1 0.1 0.02 0.02\n", encoding="utf-8")
    return root


def test_dataset_keeps_object_tiles_and_a_few_background_tiles(tmp_path):
    root = _dataset(tmp_path / "data")
    before = sorted(path.name for path in (root / "images" / "train").iterdir())
    summary = prepare_tiled_detection_dataset(root, tmp_path / "tiles", tile=640)
    assert summary["tile"] == 640 and summary["overlap"] == 128 and summary["region"] == [1600, 960]
    # 영상마다 물체 타일 1개 + 배경 타일 1개 (격자는 3×2 = 6개)
    assert summary["splits"]["train"] == {"images": 2, "tiles": 4, "object_tiles": 2, "boxes": 2}
    assert summary["splits"]["val"] == {"images": 1, "tiles": 2, "object_tiles": 1, "boxes": 1}
    tile = Image.open(tmp_path / "tiles" / "images" / "train" / "image0_y00000_x00000.png")
    assert tile.size == (640, 640)
    label = (tmp_path / "tiles" / "labels" / "train" / "image0_y00000_x00000.txt").read_text(encoding="utf-8").split()
    # 영상 (160, 96) 중심, 32×19.2px → 타일 기준 (0.25, 0.15), (0.05, 0.03)
    assert label[0] == "1" and [float(v) for v in label[1:]] == pytest.approx([0.25, 0.15, 0.05, 0.03], abs=1e-5)
    assert sorted(path.name for path in (root / "images" / "train").iterdir()) == before      # 원본 그대로


def test_scale_shrinks_before_tiling_and_records_the_original_region(tmp_path):
    pytest.importorskip("cv2")
    root = _dataset(tmp_path / "data")
    summary = prepare_tiled_detection_dataset(root, tmp_path / "tiles", tile=640, scale=0.5)
    # 1600×960 → 800×480: 가로 타일 2개, 세로는 영상이 타일보다 짧아 1개
    assert summary["scale"] == 0.5 and summary["region"] == [1600, 960]
    tiles = sorted((tmp_path / "tiles" / "images" / "train").glob("image0_*.png"))
    assert {Image.open(path).size for path in tiles} == {(640, 480)}


def test_rejects_missing_train_folder_and_bad_settings(tmp_path):
    with pytest.raises(ValueError, match="학습 이미지"):
        prepare_tiled_detection_dataset(tmp_path, tmp_path / "tiles", tile=640)
    root = _dataset(tmp_path / "data")
    with pytest.raises(ValueError, match="배율"):
        prepare_tiled_detection_dataset(root, tmp_path / "tiles", tile=640, scale=1.5)
