"""SAM2 자동 분할(격자 점 → 후보 거르기 → 합집합)과 추론 화면 연결을 검증한다.

실제 SAM2 대신 점 위치에 따라 마스크를 돌려주는 가짜 디코더를 써서 규칙만 확인한다.
공식 가중치가 있으면(설치본·가중치 폴더) 실제 SAM2로 CPU 추론까지 확인한다.
"""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "gui"), str(Path(__file__).resolve().parents[1] / "python")]
torch = pytest.importorskip("torch")

from core.sam2_automatic import automatic_mask, grid_points, settings_of  # noqa: E402

LOW = 64  # 가짜 디코더의 저해상도 마스크 크기


class FakeSam:
    """점이 결함 사각형 안이면 그 사각형, 아니면 배경 전체(높은 점수)를 돌려준다 (실제 SAM2와 같은 경향).

    background_on_defect: 배경 마스크가 결함 자리에 갖는 logit (실제 SAM2는 결함을 빼서 음수, 약하게 섞이면 0~1).
    """

    def __init__(self, defects, width, height, background_on_defect=-5.0):
        self.defects, self.width, self.height = defects, width, height
        self.background_on_defect = background_on_defect
        self.calls = 0
        self.device = "cpu"
        self._orig_hw = [(height, width)]
        self._features = {"image_embed": [torch.zeros(1)], "high_res_feats": [[torch.zeros(1)], [torch.zeros(1)]]}
        self._transforms = SimpleNamespace(transform_coords=lambda c, normalize, orig_hw: c)
        self.model = SimpleNamespace(sam_prompt_encoder=self._prompt, sam_mask_decoder=self._decode)

    def set_image(self, image):
        pass

    class _Prompt:
        def __call__(self, points, boxes, masks):
            return points[0], None

    @property
    def _prompt(self):
        prompt = FakeSam._Prompt()
        prompt.get_dense_pe = lambda: None
        return prompt

    def _decode(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings,
                multimask_output, repeat_image, high_res_features):
        self.calls += 1
        x, y = sparse_prompt_embeddings[0, 0].tolist()
        logits = torch.full((1, 1, LOW, LOW), 5.0)  # 배경 전체
        for (x0, y0, x1, y1), _score in self.defects:
            logits[0, 0, int(y0 * LOW / self.height):int(np.ceil(y1 * LOW / self.height)),
                   int(x0 * LOW / self.width):int(np.ceil(x1 * LOW / self.width))] = self.background_on_defect
        score = .99
        for (x0, y0, x1, y1), defect_score in self.defects:
            if x0 <= x < x1 and y0 <= y < y1:
                logits = torch.full((1, 1, LOW, LOW), -5.0)
                logits[0, 0, int(y0 * LOW / self.height):int(np.ceil(y1 * LOW / self.height)),
                       int(x0 * LOW / self.width):int(np.ceil(x1 * LOW / self.width))] = 5.0
                score = defect_score
        return logits, torch.tensor([[score]]), None, None


def test_grid_points_match_the_cpp_sdk_rule():
    points = grid_points(640, 480, 2)
    assert points.tolist() == [[160, 120], [480, 120], [160, 360], [480, 360]]
    with pytest.raises(ValueError):
        settings_of({"grid": 33})


def test_background_masks_are_rejected_by_area_and_covered_points_are_skipped():
    sam = FakeSam([((100, 100, 300, 160), .9), ((440, 300, 500, 360), .95)], 640, 480)
    mask, probabilities, info = automatic_mask(sam, np.zeros((480, 640, 3), np.uint8),
                                               {"grid": 16, "min_score": .5, "max_area_percent": 10})
    # 배경 마스크(100%)는 버리고, 두 결함만 합친다
    assert info["accepted"] == 2 and mask.mean() < .1
    assert mask[int(130 * LOW / 480), int(200 * LOW / 640)] == 1 and mask[0, 0] == 0
    # 배경 마스크 하나가 거의 모든 점을 덮으므로 256점 중 몇 번만 디코딩한다
    assert info["decoded"] == sam.calls and sam.calls <= 4 and info["candidates"] == 256
    assert probabilities.shape == (2, LOW, LOW) and probabilities[1][mask == 1].min() > .99
    # 최대 면적 100%(SDK 규칙)면 배경도 받아들여 결과가 이미지 전체가 된다 → 이 필터가 왜 필요한지
    everything, _, _ = automatic_mask(sam, np.zeros((480, 640, 3), np.uint8), {"grid": 16, "max_area_percent": 100})
    assert everything.all()


def test_defect_weakly_inside_a_background_mask_is_still_decoded():
    """배경 마스크가 결함 자리를 약하게(logit 0.5) 포함해도 그 위의 점은 건너뛰지 않는다."""
    sam = FakeSam([((100, 100, 300, 160), .9)], 640, 480, background_on_defect=.5)
    mask, _, info = automatic_mask(sam, np.zeros((480, 640, 3), np.uint8), {"grid": 16})
    assert info["accepted"] == 1 and mask[int(130 * LOW / 480), int(200 * LOW / 640)] == 1


def test_low_score_candidates_do_not_hide_other_points():
    sam = FakeSam([((0, 0, 640, 480), .1)], 640, 480)  # 모든 점이 낮은 점수 → 아무것도 덮지 않음
    mask, _, info = automatic_mask(sam, np.zeros((480, 640, 3), np.uint8), {"grid": 4, "min_score": .5})
    assert not mask.any() and info["decoded"] == 16


def test_sam2_checkpoints_are_recognised_for_inference():
    from core.inference_loading import _sam2_model_id
    assert _sam2_model_id("best.pt", {"type": "sam2_finetune", "backend": "sam2", "model_id": "sam2_hiera_tiny"}) == "sam2_hiera_tiny"
    assert _sam2_model_id("sam2.1_hiera_small.pt", {"model": {}}) == "sam2_hiera_small"   # 공식 파일
    assert _sam2_model_id("model.pt", {"model_state_dict": {}}) is None
    with pytest.raises(ValueError, match="모델 ID"):
        _sam2_model_id("best.pt", {"backend": "sam2", "model_id": "unknown"})


def _official_tiny():
    try:
        from builtin_assets import builtin_asset_path
        return builtin_asset_path("sam2_hiera_tiny")
    except Exception:
        return None


@pytest.mark.skipif(_official_tiny() is None, reason="SAM2 Tiny 공식 가중치가 없으면 실제 추론은 건너뛴다")
def test_real_sam2_finds_defects_on_cpu(tmp_path):
    from PIL import Image, ImageDraw
    from core.inference_loading import load_inference_engine
    from sam2_assets import load_sam2_checkpoint
    from sam2_trainer import _checkpoint_payload
    model = load_sam2_checkpoint("sam2_hiera_tiny", device="cpu")
    optimizer = torch.optim.AdamW(model.sam_mask_decoder.parameters())
    payload = _checkpoint_payload(model, optimizer, model_id="sam2_hiera_tiny", epoch=1, best_dice=.5,
                                  history=[], source_checkpoint=None)
    checkpoint = tmp_path / "best.pt"
    torch.save({**payload, "class_names": ["background", "scratch"]}, checkpoint)
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.normal(128, 10, (480, 640, 3)).clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.line((100, 100, 300, 140), fill=(30, 30, 30), width=5)
    draw.ellipse((450, 300, 490, 340), fill=(60, 60, 60))
    path = tmp_path / "part.png"
    image.save(path)
    engine = load_inference_engine(checkpoint, device="cpu", sam2={"grid": 16},
                                   segment_review={"min_area": 1, "min_score": 0, "split_dirs": []})
    result = engine.infer(str(path))
    assert result.status == "ok", result.error
    assert result.task == "segment" and result.summary == "NG 2"
    assert {d["class_name"] for d in result.details["defects"]} == {"scratch"}
    assert result.details["sam2"]["decoded"] < 20
