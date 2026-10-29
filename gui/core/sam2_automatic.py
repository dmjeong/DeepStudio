"""SAM2 자동 분할: 프롬프트 없이 검사 이미지 전체에서 결함 마스크를 만든다. Qt에 의존하지 않는다.

SAM2는 '점을 찍은 곳의 물체'를 자르는 모델이라 검사에는 점이 필요하다. C++ SDK의
``Sam2Inference::Automatic``과 같은 격자 점(칸 가운데, 행 우선)을 찍고, 받아들인 후보 마스크를 합친다.

    이미지 ─▶ Hiera 인코더(1회) ─▶ 격자 점 N×N개 ─▶ 디코더(점마다 마스크 1개 + 점수)
                                                      │
                                후보 거르기: 점수 ≥ 최소 점수, 면적 ≤ 최대 면적 %
                                                      │
                                         받아들인 마스크의 합집합 = 결함 마스크

SDK와 다른 점 두 가지(데스크톱 검사용):
- 최대 면적: 배경에 찍힌 점은 배경 전체(이미지의 98%)를 결함보다 높은 점수(0.98 대 0.88)로
  돌려주므로 점수만으로는 거를 수 없다. SDK는 점수만 보므로 결과가 이미지 전체가 된다.
- 덮인 점 건너뛰기: 이미 나온 마스크 안의 점은 디코딩하지 않는다. CPU에서 16×16 격자가 23초 → 1.5초.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SETTINGS = {"grid": 16, "min_score": 0.5, "max_area_percent": 10.0}
BACKGROUND_COVER_LOGIT = 1.0  # sigmoid(1) ≈ 0.73: 3.0이면 같은 결과에 디코딩 65번, 1.0이면 7번 (실측)


def settings_of(settings=None) -> dict:
    merged = dict(DEFAULT_SETTINGS)
    merged.update({key: value for key, value in (settings or {}).items() if value is not None})
    grid = int(merged["grid"])
    if not 1 <= grid <= 32:
        raise ValueError("SAM2 자동 분할 격자는 1~32 사이여야 합니다 (C++ SDK와 같은 범위)")
    area = float(merged["max_area_percent"])
    if not 0 < area <= 100:
        raise ValueError("SAM2 후보 최대 면적은 0보다 크고 100% 이하여야 합니다")
    return {"grid": grid, "min_score": float(merged["min_score"]), "max_area_percent": area}


def grid_points(width: int, height: int, grid: int) -> np.ndarray:
    """C++ Automatic과 같은 점: 칸 가운데 ((열 + 0.5) × W / 격자, (행 + 0.5) × H / 격자), 행 우선."""
    columns = (np.arange(grid) + 0.5) * width / grid
    rows = (np.arange(grid) + 0.5) * height / grid
    y, x = np.meshgrid(rows, columns, indexing="ij")
    return np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float32)


def automatic_mask(predictor, image, settings=None):
    """predictor: 이미지를 아직 넣지 않은 SAM2ImagePredictor. image: (H, W, 3) uint8 RGB.

    반환: (mask, probabilities, info)
      mask: 디코더 저해상도(256×256) 0/1 결함 맵 — 이미지 전체를 1024²로 늘린 좌표계라 원본 비율과 무관하다.
      probabilities: (2, 256, 256) [배경, 결함] 확률. 결함 확률 = 받아들인 후보 중 가장 큰 sigmoid(logit).
      info: 후보 수·디코더 실행 수·받아들인 수·설정 (결과 상세에 남긴다).

    속도: CPU에서 디코더는 점 하나에 약 40 ms(묶어 돌려도 줄지 않음)라 256점이면 10초가 넘는다.
    이미 나온 마스크 안에 떨어지는 점은 같은 마스크를 다시 내므로 건너뛴다: 받아들인 결함은 마스크 전체,
    너무 커서 배경으로 버린 마스크는 확신이 있는 픽셀만(확률 > 0.73). 배경 마스크 하나가 대부분의 점을 덮어
    디코더 실행이 보통 수 번~수십 번으로 준다. 점수가 낮아 버린 마스크는 믿을 수 없어 덮지 않는다.
    """
    import torch

    settings = settings_of(settings)
    height, width = image.shape[:2]
    predictor.set_image(image)
    model = predictor.model
    features = predictor._features
    points = grid_points(width, height, settings["grid"])
    # 원본 픽셀 좌표 → 모델 입력(1024²) 좌표. predictor.predict와 같은 변환을 쓴다.
    coords = predictor._transforms.transform_coords(
        torch.as_tensor(points, dtype=torch.float32, device=predictor.device),
        normalize=True, orig_hw=predictor._orig_hw[-1])
    image_embed = features["image_embed"][-1].unsqueeze(0)
    high_res = [level[-1].unsqueeze(0) for level in features["high_res_feats"]]
    dense_pe = model.sam_prompt_encoder.get_dense_pe()
    best = union = covered = None
    accepted = decoded = 0
    scores = []
    with torch.inference_mode():
        for index in range(len(points)):
            if covered is not None:
                size = covered.shape
                # 이 점이 저해상도 맵의 어느 칸인지 (이미지 전체를 256²로 늘린 좌표)
                cell_y = min(size[0] - 1, int(points[index, 1] * size[0] / height))
                cell_x = min(size[1] - 1, int(points[index, 0] * size[1] / width))
                if covered[cell_y, cell_x]:
                    continue
            chunk = coords[index:index + 1].unsqueeze(1)  # (1, 1, 2): 점 하나 = 독립 프롬프트
            labels = torch.ones((1, 1), dtype=torch.int32, device=chunk.device)
            sparse, dense = model.sam_prompt_encoder(points=(chunk, labels), boxes=None, masks=None)
            logits, iou, _, _ = model.sam_mask_decoder(
                image_embeddings=image_embed, image_pe=dense_pe,
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False,  # 미세조정 계약과 같다 (sam2_trainer._sam2_logits)
                repeat_image=False, high_res_features=high_res)
            logit, score = logits[0, 0].float(), float(iou[0, 0])
            if not torch.isfinite(logit).all() or not np.isfinite(score):
                raise ValueError("SAM2 디코더 출력이 유한하지 않습니다")
            decoded += 1
            scores.append(score)
            binary = logit > 0
            if best is None:
                best = torch.zeros_like(logit)
                union = torch.zeros_like(binary)
                covered = np.zeros(binary.shape, dtype=bool)
            area = float(binary.float().mean()) * 100
            if score < settings["min_score"] or not binary.any():
                continue
            if area <= settings["max_area_percent"]:
                covered |= binary.cpu().numpy()  # 받아들인 결함 안의 점은 같은 결함을 다시 낸다
            else:
                # 배경으로 버린 마스크: 확신이 있는 픽셀(확률 > 0.73)만 덮는다. 배경 마스크에 약하게 섞인
                # 결함 위의 점은 건너뛰지 않고 따로 디코딩한다.
                covered |= (logit > BACKGROUND_COVER_LOGIT).cpu().numpy()
            if area <= settings["max_area_percent"]:
                accepted += 1
                union |= binary
                best = torch.maximum(best, logit.sigmoid())
    if union is None:
        raise ValueError("SAM2 자동 분할에 사용할 점이 없습니다")
    mask = union.cpu().numpy().astype(np.uint8)
    foreground = np.where(mask > 0, best.cpu().numpy(), 0.0).astype(np.float32)
    probabilities = np.stack([1.0 - foreground, foreground])
    info = {**settings, "candidates": len(points), "decoded": decoded, "accepted": accepted,
            "best_candidate_score": round(max(scores), 4) if scores else None}
    return mask, probabilities, info
