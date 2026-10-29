"""분할 학습 설정 추천: 라벨된 결함의 굵기와 이미지 크기에서 학습 해상도·패치·모델을 고른다. Qt에 의존하지 않는다.

작은 결함을 놓치는 가장 흔한 원인은 모델이 아니라 학습 해상도다. 결함이 학습 입력에서 1px보다 가늘어지면
라벨에는 남아도 영상에서는 주변과 섞여 사라진다. 그래서 '가장 가는 결함이 학습 해상도에서 몇 px인가'로 정한다.

    데이터셋 스캔(영역마다 굵기 = 가장 큰 내접원 지름)
        │
        ├─ 가는 결함 굵기 = 하위 10% (라벨 잡음 몇 개에 끌려가지 않게)
        │
        ├─ 전체 축소(입력 N px, 더 많이 줄어드는 방향)에서도 THIN_TARGET_PX 이상 → 패치 학습 불필요 (가장 빠름)
        └─ 아니면 패치 학습 + 배율 = 그 굵기가 THIN_TARGET_PX 이상 남는 가장 작은 배율 (빠를수록 작은 배율)
                                    어떤 배율로도 안 되면 100% (원본)

    미세 크랙(폭 1~2px) → 100% 고정. 배율을 낮추면 추론은 빨라지지만 크랙이 사라진다.
"""

from __future__ import annotations

import numpy as np

THIN_TARGET_PX = 2.0      # 학습 해상도에서 가는 결함이 이만큼은 남게 한다
LEARNABLE_PX = 1.0        # 이보다 가늘면 경고 (데이터셋 화면의 '1px보다 작은 영역'과 같은 기준)
SCALES = (1.0, .75, .5, .35, .25)
PATCH_MAX, PATCH_MIN = 512, 128
RECOMMENDED_MODEL = "unet_resnet18"   # 9.33 실험: 같은 데이터에서 Custom CSP보다 검출·속도 모두 좋았다


def _regions(index):
    """스캔 기록 → (영역 목록, 이미지 크기 목록). 오류·크기 없는 기록은 건너뛴다."""
    regions, sizes = [], []
    for record in index.get("images", []):
        size = record.get("size")
        if not size or record.get("error"):
            continue
        sizes.append(tuple(int(v) for v in size))
        for item in record.get("region_items") or ():
            regions.append((record, item))
    return regions, sizes


def _area(project, sizes):
    """판정 영역 (가로, 세로): 중앙 크롭이 켜져 있으면 크롭(이미지보다 크지 않게), 아니면 이미지 크기 중앙값."""
    from core.paths import ensure_python_path
    ensure_python_path()
    from center_crop import configured_center_crop
    width = int(np.median([w for w, _ in sizes]))
    height = int(np.median([h for _, h in sizes]))
    crop = configured_center_crop(project.training)
    if crop:
        return min(width, crop["width"]), min(height, crop["height"])
    return width, height


def _patch_size(width, height):
    """패치 = 배율 적용 영역의 짧은 변에 들어가는 32의 배수 (최대 512)."""
    fit = min(width, height) // 32 * 32
    return int(max(min(PATCH_MAX, fit), min(PATCH_MIN, max(32, fit))))


def thin_risk(project, index):
    """현재 설정에서 학습 해상도 굵기가 1px 미만이 되는 영역 수와 가는 결함 굵기(원본 px)."""
    from core.dataset_regions import training_footprint, training_scale
    regions, _sizes = _regions(index)
    widths = [float(item.get("width", 1)) for _record, item in regions]
    lost = sum(training_footprint(item, training_scale(project, record["size"]))[1] < LEARNABLE_PX
               for record, item in regions)
    return {"regions": len(regions), "lost": int(lost),
            "thin_px": float(np.percentile(widths, 10)) if widths else None}


def advise(project, index, *, weights_ready=None):
    """추천 설정과 이유. index는 scan_dataset 결과. 분할이 아니거나 라벨된 결함이 없으면 ok=False."""
    if project.task != "segment":
        return {"ok": False, "message": "자동 설정은 분할(세그멘테이션) 프로젝트에서 사용할 수 있습니다."}
    regions, sizes = _regions(index)
    if not regions:
        return {"ok": False, "message": "라벨된 결함 영역이 없습니다. 데이터셋 화면에서 결함을 먼저 그려 주세요."}
    training = project.training
    widths = np.array([float(item.get("width", 1)) for _record, item in regions])
    thin = float(np.percentile(widths, 10))
    width, height = _area(project, sizes)
    per_class = {}
    for _record, item in regions:
        per_class.setdefault(item["class"], []).append(float(item.get("width", 1)))
    stats = {
        "images": len(sizes), "regions": len(regions), "thin_px": thin, "median_px": float(np.median(widths)),
        "area": [width, height],
        "per_class": {name: {"regions": len(values), "thin_px": float(np.percentile(values, 10))}
                      for name, values in per_class.items()},
    }
    # 결함 픽셀 비율 (라벨된 이미지 기준) → 클래스 가중치
    labeled_images = {record["path"] for record, _item in regions}
    share = float(sum(item.get("area", 0) for _r, item in regions)) / max(1, len(labeled_images) * width * height)
    stats["defect_share"] = share

    settings, reasons, notes = {}, {}, []
    # 전체 축소는 가로·세로 배율이 다르다(3500×1504 → 512²: 가로 6.8배, 세로 2.9배 축소). 세로로 뻗은 크랙의 굵기는
    # 가로 배율을 그대로 받으므로, '패치가 필요 없다'는 판단은 더 많이 줄어드는 방향으로 한다(방향과 관계없이 남게).
    resize_px = thin * min(training.input_size / width, training.input_size / height)
    if resize_px >= THIN_TARGET_PX:
        settings.update(patch_training=False, patch_scale=1.0, input_size=int(training.input_size))
        reasons["patch_training"] = (f"가장 가는 결함 {thin:.1f}px가 입력 {training.input_size}px로 줄여도(더 많이 "
                                     f"줄어드는 방향 기준) {resize_px:.1f}px 남습니다 → 전체 이미지 학습이 가장 빠릅니다.")
    else:
        scale = next((s for s in sorted(SCALES) if thin * s >= THIN_TARGET_PX), 1.0)
        scaled_w, scaled_h = max(1, round(width * scale)), max(1, round(height * scale))
        settings.update(patch_training=True, patch_scale=scale, input_size=_patch_size(scaled_w, scaled_h))
        reasons["patch_training"] = (f"전체를 입력 {training.input_size}px로 줄이면 가장 가는 결함 {thin:.1f}px가 "
                                     f"{resize_px:.2f}px가 되어 사라집니다 → 패치 학습.")
        reasons["patch_scale"] = (f"배율 {scale:.0%}에서 가는 결함이 {thin * scale:.1f}px 남습니다"
                                  + (" (이보다 낮추면 2px 아래로 내려감)." if scale < 1 else
                                     " — 미세 결함이라 줄이지 않습니다. 추론 속도가 필요하면 GPU를 쓰세요."))
        reasons["input_size"] = (f"배율 적용 영역 {scaled_w}×{scaled_h}px의 짧은 변에 들어가는 패치 "
                                 f"{settings['input_size']}px.")
        stats["compute"] = {"size": [scaled_w, scaled_h],
                            "vs_resize": round(scaled_w * scaled_h / (training.input_size ** 2), 1)}
        if thin < THIN_TARGET_PX:
            notes.append(f"가장 가는 결함이 원본에서도 {thin:.1f}px입니다. 1px 미세 크랙은 라벨 붓을 2~3px로 "
                         "그리면 학습이 안정적입니다(위치는 같고 마스크 폭만 넓어짐).")
    settings.update(model_id=RECOMMENDED_MODEL, training_mode="builtin_finetune", segmentation_loss="ce_dice",
                    class_weights="sqrt" if share < .01 else training.class_weights)
    reasons["model_id"] = ("ImageNet으로 시작하는 U-Net ResNet18: 장수가 적을 때 처음부터 학습(Custom CSP)보다 "
                           "검출이 좋고 원본 해상도 추론도 빠릅니다(9.33 실측).")
    reasons["segmentation_loss"] = "CE + Dice: 배경이 대부분인 결함 마스크에서 작은 결함의 손실을 키웁니다."
    reasons["class_weights"] = (f"결함 픽셀이 {share:.2%}로 적어 Sqrt 가중치로 결함 쪽 손실을 키웁니다."
                                if share < .01 else f"결함 픽셀 {share:.1%}: 지금 설정을 유지합니다.")
    if weights_ready is None:
        try:
            from core.paths import ensure_python_path
            ensure_python_path()
            from builtin_assets import asset_status
            weights_ready = asset_status("resnet18")[0] == "ready"
        except Exception:
            weights_ready = True
    if not weights_ready:
        notes.append("ImageNet 가중치(resnet18)가 아직 없습니다. 학습을 시작하면 받는 곳과 가져오기를 안내합니다.")
    return {"ok": True, "stats": stats, "settings": settings, "reasons": reasons, "notes": notes,
            "current": {"patch_training": bool(getattr(training, "patch_training", False)),
                        "patch_scale": float(getattr(training, "patch_scale", 1.0)),
                        "input_size": int(training.input_size),
                        "model_id": getattr(project.model, "model_id", ""),
                        "training_mode": training.training_mode,
                        "segmentation_loss": training.segmentation_loss,
                        "class_weights": training.class_weights},
            "risk": thin_risk(project, index)}
