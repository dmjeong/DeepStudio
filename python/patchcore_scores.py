"""Fixed model-relative PatchCore distance scale; this is not a probability.

Raw predict()/checkpoint thresholds remain in kNN distance units. Studio results
use d / (d + scale). A positive saved threshold sets scale and maps to 0.5;
uncalibrated/zero-threshold legacy models use unit scale, without inventing a
calibrated decision boundary. No inference images are used to fit this scale.
"""
import math


def score_normalization(raw_threshold=None):
    threshold = None if raw_threshold is None else float(raw_threshold)
    if threshold is not None and not math.isfinite(threshold):
        raise ValueError("PatchCore 임계값은 유한한 숫자여야 합니다")
    calibrated = threshold is not None and threshold > 0
    return {"method": "distance_ratio_v1", "scale": threshold if calibrated else 1.0,
            "source": "saved_threshold" if calibrated else "unit_distance_fallback"}


def validate_normalization(spec):
    if not isinstance(spec, dict) or spec.get("method") != "distance_ratio_v1":
        raise ValueError("미지원 PatchCore 점수 정규화 형식")
    try:
        scale = float(spec["scale"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("PatchCore 정규화 기준 오류") from exc
    if isinstance(spec["scale"], bool) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("PatchCore 정규화 기준은 0보다 큰 유한한 숫자여야 합니다")
    return dict(spec, scale=scale)


def normalize_score(raw_score, spec):
    scale = validate_normalization(spec)["scale"]
    score = float(raw_score)
    if not math.isfinite(score) or score < 0:
        raise ValueError("PatchCore 거리는 0 이상의 유한한 숫자여야 합니다")
    # Ratio first avoids overflow for large finite distances/scales.
    ratio = score / scale if score <= scale else scale / score
    value = ratio / (1 + ratio) if score <= scale else 1 / (1 + ratio)
    # Keep a finite distance below the asymptote and preserve the saved boundary.
    if value == .5 and score != scale:
        value = math.nextafter(.5, 0.0 if score < scale else 1.0)
    return min(value, math.nextafter(1.0, 0.0))


def normalize_threshold(raw_threshold, spec):
    if raw_threshold is None:
        return None
    threshold = float(raw_threshold)
    if not math.isfinite(threshold):
        raise ValueError("PatchCore 임계값은 유한한 숫자여야 합니다")
    return normalize_score(max(0.0, threshold), spec)


def normalized_details(raw_score, raw_threshold, spec):
    return {"engine": "patchcore", "score_space": "normalized_0_1",
            "raw_score": float(raw_score), "raw_threshold": raw_threshold,
            "score_normalization": validate_normalization(spec)}
