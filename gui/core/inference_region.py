"""Shared, immutable input-region requests for desktop and web inference."""

import hashlib
import json
from pathlib import Path

from core.paths import ensure_python_path

ensure_python_path()
from center_crop import validate_center_crop


def crop_from_json(data):
    """Read Studio export metadata, crop manifests, or saved project settings."""
    if not isinstance(data, dict):
        raise ValueError("크롭 JSON은 객체 형식이어야 합니다")
    candidates = []
    for source in (data, data.get("preprocessing")):
        if isinstance(source, dict) and "center_crop" in source:
            candidates.append(validate_center_crop(source["center_crop"]))
    training = data.get("training", data)
    if isinstance(training, dict) and "patchcore_crop_enabled" in training:
        enabled = training["patchcore_crop_enabled"]
        if not isinstance(enabled, bool):
            raise ValueError("크롭 사용 설정은 true 또는 false여야 합니다")
        candidates.append(validate_center_crop({"width": training.get("patchcore_crop_width"),
                                               "height": training.get("patchcore_crop_height")}) if enabled else None)
    if candidates:
        if any(crop != candidates[0] for crop in candidates[1:]):
            raise ValueError("JSON 안의 크롭 설정이 서로 다릅니다")
        return candidates[0]
    # Studio's original export schema predates source cropping.
    if (data.get("schema_version") == 1 and data.get("model_path")
            and "input_height" in data and "input_width" in data):
        return None
    raise ValueError('크롭 설정이 없습니다. center_crop에 width와 height를 지정해 주세요')


def read_input_region(mode="model", path=""):
    """Freeze JSON contents before a job starts; workers never reread the file."""
    if mode not in ("model", "json", "full"):
        raise ValueError("입력 영역은 모델 설정, JSON 설정, 원본 전체 중에서 선택해 주세요")
    if mode != "json":
        return {"mode": mode}
    if not str(path).strip():
        raise ValueError("크롭 JSON 파일을 선택해 주세요")
    source = Path(path).expanduser().resolve()
    try:
        with source.open("rb") as stream:
            content = stream.read(8 * 1024 * 1024 + 1)
        if len(content) > 8 * 1024 * 1024:
            raise ValueError("크롭 JSON 파일은 8 MB 이하여야 합니다")
        data = json.loads(content.decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"크롭 JSON 읽기 실패: {exc}") from exc
    return {"mode": mode, "center_crop": crop_from_json(data), "json_path": str(source),
            "json_sha256": hashlib.sha256(content).hexdigest()}


def resolve_input_region(request, saved_crop, input_size=None):
    request = request if request is not None else {"mode": "model"}
    if not isinstance(request, dict) or request.get("mode") not in ("model", "json", "full"):
        raise ValueError("추론 입력 영역 설정 오류")
    mode = request["mode"]
    saved = validate_center_crop(saved_crop)
    if mode == "json" and "center_crop" not in request:
        raise ValueError("JSON 크롭 설정이 없습니다")
    crop = validate_center_crop(request["center_crop"]) if mode == "json" else saved if mode == "model" else None
    result = {"mode": mode, "center_crop": crop, "saved_center_crop": saved}
    if mode == "json":
        result.update({key: request[key] for key in ("json_path", "json_sha256") if key in request})
    if input_size is not None:
        result["model_input_size"] = [int(input_size[1]), int(input_size[0])]
    return result


def input_region_label(region):
    mode = {"model": "모델 설정", "json": "JSON 설정", "full": "원본 전체"}[region["mode"]]
    crop = region.get("center_crop")
    area = f"중앙 {crop['width']}×{crop['height']} px" if crop else "원본 전체"
    label = f"{mode}: {area}" if mode != "full" else area
    size = region.get("model_input_size")
    return label + (f" → 모델 입력 {size[0]}×{size[1]}" if size else "")
