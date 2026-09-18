"""두 UI가 공유하는 합성 요청, 불변 후보 캐시와 검수 결과 저장."""
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import time

import numpy as np
from PIL import Image
from core.defect_generator import DefectParams, DefectSample, DefectType, generate_sample
from core.defect_io import DefectBatchWriter, read_image, read_image_snapshot, read_roi, verify_batch

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
MAX_SAMPLES = 10000


def contains(root, path):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def settings_for(project):
    defaults = {"folder": "", "output": str(Path(project.project_dir) / "synthetic"),
                "roi": "", "texture": "", "per_image": 1,
                "params": {**asdict(DefectParams(seed=42)), "types": ["scratch"]}}
    for name in ("good", "OK", "ok", "normal"):
        candidate = Path(project.data.train_dir) / name
        if candidate.is_dir():
            defaults["folder"] = str(candidate)
            break
    saved = getattr(project, "defect_generation", {})
    return {**defaults, **saved, "params": {**defaults["params"], **saved.get("params", {})}}


def validate_settings(values):
    if not isinstance(values, dict) or set(values) - {"folder", "output", "roi", "texture", "per_image", "params"}:
        raise ValueError("합성 설정 형식 오류")
    result = dict(values)
    for key in ("folder", "output", "roi", "texture"):
        value = result.get(key, "")
        if not isinstance(value, str) or value and not Path(value).is_absolute():
            raise ValueError(f"합성 {key}: 절대 경로 필요")
        result[key] = value
    per_image = result.get("per_image", 1)
    if type(per_image) is not int or not 1 <= per_image <= 100:
        raise ValueError("원본당 생성 수는 1~100 정수 필요")
    result["per_image"] = per_image
    raw = result.get("params", {})
    if not isinstance(raw, dict) or set(raw) - set(DefectParams.__dataclass_fields__):
        raise ValueError("합성 파라미터 형식 오류")
    try:
        params = DefectParams(**raw)
        if not isinstance(params.types, list) or not params.types:
            raise ValueError("불량 유형 선택 필요")
        params.types = [DefectType(t) for t in params.types]
    except (TypeError, ValueError) as exc:
        raise ValueError("합성 파라미터 또는 불량 유형 오류") from exc
    for key, minimum in (("intensity", 0), ("size_ratio", 0), ("feather", -1), ("roughness", -1)):
        value = getattr(params, key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 or value == minimum:
            raise ValueError(f"합성 {key} 범위 오류")
    if type(params.count) is not int or not 1 <= params.count <= 100 or type(params.mix_types) is not bool:
        raise ValueError("합성 개수 또는 혼합 설정 오류")
    if params.seed is not None and (type(params.seed) is not int or not 0 <= params.seed <= 4294967295):
        raise ValueError("시드는 0~4294967295 정수 또는 무작위 설정 필요")
    result["params"] = {**asdict(params), "types": [t.value for t in params.types]}
    return result


def validate_output(project, output, sources=()):
    if not output or not Path(output).is_absolute():
        raise ValueError("검수 결과 저장 폴더의 절대 경로 필요")
    for root in (project.data.root, project.data.train_dir, project.data.val_dir, project.data.test_dir):
        if root and contains(root, output):
            raise ValueError("합성 결과를 데이터셋 안에 직접 저장할 수 없음. 별도 synthetic 폴더 사용 필요")
    for source in sources:
        if contains(Path(source).parent, output):
            raise ValueError("원본 이미지 폴더 안에 합성 결과 저장 불가")


def validate_source(project, source):
    for split, root in (("val", project.data.val_dir), ("test", project.data.test_dir)):
        # 기존 이상 탐지 프로젝트의 val은 정상 학습 폴더를 가리키는 별칭이다.
        # 이 경우에만 허용하며, 실제 test 폴더 및 분류/검출/분할의 중첩은 차단한다.
        if (split == "val" and project.task == "anomaly" and root
                and Path(root).resolve() == Path(project.data.train_dir).resolve()):
            continue
        if root and contains(root, source):
            raise ValueError("검증 또는 평가 이미지를 합성 원본으로 사용 불가")
    for parent in Path(source).resolve().parents:
        manifest = parent / "manifest.json"
        if manifest.is_file():
            try:
                if json.loads(manifest.read_text(encoding="utf-8")).get("synthetic"):
                    raise ValueError("합성 결과를 원본으로 재사용 불가")
            except (json.JSONDecodeError, AttributeError):
                pass


def source_paths(project, payload, cancelled=lambda: False):
    if payload.get("images"):
        paths = [Path(p).resolve() for p in payload["images"]]
    else:
        if not payload.get("folder"):
            raise ValueError("정상 이미지 폴더 선택 필요")
        folder = Path(payload["folder"])
        if not folder.is_dir():
            raise ValueError("정상 이미지 폴더 없음")
        validate_source(project, folder / "__source__")
        paths = []
        for parent, dirs, files in os.walk(folder, followlinks=False):
            if cancelled():
                raise InterruptedError("합성 원본 스캔 중단")
            dirs[:] = sorted(name for name in dirs if not name.startswith(".") and not (Path(parent) / name).is_symlink())
            for name in sorted(files):
                path = Path(parent) / name
                if not name.startswith(".") and path.suffix.lower() in IMAGE_EXTENSIONS and not path.is_symlink():
                    paths.append(path.resolve())
                    if len(paths) > MAX_SAMPLES:
                        raise ValueError("한 작업의 원본은 최대 10,000장")
    if not paths:
        raise ValueError("합성 원본 이미지 없음")
    if len(set(paths)) != len(paths):
        raise ValueError("중복 합성 원본")
    if len(paths) * payload["per_image"] > MAX_SAMPLES:
        raise ValueError("한 작업은 최대 10,000장 생성 가능")
    hashes = {}
    for path in paths:
        if cancelled():
            raise InterruptedError("합성 원본 스캔 중단")
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"합성 원본 파일 오류: {path}")
        validate_source(project, path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                if cancelled():
                    raise InterruptedError("합성 원본 스캔 중단")
                digest.update(chunk)
        fingerprint = digest.hexdigest()
        if fingerprint in hashes:
            raise ValueError(f"내용이 같은 중복 합성 원본: {hashes[fingerprint]} / {path}")
        hashes[fingerprint] = path
    references = []
    if payload.get("texture"):
        texture = Path(payload["texture"]).resolve()
        if not texture.is_file() or texture.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("텍스처 참조 이미지 파일 오류")
        validate_source(project, texture)
        references.append(texture)
    validate_output(project, payload["output"], [*paths, *references])
    return paths


def sample_seed(master, source_hash, variant):
    value = f"{master}:{source_hash}:{variant}".encode("ascii")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


def sample_directory(directory, sample_id):
    if not isinstance(sample_id, str) or len(sample_id) != 6 or not sample_id.isascii() or not sample_id.isdigit():
        raise ValueError("합성 결과 ID 오류")
    return Path(directory) / "candidates" / sample_id


def candidate(directory, sample_id, *, verify=False):
    root = sample_directory(directory, sample_id)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 3 or manifest.get("count") != 1:
        raise ValueError("합성 후보 기록 오류")
    record = manifest["samples"][0]
    result = {**record, "id": sample_id}
    for key in ("image", "mask", "original"):
        path = (root / record[key]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("합성 후보 파일 경로 오류")
        if verify and hashlib.sha256(path.read_bytes()).hexdigest() != record[f"{key}_sha256"]:
            raise ValueError("합성 후보 파일 손상")
        result[key] = str(path)
    return result


def candidate_ids(directory):
    return [path.name for path in sorted((Path(directory) / "candidates").glob("[0-9]*"))
            if len(path.name) == 6 and path.name.isascii() and path.name.isdigit() and (path / "manifest.json").is_file()]


def candidates(directory, offset=0, limit=None):
    ids = candidate_ids(directory)
    return [candidate(directory, key) for key in ids[offset:None if limit is None else offset + limit]]


def generate_candidates(context, payload, project):
    from webapp.storage import write_json
    values = validate_settings({key: payload[key] for key in ("folder", "output", "per_image", "roi", "texture", "params") if key in payload})
    try:
        paths = source_paths(project, {**payload, **values}, context.cancelled)
    except InterruptedError:
        return {"status": "cancelled", "output": {"generated": 0, "review_required": True}}
    if payload.get("preview"):
        paths = paths[:1]
        values["per_image"] = 1
    params = DefectParams(**values["params"])
    params.types = [DefectType(t) for t in params.types]
    master_seed = params.seed if params.seed is not None else secrets.randbits(32)
    write_json(context.directory / "defect_request.json", {**values, "master_seed": master_seed,
               "images": [str(p) for p in paths], "project_path": payload["project"]["filepath"]})
    texture = read_image(values["texture"]) if values["texture"] else None
    count, total = 0, len(paths) * values["per_image"]
    for path in paths:
        if context.cancelled():
            break
        original, source_hash = read_image_snapshot(path)
        roi = read_roi(values["roi"], original.shape[:2])
        for variant in range(values["per_image"]):
            if context.cancelled():
                break
            start = time.perf_counter()
            seed = sample_seed(master_seed, source_hash, variant)
            sample = generate_sample(original, replace(params, seed=seed), roi_mask=roi, texture=texture)
            sample.recipe.update(master_seed=master_seed, variant=variant, generation_sec=time.perf_counter() - start)
            if roi is not None and not np.array_equal(sample.image[~roi], original[~roi]):
                raise RuntimeError("허용 영역 밖 원본 픽셀 변경")
            if context.cancelled():
                break
            sample_id = f"{count + 1:06d}"
            with DefectBatchWriter(context.directory / "candidates", batch_id=sample_id) as writer:
                writer.save(sample, path, original=original, source_sha256=source_hash)
            count += 1
            context.emit("defect_result", [{"id": sample_id, "source": str(path), "seed": seed}])
            context.emit("progress_updated", [count, total])
    return {"status": "cancelled" if context.cancelled() else "completed",
            "output": {"generated": count, "requested": total, "master_seed": master_seed,
                       "directory": str(context.directory), "review_required": True}}


def publish_candidates(context, payload, project):
    """검수한 캐시만 읽어 저장한다. 생성 함수를 호출하지 않는다."""
    directory = Path(payload["source_directory"])
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    if request["kind"] != "defects" or request["payload"]["project"]["filepath"] != payload["project"]["filepath"]:
        raise ValueError("다른 프로젝트의 합성 결과 저장 불가")
    ids = payload["sample_ids"]
    if not isinstance(ids, list) or not ids or len(ids) > MAX_SAMPLES or len(set(ids)) != len(ids):
        raise ValueError("저장할 합성 결과 선택 오류")
    rows = [candidate(directory, sample_id, verify=True) for sample_id in ids]
    validate_output(project, payload["output"], [row["source"] for row in rows])
    export_key = hashlib.sha256(json.dumps([str(directory.resolve()), sorted(ids)], separators=(",", ":")).encode()).hexdigest()
    root = Path(payload["output"])
    for manifest in root.glob("*/manifest.json"):
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved.get("samples") and all(row.get("export_key") == export_key for row in saved["samples"]):
            verify_batch(manifest.parent, saved)
            if len(saved["samples"]) != len(rows) or any("original" not in row for row in saved["samples"]):
                raise ValueError("이전에 저장한 합성 결과 손상: 원본 또는 결과 수 불일치")
            return {"status": "completed", "output": {"saved": len(rows), "output_dir": str(manifest.parent), "reused": True}}
    writer = DefectBatchWriter(root)
    try:
        for row in rows:
            if context.cancelled():
                writer.abort()
                return {"status": "cancelled", "output": {"saved": 0}}
            with Image.open(row["mask"]) as mask_image:
                mask = np.array(mask_image.convert("L"))
                sample = DefectSample(read_image(row["image"]), mask, [DefectType(t) for t in row["types"]],
                                      row["seed"], {**row, "origin_candidate": f"{directory.name}/{row['id']}", "export_key": export_key})
            writer.save(sample, row["source"], original=read_image(row["original"]), source_sha256=row["source_sha256"])
            context.emit("progress_updated", [writer.count, len(rows)])
        if context.cancelled():
            writer.abort()
            return {"status": "cancelled", "output": {"saved": 0}}
        writer.close()
    except BaseException:
        writer.abort()
        raise
    return {"status": "completed", "output": {"saved": writer.count, "output_dir": str(writer.output_dir),
            "archive_path": str(writer.archive_path)}}
