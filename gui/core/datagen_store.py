"""Project-scoped real defect data, immutable versions and persistent human review.

No model imports: data preparation and review also work without a GPU environment.
"""
from contextlib import contextmanager
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import time
import uuid

import numpy as np
from PIL import Image
from webapp.storage import read_json, write_json, digest
from webapp.locking import exclusive_file
from core.defect_workflow import validate_source, validate_output


def uid():
    return uuid.uuid4().hex


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Data Gen 식별자 오류")
    return value


def image8(path):
    with Image.open(path) as source:
        if source.mode not in {"L", "RGB"}:
            raise ValueError("AI Data Gen은 8비트 RGB 또는 흑백 이미지를 사용합니다")
        source.load()
        return source.copy()


def mask_image(value, size):
    if isinstance(value, dict) and value.get("png"):
        raw = base64.b64decode(value["png"].split(",")[-1], validate=True)
        if len(raw) > 32 * 1024 * 1024:
            raise ValueError("마스크 파일 크기 초과")
        source = Image.open(io.BytesIO(raw))
    else:
        source = Image.open(str(value))
    with source:
        if source.size != size:
            raise ValueError("마스크와 원본의 픽셀 크기가 다릅니다")
        result = Image.fromarray((np.array(source.convert("L")) > 127).astype("uint8") * 255)
    if not result.getbbox():
        raise ValueError("불량 영역을 표시하세요")
    return result


def save_png(path, image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(buffer.getvalue())
        stream.flush()
        import os
        os.fsync(stream.fileno())
    with Image.open(path) as saved:
        if not np.array_equal(np.array(saved), np.array(image)):
            raise ValueError("PNG 저장 후 픽셀 검증 실패")


class DataGenStore:
    def __init__(self, project):
        self.project = project
        self.root = Path(project.project_dir).resolve() / "datagen"
        self.index = self.root / "index.json"

    def state(self):
        state = read_json(self.index) if self.index.exists() else {"schema_version": 1, "items": [], "images": []}
        state["datasets"] = self.records("datasets")
        state["models"] = self.records("models")
        state["samples"] = self.records("samples")
        state["publications"] = self.records("publications")
        return state

    def records(self, kind):
        return sorted((read_json(p) for p in (self.root / kind).glob("*/manifest.json")), key=lambda row: row.get("created_at", 0))

    def record(self, kind, key):
        if kind not in {"datasets", "models", "samples", "publications"}:
            raise ValueError("Data Gen 자료 종류 오류")
        return read_json(self.root / kind / identifier(key) / "manifest.json")

    @contextmanager
    def edit(self):
        with exclusive_file(self.root / "edit.lock", "다른 Data Gen 저장이 진행 중입니다"):
            state = read_json(self.index) if self.index.exists() else {"schema_version": 1, "items": [], "images": []}
            yield state
            write_json(self.index, state)

    def item(self, state, key, active=True):
        row = next((v for v in state["items"] if v["id"] == key), None)
        if row is None or active and row["archived"]:
            raise ValueError("활성 학습 항목을 선택하세요")
        return row

    def save_settings(self, values):
        if not isinstance(values, dict):
            raise ValueError("Data Gen 설정 형식 오류")
        with self.edit() as state:
            state["settings"] = json.loads(json.dumps(values, allow_nan=False))
            return state["settings"]

    def update_item(self, name, item_id=None, class_name="", archived=False):
        name = str(name).strip()
        if not name or len(name) > 100:
            raise ValueError("학습 항목명은 1~100자입니다")
        if class_name and class_name not in self.project.data.class_names:
            raise ValueError("프로젝트에 등록된 검사 클래스를 선택하세요")
        with self.edit() as state:
            if any(v["name"].casefold() == name.casefold() and v["id"] != item_id and not v["archived"] for v in state["items"]):
                raise ValueError("같은 이름의 학습 항목이 있습니다")
            row = self.item(state, item_id, active=False) if item_id else {"id": uid()}
            row.update(name=name, class_name=class_name, archived=bool(archived))
            if not item_id:
                state["items"].append(row)
            return row

    def import_images(self, item_id, paths=None, folder="", role="defect", split="train", group="", masks=""):
        group = str(group).strip()
        if role not in {"normal", "defect"} or split not in {"train", "val"} or not str(group).strip():
            raise ValueError("실제 데이터 종류, train/val 분할과 개체/로트 그룹을 지정하세요")
        if folder:
            if not Path(folder).is_dir():
                raise ValueError("이미지 폴더가 없습니다")
            paths = [str(p) for p in sorted(Path(folder).rglob("*")) if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]
        if not paths or len(paths) > 10000:
            raise ValueError("1~10000장의 실제 이미지를 선택하세요")
        with self.edit() as state:
            self.item(state, item_id)
            if any(v["group"] == group and v["split"] != split for v in state["images"]):
                raise ValueError("같은 개체/로트 그룹을 train과 val로 나눌 수 없습니다")
            known = {v["pixel_sha256"] for v in state["images"]}
            additions = []
            for path in paths:
                source = Path(path).resolve()
                validate_source(self.project, source)
                if source.is_relative_to(self.root):
                    raise ValueError("Data Gen 저장 결과를 실제 원본으로 다시 가져올 수 없습니다")
                image = image8(source)
                pixel_hash = hashlib.sha256(str(image.size).encode() + image.convert("RGB").tobytes()).hexdigest()
                if pixel_hash in known:
                    raise ValueError(f"동일한 원본 픽셀의 중복 이미지: {source.name}")
                known.add(pixel_hash)
                key = uid()
                destination = self.root / "images" / key
                save_png(destination / "original.png", image)
                row = {"id": key, "item_id": item_id, "source": str(source), "sha256": digest(source),
                       "pixel_sha256": pixel_hash, "width": image.width, "height": image.height,
                       "image_sha256": digest(destination / "original.png"), "mask_sha256": "",
                       "mode": image.mode, "role": role, "split": split, "group": str(group).strip(), "mask": ""}
                if masks and role == "defect":
                    relative = source.relative_to(Path(folder).resolve()) if folder else Path(source.name)
                    mask_path = Path(masks) / relative.with_suffix(".png")
                    if mask_path.exists():
                        save_png(destination / "mask-import.png", mask_image(mask_path, image.size))
                        row["mask"] = "mask-import.png"
                        row["mask_sha256"] = digest(destination / row["mask"])
                additions.append(row)
            state["images"].extend(additions)
            return {"imported": len(additions)}

    def annotate(self, image_id, mask):
        with self.edit() as state:
            row = next(v for v in state["images"] if v["id"] == identifier(image_id))
            if row["role"] != "defect":
                raise ValueError("실제 불량 이미지에만 불량 라벨을 저장합니다")
            filename = f"mask-{uid()}.png"
            save_png(self.root / "images" / image_id / filename, mask_image(mask, (row["width"], row["height"])))
            row["mask"] = filename
            row["mask_sha256"] = digest(self.root / "images" / image_id / filename)
            return row

    def freeze(self, item_id):
        with self.edit() as state:
            item = self.item(state, item_id)
            rows = [v for v in state["images"] if v["item_id"] == item_id and v["role"] == "defect"]
            if not rows or any(not v["mask"] for v in rows):
                raise ValueError("모든 실제 불량 이미지의 마스크 검수를 완료하세요")
            if {v["split"] for v in rows} != {"train", "val"}:
                raise ValueError("서로 다른 그룹의 실제 train과 val 불량 데이터가 필요합니다")
            key = uid()
            root = self.root / "datasets" / key
            frozen = []
            try:
                for row in rows:
                    source = self.root / "images" / row["id"]
                    dest = root / row["id"]
                    if digest(source / "original.png") != row["image_sha256"] or digest(source / row["mask"]) != row["mask_sha256"]:
                        raise ValueError("실제 데이터 또는 검수 마스크가 외부에서 변경되었습니다")
                    dest.mkdir(parents=True)
                    for name, original in (("image.png", "original.png"), ("mask.png", row["mask"])):
                        shutil.copyfile(source / original, dest / name)
                    frozen.append({**row, "image_sha256": digest(dest / "image.png"), "mask_sha256": digest(dest / "mask.png")})
                manifest = {"id": key, "item_id": item_id, "item": dict(item), "created_at": time.time(), "images": frozen}
                write_json(root / "manifest.json", manifest)
                return manifest
            except Exception:
                shutil.rmtree(root, ignore_errors=True)
                raise

    def file(self, collection, key, kind="image"):
        key = identifier(key)
        if collection == "images":
            row = next(v for v in self.state()["images"] if v["id"] == key)
            name = row["mask"] if kind == "mask" else "original.png"
        elif collection == "samples" and kind in {"original", "image", "allowed", "requested", "changed", "reviewed"}:
            row = self.record("samples", key)
            name = row.get("review", {}).get("mask", "") if kind == "reviewed" else f"{kind}.png"
        else:
            raise ValueError("이미지 종류 오류")
        if not name:
            raise ValueError("저장된 마스크가 없습니다")
        return self.root / collection / key / name

    def review(self, sample_id, status, mask=None, reason=""):
        if status not in {"approved", "rejected", "pending"}:
            raise ValueError("검수 상태 오류")
        with self.edit():
            row = self.record("samples", sample_id)
            root = self.root / "samples" / sample_id
            for filename, expected in row["hashes"].items():
                if digest(root / filename) != expected:
                    raise ValueError("생성 후보 파일의 무결성 오류")
            old = row.get("review", {})
            name = old.get("mask", "")
            if mask:
                image = image8(root / "image.png")
                reviewed = mask_image(mask, image.size)
                allowed = np.array(Image.open(root / "allowed.png")) > 0
                if np.any((np.array(reviewed) > 0) & ~allowed):
                    raise ValueError("검수 불량 라벨이 허용 영역을 벗어났습니다")
                name = f"review-{uid()}.png"
                save_png(root / name, reviewed)
            if status == "approved" and not name:
                raise ValueError("불량 정답 마스크를 확인하고 저장한 뒤 승인하세요")
            row.setdefault("review_history", []).append(old)
            row["review"] = {"status": status, "mask": name, "reason": str(reason), "time": time.time(), "mask_sha256": digest(root / name) if name else ""}
            write_json(root / "manifest.json", row)
            return row

    def select_quality(self, model_id, checkpoint, reason):
        if not str(reason).strip():
            raise ValueError("고정 생성 샘플의 품질 검수 결과를 기록하세요")
        with self.edit():
            row = self.record("models", model_id)
            if checkpoint not in [c["id"] for c in row["checkpoints"]]:
                raise ValueError("완료된 체크포인트를 선택하세요")
            row["best_quality"] = checkpoint
            row["quality_review"] = {"reason": reason, "time": time.time()}
            write_json(self.root / "models" / model_id / "manifest.json", row)
            return row

    def publish(self, sample_ids, output):
        validate_output(self.project, output)
        if not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("중복 없는 승인 후보를 선택하세요")
        with self.edit():
            rows = [self.record("samples", identifier(key)) for key in sorted(sample_ids)]
            if any(v.get("review", {}).get("status") != "approved" for v in rows):
                raise ValueError("승인한 결과만 학습 버전으로 내보낼 수 있습니다")
            if any(not v.get("class_name") for v in rows):
                raise ValueError("모델 학습 전에 검사 클래스 연결을 지정해야 합니다")
            fingerprint = hashlib.sha256(json.dumps([(v["id"], v["review"]) for v in rows], sort_keys=True).encode()).hexdigest()
            for previous in self.records("publications"):
                if previous["fingerprint"] == fingerprint and previous["output_parent"] == str(Path(output).resolve()):
                    for entry in previous["files"]:
                        if digest(Path(previous["output"]) / entry["path"]) != entry["sha256"]:
                            raise ValueError("기존 학습 버전의 파일이 변경되었습니다")
                    return previous
            key = fingerprint[:32]
            destination = Path(output).resolve() / f"datagen-train-{key}"
            temporary = Path(output).resolve() / f".datagen-{key}.partial"
            if destination.exists():
                recovered = read_json(destination / "manifest.json")
                if recovered["fingerprint"] != fingerprint:
                    raise ValueError("학습 버전 식별자 충돌")
                for entry in recovered["files"]:
                    if digest(destination / entry["path"]) != entry["sha256"]:
                        raise ValueError("복구 대상 학습 버전 무결성 오류")
                write_json(self.root / "publications" / key / "manifest.json", recovered)
                return recovered
            shutil.rmtree(temporary, ignore_errors=True)
            files = []
            try:
                for row in rows:
                    source = self.root / "samples" / row["id"]
                    if digest(source / row["review"]["mask"]) != row["review"]["mask_sha256"]:
                        raise ValueError("검수된 불량 마스크 무결성 오류")
                    for kind, name in (("images", "image.png"), ("masks", row["review"]["mask"])):
                        if kind == "images" and digest(source / name) != row["hashes"][name]:
                            raise ValueError("생성 후보 무결성 오류")
                        target = temporary / kind / f"{row['id']}.png"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source / name, target)
                        files.append({"path": str(target.relative_to(temporary)), "sha256": digest(target)})
                manifest = {"id": key, "synthetic": True, "split": "train", "normal_training": False,
                    "fingerprint": fingerprint, "output_parent": str(Path(output).resolve()), "output": str(destination),
                    "files": files, "samples": rows, "class_mapping": {v["item_id"]: v["class_name"] for v in rows}}
                write_json(temporary / "manifest.json", manifest)
                temporary.rename(destination)
                write_json(self.root / "publications" / key / "manifest.json", manifest)
                return manifest
            finally:
                shutil.rmtree(temporary, ignore_errors=True)


def checkpoint_decision(state, value, checkpoint, min_delta=0.0, patience=0):
    """True finite minimum and early stopping are deliberately independent."""
    state = dict(state)
    if value is None or not math.isfinite(value):
        return state, False
    best = state.get("best_value")
    if best is None or value < best:
        state.update(best_value=value, best_val_loss=checkpoint)
    anchor = state.get("earlystop_value")
    if anchor is None or value < anchor - min_delta:
        state.update(earlystop_value=value, stale=0)
    else:
        state["stale"] = state.get("stale", 0) + 1
    return state, bool(patience and state.get("stale", 0) >= patience)


def mutate(project, action, values):
    store = DataGenStore(project)
    operations = {"settings": store.save_settings, "item": store.update_item, "import": store.import_images, "mask": store.annotate,
                  "freeze": store.freeze, "review": store.review, "quality": store.select_quality, "publish": store.publish}
    if action not in operations:
        raise ValueError("Data Gen 작업 오류")
    with exclusive_file(store.root / "compute.lock", "이 프로젝트의 Data Gen 학습 또는 생성이 진행 중입니다"):
        return operations[action](**values)
