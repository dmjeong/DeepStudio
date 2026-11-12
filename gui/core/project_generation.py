"""Class-aware surface synthesis with task-specific, reviewed dataset outputs.

The class name is never used to guess the defect's appearance. The user selects
the procedural method and reviews whether its result represents that class.
"""
import copy
import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path

import numpy as np
from core.annotation_journal import project_edit_lock, recover_pending
from core.class_management import IMAGE_EXTENSIONS, _walk_files
from core.dataset_editor import read_annotations, scan_dataset, sidecars, split_root
from core.defect_generator import DefectParams, DefectType, generate_sample
from core.defect_io import durable_bytes, png_bytes, read_image_snapshot
from core.project import ProjectManager
from core.segmentation_synthesis_store import checked, notify
from webapp.locking import exclusive_file
from webapp.storage import digest, read_json, write_json

TASK_OUTPUTS = {
    "classify": "Classification · 클래스별 이미지 폴더",
    "segment": "Segmentation · 픽셀 클래스 마스크",
    "detect": "Object detection · YOLO 사각 박스",
    "obb": "Oriented object detection · 회전 사각 박스",
    "anomaly": "Anomaly detection · 합성 평가 전용 이미지·마스크",
}


def class_options(project):
    if project.task not in TASK_OUTPUTS:
        raise ValueError("이 태스크는 프로젝트 클래스 생성을 지원하지 않습니다.")
    names = list(project.data.class_names)
    for name in names:
        ProjectManager.validate_name(name, kind="클래스")
    normal = {"good", "normal", "ok", "정상", "background", "배경"}
    return [(i, name) for i, name in enumerate(names)
            if not (project.task == "segment" and i == 0)
            and not (project.task == "anomaly" and name.casefold() in normal)]


def _box(mask, task, cid):
    import cv2
    y, x = np.nonzero(mask)
    if not x.size:
        raise ValueError("실제로 바뀐 픽셀이 없어 정답을 만들 수 없습니다.")
    h, w = mask.shape
    if task == "detect":
        x0, x1, y0, y1 = int(x.min()), int(x.max()) + 1, int(y.min()), int(y.max()) + 1
        coords = [(x0+x1)/(2*w), (y0+y1)/(2*h), (x1-x0)/w, (y1-y0)/h]
    else:
        # Include pixel edges, not only their centers (a one-pixel line has area).
        centers = np.column_stack((x, y)).astype(np.float32)
        hull = cv2.convexHull(centers).reshape(-1, 2)
        corners = np.concatenate([hull + offset for offset in ([0, 0], [1, 0], [1, 1], [0, 1])]).astype(np.float32)
        points = cv2.boxPoints(cv2.minAreaRect(corners))
        if np.any(points < -1e-4) or np.any(points > [w + 1e-4, h + 1e-4]):
            # At a boundary an axis-aligned rectangle is the valid enclosing OBB.
            points = np.array([[x.min(), y.min()], [x.max()+1, y.min()],
                               [x.max()+1, y.max()+1], [x.min(), y.max()+1]], np.float32)
        coords = np.clip(points / [w, h], 0, 1).ravel().tolist()
    return {"class_id": cid, "coordinates": coords}


def _labels_bytes(rows):
    return ("\n".join(str(r["class_id"]) + " " + " ".join(f"{v:.9f}" for v in r["coordinates"])
                      for r in rows) + "\n").encode("utf-8")


class ProjectGenerationStore:
    def __init__(self, project):
        self.project = project
        self.root = Path(project.project_dir) / "project_generation"
        class_options(project)

    def directory(self, kind, key):
        if kind not in {"batches", "publications"} or not re.fullmatch(r"[a-f0-9]{32}", key):
            raise ValueError("생성 기록 식별자 오류")
        return self.root / kind / key

    def batches(self):
        return sorted((read_json(p) for p in (self.root / "batches").glob("*/manifest.json")),
                      key=lambda r: r["created"], reverse=True)

    def load(self, key):
        row = read_json(self.directory("batches", key) / "manifest.json")
        if row["task"] != self.project.task or row["classes"] != list(self.project.data.class_names):
            raise ValueError("생성 후 프로젝트 클래스가 변경되었습니다. 새 후보를 생성하세요.")
        return row

    def _source_labels(self, path, shape):
        task = self.project.task
        mask = np.zeros(shape, np.uint8)
        labels = []
        if task == "segment":
            from core.mask_annotations import load_mask_document
            mask = load_mask_document(self.project, path, "train").render()
            if mask.shape != shape:
                raise ValueError(f"정답 크기 불일치: {path}")
        elif task in {"detect", "obb"}:
            related = sidecars(self.project, path, "train")
            if len(related) > 1:
                raise ValueError(f"정답 파일 중복: {path}")
            for _, label in related:
                labels.extend(read_annotations(label, task, len(self.project.data.class_names)))
            import cv2
            from core.dataset_regions import label_regions
            # A conservative exclusion margin protects existing object labels.
            h, w = shape
            for row in label_regions(labels, task, (w, h)):
                x, y, bw, bh = row["bbox"]
                mask[max(0,y-2):min(h,y+bh+2), max(0,x-2):min(w,x+bw+2)] = 1
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        return mask, labels

    def generate(self, rules, source_class="", seed=0, size_min=.01, size_max=.08,
                 strength_min=.3, strength_max=.8, context=None):
        available = dict(class_options(self.project))
        if not isinstance(rules, list) or not rules or len(rules) > 254:
            raise ValueError("생성할 프로젝트 클래스를 선택하세요.")
        ids = set()
        for row in rules:
            cid, count = row["class_id"], row["count"]
            if type(cid) is not int or cid not in available or cid in ids:
                raise ValueError("생성 클래스가 없거나 중복되었습니다.")
            if type(count) is not int or not 1 <= count <= 100:
                raise ValueError("클래스당 생성 수는 1~100입니다.")
            DefectType(row["method"])
            ids.add(cid)
        total = sum(row["count"] for row in rules)
        if total > 1000:
            raise ValueError("한 번에 최대 1,000장까지 생성할 수 있습니다.")
        if (not 0 < size_min <= size_max <= .5 or not 0 < strength_min <= strength_max <= 1
                or type(seed) is not int or not 0 <= seed <= 2**32-1):
            raise ValueError("크기(0~50%), 강도(0~100%) 범위와 시드를 확인하세요.")
        index = scan_dataset(self.project)
        records = [r for r in index["images"] if r["split"] == "train"]
        if self.project.task == "classify":
            if source_class not in self.project.data.class_names:
                raise ValueError("바탕으로 사용할 Train 클래스를 선택하세요.")
            records = [r for r in records if source_class in r["classes"]]
        elif self.project.task == "anomaly":
            records = [r for r in records if "good" in r["classes"]]
        if not records:
            raise ValueError("선택한 Train 바탕 이미지가 0장입니다. 먼저 데이터셋에 이미지를 추가하세요.")
        invalid = [r for r in records if r["error"]]
        if invalid:
            raise ValueError(f"원본 정답부터 확인하세요: {invalid[0]['path']} / {invalid[0]['error']}")
        key = uuid.uuid4().hex
        folder = self.directory("batches", key)
        temp = folder.with_name("." + key + ".partial")
        temp.mkdir(parents=True)
        rng = np.random.default_rng(seed)
        samples, failures = [], []
        try:
            for rule in rules:
                for _ in range(rule["count"]):
                    checked(context)
                    reason = "합성할 배경 위치가 부족합니다."
                    for attempt in range(12):
                        record = records[int(rng.integers(len(records)))]
                        path = Path(record["path"])
                        with project_edit_lock(self.project):
                            recover_pending(self.project)
                            image, source_hash = read_image_snapshot(path)
                            base, labels = self._source_labels(path, image.shape[:2])
                        roi = base == 0
                        roi[:2] = roi[-2:] = False
                        roi[:, :2] = roi[:, -2:] = False
                        if not roi.any():
                            continue
                        sample_seed = int(rng.integers(2**32))
                        params = DefectParams(seed=sample_seed, types=[DefectType(rule["method"])],
                                              size_ratio=float(rng.uniform(size_min, size_max)),
                                              intensity=float(rng.uniform(strength_min, strength_max)))
                        sample = generate_sample(image, params, roi_mask=roi)
                        if np.count_nonzero(sample.mask) >= 4:
                            break
                    else:
                        failures.append({"class_id": rule["class_id"], "reason": reason})
                        continue
                    sample_id = uuid.uuid4().hex
                    location = temp / sample_id
                    durable_bytes(location / "image.png", png_bytes(sample.image))
                    durable_bytes(location / "original.png", png_bytes(image))
                    durable_bytes(location / "changed.png", png_bytes(sample.mask))
                    if self.project.task == "segment":
                        target = base.copy()
                        target[sample.mask > 0] = rule["class_id"]
                        durable_bytes(location / "mask.png", png_bytes(target))
                    elif self.project.task in {"detect", "obb"}:
                        labels.append(_box(sample.mask, self.project.task, rule["class_id"]))
                        durable_bytes(location / "label.txt", _labels_bytes(labels))
                    samples.append({"id": sample_id, "class_id": rule["class_id"], "method": rule["method"],
                                    "source": str(path), "source_sha256": source_hash,
                                    "seed": sample_seed, "size_ratio": params.size_ratio, "strength": params.intensity,
                                    "hashes": {p.name: digest(p) for p in location.iterdir()}})
                    notify(context, f"{available[rule['class_id']]}: 후보 {len(samples)}장 생성", len(samples), total)
            if not samples:
                raise ValueError("생성 가능한 배경 영역이 없습니다. 바탕 이미지·크기·정답을 확인하세요.")
            from datetime import datetime, timezone
            manifest = {"schema_version": 1, "id": key, "created": datetime.now(timezone.utc).isoformat(),
                        "task": self.project.task, "classes": list(self.project.data.class_names), "seed": seed,
                        "source_split": "train", "samples": samples, "failures": failures,
                        "method_kind": "procedural_surface", "review_required": True}
            checked(context)
            write_json(temp / "manifest.json", manifest)
            temp.rename(folder)
        finally:
            shutil.rmtree(temp, ignore_errors=True)
        return {"batch_id": key, "generated": len(samples), "failures": failures}

    def verified_sample(self, batch_id, sample):
        folder = self.directory("batches", batch_id) / sample["id"]
        if not re.fullmatch(r"[a-f0-9]{32}", sample["id"]):
            raise ValueError("후보 식별자 오류")
        allowed = {"image.png", "original.png", "changed.png", "mask.png", "label.txt"}
        for name, expected in sample["hashes"].items():
            if name not in allowed or digest(folder / name) != expected:
                raise ValueError("생성 후보 또는 정답이 변경되었습니다. 다시 생성하세요.")
        return folder

    def publish(self, batch_id, sample_ids, context=None):
        batch = self.load(batch_id)
        index = scan_dataset(self.project)
        if index.get("orphan_masks") or any(row["error"] for row in index["images"]):
            raise ValueError("원본 데이터셋의 정답 오류를 먼저 해결하세요. 오류를 새 프로젝트에 복사하지 않았습니다.")
        chosen = set(sample_ids)
        if not chosen or len(chosen) != len(sample_ids) or not chosen <= {r["id"] for r in batch["samples"]}:
            raise ValueError("검수 후 승인한 후보를 선택하세요.")
        key = hashlib.sha256(json.dumps([batch_id, sorted(chosen)]).encode()).hexdigest()[:32]
        destination = self.directory("publications", key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with exclusive_file(destination.parent / "publish.lock"):
            if destination.exists():
                saved = read_json(destination / "manifest.json")
                for entry in saved["files"]:
                    if digest(destination / entry["path"]) != entry["sha256"]:
                        raise ValueError("저장한 생성 데이터가 변경되었습니다.")
                return {"project_path": str(destination / saved["project"]), "saved": len(chosen), "reused": True}
            temp = destination.with_name("." + key + ".partial")
            shutil.rmtree(temp, ignore_errors=True)
            temp.mkdir()
            try:
                task = self.project.task
                project = ProjectManager.create_new("합성평가" if task == "anomaly" else "생성데이터",
                                                    task, str(temp), batch["classes"])
                project.model = copy.deepcopy(self.project.model)
                project.training = copy.deepcopy(self.project.training)
                locations = {k: getattr(project.data, k) for k in ("root", "train_dir", "val_dir", "test_dir")}
                project.data = copy.deepcopy(self.project.data)
                for k, v in locations.items():
                    setattr(project.data, k, v)
                # Copy from the current source project under its edit lock.
                # Anomaly produces an explicitly separate synthetic evaluation
                # project, never combines generated defects with real test scores.
                with project_edit_lock(self.project):
                    recover_pending(self.project)
                    for split in (("train",) if task == "anomaly" else ("train", "val", "test")):
                        root = split_root(self.project, split)
                        for source in _walk_files(root):
                            if source.suffix.lower() not in IMAGE_EXTENSIONS:
                                continue
                            checked(context)
                            relative = source.relative_to(root)
                            if task == "anomaly" and (len(relative.parts) < 2 or relative.parts[0] != "good"):
                                continue
                            target = Path(getattr(project.data, split + "_dir")) / relative
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(source, target)
                            for kind, label in sidecars(self.project, source, split):
                                label_target = Path(project.data.root) / kind / split / relative.with_suffix(label.suffix)
                                label_target.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copyfile(label, label_target)
                for i, row in enumerate(batch["samples"]):
                    if row["id"] not in chosen:
                        continue
                    checked(context)
                    source = self.verified_sample(batch_id, row)
                    name = "generated-" + row["id"] + ".png"
                    cid = row["class_id"]
                    if task == "classify":
                        image = Path(project.data.train_dir) / batch["classes"][cid] / name
                        label = None
                    elif task == "anomaly":
                        image = Path(project.data.test_dir) / batch["classes"][cid] / name
                        label = Path(project.data.root) / "ground_truth" / batch["classes"][cid] / (Path(name).stem + "_mask.png")
                    else:
                        image = Path(project.data.train_dir) / name
                        label = Path(project.data.root) / ("masks" if task == "segment" else "labels") / "train" / Path(name).with_suffix(".png" if task == "segment" else ".txt")
                    image.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / "image.png", image)
                    if label is not None:
                        label.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source / ("mask.png" if task == "segment" else "changed.png" if task == "anomaly" else "label.txt"), label)
                    notify(context, "승인 이미지와 정답 저장 중", i+1, len(batch["samples"]))
                # The writer stores in-project paths relative to the project
                # file, so the entire publication can move to another PC.
                project_file = ProjectManager.save(project, str(temp / (project.name + ProjectManager.FILE_EXTENSION)))
                files = [{"path": str(p.relative_to(temp)), "sha256": digest(p)}
                         for p in temp.rglob("*") if p.is_file()]
                manifest = {"batch_id": batch_id, "approved": sorted(chosen), "project": Path(project_file).name,
                            "task": task, "classes": batch["classes"], "synthetic": True,
                            "evaluation_only": task == "anomaly", "files": files}
                write_json(temp / "manifest.json", manifest)
                (temp / "생성데이터안내.txt").write_text(
                    "프로젝트 클래스에 생성 기법을 직접 연결한 합성 데이터입니다. 실제 불량과 맞는지 검수하세요.\n"
                    + ("합성 평가 전용: 생성 불량은 test에만 저장했습니다. 실제 시험 정확도로 보고하지 마세요.\n"
                       if task == "anomaly" else "합성 이미지는 train에만 추가했습니다. 원본 val/test는 그대로 복사했습니다.\n"), encoding="utf-8")
                checked(context)
                temp.rename(destination)
            finally:
                shutil.rmtree(temp, ignore_errors=True)
        return {"project_path": str(destination / manifest["project"]), "saved": len(chosen), "reused": False}
