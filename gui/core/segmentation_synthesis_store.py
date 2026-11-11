"""Persistent SEG reference synthesis with split isolation and reviewed publishing."""

import copy
import hashlib
import re
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from core.defect_io import durable_bytes, png_bytes, read_image
from core.segmentation_synthesis import (
    POLICIES,
    default_policy,
    extract_regions,
    synthesize,
    validate_pair,
)

from webapp.locking import exclusive_file
from webapp.storage import digest, read_json, write_json


def checked(context):
    if context and context.cancelled():
        raise InterruptedError("가상 불량 작업을 중단했습니다.")


def notify(context, text, current=0, total=0):
    if context:
        context.emit("log_message", [text])
        if total:
            context.emit("progress_updated", [current, total])


class SynthesisStore:
    def __init__(self, project):
        if project.task != "segment":
            raise ValueError("정답 기반 합성은 Segmentation 프로젝트에서 사용합니다.")
        self.project = project
        self.root = Path(project.project_dir).resolve() / "seg_synthesis"
        if self.root.is_symlink():
            raise ValueError("합성 저장소 링크는 사용할 수 없습니다.")
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, kind, key):
        if kind not in ("profiles", "batches", "publications") or not re.fullmatch(
            r"[0-9a-f]{32}", key
        ):
            raise ValueError("합성 기록 ID 오류")
        path = self.root / kind / key
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("합성 기록 경로 오류")
        return path

    def list(self, kind):
        result = []
        for path in sorted((self.root / kind).glob("*/manifest.json"), reverse=True):
            if path.parent.name.startswith("."):
                continue
            # Class edits invalidate old profiles but must not prevent creating
            # and selecting a fresh analysis. Direct load still rejects them.
            try:
                row = read_json(
                    self.directory(kind, path.parent.name) / "manifest.json"
                )
            except (OSError, ValueError):
                continue
            if (
                isinstance(row, dict)
                and row.get("schema") == 1
                and row.get("classes") == list(self.project.data.class_names)
            ):
                result.append(row)
        return sorted(result, key=lambda row: row["created"], reverse=True)

    def load(self, kind, key):
        data = read_json(self.directory(kind, key) / "manifest.json")
        if (
            not isinstance(data, dict)
            or data.get("schema") != 1
            or data.get("classes") != list(self.project.data.class_names)
        ):
            raise ValueError(
                "프로젝트 클래스가 변경되었습니다. 정답을 다시 분석하세요."
            )
        return data

    @contextmanager
    def staged(self, kind):
        key = uuid.uuid4().hex
        parent = self.root / kind
        if parent.is_symlink() or not parent.resolve().is_relative_to(self.root):
            raise ValueError("합성 저장소 경로 오류")
        parent.mkdir(exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
        try:
            yield key, temporary
            temporary.rename(self.directory(kind, key))
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def metadata(self, key):
        return {
            "schema": 1,
            "id": key,
            "created": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - Python 3.10 supported
            "classes": list(self.project.data.class_names),
        }

    @staticmethod
    def put(folder, relative, pixels):
        path = folder / relative
        durable_bytes(path, png_bytes(pixels))
        return {"path": relative, "sha256": digest(path)}

    @staticmethod
    def file(folder, entry):
        path = folder / entry["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("기록 밖의 파일 경로")
        if digest(path) != entry["sha256"]:
            raise ValueError(f"합성 파일이 변경되었거나 손상되었습니다: {path.name}")
        return path

    def analyze(self, context=None, *, include_unlabelled=False, policies=None):
        from core.annotation_journal import project_edit_lock, recover_pending

        with project_edit_lock(self.project):
            recover_pending(self.project)
            return self._analyze(
                context, include_unlabelled=include_unlabelled, policies=policies
            )

    def _analyze(self, context=None, *, include_unlabelled=False, policies=None):
        from core.class_management import IMAGE_EXTENSIONS, _walk_files
        from core.dataset_editor import sidecars, split_root
        from core.mask_annotations import load_mask_document

        classes = list(self.project.data.class_names)
        if not 2 <= len(classes) <= 255:
            raise ValueError("배경과 불량 클래스를 먼저 등록하세요.")
        policies = policies or {}
        selected = {
            str(i): policies.get(str(i), default_policy(name))
            for i, name in enumerate(classes)
            if i
        }
        if any(p not in POLICIES for p in selected.values()):
            raise ValueError("합성 방식 오류")
        # A published synthetic project is a training result, not new real
        # evidence. Prevent repeatedly treating generated defects as real donors.
        origin = Path(self.project.project_dir) / "manifest.json"
        if origin.is_file() and read_json(origin).get("synthetic"):
            raise ValueError(
                "합성 데이터가 포함된 프로젝트입니다. 실제 원본 프로젝트에서 새 후보를 만드세요."
            )
        paths = []
        for split in ("train", "val", "test"):
            if not getattr(self.project.data, f"{split}_dir", ""):
                continue
            root = split_root(self.project, split)
            if root.is_dir():
                paths += [
                    (split, p)
                    for p in sorted(_walk_files(root))
                    if p.suffix.lower() in IMAGE_EXTENSIONS
                ]
        if not paths:
            raise ValueError("프로젝트에 이미지가 없습니다.")
        if len(paths) > 10000:
            raise ValueError("한 번의 분석은 최대 10,000장까지 지원합니다.")
        with self.staged("profiles") as (key, folder):
            data = {
                **self.metadata(key),
                "images": [],
                "donors": [],
                "policies": selected,
                "include_unlabelled": bool(include_unlabelled),
                "notes": [],
                "synthetic": True,
            }
            seen = {}
            for number, (split, source) in enumerate(paths):
                checked(context)
                image = read_image(source)
                document = load_mask_document(self.project, source, split)
                mask = document.render()
                validate_pair(image, mask, len(classes))
                identity = hashlib.sha256(
                    str(image.shape).encode() + image.tobytes()
                ).hexdigest()
                mask_hash = hashlib.sha256(mask.tobytes()).hexdigest()
                if identity in seen:
                    old_split, old_mask = seen[identity]
                    if old_split != split:
                        raise ValueError(
                            f"같은 이미지가 {old_split}/{split}에 중복됩니다: {source.name}. 분할을 정리하세요."
                        )
                    if old_mask != mask_hash:
                        raise ValueError(
                            f"같은 이미지에 서로 다른 정답이 있습니다: {source.name}"
                        )
                    continue
                seen[identity] = (split, mask_hash)
                annotated = bool(sidecars(self.project, source, split))
                image_id = f"{number:06d}"
                entry = {
                    "id": image_id,
                    "source": str(source),
                    "split": split,
                    "annotated": annotated,
                    "shape": list(image.shape),
                    "pixel_sha256": identity,
                    "image": self.put(folder, f"images/{image_id}.png", image),
                    "mask": self.put(folder, f"masks/{image_id}.png", mask),
                }
                data["images"].append(entry)
                if split != "test":
                    for class_id in range(1, len(classes)):
                        regions = extract_regions(
                            mask, class_id, selected[str(class_id)]
                        )
                        if (mask == class_id).any() and not regions:
                            data["notes"].append(
                                f"{split}/{source.name}/{classes[class_id]}: 주변 정답/테두리 때문에 참조 추출 제외"
                            )
                        for region in regions:
                            donor_id = f"{len(data['donors']):06d}"
                            x, y, w, h = region["box"]
                            data["donors"].append(
                                {
                                    "id": donor_id,
                                    "source_id": image_id,
                                    "split": split,
                                    "class_id": class_id,
                                    "policy": selected[str(class_id)],
                                    "box": region["box"],
                                    "pixels": region["pixels"],
                                    "image": self.put(
                                        folder,
                                        f"donors/{donor_id}.png",
                                        image[y : y + h, x : x + w],
                                    ),
                                    "mask": self.put(
                                        folder,
                                        f"donor_masks/{donor_id}.png",
                                        region["mask"],
                                    ),
                                }
                            )
                notify(
                    context,
                    f"정답 분석 {number + 1}/{len(paths)}: {source.name}",
                    number + 1,
                    len(paths),
                )
            if not data["donors"]:
                raise ValueError(
                    "Train/Val에 사용할 불량 정답 영역이 없습니다. 불량을 라벨링한 뒤 분석하세요."
                )
            checked(context)
            write_json(folder / "manifest.json", data)
        return {
            "profile_id": key,
            "donors": len(data["donors"]),
            "notes": data["notes"],
        }

    def generate(
        self,
        profile_id,
        counts,
        *,
        split="train",
        seed=42,
        tolerance=0.12,
        allowed_path="",
        context=None,
    ):
        if split not in ("train", "val"):
            raise ValueError("합성은 Train 또는 Val만 사용합니다.")
        if (
            not isinstance(counts, dict)
            or not counts
            or any(
                not str(k).isdigit() or type(v) is not int or not 1 <= v <= 100
                for k, v in counts.items()
            )
        ):
            raise ValueError("클래스별 생성 수는 1~100이어야 합니다.")
        if (
            sum(counts.values()) > 500
            or not np.isfinite(tolerance)
            or not 0.005 <= tolerance <= 0.3
        ):
            raise ValueError("한 배치 최대 500장, 표면 허용차는 0.005~0.3입니다.")
        profile = self.load("profiles", profile_id)
        profile_dir = self.directory("profiles", profile_id)
        targets = [
            v
            for v in profile["images"]
            if v["split"] == split and (v["annotated"] or profile["include_unlabelled"])
        ]
        if not targets:
            raise ValueError(
                "바탕 이미지가 없습니다. 정상도 빈 마스크로 검수하거나 ‘정답 없는 사진도 정상 바탕으로’ 체크 후 다시 분석하세요."
            )
        donors = {
            int(k): [
                d
                for d in profile["donors"]
                if d["class_id"] == int(k) and d["split"] == split
            ]
            for k in counts
        }
        for cid, examples in donors.items():
            if not examples:
                raise ValueError(
                    f"{split}의 클래스 {cid}에 사용 가능한 실제 결함 정답이 없습니다."
                )
        allowed = None
        if allowed_path:
            allowed = read_image(allowed_path)
            if allowed.ndim == 3:
                allowed = allowed.max(axis=2)
            allowed = allowed > 0
            if not allowed.any():
                raise ValueError("허용 영역이 비어 있습니다.")
        rng = np.random.default_rng(seed)
        with self.staged("batches") as (key, folder):
            batch = {
                **self.metadata(key),
                "profile_id": profile_id,
                "split": split,
                "seed": seed,
                "requested": counts,
                "tolerance": tolerance,
                "samples": [],
                "failures": [],
                "synthetic": True,
            }
            if allowed is not None:
                batch["allowed"] = self.put(
                    folder, "allowed.png", allowed.astype(np.uint8) * 255
                )
            fingerprints = set()
            total, done = sum(counts.values()), 0
            for class_key, count in counts.items():
                cid = int(class_key)
                for _ in range(count):
                    checked(context)
                    failure = ""
                    for attempt in range(8):
                        checked(context)
                        donor = donors[cid][int(rng.integers(len(donors[cid])))]
                        others = [t for t in targets if t["id"] != donor["source_id"]]
                        target = (others or targets)[
                            int(rng.integers(len(others or targets)))
                        ]
                        image = read_image(self.file(profile_dir, target["image"]))
                        labels = read_image(self.file(profile_dir, target["mask"]))
                        patch = read_image(self.file(profile_dir, donor["image"]))
                        mask = read_image(self.file(profile_dir, donor["mask"]))
                        if allowed is not None and allowed.shape != labels.shape:
                            raise ValueError(
                                "허용 영역은 모든 바탕 원본과 같은 크기여야 합니다."
                            )
                        try:
                            output, truth, changed, recipe = synthesize(
                                image,
                                labels,
                                patch,
                                mask,
                                donor["policy"],
                                rng,
                                tolerance=tolerance,
                                allowed=allowed,
                                cancel=(
                                    context.cancelled if context else lambda: False
                                ),
                            )
                        except ValueError as exc:
                            failure = str(exc)
                            continue
                        fingerprint = hashlib.sha256(
                            output.tobytes() + truth.tobytes()
                        ).hexdigest()
                        if fingerprint in fingerprints:
                            failure = "같은 배치에서 완전히 동일한 후보가 생성되어 제외했습니다."
                            continue
                        fingerprints.add(fingerprint)
                        sample_id = f"{len(batch['samples']):06d}"
                        relative = f"samples/{sample_id}"
                        row = {
                            "id": sample_id,
                            "class_id": cid,
                            "policy": donor["policy"],
                            "donor_id": donor["id"],
                            "target_id": target["id"],
                            "split": split,
                            "recipe": recipe,
                            "attempt": attempt + 1,
                            "status": "pending",
                            "revision": 0,
                        }
                        for name, pixels in (
                            ("image", output),
                            ("mask", truth),
                            ("original", image),
                            ("changed", changed),
                            ("reference", patch),
                        ):
                            row[name] = self.put(
                                folder, f"{relative}/{name}.png", pixels
                            )
                        batch["samples"].append(row)
                        break
                    else:
                        batch["failures"].append({"class_id": cid, "reason": failure})
                    done += 1
                    notify(
                        context,
                        f"후보 {done}/{total}: 성공 {len(batch['samples'])}, 제외 {len(batch['failures'])}",
                        done,
                        total,
                    )
            checked(context)
            write_json(folder / "manifest.json", batch)
        return {
            "batch_id": key,
            "generated": len(batch["samples"]),
            "failures": batch["failures"],
        }

    def review(self, batch_id, sample_id, status, *, mask=None, expected_revision=None):
        if status not in ("pending", "approved", "rejected"):
            raise ValueError("검수 상태 오류")
        directory = self.directory("batches", batch_id)
        with exclusive_file(directory / "review.lock"):
            batch = self.load("batches", batch_id)
            row = next(v for v in batch["samples"] if v["id"] == sample_id)
            if expected_revision is not None and row["revision"] != expected_revision:
                raise ValueError(
                    "다른 창에서 검수 내용이 변경되었습니다. 새로 불러오세요."
                )
            image = read_image(self.file(directory, row["image"]))
            current = read_image(self.file(directory, row["mask"]))
            validate_pair(image, current, len(batch["classes"]))
            if mask is not None:
                validate_pair(image, mask, len(batch["classes"]))
                # Versioned masks keep a failed manifest write from invalidating
                # the previously committed annotation.
                row["mask"] = self.put(
                    directory, f"samples/{sample_id}/mask-{uuid.uuid4().hex}.png", mask
                )
                status = "pending"
            if status == "approved" and not np.any((current > 0) & (current != 255)):
                raise ValueError(
                    "불량 정답이 비어 있습니다. 마스크를 수정하거나 제외하세요."
                )
            row.update(status=status, revision=row["revision"] + 1)
            write_json(directory / "manifest.json", batch)
        return row

    def publish(self, batch_id, context=None):
        """Create an independent ready-to-open project; source data stays intact."""
        from core.project import ProjectManager

        directory = self.directory("batches", batch_id)
        with exclusive_file(directory / "review.lock"):
            batch = self.load("batches", batch_id)
            if batch["split"] != "train":
                raise ValueError(
                    "Val 참조 합성은 검수용입니다. 학습 데이터에 넣을 수 없습니다."
                )
            approved = [v for v in batch["samples"] if v["status"] == "approved"]
            if not approved:
                raise ValueError("먼저 생성 결과와 정답을 검수하고 승인하세요.")
            profile = self.load("profiles", batch["profile_id"])
            profile_dir = self.directory("profiles", batch["profile_id"])
            with self.staged("publications") as (key, folder):
                project = ProjectManager.create_new(
                    "SEG-synthetic", "segment", str(folder), list(batch["classes"])
                )
                project.model = copy.deepcopy(self.project.model)
                project.training = copy.deepcopy(self.project.training)
                # Keep augmentation/patch settings, replace only data locations.
                new_paths = {
                    k: getattr(project.data, k)
                    for k in ("root", "train_dir", "val_dir", "test_dir")
                }
                project.data = copy.deepcopy(self.project.data)
                for k, v in new_paths.items():
                    setattr(project.data, k, v)
                entries = []
                for record in profile["images"]:
                    entries.append(
                        (
                            profile_dir,
                            record,
                            record["split"],
                            "original-" + record["id"],
                        )
                    )
                for record in approved:
                    donor = next(
                        d for d in profile["donors"] if d["id"] == record["donor_id"]
                    )
                    target = next(
                        t for t in profile["images"] if t["id"] == record["target_id"]
                    )
                    if (
                        record["split"] != "train"
                        or donor["split"] != "train"
                        or target["split"] != "train"
                    ):
                        raise ValueError("검증/시험 데이터 유입이 감지되었습니다.")
                    entries.append(
                        (directory, record, "train", "synthetic-" + record["id"])
                    )
                files = []
                for index, (source_dir, record, split, name) in enumerate(entries):
                    checked(context)
                    item = {"split": split, "name": name}
                    for field, subdir in (("image", "images"), ("mask", "masks")):
                        source = self.file(source_dir, record[field])
                        destination = folder / "data" / subdir / split / (name + ".png")
                        shutil.copyfile(source, destination)
                        item[field] = {
                            "path": str(destination.relative_to(folder)),
                            "sha256": digest(destination),
                        }
                    files.append(item)
                    notify(
                        context,
                        f"새 데이터셋 저장 {index + 1}/{len(entries)}",
                        index + 1,
                        len(entries),
                    )
                project_file = ProjectManager.save(project)
                report = {
                    **self.metadata(key),
                    "batch_id": batch_id,
                    "profile_id": batch["profile_id"],
                    "approved": [v["id"] for v in approved],
                    "files": files,
                    "project": Path(project_file).name,
                    "synthetic": True,
                }
                checked(context)
                write_json(folder / "manifest.json", report)
        return {
            "project_path": str(
                self.directory("publications", key) / report["project"]
            ),
            "count": len(approved),
        }
