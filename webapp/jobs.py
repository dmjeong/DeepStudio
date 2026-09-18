"""한 번에 한 계산 프로세스만 실행하고 상태를 디스크에 보존한다."""

from datetime import datetime, timezone
import json
import os
import shutil
import zipfile
from pathlib import Path
import subprocess
import sys
import threading
import uuid

from webapp import ROOT
from webapp.locking import exclusive_file
from webapp.storage import read_json, write_json

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self, state_dir, on_finish=None):
        self.root = Path(state_dir) / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.process = None
        self.monitor = None
        self.active_id = None
        self.on_finish = on_finish
        self.warnings = []
        self._jobs = {}
        self._root_revision = None
        for path in self.root.glob("*/job.json"):
            job = self._safe_metadata(path)
            if job is None:
                continue
            if job["status"] not in TERMINAL:
                try:
                    result = read_json(path.parent / "result.json")
                    if not isinstance(result, dict) or result.get("status") not in TERMINAL:
                        raise ValueError("완료 기록 형식 오류")
                except (OSError, ValueError, TypeError):
                    result = {"status": "interrupted", "error": "서버 종료로 중단된 작업. 저장된 결과 확인 필요"}
                job.update(result, finished_at=timestamp())
                write_json(path, job)
            self._jobs[job["id"]] = job

    def _safe_metadata(self, path):
        try:
            job = read_json(path)
            if (not isinstance(job, dict) or job.get("id") != path.parent.name
                    or not isinstance(job.get("created_at"), str)
                    or job.get("status") not in TERMINAL | {"running", "cancelling"}
                    or not isinstance(job.get("kind"), str)):
                raise ValueError("작업 기록 형식 오류")
            job.setdefault("output", {})
            job.setdefault("error", "")
            return job
        except (OSError, ValueError, TypeError) as exc:
            self.warnings.append(f"손상 작업 기록 보존: {path.parent.name}: {exc}")
            return None

    def directory(self, job_id):
        if len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id):
            raise ValueError("작업 ID 형식 오류")
        directory = self.root / job_id
        if not (directory / "job.json").is_file():
            raise FileNotFoundError("작업 기록 없음")
        return directory

    def require_idle(self):
        if self.active_id is not None:
            raise RuntimeError("실행 중인 작업 완료 또는 중단 후 변경 가능")
        with exclusive_file(self.root / "compute.lock"):
            pass

    def start(self, kind, payload):
        with self.lock:
            self.require_idle()
            job_id = uuid.uuid4().hex
            directory = self.root / job_id
            directory.mkdir()
            job = {"id": job_id, "kind": kind, "status": "running", "created_at": timestamp(),
                   "finished_at": None, "error": "", "output": {}, "project_path": payload.get("project", {}).get("filepath")}
            write_json(directory / "job.json", job)
            self._jobs[job_id] = job
            import psutil
            write_json(directory / "request.json", {"kind": kind, "payload": payload,
                       "parent_pid": os.getpid(), "parent_started": psutil.Process().create_time()})
            self.active_id = job_id
            try:
                command = ([sys.executable, "--studio-worker", str(directory / "request.json")] if getattr(sys, "frozen", False) else
                           [sys.executable, "-u", "-m", "webapp.worker", str(directory / "request.json")])
                if kind in {"datagen_train", "datagen_generate"}:
                    runtime_python = os.environ.get("DEEP_STUDIO_DATAGEN_PYTHON", "")
                    if runtime_python:
                        if not Path(runtime_python).is_absolute() or not Path(runtime_python).is_file():
                            raise ValueError("DEEP_STUDIO_DATAGEN_PYTHON에 전용 환경 Python의 절대 경로를 지정하세요")
                        command = [runtime_python, "-u", "-m", "webapp.worker", str(directory / "request.json")]
                with (directory / "console.log").open("wb") as log:
                    self.process = subprocess.Popen(
                        command,
                        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"},
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.monitor = threading.Thread(target=self._wait, args=(job_id, self.process), daemon=True)
                self.monitor.start()
            except Exception as exc:
                job.update(status="failed", error=str(exc), finished_at=timestamp())
                write_json(directory / "job.json", job)
                self.active_id = self.process = None
                raise
            return job

    def _wait(self, job_id, process):
        code = process.wait()
        with self.lock:
            directory = self.directory(job_id)
            job = self._safe_metadata(directory / "job.json") or dict(self._jobs[job_id])
            try:
                result = read_json(directory / "result.json")
                if not isinstance(result, dict) or result.get("status") not in TERMINAL:
                    raise ValueError("완료 기록 형식 오류")
            except (OSError, ValueError):
                result = {"status": "failed", "error": f"계산 프로세스 비정상 종료 ({code}). 실행 로그 확인 필요"}
            job.update(result, finished_at=timestamp())
            if self.on_finish:
                try:
                    self.on_finish()
                except Exception as exc:
                    job["project_error"] = f"작업 후 프로젝트 복원 실패: {exc}"
            write_json(directory / "job.json", job)
            self._jobs[job_id] = job
            self.active_id = self.process = None

    def cancel(self, job_id):
        with self.lock:
            directory = self.directory(job_id)
            job = read_json(directory / "job.json")
            if job["status"] not in TERMINAL:
                (directory / "cancel").touch()
                job["status"] = "cancelling"
                write_json(directory / "job.json", job)
                self._jobs[job_id] = job
            return job

    def list(self, project_path=None, limit=30):
        with self.lock:
            revision = self.root.stat().st_mtime_ns
            if revision != self._root_revision:
                for path in self.root.glob("*/job.json"):
                    if path.parent.name not in self._jobs:
                        job = self._safe_metadata(path)
                        if job is not None:
                            self._jobs[job["id"]] = job
                self._root_revision = revision
            jobs = list(self._jobs.values())
            if project_path is not None:
                jobs = [j for j in jobs if j.get("project_path") == project_path]
            return sorted(jobs, key=lambda item: item["created_at"], reverse=True)[:limit]

    def storage(self):
        with self.lock:
            records = []
            for directory in self.root.iterdir():
                if not directory.is_dir() or len(directory.name) != 32:
                    continue
                size = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
                records.append({**self._jobs.get(directory.name, {"id": directory.name, "kind": "unknown",
                    "status": "corrupt", "created_at": "", "project_path": None}), "bytes": size})
            return {"bytes": sum(j["bytes"] for j in records), "jobs": records, "warnings": self.warnings}

    def delete(self, job_ids):
        with self.lock:
            for job_id in job_ids:
                if job_id == self.active_id:
                    raise RuntimeError("진행 중인 작업 삭제 불가")
                self.directory(job_id)
            for job_id in job_ids:
                shutil.rmtree(self.directory(job_id))
                self._jobs.pop(job_id, None)
            return {"deleted": len(job_ids)}

    def export_archive(self, job_id):
        with self.lock:
            if job_id == self.active_id:
                raise RuntimeError("작업 완료 후 내보내기 가능")
            directory = self.directory(job_id)
            target = directory / "history.zip"
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in directory.rglob("*"):
                    if path.is_file() and path != target:
                        archive.write(path, path.relative_to(directory))
            return target

    def _metadata(self, path):
        job = self._safe_metadata(path)
        if job is None:
            raise ValueError("손상된 작업 기록. 저장 공간 화면에서 내보내기 또는 정리 가능")
        result = path.parent / "result.json"
        if job["status"] == "interrupted" and result.exists():
            try:
                recovered = read_json(result)
                if isinstance(recovered, dict) and recovered.get("status") in TERMINAL:
                    job.update(recovered, finished_at=timestamp())
                    write_json(path, job)
            except (OSError, ValueError, TypeError):
                pass
        return job

    def read(self, job_id, offset=0):
        directory = self.directory(job_id)
        job = self._metadata(directory / "job.json")
        events = []
        path = directory / "events.jsonl"
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(min(max(offset, 0), path.stat().st_size))
                for _ in range(250):
                    before = stream.tell()
                    line = stream.readline()
                    if not line.endswith(b"\n"):
                        stream.seek(before)
                        break
                    try:
                        events.append(json.loads(line))
                    except ValueError:
                        events.append({"event": "log_message", "args": ["손상된 로그 한 줄 건너뜀"]})
                offset = stream.tell()
        log = directory / "console.log"
        console = ""
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 12000))
                console = stream.read().decode("utf-8", errors="replace")
        return {"job": job, "events": events, "offset": offset, "console": console}

    def close(self):
        with self.lock:
            if self.active_id:
                self.cancel(self.active_id)
            process = self.process
            monitor = self.monitor
        # 프로세스는 현재 안전한 중단 지점까지 진행하고 결과를 저장한다.
        if process:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if monitor:
            monitor.join(timeout=2)
