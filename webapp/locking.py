"""프로세스 종료 시 운영체제가 자동 해제하는 Windows/POSIX 파일 잠금."""

from contextlib import contextmanager
import os
from pathlib import Path


def parallel_cpu_job(kind, payload):
    """Only explicitly selected CPU readers can coexist with another app's job."""
    return kind in {"infer", "inspect_model"} and str((payload or {}).get("device", "cpu")).lower() == "cpu"


def process_alive(pid, started):
    """A reused PID must not keep an abandoned job alive."""
    import psutil
    if not isinstance(pid, int) or pid <= 0 or not isinstance(started, (int, float)):
        return False
    try:
        process = psutil.Process(pid)
        return process.create_time() == started and process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True  # Never rewrite/delete a potentially live job on an inspection failure.


@contextmanager
def exclusive_file(path, message="다른 계산 프로세스가 종료 처리 중입니다. 잠시 후 다시 시도하세요."):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if not path.stat().st_size:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(message) from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
