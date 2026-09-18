"""Qt에 의존하지 않는 데스크톱 계산 작업 관리자."""
import os
from pathlib import Path

_manager = None


def desktop_manager():
    global _manager
    if _manager is None:
        from webapp.jobs import JobManager
        _manager = JobManager(Path(os.environ.get("DEEP_STUDIO_DESKTOP_STATE_DIR") or
                                   Path.home() / ".deep-vision-studio-desktop"))
    return _manager
