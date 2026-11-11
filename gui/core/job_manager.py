"""Qt에 의존하지 않는 데스크톱 계산 작업 관리자.

    작업 줄은 둘이다. 한 줄에는 한 번에 하나만 돈다.

        주 작업 줄      학습 · 내보내기 · 데이터 생성 · 추론        (작업 기록에 남는다)
        CPU 추론 줄     주 작업 줄이 차 있을 때의 CPU 추론 · 모델 검사   (학습과 나란히 돈다)

    GPU 추론은 CPU 추론 줄로 보내지 않는다: 학습과 GPU 메모리를 나눠 쓰면 학습이 메모리 부족으로 중단될 수 있다.
"""
import os
from pathlib import Path
import shutil

_manager = None
_cpu_manager = None


def _state_dir():
    return Path(os.environ.get("DEEP_STUDIO_DESKTOP_STATE_DIR") or Path.home() / ".deep-vision-studio-desktop")


def desktop_manager():
    global _manager
    if _manager is None:
        from webapp.jobs import JobManager
        _manager = JobManager(_state_dir())
    return _manager


def desktop_cpu_manager():
    """CPU 추론 줄. 자기 폴더와 자기 잠금(compute.lock)을 써서 주 작업 줄과 서로 막지 않는다.

    이 줄의 작업은 작업 기록 화면에 나오지 않으므로, 앱을 새로 켤 때 지난 실행의 결과를 지운다.
    """
    global _cpu_manager
    if _cpu_manager is None:
        from webapp.jobs import JobManager
        root = _state_dir() / "cpu-inference"
        shutil.rmtree(root / "jobs", ignore_errors=True)
        _cpu_manager = JobManager(root)
    return _cpu_manager


def is_cpu_device(device):
    return str(device or "cpu").lower().startswith("cpu")


def inference_manager(device):
    """추론 · 모델 검사를 돌릴 작업 줄을 고른다. 돌릴 수 없으면 RuntimeError (메시지를 그대로 보여 준다).

        주 작업 줄이 비어 있다             ─▶ 주 작업 줄 (이전과 같은 동작 · 작업 기록에 남는다)
        차 있다(학습 중 등) + 장치가 CPU   ─▶ CPU 추론 줄
        차 있다 + 장치가 GPU               ─▶ 오류: 추론 장치를 CPU로 바꾸라고 안내
    """
    main = desktop_manager()
    try:
        main.require_idle()
        return main
    except RuntimeError:
        if not is_cpu_device(device):
            raise RuntimeError("학습 등 다른 작업이 실행 중입니다. 추론 장치를 CPU로 바꾸면 그 작업과 함께 추론할 수 있습니다. "
                               "(GPU 추론은 학습과 GPU 메모리를 나눠 써 학습이 중단될 수 있어 막아 둡니다.)") from None
    lane = desktop_cpu_manager()
    try:
        lane.require_idle()
    except RuntimeError:
        raise RuntimeError("CPU 추론이 이미 실행 중입니다. 끝난 뒤 다시 시도하세요.") from None
    return lane


def clear_finished_cpu_inference():
    """CPU 추론 줄의 끝난 추론 결과를 지운다 (새 추론이 화면의 결과를 바꾸기 직전에 부른다)."""
    lane = desktop_cpu_manager()
    finished = [job["id"] for job in lane.list(limit=10000)
                if job["kind"] == "infer" and job["id"] != lane.active_id]
    if finished:
        lane.delete(finished)
