"""Qt에 의존하지 않는 데스크톱 계산 작업 관리자.

    작업 관리자 하나는 한 번에 작업 하나만 돌린다. 이 앱에는 둘을 둔다 (작업 폴더와 작업 기록은 함께 쓴다).

        주 작업 줄      학습 · 내보내기 · 데이터 생성 · 추론
        CPU 추론 줄     주 작업 줄이 차 있을 때의 CPU 모델 검사 · 추론   (학습과 나란히 돈다)

    다른 앱이 학습 중일 때의 CPU 추론은 webapp.locking.parallel_cpu_job 규칙으로 이미 허용된다.
    여기서는 같은 앱 안에서도 같은 규칙이 통하게 한다. GPU 추론은 CPU 추론 줄로 보내지 않는다:
    학습과 GPU 메모리를 나눠 쓰면 학습이 메모리 부족으로 중단될 수 있다.
"""
import os
from pathlib import Path

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
    """CPU 추론 줄: 같은 작업 폴더를 쓰는 두 번째 관리자 (다른 앱 하나가 더 떠 있는 것과 같은 관계).

    작업 기록 화면에는 두 줄의 작업이 함께 나온다. 서로의 실행 중 작업은 건드리지 않는다.
    """
    global _cpu_manager
    if _cpu_manager is None:
        from webapp.jobs import JobManager
        _cpu_manager = JobManager(_state_dir())
    return _cpu_manager


def inference_manager(kind, device):
    """모델 검사("inspect_model") · 추론("infer")을 돌릴 작업 줄을 고른다. 돌릴 수 없으면 RuntimeError.

        주 작업 줄이 비어 있다               ─▶ 주 작업 줄 (이전과 같은 동작)
        차 있다(이 앱이 학습 중) + CPU 장치  ─▶ CPU 추론 줄
        차 있다 + GPU 장치                   ─▶ 오류: 추론 장치를 CPU로 바꾸라고 안내
    """
    from webapp.locking import parallel_cpu_job
    payload = {"device": device}
    main = desktop_manager()
    if main.active_id is None:
        main.require_idle(kind, payload)
        return main
    if not parallel_cpu_job(kind, payload):
        raise RuntimeError("학습 등 다른 작업이 실행 중입니다. 추론 장치를 CPU로 바꾸면 그 작업과 함께 추론할 수 있습니다. "
                           "(GPU 추론은 학습과 GPU 메모리를 나눠 써 학습이 중단될 수 있어 막아 둡니다.)")
    lane = desktop_cpu_manager()
    if lane.active_id is not None:
        raise RuntimeError("CPU 추론이 이미 실행 중입니다. 끝난 뒤 다시 시도하세요.")
    return lane
