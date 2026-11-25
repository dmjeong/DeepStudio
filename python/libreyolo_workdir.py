"""LibreYOLO 학습 중 현재 폴더를 학습 출력 폴더로 고정한다.

    LibreYOLO 검증기(ValidationConfig.save_dir=None)는 그림 저장을 끄더라도 매 검증마다
    "runs/val/<모델>_<시각>/config.yaml"을 **현재 폴더**에 만든다.

        앱 계산 프로세스의 현재 폴더 = 앱 설치 폴더 → 설치 폴더에 기록이 쌓이고,
        쓰기 금지된 위치(예: Program Files)면 검증에서 PermissionError로 학습이 멈출 수 있다.

    → trainer.train() 동안만 현재 폴더를 학습 출력의 상위 폴더(config.project)로 바꾸고 끝나면 되돌린다.

        <config.project>/<config.name>/...   ← 학습 결과 (LibreYOLO가 train() 안에서 save_dir로 정함)
        <config.project>/runs/val/...        ← 검증 기록 (이제 여기에 남는다)

    실행 폴더(project/name)는 미리 만들지 않는다: 이미 있으면 LibreYOLO가 name2로 바꿔 앱이 결과를 못 찾는다.
"""

from contextlib import contextmanager
import os
from pathlib import Path


@contextmanager
def inside_run_directory(trainer):
    config = trainer.config
    # 상대 경로면 바꾸기 전에 절대 경로로 고정한다 (바꾼 뒤 다시 풀면 폴더가 이중으로 중첩된다)
    project = Path(config.project).resolve()
    config.project = str(project)
    project.mkdir(parents=True, exist_ok=True)
    previous = os.getcwd()
    os.chdir(project)
    try:
        yield project
    finally:
        os.chdir(previous)
