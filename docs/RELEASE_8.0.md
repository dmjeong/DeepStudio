# Deep Vision Studio React 8.0

Windows 실행 파일에서 pip 명령을 직접 입력하지 않아도 되도록 자동 설치 시작점을 추가했다.

Python 3.11 64비트를 설치하고 `Deep-Vision-Studio-React-v8.0.zip`을 새 폴더에 푼 뒤 `DeepVisionStudio/start_web.bat`을 더블클릭한다. 최초 실행은 가상환경 생성, 패키지 설치, 호환성과 import 검사, 브라우저 열기를 자동으로 진행한다. Linux는 `python start_web.py`를 사용한다.

- 설치된 환경은 재사용한다. 설치 목록이나 패키지 버전 변경 및 누락을 감지하면 설치를 다시 확인한다.
- 실패한 설치는 완료로 표시하지 않는다. 오류와 로그 경로를 보여주고 서버를 실행하지 않는다. 다음 실행에서 재시도한다.
- 활성 가상환경과 Conda 환경을 우선 사용한다. 새 환경은 CPU PyTorch를 기본 설치하며 기존 CUDA PyTorch에 CPU 설치를 강제하지 않는다.
- 설치를 동시에 실행하지 못하도록 잠금을 적용했다. 실행 중인 스튜디오가 있을 때는 먼저 서버 종료를 안내한다.
- `--setup-only`, `--port`, `--no-browser`를 지원한다. 기존 `run_web.py` 수동 실행 경로도 유지한다.

최초 패키지 다운로드에는 인터넷이 필요하다. Python 자체가 없으면 설치 경로를 안내한다. 자세한 GPU 환경과 설치 절차는 `docs/LOCAL_WEB.md`를 참고한다.

검증에는 실제 venv/pip를 사용한 오프라인 설치, 재실행, 패키지 누락 복구, 설치 실패 후 재시도, Windows 배치 인자 및 종료 코드 검사, Windows/Linux 실제 모델 패키지의 자동 설치 경로 검사를 포함한다.
