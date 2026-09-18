Deep Vision Studio의 첫 React + FastAPI 로컬 웹 버전입니다.

- 프로젝트, 데이터셋, 학습, 추론, ONNX 내보내기, Defect Gen 화면
- 기존 .dvproj와 모델 엔진 유지, Best 결과와 CSV/학습시간 연동
- 별도 계산 프로세스, 중단과 상태 복구, 저장된 추론/Grad-CAM 결과 표시
- Windows 실행기와 빌드된 React 화면 포함

**아래 Assets에서 `Deep-Vision-Studio-React-v7.9.zip`을 받으세요.** 일반 Source code ZIP에는 React 빌드가 포함되지 않습니다.

압축을 풀고 기존 Python 학습 환경을 활성화한 뒤 `DeepVisionStudio` 폴더에서:

```bash
python -m pip install -r webapp/requirements.txt
python run_web.py
```

Windows에서는 `start_web.bat`으로 실행할 수도 있습니다. 브라우저가 `http://127.0.0.1:8765`로 열립니다. 학습 중에는 서버 터미널을 유지하세요. Node.js는 실행용 ZIP 사용 시 필요하지 않습니다.

자세한 설치, 지원 범위, 캐시와 작업 복구 설명은 `docs/LOCAL_WEB.md`에 있습니다. 기존 데스크톱 화면은 `gui/main.py`로 계속 실행합니다.

이 릴리스는 Python lint, Windows/Linux 회귀 및 웹 프로세스 통합 검사, React TypeScript/빌드와 FastAPI 정적 자산 검사, C++ 계약 검사가 모두 통과한 커밋에만 생성됩니다. 실제 CUDA 성능과 사용자 데이터셋 정확도는 별도 검증 대상입니다.
