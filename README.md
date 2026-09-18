# Deep Vision Studio

로컬 환경에서 이미지 데이터 편집, 학습, 추론과 결과 검토를 수행하는 도구입니다.
애플리케이션 소스 루트는 `DeepVisionStudio`입니다.

## 태스크와 엔진

| 태스크 | 학습·추론 엔진 | 데이터 |
|---|---|---|
| 분류 | EfficientNet B0/B1, Custom CSP | 클래스별 이미지 폴더 |
| 시맨틱 분할 | Custom CSP | 이미지와 클래스 인덱스 마스크 |
| 박스 탐지 | Custom CSP | 정규화된 `class cx cy width height` 텍스트 |
| 이상 탐지 | PatchCore 또는 Custom CSP 재구성 | 정상 이미지, 선택적 평가용 불량 이미지 |
| 회전 박스 | 데이터 편집·크롭만 지원 | 클래스와 네 꼭짓점의 정규화 좌표 |

회전 박스의 학습·모델 추론 엔진은 제공하지 않습니다. 지원하지 않는 저장된 학습 모드는
실행을 거부하며, 현재 엔진을 명시적으로 선택해야 합니다.

## 실행

Python 3.11 환경에서 이 폴더로 이동합니다.

```sh
python -m pip install -r gui/requirements.txt
python gui/main.py
```

로컬 웹 UI는 Windows에서 `start_web.bat`, 다른 환경에서 `python start_web.py`로
실행합니다. [로컬 웹 안내](docs/LOCAL_WEB.md). 설치/공식 초기 가중치 취득에는 네트워크가
필요할 수 있지만 사용자 학습 이미지와 모델을 외부로 업로드하는 절차는 없습니다.
새 프로젝트 확장자는 `.dvproj`이며 프로젝트 내용은 JSON입니다.

## 코드 구성

- `gui/`: Qt 데스크톱 화면과 공통 학습·추론 엔진
- `python/`: `CustomCSP`, EfficientNet, PatchCore, 데이터 처리, ONNX 내보내기
- `web/`, `webapp/`: React UI와 로컬 FastAPI 작업 관리자
- `cpp/`: C++17 `VisionInference`, `ClassificationWorker`, 고정 C ABI `vision_runtime`, CPU 벤치마크
- `sdk/csharp/`, `sdk/cpp/`: C ABI SafeHandle 래퍼와 C++17 사용 예
- `model_runtime/`, `tools/build_model_pack.py`, `tools/install_model_pack.py`: 오프라인 `.dvmodel` 팩 생성·검증·원자 설치
- `tools/`, `tests/`: 검증·배포 도구 및 회귀 검사

## EfficientNet CPU와 C++17

CPU 자동 모드는 수치 검증을 통과한 ONNX 세션을 사용합니다. ONNX 준비에 실패하면
기존 PyTorch로 계속 추론하고 실제 엔진과 실패 사유를 표시합니다. 명시적 ONNX
내보내기는 검증에 실패한 모델을 배포하지 않습니다. Grad-CAM은 별도로 측정합니다.

```sh
python tools/cpu_benchmark.py --weights model.pt --images images --runtime auto --threads 4 --output cpu-results
```

C++ 공개 헤더는 `vision_inference.h`, 클래스는 `VisionInference`, 정적 라이브러리는
`vision_inference`입니다. 모델과 작업자는 초기화 후 재사용합니다.

- [C++17 빌드와 사용](docs/EFFICIENTNET_CPP17.md)
- [ONNX CPU 설정 비교](docs/EFFICIENTNET_ONNX_OPTIMIZATION.md)
- [실제 추론 버튼의 측정 범위](docs/EFFICIENTNET_GUI_ONNX.md)
- [EfficientNet 모델 및 학습](docs/EFFICIENTNET.md)
- [입력 영역](docs/INFERENCE_REGION.md)
- [회전 박스 데이터 편집](docs/OBB.md)

Windows/i7의 8 ms 달성 여부는 해당 장비에서 검증해야 합니다. 다른 장비에서 측정한
숫자를 보장으로 사용하지 않습니다.

## 검증

```sh
python -m pytest tests -q
cd web
npm ci
npm test
npm run build
```

데스크톱 실행에는 Qt 의존성이 필요합니다. C++17 Release/CTest 절차는 위 C++ 안내에
있습니다. 경로/API 이름 정리의 범위와 검증 결과는 [정리 보고서](docs/04-report/studio-cleanup.report.md)를 참고하세요.

## 다음 배포판 설계

기본 모델 카탈로그, Docker 모델 팩, C++17/C# ONNX SDK와 단일 Windows 설치 EXE의 설계 및 기반 계약을 구현 중입니다.
현재 실제로 SDK 검증을 통과한 기본 모델은 EfficientNet B0/B1 분류와 Custom/semantic 배포 경로이며,
Re-DETR·SAM2·PatchCore의 특수 ONNX/C ABI 출력은 카탈로그에 등록한 뒤 별도 검증 대상으로 남겨 두었습니다.
아래 문서는 구현 목표이며 위의 현재 지원 기능과 구분합니다.

- [기본 모델 목록과 지원 판정](docs/01-plan/features/model-catalog.md)
- [개정 계획](docs/01-plan/features/model-packs-windows-distribution.plan.md)
- [상세 설계](docs/02-design/features/model-packs-windows-distribution.design.md)
- [설치파일에 동봉할 README·최소사양 초안](packaging/windows/README.ko.md)

최소사양은 UI/CPU 추론, GPU 학습, Docker 확장으로 구분하며 모델별 실측 후 확정합니다.
현재 초안은 RAM16GB/CPU 실행, RAM32GB·VRAM12GB/GPU 학습을 검증 기준으로 제시하며,
모든 모델·해상도의 실행 보장을 뜻하지 않습니다.
