# Deep Vision Studio

로컬 환경에서 이미지 데이터 편집, 학습, 추론과 결과 검토를 수행하는 도구입니다.
이 저장소의 루트가 애플리케이션 소스 루트입니다.

## 태스크와 엔진

| 태스크 | 학습·추론 엔진 | 데이터 |
|---|---|---|
| 분류 | EfficientNet B0/B1, ResNet 18/50, ConvNeXt V1 Tiny, Custom CSP | 클래스별 이미지 폴더 |
| 시맨틱 분할 | SAM2 Hiera Tiny/Small/Base+/Large, DeepLab V3+ ResNet34, U-Net ResNet18, Custom CSP | 이미지와 클래스 인덱스 마스크 |
| 박스 탐지 | Re-DETR v4 Small/Medium/Large, LibreYOLO9 Tiny, Custom CSP | 정규화된 `class cx cy width height` 텍스트 |
| 이상 탐지 | PatchCore 또는 Custom CSP 재구성 | 정상 이미지, 선택적 평가용 불량 이미지 |
| 회전 박스 | 데이터 편집·크롭만 지원 | 클래스와 네 꼭짓점의 정규화 좌표 |

회전 박스의 학습·모델 추론 엔진은 제공하지 않습니다. 지원하지 않는 저장된 학습 모드는
실행을 거부하며, 현재 엔진을 명시적으로 선택해야 합니다.

## 실행

[C++17 / C# ONNX 실행·자동 테스트 예제](example/README.md)는 `example/`에서 제공한다.

ResNet 18/50, ConvNeXt V1 Tiny는 `ImageNet 가중치로 시작` 또는 `로컬 가중치로 시작`을
선택할 수 있다. DeepLab V3+와 U-Net은 ImageNet 백본을 사용하며 분할 헤드는 새로 학습한다.
공식 파일은 최초 1회 다운로드 후 캐시로 사용한다. 가중치를 설치 파일에 새로 포함하지는 않는다.
SAM2·Re-DETR v4·LibreYOLO도 설치본 기본 모델이다. Docker `.dvmodel`은 이 목록 밖에
추가하는 사용자 모델 전용이다. 이 세 모델의 Windows 학습 worker와 각 변형의 ONNX 인수는
현재 구현·검증 중이므로 `requested` 상태로 표시되며, production installer 검증은 이를
실행 가능한 기본 모델로 잘못 출고하지 않도록 막는다.

데이터셋 이미지를 열면 검출 박스와 분할 마스크를 직접 그릴 수 있다. Qt에서는
이미지 선택 후 **데이터 티칭 / 정답 그리기**, 브라우저에서는 이미지 클릭으로 연다.
[드로잉 도구 사용법](docs/DATASET_TEACHING.ko.md)과 [2.40 릴리스](docs/RELEASE_2.40.md)를 참고한다.

Python 3.11 환경에서 이 폴더로 이동합니다.

```sh
python -m pip install -r gui/requirements.txt
python gui/main.py
```

빌드 환경이 이미 준비된 Windows PC에서 소스 변경을 확인할 때는 저장소 폴더에서
다음과 같이 실행합니다. `build.bat`으로 EXE를 다시 만들 필요는 없습니다.

```bat
git pull --ff-only
python gui\main.py
```

`python`은 기존 빌드에 사용한 가상환경의 실행 파일이어야 합니다. 이미 만들어진 EXE에는
소스 업데이트가 적용되지 않으므로 위 명령으로 소스 버전을 실행합니다.

로컬 웹 UI는 Windows에서 `start_web.bat`, 다른 환경에서 `python start_web.py`로
실행합니다. [로컬 웹 안내](docs/LOCAL_WEB.md). 설치/공식 초기 가중치 취득에는 네트워크가
필요할 수 있지만 사용자 학습 이미지와 모델을 외부로 업로드하는 절차는 없습니다.
새 프로젝트 확장자는 `.dvproj`이며 프로젝트 내용은 JSON입니다.

## 코드 구성

- `gui/`: Qt 데스크톱 화면과 공통 학습·추론 엔진
- `python/`: `CustomCSP`, EfficientNet, ResNet/ConvNeXt/DeepLab/U-Net adapter, PatchCore, 데이터 처리, ONNX 내보내기
- `web/`, `webapp/`: React UI와 로컬 FastAPI 작업 관리자
- `cpp/`: C++17 `VisionInference`, `ClassificationWorker`, 고정 C ABI `vision_runtime`, CPU 벤치마크
- `sdk/csharp/`, `sdk/cpp/`: C ABI SafeHandle 래퍼와 C++17 사용 예
- `model_runtime/`, `tools/build_model_pack.py`, `tools/install_model_pack.py`: 오프라인 `.dvmodel` 팩 생성·검증·원자 설치
- `packaging/model-pack-template/`: 호스트 Python import 없이 DVW1를 말하는 Docker 모델 팩 템플릿
- `tools/`, `tests/`: 검증·배포 도구 및 회귀 검사

## EfficientNet CPU와 C++17

CPU 자동 모드는 수치 검증을 통과한 ONNX 세션을 사용합니다. ONNX 준비에 실패하면
기존 PyTorch로 계속 추론하고 실제 엔진과 실패 사유를 표시합니다. 명시적 ONNX
내보내기는 검증에 실패한 모델을 배포하지 않습니다. Grad-CAM은 별도로 측정합니다.

EfficientNet의 최적화·원본 그래프가 기본 실행 설정에서 검증에 실패하면 ONNX Runtime
실행 최적화를 줄이거나 끄고 모든 입력을 다시 검증합니다. 통과한 설정은 모델 옆 JSON에
저장하며 Python·C++·C# SDK에서 그대로 읽습니다. 이 배포 설정(schema 6)은 0.06 이상의
SDK가 필요합니다. 스레드 수를 따로 지정하지 않으면 JSON의 검증 설정을 사용합니다.
최적화를 끄면 속도가 달라질 수 있으므로 대상 PC에서 시간을 다시 측정해야 합니다.

모든 실행 설정이 실패하면 추가 수치 진단을 실행합니다.
실패한 입력으로 ONNX Runtime 최적화·스레드 설정, PyTorch FP32/FP64,
중간 레이어 출력을 비교하고 오류창에 `수치 진단:`을 표시합니다. 상세 결과는 출력 경로의
`*.export-error.json`에 기록합니다. 모델·이미지·중간 출력 배열을 전송하지 않으며,
진단은 허용 오차를 완화하거나 재학습 필요성을 확정하지 않습니다.

```sh
python tools/cpu_benchmark.py --weights model.pt --images images --runtime auto --threads 4 --output cpu-results
```

기본 ResNet/ConvNeXt/DeepLab/U-Net adapter 학습은 `python/train_builtin.py`를 사용한다.
이 스크립트는 기존 이미지·마스크 loader를 사용하고 `builtin` backend checkpoint를 저장한다.
가중치 다운로드는 수행하지 않는다.

Settings에서 사용자가 추가한 Docker 모델은 설치된 `.dvmodel`의 검증된 경로를 프로젝트에
저장한 뒤 `pack_train`·`pack_infer`·`pack_export` DVW1 작업으로 실행한다. 기본 제공 모델은
이 경로를 쓰지 않으며, 일반 Custom CSP 학습기로 다른 모델 ID를 자동 대체하지 않는다.

새 컨테이너 모델은 [Docker 모델 팩 템플릿](packaging/model-pack-template/README.ko.md)을 복사해
`manifest.json`의 digest와 입출력 계약을 고정하고, worker의 `prepare`·`train`·`infer`·`export`를
구현한다. 개발 팩은 `--allow-unsigned`로만 만들 수 있고, 출시 팩은 third-party 고지와 서명을
통과해야 한다.

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

기본 모델 카탈로그, Docker 확장 모델, C++17/C# ONNX SDK와 단일 Windows 설치 EXE의 설계 및 기반 계약을 구현 중입니다.
현재 SDK는 EfficientNet B0/B1 분류와 Custom 분류·탐지·재구성 anomaly·semantic 배포 경로를 검증하고,
고정 memory-bank를 포함해 export한 PatchCore anomaly의 score/map도 C ABI로 읽습니다. Re-DETR v4는
`pred_boxes`/`pred_logits` 두 출력 계약과 C++17/C ABI 디코더를 검증했으며, 실제 Small/Medium/Large
checkpoint의 ONNX 수치·Windows 인수 검증은 남아 있습니다. SAM2의 encoder/decoder prompt·video 경로와
기본 제공 대상인 Re-DETR v4·SAM2·LibreYOLO는 native worker·그래프·실기 검증 대상으로 관리합니다.
아래 문서는 구현 목표이며 위의 현재 지원 기능과 구분합니다.

- [기본 모델 목록과 지원 판정](docs/01-plan/features/model-catalog.md)
- [개정 계획](docs/01-plan/features/model-packs-windows-distribution.plan.md)
- [상세 설계](docs/02-design/features/model-packs-windows-distribution.design.md)
- [설치파일에 동봉할 README·최소사양 초안](packaging/windows/README.ko.md)
- [Third-party 고지와 가중치 배포 원칙](THIRD_PARTY_NOTICES.md)

최소사양은 UI/CPU 추론, GPU 학습, Docker 확장으로 구분하며 모델별 실측 후 확정합니다.
현재 초안은 RAM16GB/CPU 실행, RAM32GB·VRAM12GB/GPU 학습을 검증 기준으로 제시하며,
모든 모델·해상도의 실행 보장을 뜻하지 않습니다.
