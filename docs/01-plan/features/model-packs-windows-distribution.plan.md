# Windows 학습툴 기본 모델·ONNX SDK·단일 설치 배포 계획

2026-09-18 · 요구사항 개정 3 · 등록부/worker/팩 설치·C ABI 기반 구현 진행 중

## 1. 이번 요구사항

Windows 응용프로그램에 아래 모델군의 **학습·추론·ONNX 내보내기 및 C#/C++ 배포**를 기본 제공한다.
추가 모델은 Docker 방식의 모델 팩으로 설치한다. 프로그램·기본 모델·가중치·third-party는
**오프라인 단일 설치 EXE**에 넣고, 사용자가 Python·패키지를 따로 설치하지 않게 한다.
최소사양·모델별 제한·사용법·라이선스 고지를 README와 함께 배포한다.

| 태스크 | 사용자 지정 기본 모델군 |
|---|---|
| Classification | EfficientNet B0/B1, ResNet, ConvNeXt V1, LibreYOLO |
| Anomaly detection | PatchCore |
| Object detection | Re-DETR v4 Small/Medium/Large, LibreYOLO |
| Segmentation | SAM2 Hiera Tiny/Small/Base+/Large, DeepLab V3+, U-Net |

위 표는 출시 요구사항이다. 현재 저장소가 모두 지원한다는 뜻은 아니다.
세부 변형, 학습 방식, 내보내기 위험은 [모델 지원 설계표](model-catalog.md)에 명시한다.
LibreYOLO는 여러 모델을 포함하는 라이브러리이므로 지원할 아키텍처를 한정한다.
사용자가 새로 지정한 LibreYOLO를 정식 신규 어댑터로 도입하며 이전에 삭제한 엔진을 복원하지 않는다.

## 2. 이전 설계에서 변경하는 결정

- CPU 전용 기본판 대신 **기본 모델의 Windows CPU/CUDA 학습 worker**를 번들한다.
- Windows worker 격리는 첫 출시 범위다. GPU 학습을 모두 Docker 설치에 의존시키지 않는다.
- C++17 외에 C# SDK와 Python 없는 두 언어의 실행 예제를 필수 산출물로 추가한다.
- 기본 가중치나 런타임을 외부 파일/인터넷 설치로 나누는 방안은 단일 설치 요구를 충족하지 않는다.
- NSIS 기반 소형 설치기 대신 **WiX Burn 다중 내장 컨테이너**를 후보로 검증한다.
- Docker Desktop 수동 설치를 안내하는 방식에서 **설치기가 구성하는 앱 전용 WSL2 + Docker Engine**으로 변경한다.
- 기본 모델 전체를 설치하고, 사용자가 원하는 새 모델만 추후 `.dvmodel` 파일로 추가한다.

## 3. 권장 제품 구조

`Windows UI → 모델 등록부 → Windows 모델 worker / Docker worker → 공통 결과`

`학습한 checkpoint → 모델별 exporter → 검증된 ONNX 배포 번들 → C# / C++17 SDK`

기본 모델은 Windows에서 동작한다. Docker는 새 모델의 의존성을 격리하는 확장 수단이다.
학습툴과 무관한 C#/C++ 제품은 ONNX 모델·SDK·DLL만으로 추론한다.
Docker를 통해 추론 속도가 자동 향상된다고 가정하지 않는다.

## 4. 현재 코드와 개발 공백

| 현재 상태 | 필요한 개발 |
|---|---|
| EfficientNet B0/B1, 자체 모델, PatchCore 경로 존재 | 신규 기본 모델군 어댑터·Windows 학습·검증 |
| `training_modes.py`와 GUI/웹에 엔진 고정 분기 | 공통 등록부와 선언형 capabilities/학습 옵션 |
| 현재 PatchCore exporter는 ONNX를 거부 | memory bank·kNN·score까지 포함한 ONNX 구현 |
| C++17 분류 worker와 기존 ONNX 배포 규약 | 다중 태스크·SAM2 다중 그래프·C ABI·C# 래퍼 |
| PyInstaller 빌드와 SFX 압축 도구 | 버전 고정 런타임·가중치 수집·단일 오프라인 installer |
| 특정 CUDA 구성에 고정된 데스크톱 smoke | CPU·GPU·frozen worker·설치 환경별 검사 |
| 기존 웹 CI와 Linux C++ 테스트 | Windows 설치·학습·MSVC/C#·Docker 확장 인수 환경 |

기존 수치 검증, 입력 채널·클래스 순서, 자동 ONNX 준비 실패 시 PyTorch 복구를 유지한다.
새 모델 추가가 기존 체크포인트의 전처리·PatchCore 점수 의미를 바꾸면 안 된다.

## 5. 단일 설치 EXE의 완료 조건

- 지원되는 Windows PC에서 Setup 하나로 UI·모든 기본 모델 worker·초기 가중치·SDK·README가 설치된다.
- 별도 Python, pip, conda, Git, Node, Visual Studio, CUDA Toolkit 설치를 요구하지 않는다.
- 설치와 기본 모델 첫 실행에 인터넷 다운로드가 없다. GPU 드라이버는 장비 사전조건이다.
- Docker 확장까지 포함한 설치 프로필은 오프라인 WSL 구성요소와 앱 전용 distro/Engine을 자동 구성한다.
- 관리자 권한, BIOS 가상화, Windows 가상화 기능, 기업 정책 허용과 재부팅 가능성은 README에 명시한다.
  설치기는 BIOS/기업 정책을 우회하지 않는다. 이 사전조건을 충족하지 않는 PC에 전체 지원을 표시하지 않는다.
- 기본 모델 기능은 Docker가 없어도 동작한다. Docker 확장 미구성 상태는 분명히 표시한다.
- 설치·수정·업그레이드·제거 시 사용자 프로젝트와 학습 결과를 보존한다.

**용량 실증이 선행 조건이다.** 모든 실제 payload를 포함한 서명 가능한 단일 EXE를 먼저 시험한다.
서명할 PE 파일은 4 GB 미만이어야 한다는 Microsoft 제약이 있으므로 내부 목표를 3.5 GiB 이하로 두고
정확한 byte 수와 Authenticode 검증을 기록한다. 여러 내장 CAB으로 나눠도 전체 EXE 제한이 사라지지 않는다.
[Microsoft 서명 제약](https://learn.microsoft.com/en-us/windows/msix/package/signing-known-issues)

한도를 넘으면 런타임/가중치 중복을 줄여 다시 빌드한다. 기본 모델을 몰래 제외하거나 외부 payload,
웹 다운로드, ISO로 대체한 뒤 요구 충족으로 표시하지 않는다. 여전히 불가능하면 설치 형태/기본 변형 범위의
변경 결정을 받기 전까지 단일 EXE 출시 조건은 미충족으로 남긴다.

## 6. 출시 범위와 미확정 항목

기본 세부 모델은 모델 지원 설계표의 작은 변형을 우선한다. 라이브러리의 모든 변형·모든 해상도·모든
학습 기법을 일괄 지원하지 않는다. 작은 배치의 로컬 학습을 기본으로 하고 다중 GPU·분산 학습은 후속 범위다.

- `Re-DETR v4`: 제품 카탈로그 변형은 Small/Medium/Large로 고정했다. 실제 upstream
  구현·checkpoint·ONNX 수치 검증은 각 변형별로 남아 있다.
- SAM2: Hiera Tiny/Small/Base+/Large를 모두 카탈로그에 포함한다. 이미지·점·박스 prompt와
  video capability를 계약에 표시했으며, 각 그래프와 영상 state의 Windows/ONNX 검증은 남아 있다.
- 프리트레인드: 전체 모델 가중치인지 encoder 초기화 가중치인지 구분한다. 재배포 근거를 파일별로 확정한다.
- Windows 최소사양: README에 설계 검증 기준을 먼저 기재하고 실제 모델별 실측 후 지원 최소값으로 확정한다.

이 항목이 미확정이어도 공통 아키텍처·SDK·installer 설계는 진행한다.

## 7. 단계별 구현 계획

| 단계 | 작업 | 완료 조건 |
|---|---|---|
| P0 위험 실증 | 실제 번들 크기/서명, Windows SAM2, PatchCore 전체 ONNX, WSL 오프라인 설치 | 고위험 조건별 재현 가능한 통과/실패 보고서 |
| P1 기반 | 모델 등록부·Windows worker·런타임/가중치 resolver·프로젝트 버전 고정 | 기존 모델 회귀 통과, frozen worker 정상 실행 |
| P2 분류/시맨틱 분할 | EfficientNet·ResNet·ConvNeXt V1·Libre 분류·DeepLab V3+·U-Net | 학습 결과 checkpoint로 ONNX 생성, 결과 일치 |
| P3 탐지/특수 모델 | 확정한 DETR·Libre 탐지·PatchCore·SAM2 | 각 특수 입출력·학습·재개/fit·배포 계약 통과 |
| P4 배포 SDK | 공통 C++17 core/C ABI, C# wrapper, 예제·문서 | 깨끗한 Windows에서 Python 없이 모든 기본 모델 추론 |
| P5 확장 | 앱 전용 WSL2/Docker, 오프라인 팩·취소·업데이트·복구 | 서로 다른 의존성의 새 팩을 EXE 재빌드 없이 추가 |
| P6 설치·인수 | 전체 단일 Setup·서명·README·SBOM·라이선스·오프라인 인수 | 기본 모델별 학습→export→C#/C++ 및 Docker 추가 검증 |

P0를 통과하기 전에 전체 기능·최소사양·단일 EXE 배포 완료를 약속하지 않는다.
P2/P3의 어댑터 개발과 P4 SDK는 공통 계약을 고정한 뒤 병행할 수 있다.
일정은 P0 실측과 미확정 모델 선택 후 산정한다.

## 8. 인수·성능 기준

모든 기본 모델의 고정 버전별로 다음을 통과해야 `지원`으로 표시한다.

1. 오프라인 설치 → 번들 가중치 로드 → 작은 사용자 형식 데이터로 학습/fit.
2. 학습한 결과를 저장·재로드하고 지원하는 재개 방식 검증.
3. 그 결과 checkpoint를 ONNX로 내보내고 형상·계약·수치 검사.
4. Python ORT·C++17·C#의 결과/좌표/마스크/판정 일치.
5. 취소·오류·누락 파일·잘못된 runtime·업데이트 복구 검증.

EfficientNet B0의 224×224, batch1, 전처리+추론+후처리 8 ms는 기존 목표로 유지한다.
전체 신규 모델에 8 ms를 약속하지 않는다. 대상 Windows/i7 모델명을 기록하고 p50/p95/p99/max,
초과 비율, 초기 로드, 큐, 파일 읽기, IPC와 버튼 전체 시간을 함께 측정한다.

관련 문서: [상세 설계](../../02-design/features/model-packs-windows-distribution.design.md),
[배포 README 초안](../../../packaging/windows/README.ko.md).
