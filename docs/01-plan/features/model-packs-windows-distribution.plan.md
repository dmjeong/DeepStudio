# 모델 확장과 Windows 배포 계획

작성일: 2026-09-18 · 기준 커밋: `73e680f` · 상태: 계획 작성 완료, 구현 전

## 1. 목표와 권장안

새 모델을 넣을 때마다 프로그램 전체를 수정·재설치하는 일을 줄이고, 다른 Windows PC에서
Python 개발 환경 없이 설치 EXE로 학습·추론을 사용할 수 있게 한다.

**권장: Windows EXE 본체 + 버전이 있는 모델 팩 + 선택형 Docker 실행기.**
Docker는 새 프레임워크·학습 라이브러리의 충돌을 분리하는 데 사용한다. 검증된 ONNX 모델은
Windows/C++17에서 직접 실행한다. 이 조합으로 모델 추가의 편의성과 기존 CPU 지연시간 목표를 함께 다룬다.

일반 사용자에게는 `설치 → 모델 선택 → 이미지/데이터 선택 → 학습·추론` 흐름을 제공한다.
Docker가 필요한 모델에만 설치 조건을 표시하고, 기본 모델은 Docker 없이 동작하도록 한다.

## 2. 현재 코드에서 출발할 지점

| 현재 구현 | 확장 시 필요한 변경 |
|---|---|
| `gui/core/training_modes.py`: 엔진·모델·기능 목록 고정 | 모델 팩의 capabilities를 읽는 등록부 도입 |
| `webapp/worker.py`, `webapp/server.py`: 엔진 분기·검증 고정 | 공통 엔진 어댑터와 같은 등록부 사용 |
| `gui/core/inference_engine.py`: 태스크별 실행과 화면 연동 결합 | 실행 결과 계약을 유지하면서 모델 실행 부분을 어댑터로 이동 |
| `python/export_onnx.py`, `python/onnx_classifier.py`: 현재 모델 배포 계약 | 팩 버전·입출력 계약 연결, 지원 범위 안에서 일반화 |
| `cpp/include/vision_inference.h`, `classification_worker.h` | 기존 C++17 직접 실행·세션 재사용 유지 |
| `gui/build_exe.py`: PyInstaller 폴더형 EXE 빌드 | CPU 기본판과 Docker GPU 팩 분리, 가중치·라이선스·의존성 잠금 |
| `tools/package_desktop.py`: 7-Zip 자동 압축 해제 EXE | 설치·업데이트·제거를 지원하는 배포 방식으로 단계 전환 |
| 데스크톱 검증 도구가 특정 CUDA 구성을 요구 | CPU PC에서도 실행되는 별도 검증 프로필 |
| 현재 CI에 웹 배포 흐름 존재 | Windows 데스크톱 빌드와 깨끗한 PC 설치 검증 추가 |

현재 EXE 빌드 도구가 있다는 사실과 다른 PC에 배포 가능한 제품 검증을 완료했다는 사실은 구분한다.
이번 요청의 산출물은 계획·설계이며 EXE 생성이나 실행 코드 변경은 포함하지 않는다.

## 3. 배포와 실행 방식

| 구분 | 추천 용도 | 받는 PC의 조건 | 판단 |
|---|---|---|---|
| Windows 내장 엔진 | 현재 모델의 학습·추론·Grad-CAM | 설치 EXE의 런타임 포함 | 첫 배포의 기본 구성 |
| ONNX 데이터 팩 | 새 모델의 빠른 추론, C++17 프로그램 연동 | 지원 연산·전후처리 계약 | 저지연 추론의 우선 경로 |
| Docker 모델 팩 | 신규 모델 학습·내보내기, 다른 프레임워크 | Docker Desktop와 WSL2 등 사전조건 | 확장 기능으로 채택 |
| 독립 Windows 실행기 팩 | Docker 금지 PC에서 새로운 런타임 사용 | 별도 Windows 패키징·검증 | 후속 확장 지점, 첫 구현 범위에서는 제외 |

Docker 이미지에 들어 있는 Linux 라이브러리가 Windows DLL로 바뀌는 것은 아니다.
컨테이너에서 학습한 뒤 ONNX로 내보낼 수 있어야 Windows 직접 추론 경로로 옮길 수 있다.
ONNX에 없는 연산이나 새로운 후처리는 별도 어댑터·실행기 구현이 필요하다.

## 4. 범위와 기본 가정

- 기본 제품은 현재의 학습·추론 기능을 포함한 Windows x64 CPU 배포판으로 계획한다.
  사용자가 추론 전용 배포를 선택하면 더 작은 배포 프로필을 별도로 정의한다.
- 기준 OS는 Microsoft와 선택한 의존성이 지원하는 Windows 11 x64로 한다.
  Windows 10/Windows Server/ARM64는 요청 시 별도 검증 대상이며 초기 지원으로 표시하지 않는다.
- Qt 데스크톱 UI를 기본 EXE 화면으로 유지한다. 기존 웹 UI도 동일 모델 등록부를 사용한다.
- 모델 팩은 아키텍처, 초기 사전학습 가중치, 학습 결과, 실행 엔진을 서로 식별한다.
- 기존 EfficientNet, Custom CSP, PatchCore 동작과 `.dvproj`를 단계적으로 연결한다.
  회전 박스는 현재 편집 기능을 유지하고, 실행 가능한 새 어댑터가 등록될 때 학습 기능을 노출한다.
- 사용자 이미지·학습 결과·비공개 가중치는 로컬 경로에서만 처리한다.
- 첫 릴리스는 단일 사용자·단일 PC 기준이다. 원격 학습 서버·플러그인 마켓·다중 GPU 분산 학습은 후속 범위다.

## 5. 사용자가 받는 파일

1. `DeepVisionStudio-Setup-<version>-win-x64.exe`: 기본 CPU 프로그램과 승인된 기본 모델 팩.
2. `DeepVisionStudio-Portable-<version>-win-x64.zip`: 설치형과 동일 구성의 진단·이동용 묶음.
3. `<model-id>-<version>.dvmodel`: 나중에 추가하는 모델 팩. 대용량 팩은 별도 오프라인 파일로 제공.
4. `DeepVisionStudio-CppSDK-<version>-win-x64.zip`: 선택 산출물. C++17 헤더·라이브러리·런타임 DLL·예제·라이선스.

기본 배포판에는 Python, Qt, PyTorch CPU, ONNX Runtime CPU, OpenCV 등 필요한 의존성과
검토된 초기 가중치를 포함한다. 사용 중 pip 설치나 최초 모델 자동 다운로드를 요구하지 않는다.
첫 GPU 확장은 Docker 팩으로 제공하고 팩별 지원 드라이버를 표시한다.
Windows 네이티브 CUDA 실행기 또는 별도 CUDA EXE는 후속 범위다.
Docker Desktop, WSL2, GPU 드라이버는 EXE 하나로 대체할 수 없는 시스템 사전조건이다.

## 6. 구현 순서와 완료 기준

| 단계 | 주요 작업 | 단계 완료 기준 |
|---|---|---|
| P0 현재 배포 기반 고정 | CPU 잠금 파일, 기존 Windows EXE 빌드·스모크, 기준 결과 수집 | 새 Windows PC에서 현재 기능이 정상 실행 |
| P1 모델 등록부 | manifest·capabilities·어댑터, 기존 엔진 등록, 프로젝트 참조 | 기존 모델 결과 유지, GUI/웹 모델 목록 일치 |
| P2 ONNX 팩 | 가져오기·검증·업데이트·롤백, 배포 JSON 연결, C++ SDK | 같은 EXE에 두 번째 독립 ONNX 팩 추가·실행 |
| P3 Docker 팩 | 오프라인 이미지 취득, 지속 실행 worker, 학습/취소/내보내기 | 다른 의존성의 새 팩을 EXE 재빌드 없이 학습·내보내기 |
| P4 배포 완성 | installer, 기본 가중치·third-party 묶음, CPU 배포판·Docker GPU 팩, 서명 | 인터넷 없는 깨끗한 Windows PC 설치·학습·추론 검증 |
| P5 인수 검증 | C++17, 업그레이드, 오류 복구, 대상 i7 측정 | 기능 인수 통과, 성능 목표의 달성/미달 수치 공개 |

순서는 P0 → P1 → P2 → P3 → P4 → P5다. P4의 설치기 작업 일부는 P2 이후 병행할 수 있다.
소요 일정은 대상 Windows PC와 배포 가중치 목록을 확보한 뒤 산정한다.

공통 인수 조건:

- 지원 계약을 사용하는 새 모델 팩을 넣을 때 UI나 중앙 if/else를 수정하지 않는다.
- 필요한 프레임워크가 다른 두 Docker 팩이 각각 고정된 의존성으로 동작한다.
- Docker 없는 PC에서도 기본 CPU 모델이 정상 동작한다.
- 오프라인 환경에서 설치·초기 가중치 로드·소규모 학습·추론·프로젝트 저장이 성공한다.
- 팩 업데이트 실패 시 기존 팩·프로젝트·학습 결과가 남고 실행 버전을 되돌릴 수 있다.
- 수치 검증 실패를 감추거나 허용 오차를 자동 확대하지 않는다. 기존 자동 모드의 추론 복구 동작을 보존한다.
- 앱/팩에 포함한 코드, DLL, 초기 가중치의 버전·출처·배포 조건·고지 파일을 식별할 수 있다.

## 7. 성능 기준

224×224, batch=1의 EfficientNet B0 및 사용자 모델을 대상 Windows/i7에서 측정한다.
1채널/3채널, thread 1/2/4/8, 런타임 버전과 CPU 모델을 기록한다.

`전처리 + 모델 실행 + 후처리` 8 ms는 유지하는 목표다. 도커 도입에 따른 달성 보장은 하지 않는다.
워밍업 후 p50/p95/p99/최댓값과 초과 비율, 첫 실행·세션 로드·큐 대기·파일 읽기 시간을 별도로 기록한다.
앱 추론 버튼의 전체 시간도 함께 보여 측정 구간을 바꿔서 빨라 보이는 일을 피한다.
모든 이미지가 8 ms 이하라는 요구는 p95 목표와 별개의 엄격한 조건으로 판정한다.

## 8. 주요 위험과 대응

| 위험 | 대응 |
|---|---|
| 회사 PC에서 WSL2·가상화·Docker 설치 불가 | 기본 CPU/ONNX 경로 유지, Docker 팩만 사용 불가 표시 |
| Docker가 추론을 더 빠르게 해준다는 기대 | 격리·확장 도구로 정의, 직접 실행과 전송 포함 수치 비교 |
| 새 모델의 연산/후처리/시각화가 기존 규약 밖임 | SDK 확장 필요를 명시, 호환되지 않는 팩을 임의 실행하지 않음 |
| 사전학습 가중치의 재배포 조건 불명확 | 가중치별 검토 상태를 기록, 미확인 가중치는 공개 묶음 제외 |
| 설치 파일 대용량·DLL 충돌 | CPU/GPU 분리, onedir 기반 설치, 버전별 런타임 고정 |
| 학습과 CPU 추론의 자원 경쟁 | 기본적으로 동시 실행 제한, 큐·스레드·메모리 예산 표시 |
| 팩 교체가 기존 프로젝트 결과를 바꿈 | 프로젝트에 팩 버전·파일 hash 고정, 명시적 버전 전환 |

## 9. 근거 자료

- [Docker Windows 설치 조건](https://docs.docker.com/desktop/setup/install/windows-install/): 선택 기능의 사전조건 확인.
- [Docker WSL2 GPU 지원](https://docs.docker.com/desktop/features/gpu/): GPU 팩은 별도 장비 검증 필요.
- [PyInstaller 동작과 배포](https://pyinstaller.org/en/stable/operating-mode.html): Windows에서 Windows 번들을 만들고 Python 런타임을 포함한다.
- [TorchVision 모델/가중치 안내](https://docs.pytorch.org/vision/main/models): 코드 라이선스와 사전학습 모델의 조건을 별도로 확인한다.
- [상세 설계](../../02-design/features/model-packs-windows-distribution.design.md).
