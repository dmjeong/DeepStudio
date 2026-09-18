# Deep Vision Studio — Windows 학습툴

**배포 README 설계 초안 · 2026-09-18**
이 파일은 다음 배포판에 동봉할 문서다. 아래 모델 지원과 최소사양은 개발·검증 목표이며,
현재 저장소 또는 아직 생성하지 않은 설치파일의 기능을 보장하는 릴리스 공지가 아니다.
정식 배포 시 고정한 모델/런타임/드라이버 버전과 실제 검증된 사양으로 갱신한다.

## 제품 구성

하나의 `DeepVisionStudio-Setup-<version>-win-x64.exe`에 Windows 프로그램, 기본 모델,
재배포 조건을 확인한 초기 가중치, 필요한 런타임, C#/C++ SDK, 문서와 라이선스 고지를 넣는다.
지원사양에 맞는 PC에서는 Python·pip·conda·Node·Git·CUDA Toolkit을 따로 설치하지 않는다.
기본 모델의 최초 실행도 인터넷 다운로드 없이 동작하도록 구성한다.

| 태스크 | 기본 제공 목표 | 초기 세부 변형 제안 |
|---|---|---|
| Classification | EfficientNet B0/B1 | B0, B1 |
| Classification | ResNet | 18, 50 |
| Classification | ConvNeXt V1 | Tiny |
| Classification | LibreYOLO | MobileNetV4 Small 분류 모델 |
| Anomaly detection | PatchCore | Wide-ResNet50-2, ResNet18 백본 |
| Object detection | Re-DETR v4 | Small, Medium, Large |
| Object detection | LibreYOLO | LibreYOLO9 Tiny |
| Segmentation | SAM2 | SAM2.1 Hiera Tiny, Small, Base+, Large; image/prompt/video 계약 |
| Segmentation | DeepLab V3+ | ResNet34 encoder |
| Segmentation | U-Net | ResNet18 encoder |

ResNet/ConvNeXt/DeepLab V3+/U-Net의 native adapter는 weight-free 구조로 제공되며,
사용자가 별도로 검증한 가중치를 학습 결과에 저장한다. 외부 pretrained 파일은 기본 설치파일에
자동으로 포함하지 않는다.

모든 기본 모델군에 학습/fit·추론·ONNX 배포·C#/C++ 실행을 제공하는 것이 출시 조건이다.
LibreYOLO 라이브러리의 모든 모델/변형을 기본 제공한다는 뜻은 아니다.
세부 모델·가중치·라이선스 확인과 Windows 검증을 끝낸 버전을 정식 지원표에 적는다.
설치 payload에는 `models/default-model-catalog.json`을 함께 넣어 기본 모델 ID와 검증 상태를
오프라인에서 표시한다. 이 카탈로그는 가중치를 포함하지 않으며, 가중치와 third-party 코드는
각각 재배포 근거가 확인된 팩에만 넣는다.

## 최소사양 — 초기 검증 기준

아래는 아직 실측으로 확정하지 않은 설계 사양이다. 큰 해상도·batch·모델에서는 더 많은 메모리가 필요하다.

| 항목 | UI·라벨링·경량 CPU 추론 | 기본 GPU 학습 검증 기준 | 큰 모델/고해상도 권장 검증 기준 |
|---|---|---|---|
| OS | 지원 중인 Windows 11 x64 | 동일 | 동일 |
| CPU | AVX2 지원 4코어 이상 | AVX2 지원 8코어 이상 | 8코어 이상 |
| RAM | 16GB 이상 | 32GB 이상 | 64GB 이상 |
| GPU | 필수 아님 | 지원하는 NVIDIA CUDA GPU, VRAM12GB 이상 | VRAM24GB 이상 |
| 여유 저장공간 | SSD50GB + 데이터 | SSD100GB + 데이터 | NVMe200GB + 데이터 |
| 설치 권한 | 시스템 구성용 관리자 권한 | 동일 | 동일 |
| 앱 실행 | 일반 사용자 권한 | 동일 | 동일 |

- CPU 추론 사양은 전체 모델 학습 속도·메모리를 보장하는 사양이 아니다.
- GPU의 지원 세대/compute capability와 최소 Windows 드라이버 버전은 배포 runtime을 고정한 뒤 실측하여
  정식 README와 `release-manifest.json`에 적는다. 현재 해당 값은 확정 전이며, 미기재 상태로 정식 배포하지 않는다.
- 호환 NVIDIA 드라이버는 PC 사전조건이다. 번들 CUDA runtime은 그래픽 드라이버를 대체하지 않는다.
- 실제 설치 여유공간은 임시 해제, 설치물, 이전 버전 복구, WSL 가상디스크와 모델 이미지까지 계산한다.
  설치기가 표시하는 계산값이 위 설계값보다 크면 계산값을 적용한다. 데이터/학습 결과 공간은 별도다.
- Windows 10, Windows Server, Windows ARM64, AMD/Intel GPU 학습은 초기 지원으로 표시하지 않는다.

최소 GPU 인수 테스트는 작은 기본 변형, batch1에서 시작한다. 기본 입력은 분류224(B1은240),
탐지640 후보, semantic 분할512, PatchCore224/bank≤4096, SAM2 Tiny1024다.
SAM2의 최소 프로필은 encoder를 고정한 decoder fine-tune이다. 전체 encoder 학습과 큰 batch는
별도의 RAM/VRAM 테스트 결과를 확인해야 한다. 12GB로 모든 설정이 학습된다는 뜻은 아니다.

## Docker 모델 확장 사전조건

추가 모델은 앱의 `모델 관리 → 모델 팩 가져오기`에서 `.dvmodel`을 선택한다.
이미지와 의존성이 들어 있는 팩을 로컬로 가져와 실행하며 설치 중 Docker Hub/pip/apt 다운로드를 하지 않는다.
팩의 `manifest.json`은 선택적으로 `worker_entrypoint: "module:factory"`를 선언한다. 앱은 이 값을
호스트에서 import하지 않고, 격리된 컨테이너 안의 `DVW1` stdin/stdout worker가 로드한다.
worker는 `hello`, `describe`, `prepare`, `train`, `infer`, `export`, `cancel`, `close` 프레임을 사용하며
stdout에는 프레임 외의 로그를 쓰지 않는다. 모델 팩을 다시 만들지 않고도 이 계약을 구현한 새 모델을 추가할 수 있다.
호스트는 설치된 팩의 `runtime_requirements.container_image`를 확인한 뒤 `pack_train`, `pack_infer`,
`pack_export` 작업으로 같은 worker를 호출한다. 이미지 참조는 `@sha256:<digest>`로 고정해야 하며,
태그만 있는 이미지는 실행하지 않는다. 오프라인 설치가 필요한 팩은 `manifest.json`의
`container_image_archive`에 `docker save` tar 경로를 선언하고 checksum에 포함해야 한다. worker가
시작되기 전에 `docker load --input`으로 이미지를 가져오고, 로컬 이미지 digest가 manifest와 일치하지
않으면 실행을 거부한다. 작업 요청의 JSON은 `/data`와 `/work`에 있는 파일을 가리키고,
큰 이미지·체크포인트는 DVW1 프레임에 직접 넣지 않는다.
설치기가 앱 전용 distro를 초기화하면 실행 환경에 `DEEPVISION_WSL_DISTRO`와
`DEEPVISION_WSL_STATE_DIR`를 설정한다. 이 두 값이 있으면 worker는 모든 Docker 명령을
`wsl.exe -d <distro> -- docker ...` 인수 배열로 실행하고, `owned-distro.json` 소유 표식이 없을 때
호스트 Docker로 조용히 전환하지 않는다. 두 환경변수가 없을 때의 `docker` 기본값은 개발용이며,
Docker Desktop이나 사용자의 기존 WSL 배포판을 자동으로 변경하지 않는다.
설치가 끝나면 학습 화면의 모델 카탈로그에서 해당 팩을 선택하고 `팩 학습`, `팩 추론`,
`팩 ONNX export` 버튼으로 실행한다. 버튼은 절대 경로의 설치 팩과 데이터 루트가 확인되고
다른 데스크톱 작업이 idle일 때만 활성화되며, 작업 종료·취소·실패 후에는 화면 잠금이 해제된다.

설치 프로그램이 앱 전용 WSL2와 Docker Engine 환경을 구성한다. 다음 조건은 필요하다.
production Setup은 `build_release.ps1 -RequireOfflineWsl` gate를 통과한 payload만 만들며,
WSL 오프라인 MSI·앱 전용 distro tar·라이선스 inventory의 SHA-256과 고지 파일을 함께 검증한다.

- WSL2/SLAT를 지원하는 CPU, BIOS/UEFI 가상화 활성화.
- Windows 가상화 기능을 사용할 수 있는 회사 보안 정책.
- 관리자 설치 권한과 필요 시 재부팅 허용.
- WSL 이미지·모델 이미지·작업 복사본을 저장할 추가 공간.
- GPU 컨테이너 사용 시 WSL GPU를 지원하는 호환 Windows NVIDIA 드라이버.

BIOS 설정이나 조직 정책 때문에 가상화를 사용할 수 없다면 설치기가 이를 우회하지 않는다.
기본 Windows 모델은 사용할 수 있어도 Docker 확장까지 준비된 상태는 아니다.
별도로 Docker Desktop을 설치하거나 그 설정을 바꿀 필요가 없도록 앱 전용 환경을 사용한다.
기존 사용자 WSL 배포판이나 Docker 환경을 덮어쓰지 않는다. 앱 전용 환경은 실제 Windows 사용자별로
등록하며 두 번째 사용자는 첫 실행 시 설치된 로컬 자산으로 자동 초기화한다. 추가 다운로드는 필요하지 않다.

## 설치와 첫 사용

1. 최소사양과 GPU 드라이버/가상화 조건을 확인한다.
2. Setup을 실행해 경로·구성·라이선스 고지를 확인한다.
3. 설치기가 번들 파일과 필요한 Windows 구성요소를 검사·설치한다. 필요하면 재부팅 후 이어간다.
4. 설치 완료 검사에서 기본 모델/가중치와 Docker 확장 준비 상태를 확인한다.
5. 새 프로젝트에서 태스크와 모델을 선택하고 로컬 데이터를 등록한다.
6. 학습/fit 결과를 저장한 후 추론하거나 `ONNX 배포 내보내기`를 사용한다. 컨테이너 전용 모델은
   학습 화면에서 팩 작업 버튼을 사용한다.

한글·공백 경로와 일반 사용자 실행을 지원하도록 검증한다.
프로젝트·원본 이미지·학습 결과는 사용자가 지정한 폴더에 저장한다.
사용자 데이터나 모델을 외부 서버로 업로드하는 절차는 없다.

## 모델별 알아둘 점

- EfficientNet B1의 기본 사전학습 입력은 B0와 다르다. 모델과 함께 저장된 전처리를 적용한다.
- PatchCore 학습은 정상 이미지의 특징 bank 생성/선택과 threshold 보정이다. 일반 epoch 학습과 구분한다.
- DeepLab V3+/U-Net의 초기 가중치는 encoder 사전학습일 수 있으며, 사용자 클래스 분할 head는 학습해야 한다.
- SAM2는 Hiera Tiny/Small/Base+/Large를 모두 제공 대상으로 관리한다. 점·박스·이전 mask prompt,
  자동 mask와 영상 state는 각각의 export/runtime 검증을 통과한 기능만 활성화한다.
- Re-DETR v4는 Small/Medium/Large 세 변형만 제공 대상으로 관리하며, 실제 검증 전에는 설치기에서
  release-ready로 표시하지 않는다.
- Re-DETR 팩은 `pred_boxes`/`pred_logits` 출력 계약을 C++17/C ABI에서 검증한다. 실제 Small/Medium/Large
  checkpoint의 ONNX 수치 및 Windows 인수 검증 전에는 설치기에서 release-ready로 표시하지 않는다.
- SAM2 팩은 encoder·decoder 파일과
  point/box/mask prompt 계약을 manifest에 기록해야 한다. 계약이 없는 팩은 등록되지 않는다.

## ONNX 배포와 C#/C++

내보내기 결과에는 ONNX 그래프, 필요한 external data, 전후처리/클래스/threshold 설정,
검증 결과와 고지가 들어간다. SAM2는 encoder/decoder 그래프를 함께 배포한다.
PatchCore는 memory bank와 점수 계산을 포함한 배포 결과를 사용한다.

최종 C++/C# 전달물은 `tools/build_deployment_bundle.py <export.onnx> <model.dvdeploy>`로 묶는다.
번들은 설정 JSON, 모든 ONNX 그래프, external data와 고정 SHA-256 manifest를 포함하며 검증되지 않은
파일이 하나라도 있으면 SDK에 전달하지 않는다. C++/C#의 `OpenBundle`도 Python 없이 manifest의
모든 파일 hash와 그래프 참조를 다시 검증한다. SAM2 export 디렉터리는 디렉터리 자체를 source로 넘긴다.

C++17 또는 C# SDK는 이 배포 폴더를 열어 추론한다. 추론 프로그램에 Python·학습툴·Docker 설치를 요구하지 않는다.
현재 저장소에는 `sdk/cpp/README.ko.md`와 C# `VisionSession` SafeHandle 래퍼가 포함되어 있다.
C# 프로젝트는 `net8.0`을 대상으로 하며, 최종 설치 EXE의 self-contained 데모는 Windows runner에서
별도로 빌드·검증한다. C++ 예제는 Windows x64 Release로 제공한다.
.NET Framework 4.8 지원은 별도 검증 전까지 표시하지 않는다.
개발 프로젝트를 빌드하는 도구는 개발자에게 필요하지만 완성된 프로그램의 일반 사용자에게 요구하지 않는다.

모델을 한 번 초기화하고 여러 이미지를 반복 처리한다. 입력 buffer 수명·background 큐·결과 메모리 해제는
SDK 설명을 따른다. ONNX 파일만 복사하고 설정이나 external data를 빼면 정상 배포가 아니다.

## 속도와 오류

표시 시간은 파일 읽기·전처리·모델·후처리·큐·통신·화면 시간을 구분한다.
EfficientNet B0 224에서 8ms는 특정 Windows/i7 실측으로 확인할 목표이며 전체 모델의 보장 수치가 아니다.
Docker 추가가 자동으로 더 빠른 추론을 뜻하지 않는다.

가중치 누락, 잘못된 팩/런타임, 수치 검증 실패, GPU 메모리 부족, 큐 포화는 다른 오류로 표시한다.
수치 검증을 통과하지 못한 모델을 허용 오차를 자동 늘려 배포하지 않는다.
다른 모델의 기능과 UI는 계속 사용할 수 있어야 한다.

## 업데이트·제거·라이선스

프로젝트는 모델 팩 버전과 가중치를 고정한다. 업데이트만으로 모델을 바꾸지 않는다.
제거 시 사용자 프로젝트·학습 결과는 보존한다. 앱 전용 WSL/모델 이미지 삭제는 별도 선택 사항이다.

`licenses/`, `THIRD_PARTY_NOTICES.md`, `sbom.cdx.json`, `release-manifest.json`을 함께 제공한다.
코드, pretrained 가중치, OS 패키지, GPU 라이브러리의 배포 조건을 각각 확인한다.
상업적 이용/재배포 근거가 없는 가중치를 기본 설치파일에 넣지 않는다.
`release_ready`로 표시하는 `.dvmodel`은 `THIRD_PARTY_NOTICES.md`와 `licenses/` 파일을
반드시 포함해야 하며, 빌더와 설치기가 이 조건을 거부한다.

정식 배포 전 필수 확인: 실제 단일 EXE 생성·서명, 정확한 OS/runtime/driver 최소 버전,
모델별 Windows 학습·ONNX·C#/C++ 결과 일치, 오프라인 Docker 추가, 위 사양에서의 메모리 실측.

## 개발용 모델 팩

추가 모델을 앱에 넣을 때는 코드를 호스트에서 import하지 않고 팩으로 묶는다.
`tools/build_model_pack.py SOURCE OUTPUT.dvmodel --allow-unsigned`는 로컬 개발용
팩을 만들고 `tools/install_model_pack.py`는 경로 탈출·symlink·압축 폭탄·모든 SHA-256을
검사한 뒤 staging 디렉터리에서 원자 활성화한다. `--allow-unsigned`는 개발용에서만 사용하며,
출시 팩은 외부 서명/신뢰 키 검증을 통과한 manifest를 사용한다.

Windows 빌드 이미지는 먼저 `stage_payload.py`로 UI·worker·SDK·팩·고지·SBOM을 하나의 payload
루트에 모은다. `collect_payloads.py`와 `validate_payloads.py`가 모든 파일의 크기·SHA-256과
3.5 GiB 예산을 검사한 뒤에만 WiX MSI/Burn EXE를 생성한다. 이 단계는 인터넷 다운로드나 외부
모델 허브 호출을 수행하지 않는다.

PyInstaller GUI를 만들 때는 네이티브 C++ SDK와 ONNX Runtime/OpenCV DLL이 들어 있는 디렉터리를
`VISION_NATIVE_RUNTIME_DIR` 환경변수로 지정한다. `gui/build_exe.py`가 해당 디렉터리의 DLL을
번들에 넣고, 모델 팩 schema와 worker runtime 모듈도 함께 포함한다. 이 디렉터리를 비워 둔 개발
빌드는 UI만 만들며, 정식 payload 검증은 `vision_runtime.dll`과 의존 DLL을 포함한 staged payload에서
수행한다.
