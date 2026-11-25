# 16.90 — 이미지 이름 변경 및 모델 실행 환경 검증

## 재현·원인

- 이미지와 마스크는 상대 경로의 stem으로 연결된다. 이미지 파일만 외부에서 바꾼 뒤 `ProjectManager.save`를 실행해도 마스크는 이전 이름에 남는다. 로더는 중단하지만 기존 UI에는 미연결 마스크 조회/복구 수단이 없었다.
- timm은 기존 requirements에는 있으나 소스 업데이트가 사용 중인 Python의 패키지를 자동 설치하지 않는다. `update_segmentation_models.bat`의 `--no-deps`는 하위 패키지가 없는 환경을 보완하지 못한다. EXE 사전 의존성 목록에도 timm/SciPy/safetensors가 빠져 있었다.

## 변경 검증

- 이름 변경 관련 실제 Qt 버튼·정답 겹침 미리보기·비동기 저장·일반/패치/크롭 데이터 로더 및 기존 회귀 묶음: **135 passed, 17 skipped**. skip은 native probe/공식 가중치가 필요한 별도 통합 항목이다.
- 모델 의존성 오류·복구 constraints·EXE 패키징·사전학습·registry 검사: **93 passed**.
- timm import를 금지한 별도 Python 프로세스에서 SCTNet 1/3채널 forward/backward/optimizer 2회, 저장·복원, FP32 ONNX 수치 비교, INT8 QDQ 생성·실행 통과. 기존 timm DropPath와 0/0.1/0.5/1 확률 및 train/eval 출력·gradient가 정확히 같고 state_dict는 비어 있다.
- 실제 timm/huggingface_hub/safetensors가 없는 격리 환경을 만들어 복구 스크립트 실행: 세 패키지만 설치되고 torch 2.14.1/torchvision 0.29.1/numpy 기존 버전 보존. 새 프로세스의 실제 import 성공. 이 설치 검증은 macOS에서 수행했고 CUDA 빌드 보존은 `+cu128`이 포함된 제약 파일과 설치 인자도 별도로 검사했다.
- 복구된 환경에서 EfficientViT B0/B1, SCTNet, 기존 5종 segmentation encoder의 공식 사전학습 및 FP32/INT8·암호화·원본 좌표·C++ 통합 묶음: **24 passed**. 기본 runtime에서 SCTNet 공식 가중치 1/3채널 통합도 **2 passed**.
- 같은 복구 환경에서 EoMT DINOv2-S 1/3채널 공식 사전학습·FP32/INT8·C++·암호화·원본 좌표 통합도 **2 passed**.
- rename/relink 실패 시 이미지·마스크·프로젝트 전체 복원. 기존 정답 충돌, 다른 확장자 이름 중복, Windows 예약 이름, 잘못된 크기·클래스·분할 거부. 기존 PNG 메타데이터/픽셀 및 동반 폴리곤 바이트 보존.

- 전체 로컬 회귀(`bf448e5`): **1,624 passed, 78 skipped**, 정상 종료. 마지막 대소문자 이름 충돌·실패 복구 보강은 데이터셋 편집/실제 UI **34 passed**로 별도 확인했다.
- CI와 동일한 E/F/W 린트, 신규 모듈 Ruff, `git diff --check` 통과. 복구 창을 실제 렌더링해 원본 위 정답과 경로·버튼 표시를 확인했다.

Windows/Linux CI: [43139413048](https://github.com/dmjeong/DeepStudio/actions/runs/43139413048). 기록 시 실행 중이며 아직 전체 통과로 간주하지 않는다. Windows CI에 timm을 제거한 뒤 배포 배치로 실제 복구하는 단계도 추가했다.

## 범위

사용자의 실제 검사 데이터/Windows 설치 환경을 직접 읽지는 못했다. 파일 순서나 영상 크기만으로 옛 정답을 추측하지 않는다. 복구 창에서 사용자가 영상 위의 영역을 확인한 경우만 연결한다. 모델 정확도 개선이나 특정 CPU 지연시간을 주장하지 않는다.
