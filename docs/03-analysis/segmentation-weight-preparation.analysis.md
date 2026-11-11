# 9.57 기본 가중치 준비 검사

## 증상과 확인 범위

사용자는 Windows에서 Python으로 실행하며 DeepLab 외 모델들이 가중치 없음으로 막힌다고 답했다. 앱의 기존 학습 전 검사는 `builtin_asset_path(..., verify=False)`였다. 이 단계의 누락 메시지는 해당 Python 실행이 찾는 폴더에 필요한 파일이 없거나 비어 있음을 뜻한다. 이 메시지만으로 학습 알고리즘의 실패나 Windows 인증서 장애를 단정할 수 없다. 사용자 PC의 현재 파일 목록·환경변수·다운로드 오류는 직접 확인하지 못했다.

## 코드에서 확인한 준비/배포 문제

1. README의 Python 직접 실행 안내는 패키지 설치 후 바로 `main.py`를 실행했다. 기본 가중치 준비를 생략했다. 기존 `build.bat`에는 이미 전체 가중치 다운로드가 있으며 U-Net이나 SAM2를 목록에서 빠뜨린 것은 아니다.
2. 준비 명령은 상대 `builtin_assets` 폴더를 지정하는데, 앱은 `DVS_BUILTIN_ASSETS_DIR`를 우선했다. 환경변수가 있으면 다운로드 성공과 앱 검색 실패가 공존할 수 있었다. EXE 빌더도 검사한 경로와 별도의 고정 경로를 묶었다.
3. SAM2는 Hugging Face Hub 다운로드 경로를 사용했고, ResNet의 공통 네이티브 CA 설정을 적용하지 않았다. SAM2 공식 파일의 공개 SHA256을 코드에서 알지 못해 수동 공식 파일도 manifest가 없으면 거부했다.
4. 도움말은 SAM2에도 무작위 초기화를 대안으로 권했다. 실제로 해당 모드는 지원하지 않는다.

## 변경

Python 준비 도구·배치·EXE 빌더가 공통 경로를 사용한다. 기본 배치는 전체 기본 모델을, `segment`는 ResNet18/34 및 SAM2 4종을 준비한다. `check`는 읽기만 하며 누락 모델마다 전체 경로를 표시한다. `build.bat weights`는 PyTorch 설치/EXE 빌드를 거치지 않는다. `build.bat setup gpu`는 신규 Python 환경 준비에 사용한다.

SAM2는 Meta 공식 URL을 먼저 사용하고 네트워크 오류이면 공식 Hugging Face 미러를 시도한다. 양쪽 모두 동일하게 고정한 공개 SHA256으로 확인한다. 인증서/호스트 검증을 끄지 않으며 해시 불일치에는 미러로 우회하지 않는다. 다운받은 임시 파일은 검증이 끝나야 최종 경로로 게시된다. 기존 manifest의 해시/크기 불일치와 손상 JSON은 계속 거부한다.

## 실제 파일 검증

공식 SAM2.1 Tiny/Small/Base+/Large 네 파일 모두 다운로드 및 공개 SHA256 확인에 성공했다. 생성된 manifest를 삭제한 뒤 앱 검색 함수로 다시 검증했고, 공식 모델 생성 및 1024×1024 이미지 인코더 연산까지 수행했다. 네 모델 모두 feature 출력이 유한했다. 환경은 macOS ARM64 CPU이며 Windows CUDA 검사나 분할 정확도 측정이 아니다.

- 실행 기록: [JSON](artifacts/segmentation-weight-preparation-runtime.json)
- U-Net 공식 ResNet18 1/3채널 학습→저장→재로드→추론→Qt 표시 검사: [기존 실행 기록](artifacts/segmentation-model-audit.json), 소량 합성 데이터의 기능 검사이며 정확도 증거가 아님.
- 회귀: 가중치/인증서/배포 계약 검사 69개 통과. 전체 검사 1,258개 통과·12개 건너뜀(150.67초). 이후 상대 경로 해석 및 오류 안내 보강에 대한 관련 검사 46개 통과. 모든 검사 프로세스 종료 코드 0.

공식 근거: [Meta 다운로드 안내](https://github.com/facebookresearch/sam2#download-checkpoints), [Tiny SHA256](https://huggingface.co/facebook/sam2.1-hiera-tiny/blob/main/sam2.1_hiera_tiny.pt), [Small SHA256](https://huggingface.co/facebook/sam2.1-hiera-small/blob/main/sam2.1_hiera_small.pt), [Base+ SHA256](https://huggingface.co/facebook/sam2.1-hiera-base-plus/blob/main/sam2.1_hiera_base_plus.pt), [Large SHA256](https://huggingface.co/facebook/sam2.1-hiera-large/blob/main/sam2.1_hiera_large.pt).

## 남은 사항

사용자 Windows PC에서 다운로드/배치 실행 확인은 필요하다. SAM2 prompt 학습 체크포인트를 일반 semantic 추론 화면에서 여는 연결 문제는 가중치 준비와 별개이며 이 수정은 해당 연결을 완료하지 않는다. 모델 구조/학습 손실/ONNX 변환/추론 계산은 변경하지 않았다.
