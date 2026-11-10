# LibreYOLO 전체 영상 입력 — 11.84

## Plan / 원인
- MobileNetV4 native predict/val은 짧은 변 확대 후 중앙 자르기, train은 RandomResizedCrop을 사용했다. 앱의 전체 입력 표시 및 OpenCV 기반 ONNX/C++와 불일치했다.
- 전체 시야를 유지하는 기존 배포 계약 `opencv_linear_exact_v1`을 기본 제공 LibreYOLO 분류의 학습/검증/추론/CAM에 적용한다. 픽셀 해상도 보존이 아니라 시야 보존이며 입력은 기존 224×224다.

## Design
- 전체 RGB 또는 사용자가 명시한 ROI를 OpenCV INTER_LINEAR_EXACT로 모델 크기에 맞춘 뒤 ImageNet 정규화한다. 자동 중앙 자르기/무작위 자르기는 제거하고 기존 수평 반전 증강만 유지한다.
- 원본 전체 모드는 저장 크롭도 해제한다. JSON/모델 지정 ROI만 자르고 CAM은 해당 좌표로 복원한다. 탐지 전처리는 이번 변경 대상이 아니다.
- 서드파티 전역 함수 교체 없이 Studio 전처리 모듈, wrapper 인스턴스 어댑터, 전용 trainer/validator 서브클래스를 사용한다. DataLoader spawn에서 직렬화 가능한 최상위 transform을 쓴다.
- 새 best/last/epoch 체크포인트에 `studio_preprocessing` 계약을 저장한다. 구형 가중치는 읽기 전용으로 전체 입력에 사용하고 변경 사실을 표시한다. 구형 중단 재개는 기존 best 지표를 섞지 않도록 거부하고 추가 학습을 안내한다.
- ONNX JSON에 같은 계약/명시 ROI를 기록한다. 입력 이미지부터 Python/ORT/C++ 출력까지 확인한다.

## Check 계획
- 가로/세로/정사각형 가장자리 표식이 train/val/predict/CAM에 보존되는지 검사한다.
- 기존 모델 로드, 전체/명시 ROI 전환, GUI 실제 버튼, CAM 실패 및 캐시 회귀.
- 실제 native 1 epoch 학습, best/last 메타데이터와 새 모델 resume 검사; transform spawn 피클 검사.
- 실제 native ONNX export 후 이미지 전처리·확률 비교 및 C++ CPU 분류. Windows/GPU 검증 여부는 별도 명시한다.
