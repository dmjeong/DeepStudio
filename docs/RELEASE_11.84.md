# 11.84 — LibreYOLO 분류의 전체 영상 전처리

## 문제와 변경

LibreYOLO 1.5.0 MobileNetV4 Small의 기본 평가/추론은 짧은 변을 256으로 확대한 뒤
중앙 224×224만 선택한다. 11.83의 CAM은 이 입력을 그대로 설명했으므로 가장자리에는
설명이 없었다. 앱의 전체 입력 설정과 기존 ONNX/C++의 전체 resize 전처리도 서로 달랐다.

- 전체 RGB를 OpenCV `INTER_LINEAR_EXACT`로 224×224에 맞추고 ImageNet 정규화한다.
  입력 시야는 유지하지만, 원본 픽셀 해상도와 종횡비를 유지하는 방식은 아니다.
- 학습의 RandomResizedCrop, 검증/추론의 자동 CenterCrop을 제거했다. 기존 좌우 반전은 유지한다.
  모델 구조·입력 텐서 크기·가중치 자체를 변경하지 않는다.
- 전용 trainer/validator와 인스턴스별 입력 어댑터를 사용한다. 외부 라이브러리의 전역 함수는 교체하지 않는다.
- 명시적 ROI는 학습용 복사 데이터에 적용하고 체크포인트에 기록한다. LibreYOLO의 `model`이
  state_dict여서 ROI 속성 조회로 빠지던 문제도 수정했다. 추론의 `원본 전체`는 저장 ROI를 해제한다.
- CAM은 실제 predict와 같은 텐서를 사용하고, 전체 또는 지정 ROI의 원본 좌표에 배치한다.
- `studio_preprocessing`을 best/last/epoch에 기록하고 ONNX JSON에도 같은 계약을 전달한다.
  양자화 보정용 numpy 입력도 동일 전처리 및 upstream의 `(CHW, ratio)` 반환 계약으로 연결했다.

## 호환성

기존 `.pt`를 변경하지 않고 로드할 수 있다. 이제 가장자리도 입력에 들어가므로 이전 버전과
판정/확률이 달라질 수 있음을 모델 정보와 로그에 표시한다. 검사 이미지로 재평가해야 하며,
가장자리 정보를 학습시키려면 기존 가중치로 추가 학습할 수 있다.

구형 중앙 자르기 체크포인트의 중단 재개는 이전 best 지표와 새 입력 기준을 혼합하지 않도록
거부한다. 새 체크포인트는 동일한 입력 영역에서 재개 가능하다. C++의 기존 전체 resize 계산은
변경하지 않았으며 새 ONNX/JSON 쌍을 그대로 읽는다.

## 검증

macOS ARM64 CPU, LibreYOLO 1.5.0, PyTorch 2.14.1, ORT 1.29.0,
Python OpenCV 5 및 C++ OpenCV 4.12 Release에서 **226개 통과, 2개 건너뜀**.
건너뛴 항목은 기존 선택적 native 학습/추론 의존성 검사이며 신규 full-frame 7개는 모두 실행했다.

- 원본 네 가장자리 표식 보존, 학습 좌우 반전, Windows와 같은 spawn 방식 DataLoader.
- 실제 MobileNetV4 1 epoch 학습/검증 및 새 체크포인트를 사용한 다음 epoch 재개.
- 앱 학습 진입점의 명시 ROI 적용 및 best/last 저장. 구형 가중치 로드와 구형 중단 재개 거부.
- 가로/세로/정사각형 Grad-CAM, 원본 비교, 실제 Qt 버튼→자식 프로세스→CAM 표시.
- 실제 학습 체크포인트 ONNX 내보내기→전체/지정 ROI 이미지→C++ SDK `Classify` 실행.
  Python predict·Python ORT·C++ 확률을 `atol=1e-5, rtol=1e-4`로 비교하고 top-1 일치를 확인했다.
- 기존 탐지/분할/PatchCore 표시, 암호화 내보내기, 학습 산출물 및 C++ 전처리 회귀 검사.
- 변경 Python 코드 lint, `git diff --check` 통과. Linux/Windows CI에 실제 native 테스트 연결.

합성 이미지와 무작위 초기화 모델을 짧게 학습해 기능/계약을 검사했다. 사용자 영상의 검출률을
측정한 결과가 아니며, Windows/GPU 실행 및 이 커밋의 원격 CI 결과는 로컬 검증에 포함하지 않는다.
