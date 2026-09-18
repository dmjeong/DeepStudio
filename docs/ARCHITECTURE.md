# Deep Vision Studio 구조

## 실행 계층

Qt 데스크톱과 React 웹은 같은 프로젝트 및 학습/추론 계약을 사용합니다.
GUI는 화면·입력·상태 표시를 담당하고, webapp 작업 프로세스가 모델을 로드하여
학습·추론·내보내기를 실행합니다. Qt 없는 CLI도 공통 계산 엔진을 사용합니다.

## 태스크별 모델

| 태스크 | Python 모델 | 출력 |
|---|---|---|
| 분류 | EfficientNet, CustomCSP | 클래스 logits |
| 시맨틱 분할 | CustomCSP | 픽셀별 logits |
| 박스 탐지 | CustomCSP | 객체 박스·클래스 점수 |
| 재구성 이상 탐지 | CustomCSP | 복원 이미지 |
| 특징 기반 이상 탐지 | PatchCore | 이상 점수와 위치 맵 |

회전 박스는 편집·데이터 검증만 지원하며 실행 엔진은 제공하지 않습니다.
학습 모드의 단일 정의는 gui/core/training_modes.py입니다. 미지원 모드는 새 엔진으로
임의 전환하지 않고 거부합니다.

## 저장과 배포

프로젝트 JSON 확장자는 .dvproj입니다. 모델 state_dict와 전처리 메타데이터를 함께
저장합니다. 기존 체크포인트의 모델/채널/정규화/입력 크기 계약을 추측하여 변경하지 않습니다.

CPU EfficientNet의 auto 경로는 ONNX 수치 검증 후 실행합니다. 준비 실패 시 PyTorch로
복구하며 결과에 실제 엔진과 원래 실패 사유를 보존합니다. 사용자 모델/이미지 전송은
이 경로에 포함되지 않습니다.

C++17 공개 API는 vision_inference.h의 VisionInference와 classification_worker.h의
ClassificationWorker입니다. ONNX Runtime이 기본이며 OpenVINO는 선택 구성입니다.
버퍼/세션 재사용 및 결과 소유권 규칙은 C++ 안내를 따릅니다.

## 이름과 경로

루트 DeepVisionStudio, Python 자체 모델 CustomCSP, C++ 라이브러리 vision_inference,
데모 vision_demo로 통일합니다. 패키지 ZIP, CI와 문서도 같은 경로를 사용합니다.
