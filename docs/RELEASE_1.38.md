# 1.38 — 모델 선택·ONNX 내보내기·라벨 삭제 수정

1.28 이후의 간단한 기능 추가 릴리스다. 사용자 버전 규칙에 따라 +0.1을 적용해 1.38이다.

## 바뀐 기능

- EfficientNet B0/B1은 `사전학습`, `로컬 가중치`, `학습 재개`, `처음부터 학습`을 같은 EfficientNet 구조에서 실행한다.
- ResNet18/50, ConvNeXt V1 Tiny, DeepLab V3+, U-Net은 `처음부터 학습`을 별도 기본 모델 모드로 표시한다. 선택한 기본 모델을 Custom CSP로 바꾸지 않는다.
- ONNX 화면은 프로젝트 전환 시 이전 체크포인트·로그·진행 상태를 초기화한다. 내보내기 전 checkpoint의 task/model metadata가 현재 프로젝트와 다르면 시작하지 않는다.
- 분할 지우개는 클래스 0의 삭제 operation으로 저장된다. 화면의 객체 목록에는 보이지 않으며, 기존 영역의 클래스를 바꾸거나 저장 후 다시 열어도 삭제한 픽셀은 복원되지 않는다. 빈 배경 지우기는 변경으로 저장하지 않는다.
- 검출 라벨링은 `E`를 누른 뒤 박스를 클릭하면 해당 객체를 삭제한다. 빈 곳을 클릭해도 새 박스를 만들지 않는다.
- 추가 Docker 모델 팩은 `vendor.model` 같은 별도 ID를 사용해야 한다. 기본 제공 모델 ID를 재정의할 수 없다.

## 확인한 범위

- Qt offscreen 회귀: 분할 지우개 저장/재열기/undo, 검출 편집, export 프로젝트 전환, 모델 선택 모드.
- 모델 등록부: 기본 모델 ID 덮어쓰기 거절.
- 실제 EfficientNet B0/B1 ONNX export 및 ONNX Runtime 비교 회귀.
- 웹 TypeScript 빌드.

SAM2, RT-DETRv4, LibreYOLO의 native 학습 worker와 실제 모델별 Windows ONNX/C++/C# 인수는 이 릴리스에서 완료하지 않았다. 카탈로그 이름만으로 지원 완료로 표시하지 않는다.
