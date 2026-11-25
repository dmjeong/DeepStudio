# LibreYOLO Grad-CAM / 원본 비교 복구

## Plan
- 기본 제공 MobileNetV4 분류 `.pt`의 Grad-CAM 누락과 upstream 결과의 원본 캐시 누락을 수정한다.
- 격자 화면에서 원본 비교를 켜면 선택 이미지의 상세 비교로 전환한다.
- 탐지 모델은 박스를 표시하며, 분류용 CAM을 탐지에 임의 적용하지 않는다.

## Design
- 고정 의존성 LibreYOLO 1.5.0의 MobileNetV4 `blocks`(풀링 전 공간 특징)와 `classifier`(logits)를 기존 GradCAM에 연결한다. `conv_head`는 풀링 뒤 1×1이므로 사용하지 않는다.
- native `predict`의 판정은 유지한다. CAM은 선택했을 때만 별도 역전파하며 타겟은 실제 예측 클래스다. 실패해도 추론 결과와 원본을 보존한다.
- upstream의 `build_eval_transform`을 사용하고 실제 Resize 크기를 기록해 기존 `reproject_classification_cam`으로 중앙 잘림을 복원한다. 보지 않은 영역은 착색하지 않는다.
- 원본은 분류/탐지 모두 저장한다. UI는 원본/결과 동일 배율, 양방향 이동, 이미지 교체 및 캐시 재열기를 검증한다.

## Check 계획
- 실제 pinned LibreMobileNetV4 생성/저장/로드 → native 확률 일치 → CAM 생성과 훅/가중치 보존.
- CAM 끔/오류, 가로·세로 원본, 중앙 잘림의 좌표와 미사용 영역, 분류/탐지 캐시를 검사한다.
- Qt 화면에서 격자 → 원본 비교, 원본 파일 삭제 후 캐시 표시, 모델 변경 시 활성화, 배율 동기화를 확인한다.
- 관련 기존 heatmap, inference cache/worker, segmentation 표시 회귀 테스트. Windows/사용자 가중치 확인과 구별해서 보고한다.
