# 2.38 — 기본 제공 native 모델 worker

- LibreMobileNetV4 Small 분류, LibreYOLO9 Tiny 탐지, Re-DETR v4 Small/Medium/Large를 Docker 팩이나 Custom CSP 대체 경로 없이 native worker로 연결했다.
- 각 모델은 사전학습 시작, 로컬 가중치 전이, 중단 학습 재개, 동일 구조 스크래치 학습을 선택할 수 있다.
- 기존 검출 이미지/라벨 폴더를 upstream YOLO 데이터 설명으로 변환하며, 한글 클래스 이름과 빈 검증 분할을 보존한다.
- upstream epoch callback을 기존 학습 화면의 손실, 지표, 에폭 시간 및 Best 기록으로 전달한다.
- Re-DETR v4 S/M/L의 배포 입력은 실제 upstream 계약에 맞춰 모두 640×640으로 통일했다.

설치본은 `libreyolo==1.5.0`과 해당 LICENSE·NOTICE를 함께 포함해야 한다. pretrained 가중치는 설치 파일에 포함하지 않는다.
