# 2.40 — LibreYOLO `.pt` 추론 로드 수정

기본 제공 LibreYOLO9, Re-DETR v4, LibreMobileNetV4의 학습 체크포인트가 기존 DVS 전용 체크포인트로만 해석되어 추론에서 열리지 않던 오류를 수정했다. 앱은 checkpoint 메타데이터를 확인한 뒤 LibreYOLO의 공개 `LibreYOLO(path, device=...)` factory로 `.pt`를 복원한다.

분류 결과는 native 확률과 클래스 이름을 기존 분류 결과에 표시한다. 검출 결과는 native 정규화 `xyxy` 박스, 점수, 클래스를 기존 검출 목록과 이미지 오버레이에 전달한다. 모델 검사 화면에 native PyTorch 런타임을 표시하며, 이 경로에서 지원하지 않는 Grad-CAM 선택은 비활성화한다.
