# 2.39 — native 학습 진행 화면 수정

LibreMobileNetV4, LibreYOLO9, Re-DETR v4의 native 학습은 이제 worker가 준비되는 즉시 진행률과 모델 정보를 표시한다. 매 에폭마다 train loss, validation loss, 선택 지표, learning rate, Best 선정, 에폭 시간, 누적 시간이 기존 학습 대시보드에 전달된다.

전이학습과 재개학습에서 선택한 로컬 checkpoint 경로도 프로젝트를 다시 열었을 때 학습 화면에 유지된다.
