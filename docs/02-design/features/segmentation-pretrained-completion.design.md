# 13.85 — 새 세그멘테이션 5종 사전학습 연결 보완

사용자 요청: 이번에 추가한 모델 전부를 사전학습 가중치로 학습할 수 있게 한다.
13.84에서 빠진 BiSeNetV2/STDC1을 보완하고 기존 U-Net 3종도 실제 파일로 재검증한다.

- BiSeNetV2: CoinCheung 제공 ImageNet `backbone_v2.pth`; detail/segment/bga를 전부 엄격 로드한다.
  학습 데이터 출처는 제공자의 issues/192, issues/173 답변으로 확인한다.
- STDC1: 논문 저자 README의 Google Drive에 있는 ImageNet `STDCNet813M_73.91.tar`.
  canonical features 키와 중복 등록된 x2/x4/x8/x16/x32 키를 같은 텐서로 채워 strict 로드한다.
- 공식 출처 URL + 고정 SHA256으로 다운로드/수동 가져오기/오프라인 로드를 제공한다.
  HTTPS/인증서/해시 확인은 유지하며, 누락/불일치 시 무작위 초기화로 대체하지 않는다.
- build.bat과 update_segmentation_models.bat에서 5종의 백본을 모두 준비한다.
  분할 헤드는 사용자 클래스에 맞춰 새로 학습하고 사전학습 선택을 UI에 노출한다.
- RGB 백본을 로드한 뒤 흑백 stem 필터를 합산한다. 전체 백본이 정확히 로드됐는지 검사한다.
- 백본/새 헤드 학습률 그룹을 구분하고 백본 고정 시 BN 통계도 보존한다.
  13.84 단일 optimizer 그룹으로 학습한 두 모델의 resume 그룹 순서를 보존한다.
- 모델 구조/forward/ONNX 출력 계약은 변경하지 않는다. 기존 .pt는 다운로드 없이 복원한다.

검증: 5종 실제 사전학습 → 합성 패치 학습 → best 저장/복원 → FP32 ONNX 출력 비교 → INT8 실행,
C++ 마스크 비교 및 암호화 동일성. 640 학습/내보내기, 1채널 로드, 누락/부분 가중치 거부,
백본 고정 및 기존 재개, GUI/가중치 준비 회귀 검사. Windows/CUDA와 개인 데이터 정확도는 별도다.
