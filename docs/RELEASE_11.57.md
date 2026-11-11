# Deep Vision Studio Desktop 11.57

최신 Claude 브랜치 `claude/determined-turing-ssd181` (`04348c4`)를 main에 통합했습니다. 신규 모델과 C++ 평가 기능을 포함하므로 이전 10.57에서 +1.0 올렸습니다.

- U-Net Lite, ImageNet 인코더를 쓰는 U-Net MobileNetV2, DINOv2 분류·분할 모델, 패치 학습과 전체 판정 영역 ONNX 배포를 통합했습니다.
- 기존 main의 DINOv2 모델 ID와 체크포인트 구조, 배경 자동 등록, 라벨 없는 영상의 배경 학습, 원본 위 전경 표시와 분할 평가를 유지했습니다.
- 일부 가중치 다운로드가 실패해도 준비된 모델을 사용할 수 있습니다. 준비된 가중치의 파일 해시와 manifest 검증은 유지하며, 준비되지 않은 모델은 가져오기·다운로드 방법을 표시합니다.
- [C++17 세그멘테이션 평가 예제](../example/cpp/segmentation_eval/README.ko.md)를 추가했습니다. 명령행 인자 없이 EXE 옆 JSON으로 실행하고, 클래스별 IoU/Dice·혼동행렬·원본 크기 마스크·전경 표시·추론 시간 p50/p95를 저장합니다.
- 정답 마스크 누락, 잘못된 크기·클래스 번호, 평가 가능한 픽셀 없음은 실패로 처리합니다. 평가 성공을 모델 정확도 합격으로 표시하지 않습니다.

U-Net Lite/ResNet18/MobileNetV2와 DeepLab의 실제 모델 구조를 ONNX로 내보내 C++에서 평가하는 회귀 테스트를 추가했습니다. Windows/VS2017 실행은 Windows CI에서 별도로 검사하며, 로컬 macOS 검사 결과를 Windows 성능으로 간주하지 않습니다.
