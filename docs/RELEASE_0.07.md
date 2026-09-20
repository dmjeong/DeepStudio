# 0.07 — 모델별 사전학습 연결 수정

카탈로그에 다른 모델을 추가하면서 학습 모드 목록과 가중치 로딩이 EfficientNet 전용으로
남아 있던 오류를 수정했다. 기존 모델의 누락된 연결을 복구하는 버그 수정(+0.01)이며,
새 모델군은 추가하지 않는다. 기존 학습 체크포인트 구조와 ONNX 검증 허용 오차를 유지한다.

| 선택 모델 | ImageNet 초기화 |
|---|---|
| ResNet18 | ResNet18_Weights.IMAGENET1K_V1 |
| ResNet50 | ResNet50_Weights.IMAGENET1K_V2 |
| ConvNeXt V1 Tiny | ConvNeXt_Tiny_Weights.IMAGENET1K_V1 |
| DeepLab V3+ ResNet34 | ResNet34_Weights.IMAGENET1K_V1 백본, 분할 헤드 새 학습 |
| U-Net ResNet18 | ResNet18_Weights.IMAGENET1K_V1 백본, 분할 헤드 새 학습 |

`ImageNet 가중치로 시작`은 최초 1회 공식 파일을 받아 SHA-256을 확인한다. 이후에는 캐시를
사용한다. `로컬 가중치로 시작`에서 같은 모델의 torchvision `.pth` 또는 Studio 체크포인트를
선택할 수도 있다. 다운로드/구조 검증 실패를 무작위 가중치로 대체하지 않는다.
체크포인트 복원과 ONNX export는 가중치를 다시 다운로드하지 않는다.
가중치는 Git 저장소나 설치 프로그램에 새로 포함하지 않는다.

EfficientNet과 PatchCore의 기존 ImageNet 경로는 유지한다. SAM2, Re-DETR v4와 LibreYOLO는
구현·가중치가 들어 있는 `.dvmodel` 팩이 필요하다. 목록에 표시된 것만으로 설치된 모델이
되는 것은 아니며, 이 모델들에 EfficientNet 가중치 선택 항목을 표시하지 않는다.

[example/](../example/README.md)에 C++17과 C#의 JSON/ONNX 로드, 이미지 추론, 시간 측정,
합성 모델 자동 테스트를 추가했다. 내보내기에서 검증한 Runtime 최적화·스레드 설정을 사용한다.
합성 테스트 모델은 EfficientNet 성능 측정용이 아니다.

검증: 공식 가중치를 사용한 5개 어댑터의 역전파·백본 갱신, 체크포인트 복원, ONNX 수치 검증을
로컬 CPU에서 확인했다. 분류는 224, 분할은 512 입력으로 export했다.
C++17 및 C# 예제의 Release 빌드와 실제 ONNX 자동 테스트도 로컬에서 확인했다.
Windows/i7 속도는 이번 로컬 검사로 보장하지 않는다.

설정 근거: [torchvision 공식 가중치 문서](https://docs.pytorch.org/vision/stable/models.html).
