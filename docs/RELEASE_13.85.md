# 13.85 — 새 세그멘테이션 모델 5종 사전학습 완성

13.84에서 빠진 BiSeNetV2·STDC1 사전학습 연결을 수정한 버그 수정 릴리스(+0.01).
U-Net MobileNetV4 Conv-S / ShuffleNetV2 x1.0 / EfficientNet-B1 / BiSeNetV2 / STDC1 모두
학습 화면에서 **ImageNet 가중치로 시작**, 로컬 가중치, 처음부터 학습을 선택할 수 있다.

## 준비와 사용

1. 최신 소스를 받은 후 `gui\update_segmentation_models.bat`을 한 번 실행한다.
2. 앱을 다시 실행하고 Training → Segmentation → 모델을 선택한다.
3. 학습 방식에서 **ImageNet 가중치로 시작**을 선택한다.

배치파일은 기존 PyTorch/CUDA를 교체하지 않고 필요한 timm과 5종 백본을 준비한다.
전체 설치용 `build.bat`에도 두 백본이 포함된다. 이전에 받은 3종은 해시 확인 후 재사용한다.
회사망 때문에 다운로드가 실패하면 가중치 관리 화면의 다운로드 주소로 받은 파일을
가져올 수 있다. STDC의 `.tar`도 선택할 수 있고, manifest는 직접 만들 필요가 없다.

| 모델 | 사전학습 파일/범위 |
| --- | --- |
| U-Net MobileNetV4 Conv-S | timm ImageNet 백본 |
| U-Net ShuffleNetV2 x1.0 | torchvision ImageNet 백본 |
| U-Net EfficientNet-B1 | timm RA4 ImageNet 백본 |
| BiSeNetV2 | CoinCheung `backbone_v2.pth`, detail/segment/bga 백본 |
| STDC1 | 저자 공개 `STDCNet813M_73.91.tar`, STDCNet813 백본 |

분할 헤드는 사용자 클래스에 맞춰 새로 학습한다. 공식 파일의 SHA256과 모든 백본 키/크기를
확인한다. 잘못된 파일을 무작위 초기화로 대신하지 않는다. STDC의 features와 x2/x4/x8/x16/x32는
같은 레이어이므로 모든 별칭에도 동일한 사전학습 값을 넣는다.

백본과 새 헤드의 학습률을 구분한다. 백본 고정 시 BatchNorm 통계도 고정한다.
13.84의 단일 optimizer 그룹으로 저장한 BiSeNetV2/STDC1 재개는 기존 그룹 순서를 유지한다.
기존 .pt 복원에는 외부 사전학습 파일이 필요하지 않다. 추론 구조·SDK 계약은 그대로다.

## 검증

5종 모두 실제 공식 파일로 합성 패치 학습 → 저장/복원 → FP32 ONNX 출력 비교 → INT8 실행,
C++ 결과 비교 및 평문/암호화 동일성을 확인했다. 640×640 학습/내보내기도 확인했다.
1채널 stem 변환, 누락/부분 백본 거부, 백본 고정과 기존 체크포인트 재개도 검사했다.
관련 회귀 검사는 239개 통과, 기존 MobileNetV2 별도 가중치 검사 1개 건너뜀으로 종료했다.
Windows/CUDA 실행과 사용자 검사 데이터의 정확도는 이 macOS 검증으로 확정하지 않는다.

가중치 출처·학습 데이터 근거·라이선스 고지: [모델 출처](../licenses/segmentation-candidates.md).
실행 결과: [검증 보고서](03-analysis/segmentation-pretrained-completion.md).
