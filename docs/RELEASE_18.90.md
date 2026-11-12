# Deep Vision Studio 18.90

데스크톱 분할 프로젝트의 **학습 → 학습 이미지 → 원본 크기로 학습**을 켜면 전체 이미지를 축소하지 않고 학습합니다. 중앙 크롭을 켜면 크롭 영역 전체를 사용합니다. 원본 크기 학습을 켜면 패치 학습이 해제되고 입력 크기 칸이 비활성화됩니다.

Custom CSP 분할, U-Net Lite·ResNet18·MobileNetV2·MobileNetV4 Small·ShuffleNetV2·EfficientNet-B1, DeepLabV3+ ResNet34, BiSeNetV2, STDC1에서 사용할 수 있습니다. Train과 Val의 이미지 크기가 다르면 공통 중앙 크롭을 지정하세요.

앱 추론과 FP32·INT8 ONNX 내보내기는 학습한 판정 영역 크기를 사용합니다.

기존 Python 환경을 사용하는 Windows PC에서는 저장소 폴더에서 `git pull --ff-only` 후 `python gui\main.py`로 실행합니다. EXE로 사용하려면 업데이트한 소스로 다시 빌드합니다.
