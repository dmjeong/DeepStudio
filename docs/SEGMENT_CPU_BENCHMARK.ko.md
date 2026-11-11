# 분할 모델 CPU 속도 측정

학습 전에 **어떤 모델 · 입력 배율 · 정밀도 · 런타임이면 목표 시간 안에 드는지** 설비 PC에서 직접 잽니다.
가중치가 무작위라 **속도만** 나옵니다. 정확도는 학습한 모델로 따로 확인합니다.

```
① 준비 (torch 필요, 한 번)                       ② 측정 (numpy + onnxruntime, openvino는 있으면)
모델 × 출력 방식 ─▶ FP32.onnx ─▶ INT8.onnx  ──▶  배율 · 슬롯 크기별로 같은 파일을 크기만 바꿔 실행 ─▶ 표 + JSON
```

## 실행

```bat
python tools\segment_cpu_benchmark.py --region 3100x1500 --threads 8 --output result.json
```

| 옵션 | 뜻 |
| --- | --- |
| `--region 가로x세로` | 판정 영역 크기 |
| `--threads N` | 스레드 수. 물리 코어 수를 줍니다 |
| `--scales 1.0 0.5` | 입력 배율. 줄인 뒤 32의 배수로 채웁니다 (C++ 계약과 같은 크기) |
| `--models …` | 잴 모델. 기본은 U-Net 6종(ResNet18 · MobileNetV2 · Lite · MobileNetV4 Small · ShuffleNetV2 · EfficientNet-B1)과 BiSeNetV2 · STDC1 |
| `--heads full half` | `full` = 원본 해상도 출력(지금 내보내기), `half` = 1/2 해상도 출력 (U-Net 계열만) |
| `--slot-size 가로x세로 --slots 96` | 고정 슬롯만 잘라 넣을 때의 시간도 잽니다 |
| `--runtimes onnxruntime openvino` | 설치되지 않은 런타임은 건너뜁니다 (`pip install openvino`) |
| `--no-int8` | INT8을 만들지도 재지도 않습니다 |
| `--max-isa AVX2` | OpenVINO 명령어 세트 상한. 다른 PC 수준을 흉내 낼 때만 씁니다 |

설비 PC에 torch가 없으면 개발 PC에서 ONNX만 만들어 폴더째 복사합니다.

```bat
python tools\segment_cpu_benchmark.py --export-only --work segment_cpu_benchmark
```

## 읽는 법

- **모델 시간만** 잽니다. 전처리(축소 · 정규화)와 후처리(argmax · 결함 판정)는 들어 있지 않습니다.
- `half` 는 C++ SDK가 아직 지원하지 않는 출력 방식입니다. 마지막 확대를 빼면 얼마나 줄어드는지 보는 용도입니다.
- INT8 배속은 CPU 세대에 따라 다릅니다. VNNI가 있는 12세대 이상 Core에서 더 큽니다.
- 슬롯 방식은 조각으로 나누는 것 자체보다 **버리는 픽셀**만큼 빨라집니다.

## 참고 측정

판정 영역 3100×1500, 4스레드, OpenVINO 2026.4, 모델 시간 중앙값(ms).
서버 CPU(Xeon 2.1GHz)에서 명령어 세트를 AVX2로 제한한 값이고, 괄호는 AVX2-VNNI로 제한한 값입니다.
8개 모델을 같은 기계에서 한 번에 잰 값이라 모델끼리 비교할 수 있습니다. 설비 PC의 실제 값은 위 명령으로 직접 재야 합니다.

`full` 출력 (원본 해상도 로짓 — 지금 C++ SDK가 쓰는 모양):

| 모델 | FP32 1.0배 | INT8 1.0배 | FP32 0.5배 | INT8 0.5배 | INT8 슬롯 192×160 ×96 |
| --- | --- | --- | --- | --- | --- |
| U-Net ResNet18 | 3060 | 2008 (839) | 872 | 451 (224) | 921 (472) |
| U-Net EfficientNet-B1 | 967 | 1118 (1019) | 211 | 346 (246) | 643 (589) |
| BiSeNetV2 | 1099 | 815 (369) | 269 | 199 (109) | 483 (244) |
| STDC1 | 1071 | 662 (380) | 296 | 170 (80) | 348 (207) |
| U-Net MobileNetV2 | 458 | 285 (238) | 98 | 83 (56) | 153 (147) |
| U-Net ShuffleNetV2 | 343 | 249 (237) | 96 | 63 (46) | 135 (109) |
| U-Net MobileNetV4 Small | 325 | 237 (147) | 87 | 53 (43) | 129 (101) |
| U-Net Lite | 296 | 181 (147) | 84 | 42 (29) | 112 (69) |

`half` 출력 (1/2 해상도 로짓, U-Net 계열만):

| 모델 | FP32 1.0배 | INT8 1.0배 | FP32 0.5배 | INT8 0.5배 |
| --- | --- | --- | --- | --- |
| U-Net ResNet18 | 3302 | 1637 (805) | 781 | 390 (217) |
| U-Net EfficientNet-B1 | 850 | 1087 (1041) | 199 | 262 (300) |
| U-Net MobileNetV2 | 423 | 244 (224) | 103 | 72 (46) |
| U-Net ShuffleNetV2 | 295 | 185 (163) | 81 | 45 (46) |
| U-Net MobileNetV4 Small | 280 | 165 (136) | 73 | 45 (35) |
| U-Net Lite | 235 | 121 (90) | 55 | 39 (20) |

읽을 점:

- **가벼운 U-Net 네 종(Lite · MobileNetV4 Small · ShuffleNetV2 · MobileNetV2)이 가장 빠릅니다.** ResNet18 U-Net보다 FP32 기준 7~10배 빠릅니다.
  이 중 사전학습 백본이 있는 것은 MobileNetV4 Small · ShuffleNetV2 · MobileNetV2이고, 그 가운데 MobileNetV4 Small이 가장 빠릅니다.
- **BiSeNetV2 · STDC1은 CPU에서 가벼운 U-Net보다 느립니다.** FP32 1.0배가 약 1.1초로 MobileNetV4 Small U-Net의 3배입니다.
  1/8 해상도에서 판정하는 구조라 작은 결함에도 불리합니다. VNNI가 있는 CPU의 INT8에서는 격차가 줄어듭니다(약 370ms).
- **EfficientNet-B1 U-Net은 INT8로 바꾸면 오히려 느려집니다** (FP32 967 → INT8 1118). CPU에서 INT8 배포를 목표로 하면 맞지 않습니다.
- `full` 과 `half` 의 차이는 가벼운 모델일수록 큽니다 (Lite FP32 1.0배 296 → 235). 측정마다 수 % 흔들리므로 ResNet18처럼 차이가 작은 경우는 같다고 봅니다.
- ONNX Runtime은 명령어 세트를 제한할 수 없어 이 표에 넣지 않았습니다. C++ SDK가 쓰는 런타임이므로 설비 PC에서 직접 재는 값이 기준입니다.
