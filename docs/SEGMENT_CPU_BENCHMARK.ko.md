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
| `--heads full half` | `full` = 원본 해상도 출력(지금 내보내기), `half` = 1/2 해상도 출력 |
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

판정 영역 3100×1500, 4스레드, OpenVINO 2026.4, `half` 출력, 모델 시간 중앙값(ms).
서버 CPU(Xeon 2.3GHz)에서 명령어 세트를 AVX2로 제한한 값이고, 괄호는 AVX2-VNNI로 제한한 값입니다.
설비 PC의 실제 값은 위 명령으로 직접 재야 합니다.

| 모델 | FP32 1.0배 | INT8 1.0배 | FP32 0.5배 | INT8 0.5배 | INT8 슬롯 192×160 ×96 |
| --- | --- | --- | --- | --- | --- |
| U-Net ResNet18 | 2351 | 1190 (640) | 595 | 339 (173) | 693 (371) |
| U-Net MobileNetV2 | 390 | 221 (177) | 81 | 46 (37) | 123 (98) |
| U-Net Lite | 193 | 110 (70) | 47 | 27 (21) | 68 (51) |

- `full` 출력은 같은 조건에서 `half` 보다 대체로 느립니다 (예: Lite FP32 1.0배 244, MobileNetV2 INT8 1.0배 255).
- ONNX Runtime은 명령어 세트를 제한할 수 없어 이 표에 넣지 않았습니다.
  같은 서버에서 제한 없이 잰 ResNet18 U-Net `full` 1.0배는 FP32 1531, INT8 717 이었습니다.
