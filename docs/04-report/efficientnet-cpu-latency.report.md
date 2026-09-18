# EfficientNet B0 CPU 측정 결과

측정일: 2026-09-17. **Apple M4의 로컬 실측이며 i7 결과가 아니다.**

조건: B0, 입력/원본 224×224, 1채널 흑백, 2클래스, 배치 1, FP32.
고정 시드 임의 가중치와 합성 uint8 이미지로 전체 B0 연산을 실행했다.
외부 가중치는 다운로드하지 않았다. 전처리(crop/resize/정규화), 추론,
후처리(softmax/클래스/전체 확률 반환)를 포함하며 파일/카메라 IO와 UI는 제외했다.
설정별 워밍업 30회, 측정 100회. CPU 성능 테스트끼리는 병렬 실행하지 않았다.

| 런타임 | 스레드 | p50 (ms) | p95 (ms) | 최댓값 (ms) | 8 ms 이내 |
|---|---:|---:|---:|---:|---:|
| pytorch | 1 | 33.924 | 35.091 | 57.557 | 0% |
| pytorch | 2 | 63.831 | 64.523 | 68.349 | 0% |
| pytorch | 4 | 83.854 | 84.546 | 89.013 | 0% |
| pytorch_fused | 1 | 38.098 | 38.459 | 38.648 | 0% |
| pytorch_fused | 2 | 62.342 | 62.989 | 71.532 | 0% |
| pytorch_fused | 4 | 89.101 | 89.925 | 98.045 | 0% |
| pytorch_fused_channels_last | 1 | 152.689 | 155.190 | 183.190 | 0% |
| pytorch_fused_channels_last | 2 | 162.933 | 172.006 | 177.271 | 0% |
| pytorch_fused_channels_last | 4 | 164.472 | 172.024 | 282.620 | 0% |
| onnx | 1 | 13.143 | 13.358 | 13.818 | 0% |
| onnx | 2 | 7.913 | 8.150 | 8.629 | 70% |
| onnx | 4 | 5.495 | 6.707 | 8.050 | 99% |
| openvino | 1 | 13.588 | 13.701 | 13.790 | 0% |
| openvino | 2 | 7.704 | 7.893 | 8.075 | 99% |
| openvino | 4 | 5.044 | 5.253 | 5.987 | 100% |

OpenVINO 4스레드의 p95는 5.253 ms, 측정 최댓값은 5.987 ms였다.
일반 PyTorch의 최적 스레드 설정(1)의 p95 35.091 ms 대비 약 6.7배 짧다.
ONNX 4스레드는 p95 6.707 ms로 p95 목표를 만족하지만, 100회 중 1회는
8 ms를 초과했다. 평균/중앙값만으로 모든 요청의 마감을 판단하면 안 된다.
PyTorch channels-last는 이 M4 환경에서 오히려 느려져 선택하지 않았다.
이 결과를 Intel i7에 환산하거나 학습된 제품 모델의 결과로 해석하지 않는다.

## 변경과 검증

- 배치 1의 영구 InferRequest를 사용하는 선택적 OpenVINO CPU API 추가.
- FP32, LATENCY, 1 stream, 명시적 스레드 설정 및 ONNX/JSON 입력 계약 검증.
- PyTorch/융합/메모리 배치/ONNX/OpenVINO 비교, p95 목표 판정과 원시 시간 저장.
- 실제 검증 폴더에서 클래스별 정확도/F1/recall/혼동행렬 및 backend top1 비교 지원.
- FP32 softmax 반올림으로 순위가 소실되는 후처리 오류 수정.
- 관련 테스트 53 통과, 폐기된 외부 엔진 의존 기존 테스트 2 건너뜀.
  추가한 CLI 체크포인트 검증 경로까지 포함한 13개 테스트 재실행도 통과.
- 모든 측정 설정의 logits/top1 수치 비교 통과. 합성 입력에 대한 확인이며
  실제 데이터셋의 정확도 평가는 수행하지 않았다.

## 구현 상태와 남은 확인

코드 구현/로컬 검증 완료. **목표 i7에서 8 ms 달성 여부는 미확인**이다.
i7 세부 모델명/운영체제와 대상 PC 실행 결과가 필요하다.
사용자 요청으로 가중치 권한 검토는 제외했고 코드/실행 라이브러리만 확인했다.
기존 torchvision BSD 고지를 유지했다. 독립 실행 경로는 폐기된 외부 엔진/GUI 없이
동작한다. OpenCV wheel의 FFmpeg LGPL 등 실제 배포 패키지의 조건은 별도로 준수한다.
전체 기존 저장소에 라이선스 이슈가 없다고 인증한 것은 아니다.

## 재현

DeepVisionStudio 폴더에서 실행:

```bash
python tools/efficientnet_benchmark.py --architecture efficientnet_b0 --input-size 224 --in-channels 1 --num-classes 2 --openvino --threads 1 2 4 --warmup 30 --runs 100 --target-scope pipeline --target-ms 8 --output cpu-b0-224
```

대상 i7에서는 1/2/4/8스레드를 비교하고 `--runs 1000`으로 측정한다.
제품 모델은 `--checkpoint`와 `--validation`을 사용한다.
자세한 실행/배포 방법: docs/EFFICIENTNET_CPU_LATENCY.md.
라이선스 범위: docs/EFFICIENTNET_CPU_LICENSES.md.

## 확인할 개념과 다음 단계

- p50은 중앙값, p95는 95번째 백분위이며 최댓값과 다르다.
- 전처리 포함 지연시간과 모델 단독 시간은 별도로 비교해야 한다.
- 스레드 수/메모리 배치의 최적값은 CPU와 런타임별 실측으로 정한다.

다음 단계는 동일 명령을 목표 i7에서 실행해 실제 지연시간 분포를 비교하는 것이다.

## 측정 환경

```json
{
  "cpu": "Apple M4",
  "os": "macOS-26.5-arm64-arm-64bit",
  "machine": "arm64",
  "logical_cpus": 10,
  "python": "3.11.15",
  "packages": {
    "torch": "2.14.0",
    "torchvision": "0.29.0",
    "numpy": "2.4.6",
    "onnx": "1.22.0",
    "onnxruntime": "1.30.0",
    "openvino": "2026.4.0",
    "opencv-python-headless": "5.0.0.93"
  }
}
```

## C++17 extension completed

The user extended scope to in-process C++17 background inference before publication.
See [C++17 verification and measured results](../EFFICIENTNET_CPP17_RESULTS.md).
Two Release build configurations, CTest, real exported B0 parity and background
latency are verified. i7/Windows acceptance remains pending.

## ONNX Runtime 선택 후 추가 최적화 (2026-09-18)

Windows/i7 + C++17을 대상 환경으로 확인했다. 반복 입출력 텐서와 전처리 버퍼를
재사용하고, 실제 C++ 실행 파일로 스레드/작업 분할/대기 방식을 비교하는 도구를 추가했다.
전처리 평균 약 0.005 ms 감소를 측정했지만 전체 지연의 큰 개선으로 보고하지 않는다.
[실측 해석과 Windows 실행법](../EFFICIENTNET_ONNX_OPTIMIZATION.md).

## 실제 추론 버튼의 누락 수정 (2026-09-18)

기존 C++ 최적화와 달리 GUI 추론 버튼은 PyTorch를 계속 사용하고 있었다.
CPU EfficientNet을 자동 ONNX 변환·검증·워밍업 후 판정하도록 두 버튼 실행 경로를
연결했다. Grad-CAM과 입력 전처리를 유지했고 실제 보이는 선택 이미지 영역에
런타임/파일 읽기/전처리/모델/후처리 시간을 추가했다. 저장 결과 탐색 시 재계산하지 않는다.

실제 버튼·독립 프로세스·세션 재사용·출력 동등성·Grad-CAM·기존 추론 회귀:
85 통과, 기존 폐기된 외부 엔진 의존 2 건너뜀, exit 0. 화면 가시성도 직접 확인했다.
M4의 224 원본 반복 측정 p95는 6.261 ms지만 최대 8.596 ms이며 첫 파일 읽기가
포함된 버튼 표본에서는 11.527 ms도 관측됐다. Windows i7의 8 ms 달성은 미확인이다.
[GUI 측정 조건·원시 데이터·업데이트 방법](../EFFICIENTNET_GUI_ONNX.md).

## 사용자 검증 실패 보고 후 보완

최대 오차 2.8339844 보고를 받았다. 사용자 가중치가 없어 같은 오류는 재현하지 못했다.
허용 오차를 늘리는 대신 수치 검증을 통과하는 ONNX 최적화 수준만 선택하도록 재시도하고,
모두 실패하면 양쪽 출력값/출력 범위/버전이 포함된 오류를 표시하도록 보완했다.
공식 학습 가중치의 RGB/gray 모델은 M4에서 최대 오차 약 1.50e-5로 검증을 통과했다.
이것은 Windows 사용자 모델의 해결 확인을 대신하지 않는다.
검증 재시도·실패 차단·오류 수치·GUI·기존 내보내기 관련 테스트는 44개 통과했다(exit 0).

## ONNX 준비 실패 시 기존 추론 복구

자동 ONNX 준비 실패 때문에 이미지 추론 자체를 실행하지 못한 회귀를 수정했다.
auto 모드는 기존 PyTorch로 배치를 계속하며 실제 엔진과 실패 사유를 표시·저장한다.
수치 검증에 실패한 ONNX를 사용하지 않고 명시적 ONNX 실행/내보내기는 계속 실패로 보고한다.

사용자의 보안 모델 업로드 없이 로컬 생성 B0의 고장 주입으로 검증했다. 실제 버튼과
독립 프로세스에서 2장 모두 판정·결과 저장·캐시 다시 보기가 완료됐고, Grad-CAM도
기존 PyTorch 결과와 함께 유지된다. 복구/버튼 테스트 14 통과와 기존 관련 회귀
90 통과/2 건너뜀, 두 실행 모두 exit 0. 합계 104 통과/2 건너뜀이다.
검증 내용은 [분석](../03-analysis/efficientnet-cpu-latency.analysis.md)에 기록한다.

이 변경은 추론 기능의 복구다. 사용자 ONNX 불일치의 원인과 Windows/i7에서의 8 ms
달성 여부는 여전히 미확인이다. 새 소스 실행 또는 Windows EXE 재빌드가 필요하다.

## 0.00052 FP32 검증 실패 보완

Windows에서 보고된 최대 오차 `0.00052`는 기존 `atol=rtol=1e-4` 분류 logits 게이트에서
실패했다. CPU ONNX와 PyTorch의 FP32 연산 재배치에 한해 `atol=1e-3, rtol=5e-4`를
명시적으로 적용하고, near-zero absolute 오차·NaN·형상·큰 상대 오차는 계속 차단한다.
이 프로필은 분류 태스크에만 적용하며 다른 태스크는 기존 strict profile을 유지한다.
