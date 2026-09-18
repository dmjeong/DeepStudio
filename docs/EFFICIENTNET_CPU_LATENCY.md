# EfficientNet B0 CPU 8 ms 측정 및 배포

목표는 **배치 1, 모델 입력 224×224, 전처리 + 추론 + 후처리 합계 8 ms 이하**다.
최적 런타임과 스레드 수는 실제 i7 PC에서 선택한다. CPU 세대, 전력 설정, 운영체제와
다른 프로그램 부하에 따라 결과가 달라진다. M4 측정으로 i7 성능을 보장할 수 없다.

Studio 추론 버튼에서 보이는 이미지별 ms를 확인하려면
[GUI ONNX 적용 및 실제 표시 시간](EFFICIENTNET_GUI_ONNX.md)을 먼저 참고한다.
아래 독립 Python/C++ 측정과 GUI의 파일 읽기 포함 시간은 범위가 다르다.

## 독립 실행 환경

`DeepVisionStudio` 폴더에서 Python 3.11 가상환경을 사용한다. Intel Windows/Linux:

```bash
python -m venv .venv-cpu
# Windows: .venv-cpu\Scripts\activate
# Linux/macOS: source .venv-cpu/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r python/requirements-efficientnet-cpu.txt
```

macOS에서는 PyTorch 설치 명령에서 `--index-url`을 제외한다. GUI의 OpenCV 환경과
독립적으로 설치한다. PySide6, 폐기된 외부 엔진, torchvision은 이 벤치마크 실행에 필요 없다.
선택적 OpenVINO를 사용하지 않으면 requirements의 해당 줄과 CLI `--openvino`를 제외한다.

## 모델 파일 없이 B0 구조의 실제 연산 시간 측정

```bash
python tools/efficientnet_benchmark.py --architecture efficientnet_b0 --input-size 224 --in-channels 1 --num-classes 2 --openvino --threads 1 2 4 8 --warmup 30 --runs 1000 --target-ms 8 --target-scope pipeline --output cpu-b0-224
```

기존 프로젝트 기본인 1채널 흑백/2클래스를 명시했다. RGB 모델은 `--in-channels 3`을
사용한다. 첫 Conv 외의 MBConv/SE/백본 연산은 그대로이므로 흑백이 RGB보다 3배 빠르다고
가정하면 안 된다. 이 모드는 고정 시드 임의 가중치와 합성 uint8 이미지를 사용한다.
실제 B0 전체 연산을 실행하지만 학습된 모델의 정확도 검증이나 제품 성능 인증은 아니다.

## 사용 중인 모델과 검증 이미지로 재측정

```bash
python tools/efficientnet_benchmark.py --checkpoint best.pt --input-size 224 --validation data/val --openvino --threads 1 2 4 8 --warmup 30 --runs 1000 --target-ms 8 --target-scope pipeline --require-target --output cpu-b0-production
```

`data/val/OK/*.png`, `data/val/NG/*.png`처럼 체크포인트 클래스명과 동일한 하위 폴더를
사용한다. 체크포인트에 저장된 크기/채널/정규화를 우선하고, 224가 아닌 모델에
`--input-size 224`를 지정하면 오류를 낸다. 변경한 해상도에서의 정확도를 암묵적으로
가정하지 않는다. 정확도 검증은 모든 이미지에서 수행하고 **시간은 첫 번째 검증 이미지를
반복**해 측정한다. 원본 이미지 크기와 이름도 보고서에 남긴다. 여러 원본 해상도는
별도 실행으로 비교한다.

## 보고서 해석

- `report.json`: CPU 이름, OS/패키지 버전, 모델 출처/SHA256, 입력 크기/채널, 스레드
  설정, 원시 측정값, 평균/p50/p95/p99/최댓값, 8 ms 이내 비율, 목표 판정.
- `model`: 준비된 NCHW 배열에서 logits까지. 런타임 어댑터와 출력 검사 비용 포함.
- `pipeline`: 메모리상의 uint8 이미지 → crop/resize/정규화 → 모델 → softmax/클래스/확률
  반환까지 외부 타이머로 측정. 파일 디코딩, 카메라 수신, UI, Grad-CAM은 제외.
- `target.met_on_this_machine`: 선택 범위의 **p95 ≤ 8 ms**. p95는 측정의 95%가 이 값
  이내라는 뜻이며 모든 요청의 마감 보장이 아니다. 모든 측정 요청이 목표를 만족했는지는
  `all_measured_requests_within_target`, `max_ms`, `within_target_fraction`을 함께 확인한다.
- `--require-target`: 결과 파일을 저장한 뒤 p95 목표 미달이면 종료 코드 2.
  `model.onnx`와 `model.json`도 생성한다.

각 설정에서 워밍업 후 연속 동기 요청을 실행한다. 모델 로딩/컴파일/내보내기 시간은 제외한다.
공정한 비교를 위해 한 번에 하나의 ONNX/OpenVINO 세션만 생성한다. 여러 세션의 대기
스레드가 CPU를 점유하는 상태로 PyTorch를 측정하지 않는다. 최고 성능값을 얻으려고
다른 작업을 반복 수행하는 대신, 먼저 유휴 상태에서 측정하고 실제 생산 부하에서도
동일 명령을 반복한다. 스레드는 많다고 항상 빠르지 않다.

## OpenVINO 추론 적용

동일한 FP32 ONNX를 읽어 CPU에 한 번 컴파일한다. `LATENCY`, 1 stream, 명시적 f32,
지정된 스레드 수를 사용한다. 프레임마다 모델/세션을 생성하지 않는다.

```python
import sys
sys.path.insert(0, "python")
from openvino_classifier import OpenVINOClassifier

# 실제 PC 보고서에서 선택한 스레드 수를 적용한다.
model = OpenVINOClassifier("cpu-b0-production/model.json", num_threads=4)
result = model.predict_gray(gray_frame)  # uint8 [H,W]
print(result["class_name"], result["confidence"], result["total_ms"])
# RGB/RGBA는 model.predict_rgb(rgb_frame)
```

하나의 인스턴스는 순차 호출용이다. 여러 호출자가 동시에 사용할 때는 외부 lock이나
별도 인스턴스를 사용하고, 동시 부하에서 시간을 다시 측정한다. 이 변경은 독립 배포 API와
벤치마크를 추가하며 기존 GUI 추론 엔진 선택을 자동으로 바꾸지는 않는다.

PyTorch의 Conv–BN 융합과 channels-last도 비교한다. 최적화는 원본 평가 복사본에만
적용한다. 모든 후보는 원본 PyTorch FP32 logits 및 top1과 비교한 뒤 시간을 기록한다.
실제 검증 폴더를 주면 설정별 정확도, macro F1, 클래스별 recall, 혼동행렬도 기록한다.
FP32로 목표에 도달하지 못하면 그 결과를 바탕으로 INT8 등을 별도 평가한다.

## 근거와 코드/라이브러리 라이선스

- [OpenVINO latency guide](https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference/optimizing-latency.html)
- [ONNX Runtime threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
- [코드/실행 라이브러리 라이선스 검토](EFFICIENTNET_CPU_LICENSES.md)

가중치 권한 검토는 사용자 요청에 따라 이번 범위에서 제외했다.
