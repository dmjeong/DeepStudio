# EfficientNet CPU latency design

## C++ extension

Keep existing VisionInference API and JSON schema. Add optional runtime selection
and thread override when loading JSON, supporting ONNX Runtime (default) and optional
OpenVINO FP32 CPU classification only. Reuse existing preprocessing and validate
names/shapes/dtypes before replacing a valid model. Apply the logits ranking fix to C++.

Provide a C++17 bounded FIFO worker that exclusively owns the inference engine,
clones submitted cv::Mat frames, returns std::future results/exceptions, rejects a full
queue, and drains/joins on shutdown. Queue wait is measured separately from compute;
it never silently drops accepted frames. Producers stop before destroying the worker.
Build CLI benchmark/background examples without highgui. Existing GUI demo remains
optional. CTest checks exported fixtures, lifecycle/error/ownership semantics and
backend parity; benchmark the actual exported B0 at 224 in C++ Release mode.

Confirmed target: B0, 224×224, preprocessing + inference + postprocessing.
Maintain inherited BSD notice and document runtime/packaged dependency licenses.
Weights/data licensing review is explicitly excluded by user instruction.

## Runtime
Extract existing validated image preprocessing and prediction methods into a shared
base in onnx_classifier.py. Keep the OnnxClassifier API compatible. Add
OpenVINOClassifier in a separate optional module: static batch one, CPU device,
LATENCY hint, one stream, explicit f32 precision, requested thread count, and a
persistent InferRequest. Validate model IO against the export manifest. Return owned
logits so later calls cannot overwrite earlier results. This instance is sequential;
parallel callers use separate instances or external locking.

## Benchmark
Extend tools/efficientnet_benchmark.py. A checkpoint is authoritative for dimensions,
channels and classes; an explicit architecture creates seeded random weights for
latency-only smoke testing (224, 1ch by default). PyTorch baseline/fused/channels-last,
ONNX and optional OpenVINO run sequentially with no idle ORT spinning pools.
Warmup/setup/export/validation are excluded. Compare model-only and complete
in-memory pre/model/postprocessing paths. Use the same image and weights; check
all backend logits against unfused FP32 PyTorch. Label synthetic images and weights.
Optional class-folder validation reports accuracy/F1 and class agreement for each
candidate runtime; no accuracy claim is inferred from random/synthetic runs.

Record package versions, CPU/OS, checkpoint SHA256, source dimensions, requested and
effective runtime settings, raw timings, p50/p95/p99/max and target hit rate. Recommend
by p95 for the selected scope and classify target success only on this machine.
An optional CLI failure exit supports acceptance runs; default output is diagnostic.
OpenVINO absence is explicit, never silently substituted with another backend.

## Verification
Test shared preprocessing compatibility; real B0/B1 export and runtime numerical
parity including native gray/RGB; invalid IO rejection; output ownership; argument
validation; percentile-based target decisions. Run existing EfficientNet deployment
regressions. Measure B0 at 224 locally and document that M4 is not i7.

## 2026-09-18 ONNX Runtime 후속 설계

- 분류 모델 인스턴스가 batch=1 입력 벡터와 Ort::Value를 한 번 만들고 재사용한다.
- 전처리는 그 버퍼에 직접 기록한다. resize/cvtColor scratch를 재사용하고 같은 크기는
  resize를 생략한다. 정규화 계산 순서는 유지한다. 결과 확률은 매 호출 소유권을 유지한다.
- JSON의 선택적 onnxruntime 설정에서 spinning과 dynamic_block_base를 읽고 엄격히
  검증한다. 기본값은 기존 런타임 동작을 유지하며 자동으로 전력 사용을 바꾸지 않는다.
- 실제 C++ 벤치마크를 여러 설정으로 실행해 p95/최댓값/원시 측정값을 비교한다.
- static INT8 QDQ는 실제 학습 모델과 대표 calibration/validation 데이터 확보 후 판단한다.

## GUI CPU EfficientNet ONNX 연결

- GUI InferenceEngine의 auto 모드와 데스크톱/웹 infer 작업은 CPU EfficientNet을 ONNX로 선택한다.
- 이미 로드된 모델을 배치 1 ONNX로 1회 변환하고 FP32 logits를 검증/워밍업한다.
  모델 준비는 이미지 반복 전에 수행하고 준비 시간을 기록한다. 클래스 판정은 ONNX,
  Grad-CAM은 기존 PyTorch 모델에서 별도 계산한다.
- 기존 read_image/crop/전처리와 파일 읽기를 포함한 inference_sec 의미를 유지한다.
  runtime/device/스레드 및 decode/preprocess/model/postprocess 시간을 추가 표시한다.
- GPU/Custom/PatchCore는 기존 경로 유지. 명시적 pytorch 선택은 비교용으로 유지한다.
- auto 모드에서 ONNX 준비 실패는 경고로 보존하고 기존 PyTorch로 실행한다.
  실제 사용한 엔진과 전환 사유를 표시하며 명시적 onnx 모드는 실패를 반환한다.

## ONNX 검증 실패 대응 (2026-09-18)

- 같은 로드 모델/정적 ONNX/seeded 및 zero 입력/atol=rtol=1e-4를 유지한다.
- ALL의 출력 검증에 실패할 때만 BASIC, DISABLED를 순서대로 검증한다.
  세션은 한 번에 하나만 보유하며 검증 통과한 설정만 워밍업/추론에 사용한다.
- 출력 형상/유한성/수치 비교를 생략하지 않는다. 검증에 실패한 ONNX는 사용하지 않는다.
- 최대 절대 오차뿐 아니라 실패 원소의 양쪽 값·허용 오차 배수·출력 범위를 보고한다.
- 선택한 최적화 수준과 검증 재시도 기록을 결과에 저장하고 화면에서 fallback을 식별한다.
- 실제 학습된 공개 B0로 수치 비교를 추가하고, 재시도/전체 실패는 고장 주입으로 검증한다.
  사용자의 2.8339844 오류 재현과 Windows/i7 속도 검증은 별도로 남겨 둔다.

## 자동 모드의 추론 중단 회귀 복구 (2026-09-18)

- ONNX import/export/session/validation/warmup 예외를 auto 모드에서만 처리한다.
- 실패한 세션 참조를 제거하고 기존 로드 모델/Grad-CAM을 유지한다. 한 작업에서 준비는
  한 번만 시도하고 나머지 이미지는 PyTorch로 계속 실행한다.
- 결과에 actual runtime과 runtime_warning을 저장한다. 선택 이미지/시간 셀에는 짧은
  전환 표시, 툴팁에는 원래 오류를 표시한다. 로컬 작업 로그에도 원인을 남긴다.
- PyTorch 자체의 잘못된 출력이나 실행 오류는 계속 실패로 보고한다. 수치 불일치의
  원인을 해결했다거나 Windows/i7의 8 ms를 달성했다고 주장하지 않는다.
