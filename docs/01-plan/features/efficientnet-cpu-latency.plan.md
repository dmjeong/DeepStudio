# EfficientNet CPU latency plan

Date: 2026-09-17

User clarification: target EfficientNet **B0**, **224×224**, preprocessing + inference
+ postprocessing. Weights licensing review is excluded at the user's request.
Code/runtime licensing and required notices are included. B1 remains supported for
backward compatibility but is not the performance target.

## C++ extension requested before publication

Verify the actual exported ONNX/JSON in a compiled C++ Release executable, compare
CPU timings, document loading/dependencies, and provide a bounded background worker
with safe ownership, exception delivery and shutdown. Add optional OpenVINO to the
existing C++ classification engine, retaining ONNX Runtime as its default. Build a
headless target so Qt/FFmpeg are unnecessary. Validate and push the complete result.

## Purpose and scope
Measure batch-one EfficientNet B0/B1 at 224×224 and provide a reproducible path
to an 8 ms CPU latency target. Preserve checkpoint preprocessing and FP32 outputs.
Compare PyTorch, fused channels-last PyTorch, ONNX Runtime and optional OpenVINO.
Record hardware, versions, weight provenance, warmup, p50/p95/p99, and whether
pre/postprocessing is included. Add a checkpoint-free architecture smoke benchmark,
explicitly labelled as random weights, without presenting it as accuracy evidence.

## Success criteria
- Existing trained checkpoints remain supported without changing input contracts.
- Intel deployment can use a persistent, single-request OpenVINO CPU classifier.
- Backend parity is checked before recommending a configuration.
- The target verdict uses p95; no i7 claim is made from Apple M4 measurements.
- Regression tests and an actual 224×224 local run pass; publish code to deepstudio3.

## Risks and schedule
Plan, design, implement, test, then report in this task. The target i7 model, OS and
trained checkpoint have not yet been supplied. Actual i7 acceptance therefore remains
pending. FP32 avoids an unmeasured quantization accuracy tradeoff. No INT8 release,
GUI backend migration, camera IO or Grad-CAM latency is included in this change.

## 2026-09-18 ONNX Runtime 후속 최적화

사용자는 ONNX Runtime을 선택했다. FP32와 기존 전처리/출력 계약을 유지하면서 반복 입력
버퍼/텐서 생성 비용을 줄이고, 대상 CPU에서 스레드/작업 분할 옵션을 비교할 수 있게 한다.
기존 실행 파일과 변경 실행 파일을 같은 조건으로 번갈아 측정한다. i7 모델/OS와 실제
검증 데이터가 아직 없으므로 i7 성능 및 INT8 정확도를 달성했다고 표시하지 않는다.

## 실제 프로그램 추론 버튼의 20 ms 보고

사용자는 C++ 벤치마크가 아니라 GUI에서 표시하는 이미지별 시간을 보고 있었다.
기존 버튼은 PyTorch 경로여서 ONNX 최적화와 연결되지 않았다. CPU EfficientNet 분류의
실제 버튼/프로세스 워커를 ONNX로 연결하고 동일한 입력/출력/Grad-CAM을 검증한다.

## ONNX 검증 실패 후속 조사 (2026-09-18)

사용자 Windows 실행에서 최대 출력 오차 2.8339844로 준비 검증이 중단됐다.
같은 가중치가 없으므로 원인은 아직 확정하지 않는다. 허용 오차/FP32/입력 크기를
유지하고, 최적화에 의한 차이인지 구분하도록 검증을 통과하는 ONNX 설정만 선택한다.
모든 설정이 실패하면 출력 크기와 실패 원소 값을 포함한 오류를 남긴다.

## ONNX 실패 시 기존 추론 복구 (2026-09-18)

자동 ONNX 준비 실패가 전체 추론 작업을 막은 회귀를 수정한다. 보안 모델을 업로드하지
않고도 기존 PyTorch 추론을 계속할 수 있어야 한다. 자동 모드는 실패 원인을 보존하고
실제 사용한 PyTorch 엔진을 화면에 표시한다. 명시적 ONNX 실행/내보내기는 계속 엄격히
검증한다. 고장 주입과 실제 버튼/작업 워커로 복구를 확인하고 push한다.
