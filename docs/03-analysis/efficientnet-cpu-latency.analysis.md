# EfficientNet CPU latency implementation check

Date: 2026-09-17. Scope: B0 224×224, in-memory pre/model/postprocessing.

| Design item | Evidence | Result |
|---|---|---|
| Shared unchanged preprocessing | Existing OpenCV pipeline and B0/B1 gray/RGB deployment tests | Pass |
| Persistent OpenVINO FP32, CPU, latency, one stream | Real OpenVINO inference and effective-property tests | Pass |
| Manifest validation and owned outputs | Mismatched names/shapes/classes, invalid inputs, output reuse tests | Pass |
| Baseline/fusion/layout/ORT/OV comparison | CLI smoke and 224×224 local run | Pass |
| p95-based target, raw samples and scope | Deterministic timing/verdict tests; CLI failure exit after report | Pass |
| Checkpoint authority and explicit random mode | Conflicting size test, synthetic provenance in report | Pass |
| Validation metrics and backend parity | Subprocess checkpoint/class-folder validation test | Pass |
| Dependency licensing scope and notices | Existing BSD notice retained; standalone requirements and license document | Pass |

Implementation match: 8/8 items (100%). This is design coverage, not a measured i7
speed claim or a legal certification. The target i7 and final workload remain untested.

Regression command:
```
python -m pytest tests/test_efficientnet_cpu_benchmark.py tests/test_openvino_classifier.py tests/test_efficientnet_deployment.py tests/test_efficientnet_channels.py tests/test_opencv_pipeline.py tests/test_export_contracts.py -q
```
Result: 53 passed, 2 skipped. The skips require legacy 폐기된 외부 엔진, deliberately absent
from this environment. After adding checkpoint validation to the CLI integration test,
the 13 benchmark tests were rerun and passed. No external pretrained weights downloaded.

The regression exposed a pre-existing softmax-rounding tie: tiny unequal logits can
produce equal FP32 probabilities and change argmax. Shared postprocessing now selects
the class from logits, and includes probability-list conversion in its postprocessing
timer. A direct regression test covers the rounding case. Export remains numerically
checked against unfused PyTorch.

User clarifications supersede the initial B0/B1 measurement plan: B0 only is timed,
and weight/data licensing is excluded. The report identifies synthetic timing inputs
solely to keep the performance evidence interpretable.

## C++17 scope extension

User requested an in-process C++17 function/component, not a background service.
Added and checked: optional OpenVINO C++ loading; unchanged ORT defaults; bounded FIFO
worker and owned frames; readiness/result exception delivery; drain/join lifecycle;
headless dependency setup; actual exported B0 parity; sync/background wall benchmarks.
Eight additional design items are implemented (16/16 implementation coverage).
CTest passes with OpenVINO enabled and disabled. See EFFICIENTNET_CPP17_RESULTS.md
for measured latency and the explicit outstanding i7/hard-deadline limitations.

## ONNX 후속 검증 (2026-09-18)

입출력 텐서/전처리 버퍼 재사용과 identity resize 생략은 색상·crop·strided ROI·
소유권·재초기화 오류 검증을 통과했다. 실제 gray/RGB B0의 24개 Python/C++ 조합도 통과했다.
동기 p95는 3회 중앙값 5.423→5.461 ms, 백그라운드는 5.857→5.460 ms이나 실행 간
변동과 최대 35.129 ms 표본이 있어 전체 지연 개선/8 ms 마감 달성을 주장하지 않는다.
전처리 평균 감소는 약 0.005 ms다. 선택된 SDK 옵션의 동작을 확인하는 Windows용
설정 비교 도구를 제공하며 실제 대상 Windows/i7 결과는 여전히 남아 있다.

## 실제 Studio 버튼 수정 검증 (2026-09-18)

사용자가 명확히 한 20 ms는 배치 표의 inference_sec였다. 이전 작업에서 이 GUI
호출 경로를 연결하지 않은 누락을 확인했다. InferenceWorker와 DesktopInferenceJob의
실제 CPU EfficientNet 판정을 ONNX Runtime에 연결했다. 같은 전처리/FP32/클래스 순서와
체크한 Grad-CAM을 유지하며 파일 읽기까지 시간에 포함한다.

실제 화면 확인에서 기존 ResultCard가 숨겨진 호환용 위젯임을 확인했다.
실제 보이는 선택 이미지 패널과 표의 시간 셀 툴팁에 런타임/단계 시간을 연결하고,
통합 테스트에서 QLabel.isVisible까지 검증했다.

검증: 85 passed, 2 skipped, 프로세스 exit 0. 건너뛴 항목은 기존 폐기된 외부 엔진 의존 로더다.
새 테스트는 실제 로컬 B0를 사용하며 forward를 모킹해 만든 가짜 성능 테스트가 아니다.
준비 후 PyTorch forward를 실패시키는 경우는 ONNX 경로 사용을 입증하는 기능 검증이다.
Qt 테스트에 하나의 QApplication을 유지해 통합 실행의 종료 시점 자원 해제 충돌도 해결했다.

```bash
python -m pytest tests/test_gui_efficientnet_onnx.py tests/test_inference_contracts.py tests/test_inference_timing.py tests/test_async_inference.py tests/test_inference_region.py tests/test_inference_cache.py tests/test_inference_review.py tests/test_efficientnet_channels.py tests/test_model_selection.py tests/test_inference_loading.py tests/test_webapp.py::WebAppTests::test_cached_gradcam_range_and_toggle_without_model tests/test_webapp.py::WebAppTests::test_disk_cache_failure_preserves_computed_prediction -q
```

원본 224/모델 224의 앱 엔진 반복 측정은 M4에서 ONNX p50 6.037, p95 6.261,
최대 8.596 ms였다. 첫 실제 버튼 표본은 파일 읽기 4.476 ms를 포함해 11.527 ms였다.
원본 2048의 같은 엔진은 p95 34.380 ms로 크기/파일 읽기 영향도 확인했다.
따라서 모든 이미지 8 ms 또는 Windows i7에서의 달성을 주장하지 않는다.
[상세 결과와 사용법](../EFFICIENTNET_GUI_ONNX.md).

## 최대 오차 2.8339844 조사와 검증 경로 보완

사용자가 Windows에서 보고한 최대 절대 오차만으로 원인을 결정하지 않았다.
공식 학습 B0의 RGB/gray 적응 모델은 로컬 M4에서 seeded/zero 검증을 통과했으며
최대 오차 약 1.50e-5, top1 일치였다. 사용자 가중치의 오류는 재현하지 못했다.

기존 오류를 큰 오차로 차단하는 기준은 유지하되, CPU FP32 분류 logits의 연산 재배치에서
생기는 작은 상대 오차에는 `atol=1e-4, rtol=5e-4` parity profile을 적용한다. ALL → BASIC →
DISABLED 재시도와 선택 설정 기록은 계속 유지한다.
실패 세션을 해제한 후 다음 세션을 만들고, 전부 실패하면 실행을 중단한다.
기준 PyTorch 출력부터 NaN/무한대인 경우는 ONNX를 실행하기 전에 명시적으로 구분한다.
오류 메시지는 최대 절대 오차 외에 실패 원소 값/허용 오차 배수/출력 범위/버전을 담는다.

0.00052 최대 오차 회귀: `1.0 → 1.00052` 분류 logits는 새 profile을 통과하고,
`0.0 → 0.00052` near-zero 오차와 `1.0 → 1.01` 상대 오차는 계속 실패하도록 단위 검사를 추가했다.

검증 명령:
```bash
python -m pytest tests/test_gui_efficientnet_onnx.py tests/test_export_contracts.py tests/test_inference_timing.py tests/test_efficientnet_deployment.py -q
```

실제 ORT에 2.8339844 오차를 주입해 BASIC/DISABLED 선택 및 전체 실패를 검증한다.
결과: 44 passed, 프로세스 exit 0.
이 고장 주입은 사용자 체크포인트 오류를 재현했다는 의미가 아니다.
사용자 환경에서의 문제 해결 여부와 선택된 수준에서의 속도는 미확인이다.

## 자동 ONNX 실패가 전체 추론을 막은 회귀 복구

이전 변경에서 ONNX 검증을 자동 모드의 필수 조건으로 만든 것이 추론 작업 중단의
직접 원인이었다. auto 모드의 import/export/session/validation/warmup 실패는 이제
기존 PyTorch 실행으로 전환하고, 실패 세션은 사용하지 않는다. 원래 오류와 실제 런타임을
화면·로컬 작업 로그·결과 JSON·캐시에 보존한다. 명시적 onnx/내보내기는 엄격히 검증한다.

- 로컬 생성 B0로 변환 실패/라이브러리 오류와 ALL/BASIC/DISABLED 수치 불일치를
  주입했다. 기존 PyTorch 확률과 정확히 같은 결과, Grad-CAM, 배치당 1회 준비를 확인했다.
- 실제 Qt 버튼에서 2장 배치의 준비 실패 후 판정·선택 결과 표시·캐시 탐색을 확인했다.
- 별도 실제 작업 프로세스의 ONNX import를 실패시킨 버튼 실행도 2/2 완료, 오류 0이었다.
  저장 캐시 복원과 `PyTorch cpu [ONNX 준비 실패 → PyTorch]` 표시를 직접 확인했고 exit 0이었다.
- CPU 측정 CLI의 auto 실패도 PyTorch로 완료하며 stdout/JSON에 actual_runtime과
  runtime_warning을 남긴다. 이 실행은 복구 확인용이며 새 성능 벤치마크가 아니다.
- 기존 추론·내보내기·시간 측정·캐시·리뷰·로딩·웹 저장 회귀는 90 passed, 2 skipped,
  exit 0이었다. 건너뛴 2개는 기존 폐기된 외부 엔진 의존 테스트다.
- ONNX/자동 복구/실제 버튼/워커 저장 테스트는 14 passed, exit 0이었다. 두 검증 실행의
  합계는 104 passed, 2 skipped다. 오류 주입 예외의 traceback이 완료된 Qt 워커를 붙잡아
  초기 통합 테스트의 종료에서 충돌했다. 테스트 정리 시 traceback/순환 참조를 해제하고
  QApplication이 살아 있을 때 지연 삭제를 처리한 후 14개 전체의 정상 종료를 확인했다.

```bash
python -m pytest tests/test_gui_efficientnet_onnx.py -q
python -m pytest tests/test_export_contracts.py tests/test_inference_contracts.py tests/test_inference_timing.py tests/test_async_inference.py tests/test_inference_region.py tests/test_inference_cache.py tests/test_inference_review.py tests/test_efficientnet_channels.py tests/test_model_selection.py tests/test_inference_loading.py tests/test_webapp.py::WebAppTests::test_cached_gradcam_range_and_toggle_without_model tests/test_webapp.py::WebAppTests::test_disk_cache_failure_preserves_computed_prediction -q
```

보안 모델이나 사용자 이미지를 업로드하지 않았다. 사용자 ONNX 수치 불일치의 근본 원인과
Windows/i7 8 ms 달성 여부는 미확인이다. 이번 변경은 기존 실행 기능의 복구다.
