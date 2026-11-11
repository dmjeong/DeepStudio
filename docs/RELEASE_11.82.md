# 11.82 — C++ 분할 추론 구간별 시간

단순 기능 추가 +0.1. `SegmentResult`에서 다음 시간을 ms로 읽을 수 있다.

| 필드 | 구간 |
|---|---|
| `preprocess_ms` | ROI 선택, 색 변환, 크기 조정, 정규화, ORT 입력 준비 |
| `model_ms` | ONNX Runtime `Session::Run` |
| `validation_ms` | 출력 형식·크기 및 모든 값의 NaN/Inf 검사 |
| `decode_ms` | 픽셀별 클래스 선택·필요 시 softmax 확률 계산 |
| `rules_ms` | 결함 덩어리 계산, 면적/점수 기준 적용, 원본 좌표 변환, 확률 버퍼 해제 |
| `render_ms` | 출력 마스크 크기 조정·복사·색상 이미지 생성·픽셀 집계 |
| `inference_ms` | 위 6개 구간의 합 |
| `postprocess_ms` | validation+decode+rules+render 소계 |

`std::chrono::steady_clock`으로 인접 경계를 측정한다. 모델·입력 크기·판정 알고리즘과 스레드는 바꾸지 않는다. 로그 출력은 구간 밖에서 한다. 함수 반환 시 나머지 로컬 객체 정리와 호출자 결과 대입은 내부 합계에 포함되지 않는다.

무인자 `example/cpp/segmentation_eval` 예제의 콘솔과 `report.json`에 구간을 추가했다. 각 이미지의 `timing_samples_ms`에 워밍업 제외 매 반복을 기록한다. `timing_sample_ms`는 전체 호출 중앙값과 가장 가까운 실제 실행 한 회이며, 구간별 중앙값을 조합하지 않는다. `call_total_ms`는 반환·대입까지 포함한 외부 시간이고 `sdk_total_ms`는 내부 시간이다. 파일 읽기·그림 표시·파일 저장은 제외한다.

Windows GUI/MFC에서 `OutputDebugStringA`로 출력하는 코드는 `example/cpp/segmentation_eval/README.ko.md`에 있다. Release/x64 상태로 F5 실행 후 출력 창(Ctrl+Alt+O)의 디버그 출력을 확인한다. 디버거가 없는 실행은 콘솔/파일 로그를 사용한다.

## 검증

- macOS C++17 Release 빌드: SDK와 평가 예제 성공.
- CTest 7/7 통과: 구간의 유한성·음수 아님·합계 일치, 결함 기준 유무, 기존 마스크, 배포·암호화·C ABI·분류 동시성 포함.
- Python 7개 통과: C++ 분할 판정 비교(원배율/50%), 실제 U-Net Lite/ResNet18 ONNX export → 평가 예제 → 반복 구간 JSON·정답 기반 지표 확인. 변경과 무관한 DINOv2/DeepLab/MobileNet 모델 5개는 이번 검사에서 제외.
- Windows 실행과 사용자의 1.2초 원인은 아직 측정하지 않았다. 이 기능으로 대상 PC의 구간 값을 수집해야 한다.

## 적용

- `cpp/include/vision_inference.h`와 `cpp/src/vision_inference.cpp`를 함께 갱신한다.
- 결과 구조체가 확장되므로 SDK와 호출 프로젝트를 모두 Release/x64로 재빌드한다. 새 헤더와 이전 LIB/DLL을 섞지 않는다.
- C ABI 구조체는 변경하지 않았다. 재학습·모델 재내보내기는 필요 없다.
