# C++17 EfficientNet B0 검증 결과

2026-09-17. Apple M4 / macOS 26.5 / Apple Clang / C++17 / Release 실측.
**Intel i7 또는 Windows에서 측정한 결과가 아니다.**

B0, 배치 1, 224×224, 1채널, 2클래스, FP32, 시드 고정 임의 가중치.
Python 측정에서 내보낸 **동일 ONNX/JSON과 동일 이미지**를 사용했다.
ONNX Runtime 1.30.0, OpenVINO 2026.4.0, 최소 OpenCV 4.12.0.
아래 결과는 4스레드, 워밍업 30회 후 1,000회다. 1/2스레드도 각 100회 비교했다.
[환경·모델 해시·전체 측정값 JSON](benchmarks/efficientnet-b0-m4-cpp17.json)도 보관했다.

| Runtime | 호출 방식 | 반복 | p50 ms | p95 ms | p99 ms | 최대 ms | 8 ms 이내 |
|---|---|---:|---:|---:|---:|---:|---:|
| onnxruntime | 동기 | 1000 | 5.232 | 5.599 | 5.860 | 6.103 | 100.0% |
| onnxruntime | 백그라운드 | 1000 | 5.284 | 5.821 | 6.795 | 8.115 | 99.9% |
| openvino | 동기 | 1000 | 4.944 | 5.133 | 6.139 | 11.257 | 99.5% |
| openvino | 백그라운드 | 1000 | 4.968 | 5.198 | 6.151 | 10.339 | 99.7% |

동기 측정은 호출 전체 전처리+모델+후처리다. 백그라운드 측정은 Submit의 프레임 복사,
대기열, 전처리+추론+후처리, future 결과 수신까지 포함한다. 파일 디코딩/모델 로드/컴파일은
제외했고 한 번에 한 요청을 제출했다. **생산 부하에서 대기열이 쌓일 때의 시간은 다르다.**

OpenVINO 백그라운드 p95는 5.198 ms로 이 조건의 p95 목표를 만족했다.
그러나 최대 10.339 ms, 1,000회 중 3회는 8 ms 초과였다. ONNX Runtime 백그라운드도
최대 8.115 ms로 1회 초과였다. 따라서 모든 요청의 8 ms 마감 달성으로 보고하지 않는다.
목표 i7의 실제 실행 결과와 요구되는 p95/최댓값 기준을 적용해야 한다.

## 제공한 C++ 기능

- `ClassificationWorker`: 앱 내부에서 생성하는 C++17 기능 객체. 모델 로드/워밍업/
  추론이 하나의 전용 std::thread에서 실행된다. 별도 서비스/데몬이 아니다.
- Submit은 입력 cv::Mat을 복사하고 future를 반환한다. 부하가 쌓이면 대기열 상한에서
  명확한 예외를 전달한다. 허용된 요청은 FIFO로 끝까지 처리한다.
- Ready future에 초기화 오류, 결과 future에 개별 추론 오류를 전달한다.
  Stop/소멸자는 접수된 작업을 완료하고 join한다.
- C++ 런타임은 model.json의 상대 ONNX 경로, 채널/정규화/입출력 계약을 읽는다.
  ONNX Runtime 기본 경로와 선택적 OpenVINO FP32 CPU 경로를 제공한다.
- 일반/백그라운드 벤치마크 실행 파일과 컴파일 가능한 앱 통합 예제를 추가했다.

## 검증

- C++17 Release 빌드: OpenVINO ON/OFF 두 구성 성공. 각 구성의 CTest 4개 통과.
- worker: 소유한 프레임 복사, FIFO/대기열 상한, 개별 실패 후 복구, 초기화 실패,
  종료 후 거부, 중복 Stop, 4개 생산자 160개 요청, 실제 ONNX/OpenVINO 모델 실행 확인.
- 실제 B0 export: 1채널 모델과 중앙 crop을 포함한 3채널 모델(동적 배치 export)을 사용.
  각 모델을 GRAY/RGB/RGBA, ONNX/OpenVINO, 동기/백그라운드로 총 24개 조합 검증.
  Python과 클래스 일치 및 확률 오차 ≤ 1e-5를 확인했다.
- C++ 전처리의 실제 컴파일 실행 파일을 Python/OpenCV와 비교하는 4개 테스트 통과.
- Python 변경 관련 테스트 53 통과/2 skip. skip은 기존 폐기된 외부 엔진 의존 경로다.
- 실제 백그라운드 예제 실행에서 모델 로딩/추론 동안 메인 루프가 계속 진행되고 결과가
  정상 반환되는 것을 확인했다. 목표 미달 CLI가 JSON을 저장한 뒤 exit 2로 끝나는 것도 확인.

## 사용과 의존성

자세한 내용: [C++17 사용 안내](EFFICIENTNET_CPP17.md).
C++17 컴파일러/CMake, OpenCV core/imgproc/imgcodecs, nlohmann/json, ONNX Runtime SDK,
선택적 OpenVINO Runtime이 필요하다. 현재 결합 라이브러리는 OpenVINO를 선택해도
ONNX Runtime SDK에 의존한다. C++ 앱 실행에는 Python이 필요 없다.

이번 C++ 검증 빌드는 highgui/videoio/FFmpeg/Qt를 제외했다. BSD/MIT/Apache 및 PNG/JPEG
등 실제 구성요소 고지는 유지한다. 가중치 권한 검토는 사용자 요청대로 제외했다.
[코드/라이브러리 라이선스 범위](EFFICIENTNET_CPU_LICENSES.md)

## 다음 확인

목표 i7에서 Release 빌드로 동일 명령을 실행한다. 현재 실제 i7 모델명과 OS가
제공되지 않았으므로 그 환경의 8 ms 달성과 Windows 빌드 성공을 주장하지 않는다.

확인할 개념: ONNX는 런타임에서 읽는 모델 파일, future는 비동기 결과 전달,
p95와 최댓값은 서로 다른 지연시간 기준이다.
