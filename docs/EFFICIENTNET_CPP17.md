# EfficientNet B0를 C++17 앱의 백그라운드 기능으로 사용

`ClassificationWorker`를 기존 C++ 앱의 멤버로 한 번 생성하고 `Submit()`으로 이미지를
넘긴다. 내부 스레드가 모델 로드/컴파일/워밍업/추론을 담당한다. 호출자는
`std::future`로 결과나 오류를 받는다. 별도 서비스, Python 프로세스, GUI 프로그램을
상주시킬 필요가 없다. **C++17**의 thread/condition_variable/future만 사용한다.

[실제 C++17 검증 및 1,000회 측정 결과](EFFICIENTNET_CPP17_RESULTS.md)

## 1. 내보내면 생기는 파일과 C++에서 읽는 방법

프로젝트 폴더 `DeepVisionStudio`에서:

```bash
python python/export_onnx.py --checkpoint best.pt --output deploy/model.onnx
```

성공하면 `model.onnx`와 `model.json`이 생성된다. ONNX는 가중치와 연산 그래프,
JSON은 입력 크기/채널/정규화/클래스/ONNX 파일명이다. 두 파일을 같은 디렉터리에
복사한다. 모델을 C++ 소스 코드로 번역하는 방식이 아니라 **C++ 런타임이 ONNX를 읽어
실행**한다. export는 Conv–BN 융합을 적용하고 원본 PyTorch FP32 출력과 수치 비교한다.
동일한 ONNX/JSON을 ONNX Runtime과 OpenVINO 모두 읽는다.

직접 동기 호출부터 확인하려면:

```cpp
#include "vision_inference.h"
#include <stdexcept>

VisionInference model; // 기존 클래스명 유지; JSON backend는 efficientnet
if (!model.InitializeFromJson("deploy/model.json", "onnxruntime", 4)) {
    throw std::runtime_error("model load failed");
}
ClassifyResult result = model.Classify(gray_frame); // uint8 cv::Mat, GRAY/BGR/BGRA
// result.class_id, class_name, confidence, probabilities
// result.inference_ms == preprocess_ms + model_ms + postprocess_ms
```

`"onnxruntime"`은 기본 경로다. `"openvino"`는 아래 빌드 옵션이 필요하다.
한 모델 인스턴스에 동시에 `Initialize`/`Release`/`Classify`하지 않는다.
백그라운드 사용에는 다음 worker가 소유권과 순차 실행을 관리한다.

## 2. C++17 앱에 넣는 최소 사용 코드

```cpp
#include "classification_worker.h"
#include <chrono>
#include <future>

// 앱 시작: 모델은 worker에서 로드된다. 4 threads, 최대 대기 2장, 워밍업 30회.
ClassificationWorker inference("deploy/model.json", "onnxruntime", 4, 2, 30);
auto initialized = inference.Ready();

// UI tick 등에서 준비 여부를 확인. get()은 ready일 때 호출한다.
if (initialized.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
    initialized.get(); // 초기화 실패 시 예외
}

// 준비된 뒤 카메라/검사 코드에서 제출. 입력은 호출이 반환되기 전에 복사된다.
std::future<BackgroundClassification> pending = inference.Submit(frame_id, gray_frame);
// 이 시점부터 원래 gray_frame/카메라 버퍼를 재사용할 수 있다.

// 다른 작업을 하면서 다음 tick에서 결과 확인:
if (pending.valid() && pending.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
    auto completed = pending.get(); // 해당 추론에서 발생한 예외도 여기서 전달
    // UI 갱신은 이 호출자/UI 스레드에서 수행
    UseResult(completed.frame_id, completed.prediction);
}

// 앱 종료: 제출하는 스레드를 먼저 멈추고, 접수된 작업을 끝낸 뒤 join.
inference.Stop(); // 소멸자도 동일하게 정리한다.
```

위 코드는 앱 lifecycle에 나눠 넣는 예시다. `gray_frame`, `frame_id`, `UseResult`는
기존 앱의 입력/처리 함수다. 완전히 컴파일 가능한 최소 예제는
`cpp/src/efficientnet_background_demo.cpp`에 있다. worker 생성과 추론 사이에
메인 루프가 계속 실행되는 것도 이 예제로 확인할 수 있다.

### 소유권과 오류 처리

- worker를 프레임마다 만들지 않는다. 한 번 생성하고 재사용한다.
- 대기열은 기본 2장이고 실행 중인 1장이 추가될 수 있다. 가득 차면 `Submit()`이 예외를
  낸다. 호출자가 대기/건너뛰기/경보를 결정하며, 접수된 이미지를 조용히 버리지 않는다.
- `Submit()`은 추론 완료를 기다리지 않지만 이미지 복사와 짧은 mutex 대기는 수행한다.
  여러 생산자가 제출할 수 있다. 같은 입력 버퍼를 복사 중에 다른 스레드가 수정하면 안 된다.
- `Ready().get()`의 모델 초기화 오류, `Submit()`의 접수 오류,
  `pending.get()`의 개별 추론 오류를 각각 `try/catch`로 처리한다.
- `Stop()`은 실행 중/대기 중 작업을 완료할 때까지 기다린다. 호출자 스레드에서 실행한다.
  worker 내부 predictor에서 Stop/소멸하지 않는다. 소멸 중에는 Submit하지 않는다.
- 직접 실행 가능한 예제는 완료까지 `wait_for(0)`으로 폴링하지만, 실제 GUI에서는
  프레임별 무한 대기 루프 대신 기존 timer/event loop에 결과 확인 코드를 넣는다.
- 백그라운드화는 UI 차단을 줄인다. CPU 연산을 자동으로 더 빠르게 만드는 것은 아니다.
  8 ms 판정에는 추론 합계와 함께 복사/대기/결과 전달 시간을 별도로 확인한다.

## 3. 필요한 개발·실행 파일

| 항목 | 빌드 때 | 앱 실행 때 |
|---|---|---|
| C++17 컴파일러, CMake | 필요; MSVC 2022/GCC/Clang | 컴파일러 불필요 |
| OpenCV 4.x core/imgproc/imgcodecs | 헤더와 import/static library | 동적 빌드의 해당 DLL/so/dylib |
| nlohmann/json | CMake package와 헤더 | 별도 DLL 불필요 |
| ONNX Runtime CPU SDK | 헤더와 라이브러리 | SDK의 CPU runtime DLL/so/dylib |
| OpenVINO Runtime (선택) | `VISION_WITH_OPENVINO=ON`이면 필요 | runtime, CPU plugin, ONNX frontend, TBB 등 SDK 실행 의존성 |
| `model.onnx`, `model.json` | 없어도 C++ 컴파일 가능 | 두 파일 필요 |
| Python + PyTorch/onnx/NumPy/OpenCV | 모델 export 및 검증 도구 실행 때 | C++ 앱 실행에는 불필요 |

현재 C++ 라이브러리는 ONNX Runtime과 선택적 OpenVINO를 함께 제공하므로 OpenVINO를
선택해도 **ONNX Runtime SDK는 빌드/실행 의존성**이다. Python 구현의 의존성과 혼동하지
않는다. Intel i7 Windows는 컴파일러/SDK/DLL을 모두 x64 Release로 맞춘다.
Debug/Release DLL이나 x86/ARM64 파일을 혼합하지 않는다.

필요한 모델 파일 옆에 `python/EFFICIENTNET_NOTICE.txt`와 실제 배포 라이브러리의
라이선스/제3자 고지를 포함한다. 자세한 조건은 [라이선스 문서](EFFICIENTNET_CPU_LICENSES.md).

## 4. Windows x64 Release 빌드

각 SDK의 실제 설치 경로로 바꾼다. `OpenCV_DIR`/`OpenVINO_DIR`는 각각
`OpenCVConfig.cmake`/`OpenVINOConfig.cmake`가 있는 디렉터리다.

```bat
cmake -S cpp -B build-cpp -G "Visual Studio 17 2022" -A x64 ^
  -DVISION_BUILD_DEMO=OFF -DVISION_WITH_OPENVINO=OFF -DBUILD_TESTING=OFF ^
  -DONNXRUNTIME_ROOT=C:/deps/onnxruntime ^
  -DOpenCV_DIR=C:/deps/opencv/lib/cmake/opencv4 ^
  -Dnlohmann_json_DIR=C:/deps/json/share/cmake/nlohmann_json
cmake --build build-cpp --config Release --parallel 4
```

실행 전 ONNX Runtime/OpenCV DLL 디렉터리를 PATH에 추가한다. 위 ONNX Runtime 전용
구성에는 OpenVINO가 필요 없다. 선택적으로 OpenVINO를 사용할 때는
`-DVISION_WITH_OPENVINO=ON`과 실제 `OpenVINO_DIR`를 지정하고 배포판의 `setupvars.bat`로
실행 경로를 설정한다. OpenVINO DLL 하나만 복사하면 CPU plugin이나
ONNX frontend를 찾지 못할 수 있으므로 SDK의 실행 디렉터리 구성을 유지한다.
배포 시에는 선택한 SDK 버전의 배포 안내에 맞춰 runtime 의존성을 함께 포함한다.

현재 선택한 경로는 `-DVISION_WITH_OPENVINO=OFF`와 runtime `onnxruntime`이다. 포함되지 않은 OpenVINO를 요청하면 명확한 오류를 반환하며
다른 런타임으로 암묵적으로 바꾸지 않는다.

다른 CMake 앱에 연결할 때는 `cpp`를 `add_subdirectory()`로 추가하고
`target_link_libraries(your_app PRIVATE vision_inference)`로 연결한다. 이 target이 C++17,
헤더 경로 및 런타임 의존성을 전달한다. 최상위에서 `VISION_BUILD_DEMO=OFF`를 설정하면
기존 highgui 데모 없이 기능 라이브러리/콘솔 예제만 빌드한다.

## 5. 실행 및 속도 측정

```bat
build-cpp\Release\efficientnet_background_demo.exe deploy/model.json test.png onnxruntime
build-cpp\Release\efficientnet_cpu_benchmark.exe --config deploy/model.json --image test.png --runtime onnxruntime --threads 4 --warmup 30 --runs 1000 --output cpp-direct.json
build-cpp\Release\efficientnet_cpu_benchmark.exe --config deploy/model.json --image test.png --runtime onnxruntime --threads 4 --warmup 30 --runs 1000 --background --output cpp-background.json
```

Linux/macOS 단일 구성 빌드는 `-DCMAKE_BUILD_TYPE=Release`로 지정하고 실행 파일은
`build-cpp/efficientnet_cpu_benchmark`에 생성된다. 실행 인자는 동일하다.
이미지 없이 구조 시간만 확인하려면 `--image` 대신 `--synthetic`을 사용한다.
`--threads 1`, `2`, `4`, `8`을 각각 실행해 p95가 가장 낮은 설정을 선택한다.

- direct의 `wall`: 함수 호출 전체 전처리+모델+후처리 시간.
- background의 `wall`: Submit의 이미지 복사+대기열+전체 추론+future 결과 수신까지.
  한 번에 한 요청을 제출하므로 정상 상태의 낮은 부하 지연시간이다.
- `pipeline`: worker 안에서 실행한 전처리+모델+후처리 시간.
- `queue`: 접수 시작부터 worker가 작업을 꺼낼 때까지. 입력 복사 시간도 포함한다.
- `--threads`를 생략하면 JSON의 `num_threads`를 따른다.
- 모델 로드/컴파일/워밍업, 파일 디코딩은 제외한다. 백그라운드 대기열이 실제 생산
  부하에서 쌓이면 전체 지연이 늘어난다. 8 ms의 최종 판단은 대상 i7 실측으로 한다.
- p50/p95/p99/최댓값/원시 측정값/8 ms 이내 비율을 JSON에 기록한다.
  `--require-target --target-ms 8`은 wall p95가 목표를 넘으면 결과 저장 후 exit 2.

## 6. 검증 명령과 확인 범위

`BUILD_TESTING=ON`, 테스트용 Python에 `onnx`가 설치된 상태로 빌드한다.
필요하면 `-DPython3_EXECUTABLE=/path/to/python`으로 지정한다.

```bash
ctest --test-dir build-cpp -C Release --output-on-failure
python tools/verify_efficientnet_cpp.py --executable build-cpp/Release/efficientnet_cpu_benchmark.exe --config deploy/model.json --runtimes onnxruntime --output cpp-parity.json
```

CTest는 fixture 모델 생성, 출력 계약, 잘못된 JSON/모델 거부, 결과 버퍼 소유권,
worker 대기열/예외/종료/다중 생산자를 검증한다. 두 번째 명령은 사용자가 내보낸 실제
ONNX를 RGB/RGBA/gray 이미지와 두 호출 방식으로 실행하여 Python의 클래스·전체 확률과
비교한다. OpenVINO를 포함한 빌드는 `--runtimes onnxruntime openvino`로 두 런타임을 비교할 수 있다. 테스트 입력은 합성이며 실제 데이터셋의 분류 정확도 평가는 별도다.

대상 환경은 Windows/i7이며 세부 i7 모델은 아직 확인되지 않았다.
[ONNX Runtime 추가 최적화와 Windows 자동 비교](EFFICIENTNET_ONNX_OPTIMIZATION.md)를 참고한다.

이번 검증은 macOS/Apple M4의 C++17 Release에서 실행했다. **Windows/i7에서 실행한
결과로 표시하지 않는다.** Windows 빌드 명령은 제공하지만 해당 장비 검증은 남아 있다.
