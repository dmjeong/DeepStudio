# eVision BW8 + ONNX Runtime (C++17)

**OpenCV 필요 없음.** 기존 eVision 프로그램의 `EImageBW8` 또는 `EROIBW8`를 전달한다.
이 예제는 Studio에서 내보낸 **분류 모델**용이며 컬러·BW16 버퍼는 받지 않는다.

암호화 모델은 [암호화 예제](../../ENCRYPTED_MODELS.md)를 따른다. `encrypted.cpp`는
`.dvsenc` 로드·복호화·준비 추론 후 같은 `InferEvision()` / `InferBW8()`을 사용한다.

## 기존 프로그램에 넣기

### 이미 있는 EROIBW8을 입력하기

`roi_example.h`는 실제 Open eVision의 **EROIBW8 참조를 받는 예제**다.
기존 eVision 프로젝트에 `classifier.h`, `bw8_preprocess.h`, `roi_example.h`, 상위 폴더의 `model_crypto.h`와
상위 폴더의 `nlohmann` 폴더를 함께 복사한다. 기존 eVision 헤더·라이브러리 설정은 필요하다.

```cpp
#include "roi_example.h"

// 프로그램 시작 시 생성하고 멤버 변수 등으로 계속 보관한다.
// 생성자에서 모델 로드와 준비 추론 1회를 완료한다.
EvisionRoiExample inspector(L"C:/models/model.json");

// 검사할 때: myRoi는 프로그램에서 이미 사용하는 EROIBW8 객체다.
auto result = inspector.Inspect(myRoi);
// 클래스 번호: result.class_id
// 클래스 이름: result.class_name
// 신뢰도(0~1): result.confidence
// 전처리 + 추론 + 후처리 시간(ms): result.inference_ms
```

`EROIBW8*` 포인터라면 null 여부를 확인한 후 `inspector.Inspect(*myRoi)`로 호출한다.
ROI 위치와 크기는 전달한 객체의 값을 그대로 사용하며, ROI의 첫 픽셀과 실제 행 간격으로
읽는다. 호출이 끝날 때까지 부모 이미지가 유효하고 카메라가 버퍼를 덮어쓰지 않아야 한다.
ROI를 모델 입력 크기로 직접 바꿀 필요는 없다. 내보낸 JSON에 맞춰 내부에서 전처리한다.

`roi_example.cpp`는 **이미지 로드 → ROI Attach/SetPlacement → ROI 추론**까지 포함한
별도 실행 예제다. 경로와 ROI 좌표를 바꿔 사용한다. 기존 MFC 프로젝트에는 이 파일의
`main()`을 추가하지 말고 위의 생성·검사 호출을 옮긴다. `main.cpp`는 SDK 없이 실행하는
합성 버퍼 테스트이며 실제 ROI 사용 예제는 `roi_example.cpp`다.

실제 Open eVision SDK가 없는 자동 테스트 환경에서는 이 두 파일의 SDK 빌드를 검증하지 않는다.
API 사용은 [Euresys 공식 예제](https://documentation.euresys.com/products/open_evision/open_evision_22_04/en-us/content/11_Pdf/D124ET-Using_Matching_and_Measurement_Tools_C%2B%2B-Open_eVision-22.04.0.1166.pdf)의 ROI 연결 방식을 따른다.

### 공통 파일과 링크 설정

1. 이 폴더의 **`classifier.h`, `bw8_preprocess.h`와 상위 폴더의 `model_crypto.h`**를 프로젝트에 복사한다.
2. 상위 폴더의 **`nlohmann` 폴더 전체**를 `classifier.h` 옆에 복사한다. `json.hpp`가 포함돼 있다. VS 추가 포함 디렉터리에 그 프로젝트 폴더와 ONNX Runtime 1.29.0의 `include`를 추가한다.
3. ONNX Runtime의 `lib`를 라이브러리 경로에 추가하고 `onnxruntime.lib`를 링크한다. Windows 암호화용 `bcrypt.lib`는 헤더에서 자동 지정하며 OS 기본 구성이라 추가 DLL은 없다.
4. x64 / C++17 / UTF-8 소스(`/utf-8`), 정확한 부동소수점(`/fp:precise`)으로 빌드한다.
5. `onnxruntime.dll`은 프로그램 EXE 옆에 둔다. 모델 JSON과 ONNX도 함께 준비한다.

기존 프로그램에서 사용 중인 eVision SDK/라이선스 설정은 그대로 필요하다.
`cpp/`의 기존 추론 엔진, `vision_runtime.dll`, OpenCV는 이 경로에서 사용하지 않는다.

```cpp
#include "classifier.h"

// 프로그램 시작 시 한 번 생성하고 앱의 멤버로 유지한다.
// 모델 로드 + 준비 추론 1회가 끝난 뒤 반환한다.
dvs_bw8::Classifier model(std::filesystem::u8path(u8"C:/모델/model.json"));

// image는 기존 프로그램의 EImageBW8 또는 EROIBW8 객체.
// 추론 버튼/작업 스레드에서 호출한다.
auto result = model.InferEvision(image);
// result.class_name, result.confidence, result.inference_ms
```

포인터를 직접 넘길 수도 있다. `GetRowPitch()`는 바이트 단위다.

```cpp
auto result = model.InferBW8(
    image.GetImagePtr(0, 0), image.GetWidth(), image.GetHeight(), image.GetRowPitch());
```

시작할 때 실제 프레임이 있으면 `Classifier(json, {pointer, width, height, pitch})`로
전달해 그 프레임으로 준비 추론한다. 없으면 검은 이미지로 준비한다. 오류는
`std::exception`으로 전달하므로 초기화/추론 호출부에서 잡아 표시한다.

기본 `InferEvision(image)` / `InferBW8(...)` 호출은 같은 객체를 여러 스레드에서
사용해도 내부 잠금으로 순차 처리한다. 호출이 끝날 때까지 원본 버퍼를 해제하거나
카메라가 덮어쓰지 않도록 한다. 원본 전체 이미지 복사는 하지 않으며 모델용 float 텐서는
내부 버퍼에 생성한다. 전처리 좌표와 텐서를 재사용한다.

## OpenMP로 동시에 추론하기

**모델은 하나만 읽고, 작업자마다 Context를 하나씩 만든다.** Context는 각 작업자가
쓰는 전처리 좌표·입력·출력 버퍼다. 다른 Context는 같은 ONNX Runtime CPU 세션을
동시에 사용하므로 모델을 여러 번 읽거나 복호화하지 않는다. OpenVINO 전환이나
배치 export는 이 공유 버퍼 오류를 고치는 데 필요하지 않다.

초기화 시 모델과 Context를 멤버에 보관한다. 다음은 평문 모델 초기화다.
암호화 모델은 기존 `Classifier(encrypted_path, key)` 생성자로 바꾸면 된다.

```cpp
#include "classifier.h"
#include <array>
#include <exception>
#include <omp.h>

constexpr int workers = 2;
dvs_bw8::Classifier model(L"C:/models/model.json");
std::array<std::unique_ptr<dvs_bw8::Classifier::Context>, workers> contexts;
for (auto& context : contexts) {
    context = model.CreateContext();
    model.InferEvision(*context, startupRoi); // 실제 ROI로 작업자별 준비 추론.
}
```

검사할 ROI들을 먼저 준비하고, 결과 배열도 검사 개수만큼 미리 만든다.
`rois`는 서로 다른 `EROIBW8` 객체를 가리키는 포인터 배열이다.

```cpp
std::vector<dvs_bw8::Result> results(rois.size());
std::vector<std::exception_ptr> errors(rois.size());
const int count = static_cast<int>(rois.size());
#pragma omp parallel for num_threads(workers)
for (int i = 0; i < count; ++i) {
    try {
        results[i] = model.InferEvision(*contexts[omp_get_thread_num()], *rois[i]);
    } catch (...) {
        errors[i] = std::current_exception(); // OMP 영역 밖에서 표시한다.
    }
}
for (const auto& error : errors) if (error) std::rethrow_exception(error);
```

VS2017에서 **C/C++ → 언어 → OpenMP 지원: 예(`/openmp`)**를 켠다.
전체 실행 예제 `parallel.cpp`는 인자 없이 암호화 예제 모델을 한 번 읽고,
작업자별 Context 생성·준비 추론·병렬 호출·결과 검사를 수행한다.
OpenMP가 없는 빌드는 같은 예제를 `std::thread`로 실행한다.

- 작업자별로 다른 Context를 쓴다. 같은 Context를 공유하면 안전하지만 잠금 때문에 순차 처리한다.
- 같은 ROI 객체를 다른 스레드에서 `SetPlacement()`로 변경하지 않는다. ROI와 부모 이미지의 픽셀은 모든 호출이 끝날 때까지 그대로 유지한다.
- 결과는 `results[i]`처럼 각 작업이 자기 위치에 쓴다. 공유 `result` 변수나 동시 `push_back()`을 쓰지 않는다.
- 모델과 Context를 해제하기 전에 모든 작업자를 종료한다. 추론 중 모델을 재초기화하지 않는다.
- 작업자 수는 2개부터 비교한다. ORT 내부 스레드도 CPU를 사용하므로 작업자가 많다고 항상 빨라지지는 않는다. JSON에 기록된 검증된 실행 설정은 그대로 사용한다.
- `inference_ms`는 전처리·모델·후처리 시간이다. 잠금 대기 시간과 전체 ROI 처리 시간은 호출부의 시계로 별도 측정한다.

배치는 여러 ROI를 `[N,C,H,W]` 입력 하나로 묶는 별도 방식이다. 이를 쓰려면
dynamic batch로 내보낸 모델과 batch API가 필요하다. 위 Context API는 기존
batch 1 모델로 동작한다. 배치와 Context 중 빠른 쪽은 대상 PC에서 비교해야 한다.

## 예제 자체 빌드·테스트

### Visual Studio 2017 사용 시

**VS2017 15.9 최신 업데이트, x64, C++17(`/std:c++17`)**을 사용한다.
`/Zc:noexceptTypes`를 켜고 `/Zc:noexceptTypes-` 옵션은 제거한다.
`Float16_t`/`BFloat16_t`의 C3615는 C++17만 켜서는 해결되지 않는 SDK 헤더 호환성 오류다.

아래 CMake 빌드 명령의 생성기를 `"Visual Studio 15 2017"`로 바꾸고 **새 빌드 폴더**를
사용하면 호환용 헤더가 자동 생성된다. SDK 원본이나 DLL은 수정하지 않는다.

**기존 VS 프로젝트에서는 명령어를 입력할 필요 없이 다음 순서로 진행한다.**

1. `example/cpp/setup_vs2017.bat`을 더블클릭한다.
2. ONNX Runtime SDK 폴더(예: `C:/libs/onnxruntime-win-x64-1.29.0`)를 붙여넣고 Enter를 누른다. `include` 폴더를 넣어도 된다.
3. 생성이 완료되면 화면에 표시되는 `include-vs2017` 경로를 복사한다.
4. VS의 **C/C++ → 일반 → 추가 포함 디렉터리**에 기존 ONNX include보다 앞에 넣고 **솔루션 다시 빌드**한다.

Python 3.8 이상이 필요하며, CMake나 PowerShell은 필요 없다. 배치파일은 같은 폴더의
`setup_vs2017.py`를 실행한다. Python 실행 환경이 이미 있으면 `.py`를 직접 실행해도 된다.
원본 SDK를 보존하고 같은 SDK 안의 별도 `include-vs2017` 폴더에 호환용 헤더를 만든다.
`onnxruntime.lib`와 DLL은 기존 SDK 파일을 사용한다. 설치본에서도
`Examples/cpp/setup_vs2017.bat`과 `.py`를 함께 제공한다.

### 빌드 명령

저장소 루트에서 PowerShell로 실행한다. ONNX Runtime SDK가 필요하다. nlohmann-json은 예제에 포함돼 있어 별도 설치가 필요 없다.

```powershell
cmake -S example/cpp/with_evision -B example/build-evision -G "Visual Studio 17 2022" -A x64 `
  "-DONNXRUNTIME_ROOT=C:/libs/onnxruntime-win-x64-1.29.0" `
  "-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build example/build-evision --config Release
ctest --test-dir example/build-evision -C Release --output-on-failure
.\example\build-evision\Release\evision_onnx_example.exe
```

실행 인자가 없다. 기본 실행은 합성 BW8 버퍼로 ONNX를 실행하므로 eVision 설치도 필요 없다.
실제 SDK 객체와의 연결 코드는 위 `InferEvision(image)`다. 자체 테스트는 포인터 접근 계약을
검사하며, eVision SDK/카메라 실장비 실행을 대신하지 않는다.

배포에는 자신의 EXE, `onnxruntime.dll`, 모델 JSON/ONNX, 기존 eVision 실행 구성과
Visual C++ x64 런타임이 필요하다. nlohmann-json은 헤더 라이브러리라 DLL이 없다.

### 개발용: Studio 내보내기 산출물까지 연결 검사

Windows CI는 LibreYOLO 공개 API 형태의 작은 합성 모델을 Studio의 실제
`export_checkpoint()`로 내보낸 뒤, **생성된 ONNX·JSON을 수정 없이** C++로 읽는다.
검정·회색·흰색 BW8 ROI의 클래스와 모든 확률을 PyTorch 기준값과 비교하므로,
입출력 이름 불일치와 softmax 중복 적용도 검출한다. VS2022와 VS2017 v141 CI에 연결돼 있다.
이는 배포 연결 회귀 검사이며 실제 MobileNetV4 가중치의 정확도나 eVision SDK 인증을 뜻하지 않는다.

로컬에서 검사하려면 torch·onnx·onnxruntime이 있는 Python 환경으로 저장소 루트에서
아래 명령을 실행한다. 기존 사용자 예제 빌드에는 이 Python 환경이 필요 없다.

```bat
python example/generate_export_test_assets.py example/build-export-assets
cmake -S example/cpp/with_evision -B example/build-export-check -G "Visual Studio 17 2022" -A x64 ^
  "-DONNXRUNTIME_ROOT=C:/libs/onnxruntime-win-x64-1.29.0" ^
  "-DSTUDIO_EXPORT_TEST_ASSETS=%CD%/example/build-export-assets"
cmake --build example/build-export-check --config Release
ctest --test-dir example/build-export-check -C Release --output-on-failure
```

## 전처리 일치 검사

BW8 입력을 모델 채널 수(1 또는 3)에 맞게 사용하고, JSON의 중앙 crop, uint8 bilinear
resize, 채널별 정규화를 적용한다. 3채널 모델에는 같은 흑백 값을 세 채널로 복제한다.
미지원 전처리는 오류로 알린다. 학습툴과의 일치 확인용으로만
`-DEVISION_CHECK_OPENCV_PARITY=ON`을 사용하면 OpenCV 비교 테스트를 추가 빌드한다.
일반 빌드에는 이 옵션을 켜지 않는다.

## 기존 Visual Studio 프로젝트에서 LNK2001: OrtGetApiBase가 나올 때

프로젝트 속성 위쪽에서 **모든 구성**, 플랫폼 **x64**를 선택한다.
- **링커 → 일반 → 추가 라이브러리 디렉터리**: `onnxruntime.lib`가 실제 들어 있는 폴더를 추가한다.
- **링커 → 입력 → 추가 종속성**: `onnxruntime.lib`를 추가한다. 최신 classifier.h에는 자동 지정도 들어 있다.
- 기존 항목을 지우지 말고 추가하고, Debug와 Release를 각각 다시 빌드한다.

예를 들어 파일이 `D:/SVM/SRC/HVision_260917/onnx/lib/onnxruntime.lib`라면 추가할 폴더는
`D:/SVM/SRC/HVision_260917/onnx/lib`다. 실제 파일 위치가 다르면 그 위치를 사용한다.
Debug/Release 모두 같은 공식 x64 ONNX Runtime import library를 사용한다.
실행 시에는 같은 SDK의 `onnxruntime.dll`을 EXE 옆에 둔다. DLL을 두는 것만으로 링크 오류는 해결되지 않는다.
