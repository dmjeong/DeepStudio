# C++17 / C# ONNX 로딩·추론 예제

**C++ 입력 방식에 따라 둘 중 하나를 선택한다.** 둘 다 시작 시 준비 추론을 1회 수행한다.

| 버전 | 입력 | OpenCV 필요 | 사용법 |
|---|---|---|---|
| `cpp/with_opencv` | `cv::Mat` / 이미지 파일 | 예 | [OpenCV용](cpp/with_opencv/README.md) |
| `cpp/with_evision` | eVision BW8 객체 / 버퍼 포인터 | 아니요 | [eVision용](cpp/with_evision/README.md) |

**아래 기존 C++ 설명은 OpenCV 버전이다. eVision은 위 전용 사용법을 따른다.**

앱에서 내보낸 **분류 모델의 `.json`** 경로로 시작한다. 같은 폴더의 `.onnx`를 자동으로
읽고 전처리·클래스 이름·검증된 ONNX Runtime 최적화 및 스레드 설정을 적용한다.
EfficientNet B0/B1, ResNet, ConvNeXt 등 Studio 분류 export를 사용한다.
임의의 외부 ONNX에 JSON만 붙이면 모든 모델이 동작한다는 의미는 아니다.

- [C++17 전체 코드](cpp/main.cpp): **명령행 인자 없이 실행**. 코드의 경로 두 개만 수정한다.
- [C++ 함수 호출용 코드](cpp/classifier.h): `Classifier`를 한 번 만들고 `Infer(cv::Mat)`를 호출한다.
- [C# 전체 코드](csharp/Program.cs): 카메라/이미지의 byte 배열 → `VisionSession` → native SDK.
- [테스트 데이터](assets): 224×224 합성 이미지와 작은 ONNX. 학습 가중치나 외부 이미지가 없다.

앱/Python/PyTorch/Docker를 띄울 필요 없이 프로그램 내부 함수로 호출한다.
프로그램 시작 시 `Classifier`를 생성하면 모델 로드와 준비 추론 1회를 끝낸 뒤 반환한다.
생성이 성공한 뒤에만 추론 버튼을 활성화하고, 해당 객체를 계속 재사용한다.
이 SDK의 같은 세션을 여러 스레드에서 동시에 호출하지 않는다.
비동기 큐가 필요하면 설치된 `cpp/vision-runtime/include/classification_worker.h`의
`ClassificationWorker`를 사용한다.

## Windows 준비

빌드 PC: Windows x64, Visual Studio 2022의 C++ 데스크톱 개발 도구, CMake,
vcpkg의 `opencv4:x64-windows`와 `nlohmann-json:x64-windows`,
[ONNX Runtime 1.29.0 win-x64](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0),
C#용 .NET 8 SDK가 필요하다. 아래 명령은 **저장소 루트의 PowerShell**에서 실행한다.
`build.bat`으로 학습툴 전체를 다시 만들 필요가 없다.

```powershell
$env:VCPKG_ROOT = 'C:\vcpkg'
$env:ONNXRUNTIME_ROOT = 'C:\libs\onnxruntime-win-x64-1.29.0'
& "$env:VCPKG_ROOT\vcpkg.exe" install opencv4:x64-windows nlohmann-json:x64-windows

cmake -S example/cpp -B example/build -G "Visual Studio 17 2022" -A x64 `
  "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  "-DONNXRUNTIME_ROOT=$env:ONNXRUNTIME_ROOT"
cmake --build example/build --config Release --target onnx_cpp_example onnx_cpp_self_test vision_runtime

# CMake가 선택한 onnxruntime.dll을 EXE 옆에 복사한다.
# vcpkg는 OpenCV 종속 DLL을 복사한다. 아래 PATH는 C# 실행에도 사용한다.
$env:PATH = "$env:ONNXRUNTIME_ROOT\lib;$env:VCPKG_ROOT\installed\x64-windows\bin;$env:PATH"
ctest --test-dir example/build -C Release --output-on-failure
```

## 설치파일에 포함된 예제

Windows Setup은 최상위에 프로그램과 `Examples`만 설치한다. `Examples/cpp`와
`Examples/csharp`는 위와 같은 ONNX 실행 예제이며, 각각 필요한 native runtime 소스를
`vision-runtime/`에 같이 둔다. 설치본에서 예제를 빌드할 때는 다음처럼 경로만 바꾼다.

```powershell
cmake -S "$env:ProgramFiles\DeepVisionStudio\Examples\cpp" -B "$env:TEMP\dvs-cpp-example" `
  -G "Visual Studio 17 2022" -A x64 `
  "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  "-DONNXRUNTIME_ROOT=$env:ONNXRUNTIME_ROOT"
cmake --build "$env:TEMP\dvs-cpp-example" --config Release --target onnx_cpp_example onnx_cpp_self_test vision_runtime

$env:DEEP_VISION_NATIVE_RUNTIME_DIR = "$env:TEMP\dvs-cpp-example\vision\Release"
$env:PATH = "$env:DEEP_VISION_NATIVE_RUNTIME_DIR;$env:ONNXRUNTIME_ROOT\lib;$env:VCPKG_ROOT\installed\x64-windows\bin;$env:PATH"
dotnet build "$env:ProgramFiles\DeepVisionStudio\Examples\csharp\OnnxExample.csproj" -c Release
```

설치 예제에는 합성 테스트용 ONNX/이미지가 들어 있다. 사용자 학습 모델·데이터·Python
소스 저장소는 포함하지 않는다. C++에서 내 모델을 사용할 때는 아래 코드의 경로를 수정한다.

`100% tests passed`가 나오면 C++ 모델 로드와 실제 추론이 통과한 것이다.

## C++: 내 모델 실행

`cpp/main.cpp`의 세 값만 바꾸고 빌드한다. `argc`, `argv`, 실행 인수 설정이 없다.

```cpp
const auto model_json = std::filesystem::u8path(u8"C:/모델/model.json");
const auto image_file = std::filesystem::u8path(u8"C:/이미지/test.png");
const int runs = 100;
```

```powershell
cmake --build example/build --config Release --target onnx_cpp_example
.\example\build\Release\onnx_cpp_example.exe
```

수정 전 기본값으로 실행하면 EXE 옆에 자동 복사된 `assets/test.json`과 `white.pgm`을
사용한다. 상대 경로는 EXE 폴더 기준이므로 Visual Studio F5 실행과 다른 작업 폴더에서도
동일하게 동작한다. 내보낸 JSON과 ONNX는 함께 둔다. JSON의 ONNX 경로는 JSON 폴더 기준이다.

**기존 Windows 프로그램의 기능으로 넣을 때**는 `classifier.h`를 포함하고
`vision_inference`에 링크한다. 아래 `classifier`를 창/기능 객체의 멤버로 유지하고,
추론 버튼이나 카메라 처리 함수에서 `Infer`만 호출한다. 이미지마다 모델을 다시 로드하지 않는다.

```cpp
#include "classifier.h"

// 프로그램 시작 시 실행한다. 모델 로드 + 준비 추론 1회가 완료된 후 반환한다.
example::Classifier classifier(std::filesystem::u8path(u8"C:/모델/model.json"));
// 여기서부터 추론 버튼을 활성화한다. 아래 호출은 이미지마다 반복한다.
cv::Mat pixels = example::ReadImage(std::filesystem::u8path(u8"C:/이미지/test.png"));
ClassifyResult result = classifier.Infer(pixels);
// 파일을 바로 처리할 경우: classifier.InferFile(image_path)
// 카메라의 GRAY/BGR/BGRA cv::Mat도 Infer(pixels)에 그대로 넣는다.
std::cout << result.class_name << " " << result.confidence
          << " " << result.inference_ms << " ms\n";
```

카메라 프레임을 시작 시 확보할 수 있다면 `Classifier(model_json, startup_frame)`처럼
실제 입력과 같은 크기·자료형의 이미지를 전달한다. 생략하면 모델 입력 크기의 검은 이미지로
준비 추론한다. 준비 추론 결과는 버리며 사용자 판정 결과에 포함하지 않는다.
준비 추론에 실패하면 생성자가 예외를 발생시키므로 준비 완료로 표시하지 않는다.
초기화가 UI를 막지 않도록 하려면 앱의 작업 스레드에서 생성하고, 완료 후 버튼을 활성화한다.
모델을 바꿀 때도 새 객체의 준비 추론이 끝난 뒤 사용한다.

전처리·ONNX 실행·softmax는 내부에서 수행한다. `result.inference_ms`는 이 세 단계의
합계이며 파일 읽기·디코딩은 제외한다. 실패는 `std::exception`을 잡아 프로그램에서 표시한다.
같은 객체에 대한 동시 호출은 피하고, UI를 멈추지 않으려면 앱의 작업 스레드에서 순차 호출한다.
객체가 살아 있는 동안 입력 `cv::Mat`을 보관할 필요는 없지만, 호출 중에는 픽셀을 수정하지 않는다.

다른 PC로 옮길 때는 EXE, EXE 옆의 DLL들, JSON/ONNX를 함께 복사한다.
**ONNX Runtime DLL을 PATH에만 두지 않는다.** Windows 시스템 경로의 다른 버전이 먼저
선택될 수 있으므로 빌드에 사용한 `onnxruntime.dll`이 EXE 옆에 있어야 한다.

`onnx_cpp_self_test`도 인자 없이 실행된다. ONNX 실제 출력, 세션 재사용, 한글 파일 경로,
GRAY/BGR/BGRA 및 연속되지 않은 카메라 ROI, 잘못된 파일 처리와 오류 후 재추론을 검사한다.

## C#: 빌드 및 자동 테스트

```powershell
dotnet build example/csharp/OnnxExample.csproj -c Release --artifacts-path example/dotnet-build
$env:DEEP_VISION_NATIVE_RUNTIME_DIR = (Resolve-Path example/build/vision/Release).Path
$env:PATH = "$env:DEEP_VISION_NATIVE_RUNTIME_DIR;$env:PATH"
dotnet example/dotnet-build/bin/OnnxExample/release/OnnxExample.dll --self-test example/assets
```

기대 결과는 `class_id=0`, `confidence=0.880797`, 마지막 줄 `PASS`다.
단순히 파일 존재 여부를 보는 테스트가 아니라 ONNX를 실행하고 두 클래스의 확률을 비교한다.
JSON에 저장한 `disabled` 최적화 설정을 무시하는 구형 SDK도 검출할 수 있게 구성했다.
파일 누락·형식 오류·출력 불일치 시 종료 코드 1, 사용법 오류 시 2를 반환한다.

실제 카메라 이미지라면 압축하지 않은 8-bit **BGR 또는 GRAY** 배열을 바로 넘긴다.

```csharp
using DeepVisionStudio;
using var model = VisionSession.Open("C:/models/model.json");
// pixels: 원본 이미지의 byte[], channels: 1 또는 3
var result = model.InferClassification(pixels, width, height, channels);
Console.WriteLine($"{result.ClassName}: {result.Confidence:P2}");
```

C# 예제의 CLI는 카메라 프레임처럼 이미 디코딩된 raw BGR/GRAY 배열을 받는다. PNG/JPEG을
사용하는 프로그램은 사용하는 이미지 라이브러리로 먼저 해당 배열을 만들고, 실제 `width`,
`height`, `channels`를 인수로 전달한다. C# SDK 자체에는 Python/Pillow 의존성이 없다.

다른 PC에 C# 실행 파일을 배포하려면:

```powershell
dotnet publish example/csharp/OnnxExample.csproj -c Release -r win-x64 --self-contained true -o example/publish
```

publish 폴더에 **현재 소스에서 빌드한** `vision_runtime.dll`, ONNX Runtime/OpenCV의 종속 DLL들,
모델 JSON/ONNX를 함께 배포한다. 대상 PC에는 .NET SDK나 Python이 필요하지 않다.
C++ 런타임은 Visual C++ 2015–2022 x64 재배포 패키지 또는 회사 설치 프로그램에서 제공해야 한다.
구형 `vision_runtime.dll`과 새 관리형 SDK를 섞지 않는다.

## 시간과 다른 태스크

C++ 예제는 시작 시 모델 로드와 준비 추론 **1회**를 끝낸 뒤 `READY`를 출력하고,
사용자 추론을 지정한 횟수로 반복한다. `model_load_and_warmup_ms`와 `warmup_ms`는
시작 비용이고, `first_call_ms`는 준비 완료 후 첫 사용자 추론의 호출 시간이다.
C# 예제는 파일 디코딩 후 3회 준비 추론을 한다.
`preprocess_ms`, `model_ms`, `postprocess_ms`는 각 단계 평균이며,
`call_p50_ms`/`call_p95_ms`는 호출 전체 시간의 중앙값/95백분위다.
C# 호출 시간에는 P/Invoke와 결과 복사도 포함된다. 디스크 읽기·모델 로드·첫 추론은 제외된다.
합성 모델의 시간은 EfficientNet 속도가 아니므로 실제 내보낸 모델로 측정해야 한다.
준비 추론은 최초 런타임 초기화 비용을 시작 단계에서 치르도록 한다. 이후 입력 크기·자료형이
바뀌거나 CPU 절전·OS 스케줄링이 개입하면 추가 지연은 생길 수 있으므로 0ms 지연을 보장하지 않는다.

이 실행 예제의 결과 처리는 **분류 전용**이다. 분할·검출·PatchCore·SAM2 API는
설치된 `cpp/vision-runtime/include/vision_runtime_c.h`와
`csharp/vision-runtime/VisionRuntime.cs`를 참고한다.
`VisionSession.OpenBundle("model.dvdeploy")`는 해시 검증을 포함한 배포 폴더를 연다.

테스트 파일을 재생성할 때만 `python example/generate_test_assets.py`를 실행한다 (`onnx` 필요).
이미 파일이 들어 있으므로 일반 빌드·실행에는 재생성이 필요 없다.
