# C++17 / C# ONNX 로딩·추론 예제

앱에서 내보낸 **분류 모델의 `.json`** 경로로 시작한다. 같은 폴더의 `.onnx`를 자동으로
읽고 전처리·클래스 이름·검증된 ONNX Runtime 최적화 및 스레드 설정을 적용한다.
EfficientNet B0/B1, ResNet, ConvNeXt 등 Studio 분류 export를 사용한다.
임의의 외부 ONNX에 JSON만 붙이면 모든 모델이 동작한다는 의미는 아니다.

- [C++17 전체 코드](cpp/main.cpp): 이미지 파일 → OpenCV → `VisionInference`.
- [C# 전체 코드](csharp/Program.cs): 카메라/이미지의 byte 배열 → `VisionSession` → native SDK.
- [테스트 데이터](assets): 224×224 합성 이미지와 작은 ONNX. 학습 가중치나 외부 이미지가 없다.

앱/Python/PyTorch/Docker를 띄울 필요 없이 프로그램 내부 함수로 호출한다.
세션은 한 번 열고 재사용한다. 이 SDK의 같은 세션을 여러 스레드에서 동시에 호출하지 않는다.
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
cmake --build example/build --config Release --target onnx_cpp_example vision_runtime

# 실행 시 ONNX Runtime/OpenCV 및 이들의 종속 DLL을 찾을 경로
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
cmake --build "$env:TEMP\dvs-cpp-example" --config Release --target onnx_cpp_example vision_runtime

$env:DEEP_VISION_NATIVE_RUNTIME_DIR = "$env:TEMP\dvs-cpp-example\vision\Release"
$env:PATH = "$env:DEEP_VISION_NATIVE_RUNTIME_DIR;$env:ONNXRUNTIME_ROOT\lib;$env:VCPKG_ROOT\installed\x64-windows\bin;$env:PATH"
dotnet build "$env:ProgramFiles\DeepVisionStudio\Examples\csharp\OnnxExample.csproj" -c Release
```

설치 예제는 모델·학습 데이터·Python 소스 저장소를 포함하지 않는다. 앱에서 내보낸
ONNX/JSON 또는 `.dvdeploy` 번들을 예제 실행 인수로 지정한다.

`100% tests passed`가 나오면 C++ 모델 로드와 실제 추론이 통과한 것이다.

## C++: 내 모델 실행

```powershell
.\example\build\Release\onnx_cpp_example.exe C:\models\model.json C:\images\test.png 100
```

```cpp
VisionInference model;
if (!model.InitializeFromJson("C:/models/model.json", "onnxruntime"))
    throw std::runtime_error("model load failed");
cv::Mat pixels = cv::imread("C:/images/test.png", cv::IMREAD_COLOR); // BGR
auto result = model.Classify(pixels); // resize + normalize + ONNX + softmax
std::cout << result.class_name << " " << result.confidence << '\n';
```

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

예제는 모델 로드와 파일 디코딩 후 3회 준비 추론을 하고, 지정한 횟수로 반복한다.
`preprocess_ms`, `model_ms`, `postprocess_ms`는 각 단계 평균이며,
`call_p50_ms`/`call_p95_ms`는 호출 전체 시간의 중앙값/95백분위다.
C# 호출 시간에는 P/Invoke와 결과 복사도 포함된다. 디스크 읽기·모델 로드·첫 추론은 제외된다.
합성 모델의 시간은 EfficientNet 속도가 아니므로 실제 내보낸 모델로 측정해야 한다.

이 실행 예제의 결과 처리는 **분류 전용**이다. 분할·검출·PatchCore·SAM2 API는
설치된 `cpp/vision-runtime/include/vision_runtime_c.h`와
`csharp/vision-runtime/VisionRuntime.cs`를 참고한다.
`VisionSession.OpenBundle("model.dvdeploy")`는 해시 검증을 포함한 배포 폴더를 연다.

테스트 파일을 재생성할 때만 `python example/generate_test_assets.py`를 실행한다 (`onnx` 필요).
이미 파일이 들어 있으므로 일반 빌드·실행에는 재생성이 필요 없다.
