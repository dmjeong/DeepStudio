# eVision BW8 + ONNX Runtime (C++17)

**OpenCV 필요 없음.** 기존 eVision 프로그램의 `EImageBW8` 또는 `EROIBW8`를 전달한다.
이 예제는 Studio에서 내보낸 **분류 모델**용이며 컬러·BW16 버퍼는 받지 않는다.

## 기존 프로그램에 넣기

1. 이 폴더의 **`classifier.h`, `bw8_preprocess.h` 두 파일**을 프로젝트에 복사한다.
2. ONNX Runtime 1.29.0의 `include`와 nlohmann-json의 `include`를 포함 경로에 추가한다.
3. ONNX Runtime의 `lib`를 라이브러리 경로에 추가하고 `onnxruntime.lib`를 링크한다.
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

동일 객체의 동시 호출은 지원하지 않는다. 호출이 끝날 때까지 원본 버퍼를 해제하거나
카메라가 덮어쓰지 않도록 한다. 원본 전체 이미지 복사는 하지 않으며 모델용 float 텐서는
내부 버퍼에 생성한다. 전처리 좌표와 텐서를 재사용한다.

## 예제 자체 빌드·테스트

저장소 루트에서 PowerShell로 실행한다. ONNX Runtime SDK와 nlohmann-json은 준비돼 있어야 한다.
nlohmann-json만 vcpkg로 설치한다면 `vcpkg install nlohmann-json:x64-windows`를 사용한다.

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

## 전처리 일치 검사

BW8 입력을 모델 채널 수(1 또는 3)에 맞게 사용하고, JSON의 중앙 crop, uint8 bilinear
resize, 채널별 정규화를 적용한다. 3채널 모델에는 같은 흑백 값을 세 채널로 복제한다.
미지원 전처리는 오류로 알린다. 학습툴과의 일치 확인용으로만
`-DEVISION_CHECK_OPENCV_PARITY=ON`을 사용하면 OpenCV 비교 테스트를 추가 빌드한다.
일반 빌드에는 이 옵션을 켜지 않는다.
