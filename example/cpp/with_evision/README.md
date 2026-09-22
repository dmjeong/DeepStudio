# eVision BW8 + ONNX Runtime (C++17)

**OpenCV 필요 없음.** 기존 eVision 프로그램의 `EImageBW8` 또는 `EROIBW8`를 전달한다.
이 예제는 Studio에서 내보낸 **분류 모델**용이며 컬러·BW16 버퍼는 받지 않는다.

## 기존 프로그램에 넣기

1. 이 폴더의 **`classifier.h`, `bw8_preprocess.h` 두 파일**을 프로젝트에 복사한다.
2. 상위 폴더의 **`nlohmann` 폴더 전체**를 `classifier.h` 옆에 복사한다. `json.hpp`가 포함돼 있다. VS 추가 포함 디렉터리에 그 프로젝트 폴더와 ONNX Runtime 1.29.0의 `include`를 추가한다.
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
