# OpenCV + ONNX Runtime (C++17)

`cv::Mat` 또는 이미지 파일을 쓰는 버전이다. 상위 `cpp/`의 공통 예제와 저장소 루트의
`cpp/` 추론 엔진을 사용하므로 **이 폴더만 복사하지 말고 기존 폴더 구조를 유지**한다.
설치본은 `Examples/cpp/vision-runtime`에 엔진이 함께 들어 있다.

```cpp
#include "classifier.h"
// 프로그램 시작 시 모델 로드 + 준비 추론 1회
example::Classifier model("C:/models/model.json");
// 원본 GRAY/BGR/BGRA cv::Mat을 전달한다.
auto result = model.Infer(image);
```

OpenCV와 ONNX Runtime은 별도로 준비한다. nlohmann-json은 상위 `nlohmann` 폴더에 포함돼 있다. 수동 프로젝트에는 그 폴더도 복사하고 추가 포함 디렉터리에 그 상위 폴더를 등록한다. 저장소 루트에서:

VS2017은 **15.9 최신 업데이트 + C++17 + x64**를 사용한다. 아래 생성기를
`"Visual Studio 15 2017"`로 바꾸고 새 빌드 폴더를 지정하면 ONNX Runtime의
Float16/BFloat16 헤더 호환성 보정이 자동 적용된다. 기존 VS 프로젝트에 수동으로
넣을 때는 [VS2017 헤더 생성 및 포함 경로 설정](../with_evision/README.md#visual-studio-2017-사용-시)을 따른다.

```powershell
cmake -S example/cpp/with_opencv -B example/build-opencv -G "Visual Studio 17 2022" -A x64 `
  "-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  "-DONNXRUNTIME_ROOT=C:/libs/onnxruntime-win-x64-1.29.0"
cmake --build example/build-opencv --config Release --target onnx_cpp_example onnx_cpp_self_test
ctest --test-dir example/build-opencv -C Release --output-on-failure
.\example\build-opencv\Release\onnx_cpp_example.exe
```

모델/이미지 경로는 상위 `cpp/main.cpp`에서 지정하며 실행 인자는 없다.
EXE 옆에 복사된 DLL들과 JSON/ONNX를 함께 배포한다.

## 기존 Visual Studio 프로젝트에서 LNK2001: OrtGetApiBase가 나올 때

프로젝트 속성 위쪽에서 **모든 구성**, 플랫폼 **x64**를 선택한다.
- **링커 → 일반 → 추가 라이브러리 디렉터리**: `onnxruntime.lib`가 실제 들어 있는 폴더를 추가한다.
- **링커 → 입력 → 추가 종속성**: `onnxruntime.lib`를 추가한다. 최신 classifier.h에는 자동 지정도 들어 있다.
- 기존 항목을 지우지 말고 추가하고, Debug와 Release를 각각 다시 빌드한다.

예를 들어 파일이 `D:/SVM/SRC/HVision_260917/onnx/lib/onnxruntime.lib`라면 추가할 폴더는
`D:/SVM/SRC/HVision_260917/onnx/lib`다. 실제 파일 위치가 다르면 그 위치를 사용한다.
Debug/Release 모두 같은 공식 x64 ONNX Runtime import library를 사용한다.
실행 시에는 같은 SDK의 `onnxruntime.dll`을 EXE 옆에 둔다. DLL을 두는 것만으로 링크 오류는 해결되지 않는다.
