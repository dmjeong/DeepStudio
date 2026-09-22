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
