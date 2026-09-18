# C++17 / C ABI SDK

`cpp/include/vision_runtime_c.h`와 `vision_runtime` shared library는 C++17,
C#, 다른 언어에서 같은 ONNX 세션을 호출하는 고정 ABI다. 세션은 한 번 열고 여러
이미지를 반복 처리한다. `dv_infer`는 동기 호출이므로 입력 버퍼는 호출이 끝날 때까지
유효해야 하며, 결과는 반드시 `dv_release_result`로 해제한다.

`builtin` backend manifest도 generic classify/segment 계약으로 읽는다. ResNet,
ConvNeXt, DeepLab V3+, U-Net의 export 파일은 Python 없이 같은 C ABI를 사용한다.

현재 C ABI에서 검증된 결과 종류는 `classify`, semantic `segment`, generic
`detect`, Re-DETR v4 `detect`, reconstruction `anomaly`다. generic `detect`는
`[1,N,5+C]` ONNX 출력에 sigmoid score와 class-aware NMS를 적용한다. Re-DETR v4는
`pred_boxes=[1,N,4]`와 `pred_logits=[1,N,C]`를 함께 읽고 sigmoid/softmax class score와
같은 NMS 결과를 반환한다. `anomaly`는 reconstruction과 입력의
채널 평균 절대오차 map을 반환한다. 고정 memory-bank를 포함해 export한 PatchCore
그래프도 두 번째 anomaly map 출력과 score를 같은 anomaly 결과로 읽는다. SAM2의
encoder/decoder prompt·video 그래프는 카탈로그에 등록되어 있지만 Windows 실기 검증이
끝날 때까지 release-ready로 표시하지 않는다.

## 최소 사용 예

```cpp
#include "vision_runtime_c.h"

dv_session_options options{
    sizeof(dv_session_options), DV_ABI_VERSION, "onnxruntime", 4};
dv_session* session = nullptr;
if (dv_create_session("model.json", &options, &session) != DV_STATUS_OK) {
    // 생성 실패 직후에도 dv_last_error(nullptr)를 읽을 수 있다.
    return 1;
}

dv_image_view image{
    sizeof(dv_image_view), DV_ABI_VERSION, pixels, width, height, 3, stride};
dv_result* result = nullptr;
const dv_status status = dv_infer(session, &image, &result);
if (status == DV_STATUS_OK) {
    // result->class_id, result->probabilities, result->total_ms 등을 사용한다.
    dv_release_result(result);
} else {
    // 오류 문자열은 세션이 소유한다.
    const char* message = dv_last_error(session);
}
dv_close_session(session);
```

MSVC에서는 `vision_runtime.dll`과 ONNX Runtime/OpenCV의 실행 DLL을 같은 배포
프로필로 묶는다. CMake의 `vision_runtime` target과 `install()` 규칙은 헤더와
라이브러리 설치를 함께 처리한다. Python, PyTorch, Docker는 실행 PC에 필요하지 않다.
Windows x64 Release와 실제 `.dvdeploy`의 JSON/그래프를 함께 검증한 뒤 배포한다.
