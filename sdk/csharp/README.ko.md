# C# SDK

`VisionRuntime.csproj`는 C ABI DLL `vision_runtime`를 `SafeHandle`로 감싼다.
Python, PyTorch, Docker 없이 ONNX 배포 폴더의 JSON 설정을 열고 이미지 byte buffer를
동기 추론한다. 현재 C ABI 결과 종류는 classification, semantic segmentation, generic
detection(``[1,N,5+C]``), Re-DETR v4 detection(``pred_boxes``/``pred_logits``),
reconstruction anomaly다. 고정 memory-bank를 포함해 export한 PatchCore는 score와
anomaly map을 anomaly 결과로 제공한다. SAM2는 `EncodeSam`으로 encoder 결과를 한 번 만들고
`SegmentSam`에 점·박스·이전 mask prompt를 넘기는 다중 그래프 C ABI를 사용한다. 영상 state와
실제 Hiera 변형의 Windows 검증 전에는 release-ready가 아니다.
`AutomaticSam`은 지정한 격자(축마다 1~32)의 positive point를 반복해 선택 mask를 합치는
이미지 자동 마스크 primitive다. upstream 영상 memory propagation API와는 별도다.
SAM2 prompt 좌표는 manifest의 `contracts.prompt_coordinate_space` 계약을 따른다. 기본
`resized_input` graph에서는 native SDK가 원본 이미지 좌표를 encoder 입력 크기에 맞춰 변환한다.

ResNet, ConvNeXt V1, DeepLab V3+, U-Net의 `builtin` manifest도 동일한
classification/segmentation API로 읽는다. 모델 가중치는 SDK에 포함하지 않는다.

배포할 때는 ONNX와 JSON을 개별 복사하지 않고 `.dvdeploy` 번들을 만든다.
`tools/build_deployment_bundle.py`가 그래프, SAM2 encoder/decoder, PatchCore bank와 external data,
전처리 설정, SHA-256 manifest를 묶으며 `VisionSession.Open`에는 번들 안의 설정 경로를 넘긴다.
디렉터리 번들을 바로 열 때는 `VisionSession.OpenBundle("model.dvdeploy")`를 사용한다.

```csharp
using DeepVisionStudio;

using var session = VisionSession.Open("classify.json", numThreads: 4);
var result = session.InferClassification(bytes, width, height, channels);
Console.WriteLine($"{result.ClassName}: {result.Confidence:P2}");
```

탐지와 이상 결과는 다음처럼 읽는다.

```csharp
var detections = session.InferDetection(bytes, width, height, channels).Detections;
var anomaly = session.InferAnomaly(bytes, width, height, channels);
Console.WriteLine($"{detections.Length} boxes, score={anomaly.Score}");
```

SAM2는 encoder context를 재사용해 여러 prompt를 처리한다.

```csharp
using var context = session.EncodeSam(bytes, width, height, channels);
var mask = session.SegmentSam(context, new SamPrompt(
    new[] { 120f, 80f }, new[] { 1 }));
```

설치 payload에서는 관리형 `VisionRuntime.dll`과 native `vision_runtime.dll` 및 ONNX Runtime/OpenCV DLL을
`sdk/`와 `sdk/native/`에 둔다. `VisionSession`은 `DEEP_VISION_NATIVE_RUNTIME_DIR`, 자신의 `native/`,
설치기의 `app/`, 실행 폴더 순서로 native DLL을 찾으므로 호출 프로그램의 현재 작업 폴더에 의존하지 않는다.
`VisionSession`이 닫힌 뒤 결과 객체는 독립 managed 배열이므로 native 메모리를 참조하지 않는다.
