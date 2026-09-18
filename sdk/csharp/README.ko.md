# C# SDK

`VisionRuntime.csproj`는 C ABI DLL `vision_runtime`를 `SafeHandle`로 감싼다.
Python, PyTorch, Docker 없이 ONNX 배포 폴더의 JSON 설정을 열고 이미지 byte buffer를
동기 추론한다. 현재 C ABI 결과 종류는 classification, semantic segmentation, generic
detection(``[1,N,5+C]``), Re-DETR v4 detection(``pred_boxes``/``pred_logits``),
reconstruction anomaly다. 고정 memory-bank를 포함해 export한 PatchCore는 score와
anomaly map을 anomaly 결과로 제공한다. SAM2는 `EncodeSam`으로 encoder 결과를 한 번 만들고
`SegmentSam`에 점·박스·이전 mask prompt를 넘기는 다중 그래프 C ABI를 사용한다. 영상 state와
실제 Hiera 변형의 Windows 검증 전에는 release-ready가 아니다.

ResNet, ConvNeXt V1, DeepLab V3+, U-Net의 `builtin` manifest도 동일한
classification/segmentation API로 읽는다. 모델 가중치는 SDK에 포함하지 않는다.

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

`vision_runtime.dll`과 ONNX Runtime/OpenCV DLL은 같은 Windows x64 배포 폴더에 둔다.
`VisionSession`이 닫힌 뒤 결과 객체는 독립 managed 배열이므로 native 메모리를 참조하지 않는다.
