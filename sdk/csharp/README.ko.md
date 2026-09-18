# C# SDK

`VisionRuntime.csproj`는 C ABI DLL `vision_runtime`를 `SafeHandle`로 감싼다.
Python, PyTorch, Docker 없이 ONNX 배포 폴더의 JSON 설정을 열고 이미지 byte buffer를
동기 추론한다. C ABI가 현재 제공하는 태스크는 classification과 semantic segmentation이다.
Re-DETR, PatchCore, SAM2는 각 exporter와 그래프 계약이 구현된 뒤 같은 API의 결과 종류를 추가한다.

```csharp
using DeepVisionStudio;

using var session = VisionSession.Open("classify.json", numThreads: 4);
var result = session.InferClassification(bytes, width, height, channels);
Console.WriteLine($"{result.ClassName}: {result.Confidence:P2}");
```

`vision_runtime.dll`과 ONNX Runtime/OpenCV DLL은 같은 Windows x64 배포 폴더에 둔다.
`VisionSession`이 닫힌 뒤 결과 객체는 독립 managed 배열이므로 native 메모리를 참조하지 않는다.
