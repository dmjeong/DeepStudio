# C# SDK

`VisionRuntime.csproj`는 C ABI DLL `vision_runtime`를 `SafeHandle`로 감싼다.
Python, PyTorch, Docker 없이 ONNX 배포 폴더의 JSON 설정을 열고 이미지 byte buffer를
동기 추론한다. 현재 C ABI 결과 종류는 classification, semantic segmentation, generic
detection(``[1,N,5+C]``), reconstruction anomaly다. 고정 memory-bank를 포함해
export한 PatchCore는 score와 anomaly map을 anomaly 결과로 제공한다. Re-DETR v4의
탐지 특수 출력과 SAM2 encoder/decoder prompt/video 계약은 별도 그래프로 관리하며
Windows 검증 전에는 release-ready가 아니다.

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

`vision_runtime.dll`과 ONNX Runtime/OpenCV DLL은 같은 Windows x64 배포 폴더에 둔다.
`VisionSession`이 닫힌 뒤 결과 객체는 독립 managed 배열이므로 native 메모리를 참조하지 않는다.
