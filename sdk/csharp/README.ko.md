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

## Windows self-contained 샘플

저장소의 `sample/VisionRuntime.Sample.csproj`는 C# SDK와 native C ABI를 호출하는
`win-x64` 단일 실행 파일 예제다. 예제는 별도 이미지 라이브러리 없이 8-bit raw
BGR/gray 파일을 읽으므로, 배포 번들의 전처리·모델 검증 경로를 그대로 확인할 수 있다.

```powershell
dotnet publish sdk/csharp/sample/VisionRuntime.Sample.csproj -c Release -r win-x64 --self-contained true
```

publish 결과에 `vision_runtime.dll`, ONNX Runtime/OpenCV DLL과 `.dvdeploy` 폴더를
함께 배치한 뒤 다음처럼 실행한다.

```powershell
VisionRuntime.Sample.exe C:\models\classify.dvdeploy C:\images\frame.bgr 224 224 3
```

관리형 .NET 런타임은 샘플에 포함되지만, native DLL과 모델 번들은 별도 payload로
검증·해시해 설치한다. PNG/JPEG 입력은 학습툴의 이미지 디코더에서 raw buffer로
변환한 뒤 SDK에 전달한다.

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
