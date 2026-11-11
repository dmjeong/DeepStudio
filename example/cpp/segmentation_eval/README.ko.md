# Segmentation ONNX C++ 평가 예제

명령행 인자 없이 EXE를 실행합니다. 모델은 한 번 불러오고 각 영상에서 워밍업 후 측정합니다.
기존 프로그램에 넣을 핵심 호출은 다음과 같습니다.

```cpp
VisionInference model;  // 초기화 후 유지
if (!model.InitializeFromJson("C:/inspection/model.json", "onnxruntime", 4))
    throw std::runtime_error("모델 로드 실패");
model.Segment(first_image);  // 실제 입력 크기로 시작 시 워밍업
auto result = model.Segment(image);  // image: OpenCV BGR 또는 흑백, JSON과 같은 입력 형식
// result.mask: 원본 영상 크기의 클래스 ID, 0=배경
// result.valid_mask: 실제 검사한 영역. 중앙 크롭 바깥은 0.
```

## Release에서 구간별 속도 확인 (11.82 이상)

SDK와 호출 프로젝트 모두 **Release / x64로 다시 빌드**하세요. `SegmentResult`에 새 필드가 추가되어 이전 LIB/DLL과 새 헤더만 섞으면 안 됩니다. 모델 재학습·재내보내기는 필요 없습니다.

Windows GUI/MFC 프로그램에서는 다음처럼 호출합니다. Visual Studio에서 **Release / x64 → F5**로 실행하고 **출력 창(Ctrl+Alt+O) → 디버그**를 선택하면 로그가 보입니다. `TRACE`는 Release에서 사라질 수 있으므로 여기서는 `OutputDebugStringA`를 씁니다. 디버거 없이 EXE를 실행하면 이 로그를 표시할 창이 없으므로 콘솔 또는 파일 로그로 바꾸세요.

```cpp
#include <cstdio>
#include <windows.h>

// 초기화 뒤 최초 1회만, 실제 입력 크기로 예열합니다.
// model.Segment(first_image);

const auto r = model.Segment(image);
char message[512];
std::snprintf(message, sizeof(message),
    "[SEG ms] pre=%.2f model=%.2f validate=%.2f decode=%.2f "
    "rules=%.2f render=%.2f total=%.2f\n",
    r.preprocess_ms, r.model_ms, r.validation_ms, r.decode_ms,
    r.rules_ms, r.render_ms, r.inference_ms);
OutputDebugStringA(message); // 측정이 끝난 다음 출력
```

| 표시 | 포함 범위 |
|---|---|
| pre | 크롭 영역 선택 + 색 변환 + 리사이즈 + 정규화 + 입력 텐서 준비 |
| model | ONNX Runtime `Session::Run` |
| validate | 출력 크기·형식·NaN/Inf 검사 |
| decode | 픽셀별 클래스 선택·확률 계산 |
| rules | 연결된 덩어리 계산·면적/점수 기준 판정·좌표 변환 |
| render | 마스크 리사이즈·복사·색상 이미지 생성·픽셀 집계 |
| total | 위 6개 구간 합계 |

`postprocess_ms`는 validate+decode+rules+render 소계입니다. 전체에 다시 더하지 마세요. `total`은 파일 읽기·화면 표시·파일 저장을 포함하지 않습니다. 함수 반환 시 내부 버퍼 해제나 호출자 결과 대입도 제외하므로 전체 호출의 벽시계 시간과 소폭 차이가 날 수 있습니다.

이 평가 예제를 실행하면 콘솔에 구간별 시간이 자동 출력됩니다. `report.json`의 이미지별 `timing_samples_ms`는 워밍업을 제외한 매 반복의 기록입니다. `timing_sample_ms`는 전체 호출 중앙값에 가장 가까운 실제 1회의 구간이며, 각 구간의 독립적인 중앙값을 조합한 값이 아닙니다. `sdk_total_ms`는 위 `total`, `call_total_ms`는 함수 반환·기존 결과 해제/대입까지 포함한 호출부 시간입니다. 최종 성능은 워밍업 뒤 20회 이상 반복하고 중앙값과 p95로 비교하세요.

## Windows에서 실행

1. 앱에서 학습한 모델을 ONNX로 내보냅니다. 함께 생성된 `.json`과 `.onnx`를 모두 가져옵니다. 패치 학습 모델은 실제 판정 영역 크기로 export합니다.
2. 이 저장소 전체를 내려받습니다. 이 예제는 `../../../cpp`의 SDK와 저장소에 포함된 nlohmann 헤더를 사용합니다. 설치본에서는 `Examples/cpp/vision-runtime`을 자동으로 찾습니다.
3. `paths.example.bat`을 `paths.bat`으로 복사합니다. 이미 사용 중인 ONNX Runtime 폴더(include/lib가 있는 곳)와 OpenCV 폴더(OpenCVConfig.cmake가 있는 곳)를 적습니다.
4. CMake 및 Visual Studio C++ 개발 도구가 설치된 PC에서 `build.bat`을 실행합니다. `build/Release/segmentation_eval.exe`가 생성됩니다. VS2017은 15.9/v141 이상, C++17이며 SDK의 기존 ORT 헤더 호환 처리를 사용합니다. Windows 실행 검증은 별도로 필요합니다.
5. 아래처럼 EXE 옆에 파일을 놓고 `evaluation.json`을 수정합니다. JSON에는 경로 구분자로 `/`를 사용합니다. JSON 상대 경로는 항상 JSON 위치 기준입니다.

```text
Release/
  segmentation_eval.exe
  onnxruntime.dll            ← 빌드에서 같은 SDK의 DLL 복사
  evaluation.json
  model/
    model.json              ← 앱이 생성한 파일 그대로, 실제 파일명에 맞춰 설정
    model.onnx
  images/
    tray001.png
  masks/
    tray001.png
```

공식 OpenCV SDK의 DLL은 빌드에서 EXE 옆으로 복사합니다. 별도 의존 DLL이 필요하면 같은 실행 폴더에 두거나 `paths.bat`의 PATH에 해당 폴더를 지정하고 `run.bat`을 실행합니다. Debug/Release, x64, 헤더·LIB·DLL은 일치해야 합니다.

## 정답 마스크

- 컬러로 칠한 화면 캡처가 아니라 **원본 이미지와 같은 크기의 8비트 단일 채널 PNG**입니다. 픽셀 값은 export JSON의 클래스 번호입니다(0=배경, 1=첫 결함, 2=둘째 결함, 255=평가 제외).
- 앱 프로젝트의 `masks/test`를 사용합니다. 이미지 이름과 같은 이름의 PNG로 맞춥니다(`abc.jpg` → `abc.png`). 하위 폴더도 동일하게 유지합니다.
- 팔레트 PNG는 OpenCV에서 색상으로 해석될 수 있으므로 클래스 번호를 보존해 L 모드 PNG로 저장하세요. RGB 마스크나 크기가 다른 마스크는 실패로 처리합니다.
- 정답 누락을 정상으로 간주하지 않습니다. 정상 시험 영상은 명시적으로 전부 0인 마스크를 만드세요.
- 중앙 크롭 바깥은 평가에서 제외합니다. 정답을 모델 입력 크기로 축소하지 않습니다.

## 생성 결과

새 `evaluation_output/` 폴더에 `report.json`, 클래스별 `classes.csv`, 예측 ID PNG(`predictions/`), 원본 위 전경 표시(`overlays/`)를 저장합니다. 기존 결과 폴더는 덮어쓰지 않습니다.

- IoU/Dice는 영상별 평균이 아니라 전체 영상 TP/FP/FN 합으로 계산합니다. 배경 포함/제외 mIoU를 따로 기록합니다.
- 정답·예측 모두 없는 클래스는 null/빈칸으로 표시하고 평균에서 제외합니다. 정답이 없어도 오검출이 있으면 IoU=0으로 포함합니다.
- JSON의 `defect_rules`가 있으면 **그 기준 적용 후 마스크**를 평가합니다. 해당 기준을 정한 검증셋과 별개의 시험셋을 사용하세요.
- `p50_ms`/`p95_ms`는 전처리+ORT+후처리 전체 호출 시간입니다. 모델 로드·파일 읽기·워밍업·지표 계산·파일 저장은 제외합니다. 5회 p95는 참고치이며 성능 측정은 `repeat`를 늘려 진행하세요.
- 종료 0은 평가가 완료됐다는 뜻입니다. 합격 정확도나 FP32↔INT8 동등성을 보장하지 않습니다. 오류는 종료 2이며 `report.json`을 만들지 않습니다.

이 예제는 U-Net Lite/ResNet18/MobileNetV2, DeepLab, DINOv2 등 **단일 semantic logits ONNX**용입니다. SAM2의 encoder/decoder 프롬프트 배포는 `cpp/include/sam2_inference.h`의 별도 API를 사용합니다. 암호화 파일은 일반 `.onnx`처럼 읽을 수 없으며 SDK의 키 전달 초기화 API가 필요합니다. 본 예제는 평문 export 기준입니다.

Linux/macOS: `cmake -S . -B build -DONNXRUNTIME_ROOT=... -DOpenCV_DIR=...` 후 `cmake --build build --target segmentation_eval -j4`.
