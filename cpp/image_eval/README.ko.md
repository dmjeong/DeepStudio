# 영상 평가 예제 (C++)

학습한 모델을 ONNX로 내보내고, **설비와 같은 C++ SDK**에 영상을 넣어 판정과 시간을 봅니다.
분할 · 이상 탐지(PatchCore 포함) · 검출 · 분류 모델이 같은 프로그램으로 돕니다.

```
학습 결과(best.pt) ─▶ ① ONNX 내보내기 ─▶ 모델.json + 모델.onnx
                                              │
                        ② 솔루션 빌드 ─▶ image_eval.exe
                                              │
        ③ image_eval 모델.json 영상폴더 ─▶ 영상별 판정 · ms / 묶음별 판정 수 / 결과 그림 / eval.csv
```

## 1. ONNX로 내보내기

데스크톱 앱의 내보내기 화면을 쓰거나, 명령으로 내보냅니다. `모델.onnx` 옆에 `모델.json`이 함께 생깁니다.

```bat
rem 분할 (패치 학습 모델은 판정 영역 크기를 준다. 학습 때 기록돼 있으면 생략)
python python\export_onnx.py --checkpoint best.pt --output seg.onnx --region 3100x1500

rem PatchCore (임계값을 보정한 모델만 내보낼 수 있다)
python python\export_onnx.py --checkpoint patchcore.pt --output patchcore.onnx
```

- 분할 결함 목록은 **판정 기준**이 있어야 나옵니다. 데스크톱에서 '오검출 0 기준 찾기'로 저장한 뒤 내보내세요.
- 이전 버전으로 내보낸 PatchCore JSON은 `input_name`이 없어 C++에서 열리지 않습니다. 다시 내보내거나
  JSON에 `"input_name": "input_image"` 한 줄을 넣으세요.

## 2. 솔루션 만들기 (Windows, Visual Studio 2022)

```bat
cd cpp\image_eval
make_solution.bat C:\onnxruntime-win-x64-1.29.0 C:\opencv\build C:\vcpkg
```

`build\ImageEval.sln`을 열고 **Release**로 빌드합니다. 시작 프로젝트는 `image_eval`입니다.

- `onnxruntime.dll`은 빌드 후 실행 파일 옆에 자동으로 복사됩니다.
- OpenCV DLL(`opencv_world*.dll`)은 PATH에 있거나 실행 파일 옆에 있어야 합니다.

## 3. 실행

```bat
image_eval.exe seg.json D:\images --roi 240,300,3100,1500 --threads 8
image_eval.exe patchcore.json D:\images
```

| 인자 | 뜻 |
| --- | --- |
| `모델.json` | 내보낸 모델 설정. 같은 폴더의 `.onnx`를 읽습니다 |
| `영상 또는 폴더` | 폴더면 하위 폴더까지 전부. 첫 하위 폴더 이름(`ok`, `ng` 등)이 묶음이 됩니다 |
| `--roi x,y,w,h` | 영상에서 그 사각형만 잘라 모델에 넣습니다. 결함 좌표와 결과 그림은 잘라 낸 영역 기준입니다 |
| `--threads N` | 스레드 수. 0(기본)은 자동 |
| `--repeat N` | 영상마다 반복 횟수. 시간은 중앙값 (기본 1) |
| `--output 폴더` | 결과 그림과 `eval.csv`를 두는 폴더 (기본 `eval_result`) |
| `--no-save` | 결과를 저장하지 않습니다 |

**인자 없이 실행**하면 `image_eval.cpp` 맨 위의 기본값(`kDefaultModel`, `kDefaultImages`)을 씁니다.
Visual Studio에서 그 두 경로만 고쳐 바로 실행할 수 있습니다.

출력 예 (분할, 3580×2100 영상에서 판정 영역 3100×1500):

```
모델 seg.json · 태스크 segment · 스레드 4 · 반복 3 · 영상 2장
       ms  판정  영상 · 요약
    144.7  NG      ng/tray_ng.png · 결함 12개 | crack 면적 640 점수 0.99 | fm 면적 680 점수 0.99 | ...
    131.1  OK      ok/tray_ok.png · 결함 0개
시간(ms): 평균 137.9 · 중앙값 144.7 · 최대 144.7  (전처리 + 모델 + 후처리, 영상 읽기 제외)
판정:
  ng            NG 1
  ok            OK 1
```

- **판정**: 분할은 결함이 하나라도 남으면 NG, 이상 탐지는 점수가 임계값 이상이면 NG, 분류는 클래스 이름입니다.
- **묶음별 판정 수**: `ok` 폴더에 NG가 있으면 오검출, `ng` 폴더에 OK가 있으면 미검출입니다.
- **결과 그림**: 분할은 결함 색과 빨간 상자, 이상 탐지는 히트맵(색 범위 0 ~ 임계값의 2배)을 원본에 겹칩니다.
- **시간**: SDK가 잰 전체 시간입니다. 모델만 잰 시간(`tools/segment_cpu_benchmark.py`)보다 전처리·후처리만큼 깁니다.
- 종료 코드: 0 = 모든 영상 실행, 1 = 일부 영상 오류, 2 = 사용법·모델 오류.

## 내 코드에 넣기

```cpp
#include "vision_inference.h"

VisionInference engine;                                   // 프로그램 시작 때 한 번 만들고 계속 쓴다
if (!engine.InitializeFromJson("seg.json", "", 8)) return; // 모델 로드 (스레드 8개)

cv::Mat roi = image(cv::Rect(240, 300, 3100, 1500));      // 판정 영역만 (복사 없음)
SegmentResult result = engine.Segment(roi);               // 이상 탐지는 engine.Anomaly(image)
bool ng = !result.defects.empty();                        // 판정 기준을 통과한 결함이 있으면 NG
for (const auto& defect : result.defects) {
    // defect.class_id · defect.box(판정 영역 좌표) · defect.area(px) · defect.score
}
```
