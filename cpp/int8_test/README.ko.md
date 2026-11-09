# INT8 테스트 솔루션 (실험)

분할 모델(FP32 ONNX)을 INT8로 바꾸고, **설비와 같은 C++ SDK**로 FP32와 속도·결함 목록을 비교합니다.
SDK 코드는 바꾸지 않습니다. INT8 모델도 기존 `InitializeFromJson` → `Segment` 로 그대로 돕니다.

```
데스크톱에서 내보내기 ─▶ 모델.json + 모델.onnx (FP32)
        │
        ▼  python/quantize_onnx.py  (보정 영상 16장 정도, 768 타일 단위 보정)
모델_int8.json + 모델_int8.onnx (가중치 int8 · 활성값 uint8, QDQ)
        │
        ▼  int8_compare.exe  (이 솔루션)
영상별: FP32 ms · INT8 ms · 배속 · 마스크 픽셀 일치 · 결함 짝(같은 클래스 + 상자 IoU ≥ 0.3)
```

## 1. INT8 모델 만들기 (Python, 데스크톱 앱 설치 폴더의 python 사용 가능)

```bat
python python\quantize_onnx.py 모델.json 학습영상폴더 --count 16 --compare 검증영상폴더
```

- 보정 영상: 정상과 불량을 섞어서 넣습니다. 학습 영상을 그대로 써도 됩니다.
- 결과: `모델_int8.json`, `모델_int8.onnx`, `모델_int8.report.json`(픽셀 일치율, 클래스별 IoU, 시간)
- 6~11세대처럼 VNNI가 없는 구형 i7에서 정확도가 떨어지면 `--reduce-range` 를 붙여 다시 만듭니다.

## 2. 솔루션 만들기 (Windows, Visual Studio 2022)

```bat
cd cpp\int8_test
make_solution.bat C:\onnxruntime-win-x64-1.20.1 C:\opencv\build C:\vcpkg
```

`build\Int8Test.sln` 을 열고 **Release** 로 빌드합니다. 시작 프로젝트는 `int8_compare` 입니다.

- `onnxruntime.dll` 은 빌드 후 실행 파일 옆에 자동으로 복사됩니다.
- OpenCV DLL(`opencv_world*.dll`)은 PATH에 있거나 실행 파일 옆에 있어야 합니다.

## 3. 비교 실행

```bat
int8_compare.exe 모델.json 모델_int8.json 검증영상폴더 8 5 20
                 └ FP32     └ INT8          └ 영상/폴더   │ │ └ 비교 최소 면적(px)
                                                           │ └ 반복 횟수(중앙값)
                                                           └ 스레드 (0 = 자동 = 물리 코어 수)
```

- 결함 목록은 모델에 **판정 기준**이 있어야 나옵니다. 데스크톱에서 '오검출 0 기준 찾기'로 저장한 뒤 내보내세요.
  기준이 없으면 마스크 픽셀 일치율만 비교합니다.
- 종료 코드: 0 = 모든 결함 짝이 맞음, 1 = 한쪽에만 있는 결함이 있음 (출력 목록의 영상을 직접 확인).

## 참고 측정 (합성 트레이 3504×1504, ResNet18 U-Net, 4스레드)

| | FP32 | INT8 |
|---|---|---|
| C++ Segment | 2656 ms | **1400 ms (1.90배)** |
| 마스크 픽셀 일치 | 기준 | 0.99999 |
| 결함 (최소 면적 20) | 36개 | 36개 모두 짝 맞음 |

- 차이가 난 것은 면적 1~11 px, 점수 0.4~0.6의 경계 조각뿐입니다. 판정 기준의 최소 면적으로 걸러집니다.
- 명령어 세트를 i7 수준으로 제한해 OpenVINO로 잰 INT8 배속: AVX2 약 1.9배, AVX-VNNI(12세대 이상) 약 3.5배.
  실제 배속은 이 솔루션을 설비 PC에서 직접 돌려 확인하세요.
