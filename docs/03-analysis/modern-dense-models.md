# 15.88 구현·검증 보고서

## 구현 범위

기본 모델에 분할 4종(EfficientViT B0/B1, SCTNet-S, EoMT DINOv2-S)과 이상 탐지
Dinomaly DINOv2-S/14를 추가했다. 분할 모델에는 원본 해상도의 세부 특징 경로를 붙였다.
공식 구현과 다른 부분은 [출처·변형 기록](../../licenses/modern-dense-models.md)에 명시했다.
SegMAN은 CUDA selective-scan/NATTEN 의존 때문에 이번 CPU 배포 대상에서 제외했다.

| 기능 | 구현과 확인 |
|---|---|
| 사전학습 | 실제 공식 EfficientViT B0/B1, SCTNet-S, DINOv2-S 파일을 로드하고 SHA256 및 전체 백본 텐서 검사 |
| 학습 | RGB/BW8 forward·backward, 분할/정상 재구성 학습, 저장·복원·재개 |
| 증강·패치 | 수평 반전·회전·색상 변화, 패치 배율, 중앙 크롭, 학습 크기를 유지하는 겹침 타일 |
| 분할 증류 | 교사와 학생에게 같은 증강 입력, 255 제외, 교사 고정, 클래스 순서와 해시 검사, 학생만 배포 |
| 화면 | 모델 선택·학습 모드·증강·패치·증류 설정 저장/복원, 에폭·손실·LR·Best 갱신 |
| FP32 | 원본 PyTorch 대비 ONNX 출력 검증. 분할 448, Dinomaly 224의 기본 크기에서도 검사 |
| INT8 | 실제 보정 영상의 고정 타일로 QDQ 변환, FP32 대비 픽셀/IoU 또는 이상 점수 차이 기록 |
| C++ | C++17 Release, schema 8 타일 전처리·스티칭·원본 좌표 복원, FP32/INT8 Python 대조 |
| 암호화 | FP32/INT8 모두 평문과 복호화 후 C++ 출력 일치. 내보내기 서비스의 INT8→암호화 경로도 실행 |
| 원본 비교 | 흑백 입력 모델에서도 컬러 원본 캐시 보존, 배경을 포함한 원본 좌표 유지 |
| 배포 | 기본 가중치 준비·업데이트 배치, PyInstaller 수집, 설치 페이로드·모델 카탈로그, 인자 없는 C++ 예제 |

## 검사 환경과 결과

Apple M4/macOS, Python 3.12, PyTorch 2.14.1, timm 1.0.31, ONNX Runtime 1.29.0,
Python OpenCV 5.0.0 / native OpenCV 4.12.0, AppleClang C++17 Release에서 실행했다.
실제 공식 사전학습 백본과 작은 합성 정답을 사용했다. 사용자 영상·가중치는 사용하지 않았다.

| 실행 묶음 | 결과 |
|---|---|
| 신규 모델 + 관련 기존 회귀 488개 | 481 통과, C++ 평가기 경로가 미설정된 7개 보류; 172.92초, 정상 종료 |
| C++ 평가기를 빌드한 뒤 평가·가중치 준비 재검사 | 22 통과; 보류했던 7개 전부 통과; 12.10초 |
| 학습 UI·재개·패치·배포·기존 EfficientNet export 등 추가 회귀 | 144 통과, Windows 배치 실행 1개 플랫폼 사유로 제외; 49.81초 |
| Dinomaly 실제 재개/LR/CSV + 다중 타일·크롭·배율·인자 없는 예제 | 3 통과; 9.06초 |
| 내보내기 서비스의 EfficientViT INT8 암호화 → C++ 재검사 | 1 통과; 4.15초 |
| native CTest | 전처리·다중 스레드 분류·배포·암호화·C ABI 7개 통과, 분할 평가 지표 1개 통과 |
| 정적 검사 | CI 범위 Ruff, 새 모듈 Ruff, 변경 Python 40개 구문 검사, YAML 파싱, git diff 공백 검사 통과 |

중복 실행을 제외하면 Python 검사 **634개 통과, Windows 전용 1개 제외**다.
신규 모델 검사 파일은 30개이며 모두 실행했다. ONNX legacy exporter/상수 폴딩 경고와
Qt 폰트/폐기 예정 API 경고는 남아 있지만 실패나 비정상 종료는 없었다.

검사 로그는 개발 환경의 `/tmp/dvs-modern-models/`에 보관했다.
재실행 시 공식 가중치를 먼저 준비하고 `DVS_RUN_MODERN_INTEGRATION=1`,
`DVS_MODERN_INPUT_SIZE=448`, `DVS_DINOMALY_INPUT_SIZE=224`,
`SEGMENT_PROBE`, `ANOMALY_PROBE`를 빌드 결과 경로로 설정한 다음
`python -m pytest tests/test_modern_dense_models.py -q -o tmp_path_retention_policy=failed`를 실행한다.
`dense_model_example`도 probe와 같은 폴더에 빌드하거나 `DENSE_EXAMPLE`로 지정한다.
Linux/Windows CI에 실제 가중치·FP32·INT8·암호화·C++ 검사를 연결했다.

## 검사 중 수정한 결함

1. **재개와 Best 파일명:** 데스크톱이 `best.pt`를 지표 포함 이름으로 바꾼 뒤 Dinomaly 재개가 실패했다.
   `best_selection.json`의 실제 파일을 복원하도록 공통화했다. 기존 builtin 모델도 `last.pt`를
   Best로 잘못 대체하지 않도록 수정하고, 더 나빠진 다음 에폭에서도 이전 Best가 유지되는지 검사했다.
2. **원본 컬러 손실:** 흑백 모델 입력을 원본 비교에 재사용하던 경로를 고쳤다.
   컬러 원본을 한 번 읽어 보존하고 모델용 흑백 배열을 별도로 만든다.
3. **마스크 확대의 한 줄 차이:** Python Pillow와 native OpenCV의 최근접 보간은 정확한 중간 좌표에서
   다른 픽셀을 고를 수 있었다. C++ 마스크·영역 면적 계산을 데스크톱의 좌표 계산 순서와 맞췄다.
4. **패치 배율 반올림:** `.5` 경계에서 C++와 Python의 크기 반올림을 일치시켰다.
   여러 타일, 홀수 ROI, 중앙 크롭, 50% 배율, 원본 좌표를 직접 대조했다.
5. **ViT 패딩:** 14의 배수가 아닌 입력에서 패딩 영역까지 원본으로 축소하지 않도록
   EoMT/Dinomaly 출력을 패딩 크기로 확대한 뒤 실제 영상 크기로 잘랐다.
6. **화면 지표:** Dinomaly에 PatchCore 지표가 표시되지 않도록 정상 검증 `val_loss`와 LR 이벤트를 연결했다.
7. **공개 예제:** runtime을 하위 프로젝트로 추가할 때 CMake 타깃이 중복되던 조건을 고쳤다.
   한글 Windows 경로는 filesystem 기반 I/O를 사용하고, 옵션 TIFF 코덱이 없어도 원시 이상 점수를 저장한다.
8. **검사 프로세스 종료:** ORT 1.29의 macOS telemetry 업로더가 종료 중 충돌한 사례를 확인했다.
   ORT import 전에 공식 `ORT_DISABLE_TELEMETRY=1` 설정을 적용했다.

## 남은 실측 범위

- 이 결과는 학습·배포 연결과 출력 계약의 검증이다. 실제 crack/FM/shift/double/flip 검출력이 개선됐다는 결과는 아니다.
- INT8은 `int8_compared`로 표시한다. 보정 영상 비교가 독립 시험 정확도 검증을 대신하지 않는다.
- Windows/i7의 전체 ROI 지연시간, GPU 학습 속도, Windows 설치 EXE 실행은 이 macOS 환경에서 실측하지 않았다.
  CI Windows 검증 결과와 실제 검사 PC의 Release/x64 측정을 별도로 확인해야 한다.
- Dinomaly는 batch 1로 ONNX 배포한다. 고정 인코더에는 사전학습이 필요하며 재구성기는 현장 정상 영상으로 학습한다.
- 새 모델의 이름은 Studio 변형을 뜻한다. 공식 논문 수치나 기존 모델보다 빠른 속도를 보장하지 않는다.

사용 절차는 [모델 안내](../MODERN_DENSE_MODELS.md), C++ 호출은
[dense_models 예제](../../example/cpp/dense_models/README.ko.md)를 따른다.
