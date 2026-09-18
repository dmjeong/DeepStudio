# 21.7: 선택 레이어 관찰과 C++ 전처리 일치

20.7에서 기능 업데이트 규칙 +1.0을 적용한 버전입니다.

## 변경 사항

- EfficientNet B0/B1과 Custom CSP 학습에 선택 레이어 관찰을 연결했습니다. 데스크톱과 웹의 학습 설정에서 활성화하고 레이어 이름이나 `*`, `?` 패턴을 쉼표로 지정합니다. 추천 패턴 버튼으로 모델별 기본값을 채울 수 있습니다.
- 초기 1~10개 배치, 최대 128개 레이어의 입력/출력 shape, 출력 min/max/mean, 유한성, 역전파 Gradient 통계를 기록합니다. AMP Gradient는 배율을 제거합니다. 관찰을 끝내도 전체 학습은 계속됩니다. 동결 레이어처럼 Gradient가 없으면 N/A로 표시합니다.
- 관찰 파일은 해당 실행의 `runs/<run_id>/layer_debug.json`에 저장합니다. 원본 이미지나 활성값 배열은 저장하지 않습니다. 화면과 프로젝트 기록에는 해당 Run의 최신 스냅샷을 연결합니다. 검증 전과 오류/중단 시 훅을 해제합니다.
- EfficientNet 재개 시 가중치, optimizer, 스케줄러와 난수 복원은 유지하고 관찰 설정만 현재 요청을 적용합니다. 관찰 활성화는 배치 수를 줄이는 시험 학습으로 취급하지 않습니다.
- 학습 엔진, 모델 목록, 지원 증강, 채널 및 레이어 관찰 지원 여부를 `training_capabilities`에서 정의합니다. API, 데스크톱과 웹이 이를 사용합니다. 이전 UI의 수직 반전과 Mixup은 학습에 전달되지 않았으므로 현재 지원 옵션에서 제외하고, 저장된 값이 있으면 명시적으로 0으로 변경하도록 안내합니다.
- Python의 Pillow 8-bit bilinear 축소/확대, 안티앨리어싱, RGB→L 변환과 일치하는 C++ 전처리를 추가했습니다. 이후 정규화는 float32 NCHW입니다. 기존 Python 학습 변환과 가중치는 변경하지 않습니다.
- ONNX 배포 JSON의 지원 C++ 모델은 schema 3을 사용하며 `resize_implementation=pillow_u8_bilinear`, `antialias=true`를 명시합니다. 새 JSON 사용 시 C++ 라이브러리도 다시 빌드해야 합니다. 구형 C++가 새 계약을 잘못 읽는 것을 방지합니다. 새 C++는 기존 schema 1/2도 읽습니다.

## 사용

학습 화면에서 EfficientNet B0/B1을 선택한 뒤 **선택 레이어 관찰**을 켜고 추천 패턴과 관찰 배치 수를 지정합니다. 데스크톱의 **레이어 관찰** 탭 또는 웹 학습 결과 아래에서 통계를 확인합니다. 상세 배치별 기록은 실행 폴더의 JSON에서 확인할 수 있습니다. 공식 이전 모델 및 PatchCore 워커에는 이 관찰 기능을 적용하지 않습니다.

C++에서는 기존 `InitializeFromJson("model.json")`과 `ClassifyFile("image.png")` 사용법을 유지합니다. 새 `pillow_preprocess.h`를 포함한 라이브러리를 다시 빌드합니다. ONNX 그래프는 기존과 동일하게 정규화된 입력을 받으며, 이미지 전처리는 C++에서 수행합니다.

## 검증 및 한계

로컬 회귀 검사: Python unittest 86개 실행, 85개 통과, PyTorch 필요 검사 1개 건너뜀. 웹 순수 TypeScript 검사 11개 통과. Python 구문 컴파일과 git diff 공백 검사 통과.

C++17로 실제 전처리 함수를 컴파일했습니다. 무작위 RGB/L 리사이즈 100건에서 Pillow 출력과 바이트 단위로 일치했고, 64,507개 RGB 픽셀의 흑백 변환, B0/B1 입력 크기의 float32 NCHW 정규화 및 단일 픽셀/극단값도 비교했습니다. 정규화 비교 허용 오차는 절대값 1e-6입니다. 비교 대상은 동일하게 디코딩된 픽셀입니다. JPEG 등 파일 디코더 차이까지 동일하다고 보장하지 않습니다.

실제 PyTorch 역전파에서 관찰 전후 가중치와 RNG가 같은지 확인하는 검사, EfficientNet 재개 검사, Qt 화면 검사, C++ ONNX Runtime smoke 검사를 CI에 포함했습니다. 이 환경에는 PyTorch, PySide6, OpenCV/ONNX Runtime 개발 의존성이 없으므로 해당 전체 검사는 로컬에서 실행하지 못했습니다. 프런트엔드 전체 빌드도 패키지 다운로드 403 문제로 미검증입니다. 실제 학습 정확도나 공식 모델 대비 성능 검증이 완료됐다는 의미는 아닙니다.

직전 CI는 runner 배정 전 실패했습니다. 이번 커밋도 CI 통과 여부를 별도로 확인하며, 통과 전 자동 릴리스 태그는 생성하지 않습니다.

전처리 계약 참고: [Pillow Resample.c](https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c). 관련 라이선스는 `cpp/licenses/Pillow.txt`에 포함했습니다.
