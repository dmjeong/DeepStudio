# 11.72 — C++ 전처리·출력 검사 지연 수정

## 원인과 변경

사용자 측정은 Debug 빌드였다. 전체 영역 크롭은 `cv::Mat`의 ROI 참조이지만 이후 색 변환·정규화·텐서 생성은 실제 데이터를 처리한다. `num_threads`는 ONNX Runtime 모델 실행에 전달되므로 수동 전처리 반복문과 출력 검사의 병렬 스레드 수가 아니다.

1. `ValidateOutput`은 원소마다 `GetElementCount()`를 호출했다. 이 함수는 ORT C API를 호출한다. `[1,5,1504,3520]`에서 26,470,401번이던 크기 조회를 한 번으로 줄였다. Re-DETR 점수·박스 검사에도 적용했다.
2. 전체 영역 8비트 전처리의 정규화 식을 채널별 256개 값에 미리 계산하고 재사용한다. RGB/gray 입력을 채널별 연속된 NCHW 버퍼에 쓴다. 색 변환·크기·보간·정규화 뒤 0 패딩은 유지한다.
3. 모든 출력 원소의 NaN/Inf 검사와 shape/type 검사는 유지한다. 검사 샘플링, fast-math, 해상도 축소, 정밀도 변경을 적용하지 않았다.

## 측정

Apple M4 / macOS / Clang C++17 / ORT 1.29.0 / OpenCV 4.12.0. 첫 2회 제외 후 13회 중앙값, 구/신 함수를 번갈아 실행했다. Windows/MSVC 측정값이 아니다.

| 항목 | Release 기존 | Release 수정 | 비최적화 기존 | 비최적화 수정 |
|---|---:|---:|---:|---:|
| 출력 검사 `[1,5,1504,3520]` | 99.150ms | 6.116ms | 182.075ms | 24.195ms |
| 전체 영역 전처리 | 8.908ms | 6.075ms | 125.285ms | 89.433ms |

- 출력 검사: 실제 ORT 텐서, 동일한 검사 함수와 데이터. 모델 실행은 포함하지 않는다.
- 전처리: 합성 BGR 3800×2100의 비연속 3500×1500 ROI → RGB 정규화 → 3520×1504 패딩. OpenCV 1스레드. 색 변환·메모리 할당 포함. 전체 텐서를 `memcmp`로 비교해 동일함을 확인했다.
- Release는 `-O3 -DNDEBUG`, 비최적화는 `-O0 -g`. Visual Studio Debug의 STL 검사 등과 동일한 환경이라고 해석하면 안 된다.
- 모델의 600ms 계산 자체를 줄인 수정이 아니다. 사용자 PC의 전체 시간은 Release/x64에서 다시 측정해야 한다.
- 로컬 재현 자료: `runs/output-validation-20261110/benchmark.cpp`, `results.json`, `preprocess_benchmark.cpp`, `preprocess-results.json` (runs 디렉터리는 Git 제외).

## 회귀 검증

- C++ CTest 7/7 통과: 전처리 계약, 배포, 암호화, C ABI, 분류 워커·동시성 및 fixture 생성.
- 실제 Segment 출력에 NaN/+Inf/-Inf를 첫·중간·마지막 위치에 주입한 9개 사례 모두 거부.
- Re-DETR 점수·박스 마지막 원소에 NaN/+Inf/-Inf를 주입한 6개 사례 모두 거부.
- 정규화: RGB/gray 모든 0..255 값, ImageNet/흑백/identity 계수, 비연속 행, 패딩/무패딩, 원본 불변을 검사. 전 원소 bitwise 일치.
- Python 8/8 통과 (`test_cpp_preprocessing.py`, `test_cpp_segment_rules.py`): 실제 학습·ONNX 내보내기·C++ 분할 판정 비교 포함, 원배율/50% 배율.
- Windows/VS2017 빌드 및 실제 생산 영상의 실행 시간은 이 환경에서 측정하지 않았다.

## 적용

직접 소스를 포함하는 C++ 프로젝트에서는 두 파일을 함께 갱신한다.

- `cpp/src/vision_inference.cpp`
- `cpp/include/opencv_preprocess.h`

DLL/정적 라이브러리를 쓰는 경우에도 해당 SDK를 재빌드해서 교체해야 한다. 솔루션 및 SDK 프로젝트를 Release/x64로 빌드한다. 재학습, ONNX 재내보내기, JSON 변경은 필요 없다. 기존 산출물의 판정 계산과 암호화 로드는 유지된다.
