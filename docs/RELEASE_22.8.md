> 과거 릴리스 기록입니다. 폐기된 엔진 관련 항목은 현재 기능이 아닙니다. 현재 지원 범위는 README를 참고하세요.

# 22.8: OpenCV 전처리로 통일

이전 모델 분류와 EfficientNet의 Python 학습, 검증, 추론 및 C++ 추론에서 리사이즈를 OpenCV로 통일했습니다. 버그 수정 버전 규칙에 따라 22.7에서 22.8로 올립니다.

## 변경 내용

- Python `cv2.resize`와 C++ `cv::resize`에 동일한 `INTER_LINEAR_EXACT` 옵션을 적용합니다. 이 옵션은 OpenCV가 제공하는 8-bit bilinear 보간이며 Pillow의 축소 안티앨리어싱을 재구현하지 않습니다.
- Python과 C++의 RGB/흑백 변환은 OpenCV `cvtColor`를 사용합니다. 기존 모델별 크롭 영역과 RGB 정규화 계수는 유지합니다.
- Custom CSP와 EfficientNet은 공통 데이터셋 및 추론 경로를 변경했습니다. Custom의 세그멘테이션, 검출 및 재구성 모델 이미지 resize도 같은 함수를 사용하고 정답 마스크는 OpenCV nearest로 처리합니다.
- 공식 이전 모델 분류는 전용 데이터셋, Trainer, Validator, Predictor를 연결합니다. 학습의 random resized crop, 검증 및 추론 resize 모두 OpenCV를 사용합니다. 기존 torchvision 학습 증강은 텐서로 수행하므로 PIL 이미지 리사이즈 경로를 사용하지 않습니다. 네트워크 구조와 가중치 형식은 변경하지 않습니다.
- 기존 `pillow_preprocess.h`와 C++ 배포의 `licenses/Pillow.txt`를 제거했습니다. UI 파일 처리 및 다른 기능의 Python 의존성으로 Pillow가 남아 있는 것은 별개입니다. PatchCore의 메모리 뱅크 전처리는 이번 분류 모델 변경 대상이 아닙니다.
- JSON은 schema 5, `resize_implementation=opencv_linear_exact_v1`, `interpolation=INTER_LINEAR_EXACT`, `antialias=false`를 명시합니다.

## 기존 프로젝트 적용

C++ 프로젝트에서 다음 파일을 추가하거나 교체한 뒤 다시 빌드합니다.

1. `cpp/include/vision_inference.h`
2. `cpp/src/vision_inference.cpp`
3. `cpp/include/opencv_preprocess.h`

`pillow_preprocess.h`를 프로젝트에서 제거합니다. 기존 OpenCV, ONNX Runtime, nlohmann/json 연결 설정은 유지합니다. API는 `InitializeFromJson("model.json")` 후 원본 이미지로 `Classify(image)`를 호출하는 방식입니다. 호출부에서 중복 resize 또는 normalize하지 않습니다.

22.8의 내보내기 기능으로 `.onnx`와 `.json`을 다시 생성하고 함께 배포합니다. 구형 JSON은 새로운 전처리로 잘못 해석하지 않도록 C++ 로더가 거부합니다. 기존 가중치는 계속 사용할 수 있지만 전처리가 바뀌므로 기존 판정 성능과 완전히 같다고 보장하지 않습니다. 구형 Custom/EfficientNet 체크포인트를 읽거나 구형 이전 모델를 내보낼 때 전환 경고 및 `migrated_from` 메타데이터를 기록합니다.

이전 Pillow 전처리에서 중단된 학습의 optimizer/RNG 상태를 그대로 재개하면 기존 Best 점수와 새 전처리 점수가 섞입니다. 따라서 해당 구형 체크포인트의 정확한 중단 재개는 거부하고 `내 가중치로 추가 학습`을 사용하도록 안내합니다. 22.8에서 저장한 OpenCV 체크포인트는 정상적으로 중단 재개할 수 있습니다.

## 검증 기록

로컬 회귀 검사: 102개 실행, 89개 통과, 13개 생략. 구형 설정의 명시적 전환, 새로운 JSON, 정규화 계수 보존, 중단 재개 조건, 확률 출력, 결과 저장 및 크롭 좌표를 확인했습니다. C++ 확률 검사는 컴파일하여 실행했습니다.

현재 환경에 Python OpenCV와 C++ OpenCV 개발 패키지, PyTorch/폐기된 외부 엔진/ONNX Runtime이 없으며 OpenCV 패키지 설치도 실패했습니다. 따라서 실제 OpenCV 픽셀 비교와 전체 모델 학습/추론을 로컬에서 수행했다고 주장하지 않습니다.

CI에는 OpenCV Python/C++ 비교, 실제 이전 모델 학습 및 검증 데이터셋, Pillow resize 호출 금지 검사, ONNX 내보내기 비교, C++ ONNX 실행 검사를 연결했습니다. 이 검사들이 실행되어 통과하기 전까지 전체 실행 검증은 미완료입니다.
