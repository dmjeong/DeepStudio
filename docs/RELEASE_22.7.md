> 과거 릴리스 기록입니다. 폐기된 엔진 관련 항목은 현재 기능이 아닙니다. 현재 지원 범위는 README를 참고하세요.

# 22.7: 공식 이전 모델 분류 모델의 공통 C++ 배포

공식 폐기된 외부 엔진 이전 모델 분류 모델을 EfficientNet 및 Custom CSP와 같은 `VisionInference` API로 로드하도록 추가했습니다. 기능 추가에 따라 앱 버전을 21.7에서 22.7로 올립니다.

## 적용 순서

1. `deepstudio3`의 22.7 코드를 받습니다.
2. 기존 C++ 프로젝트에 `cpp/src/vision_inference.cpp`, `cpp/include/vision_inference.h`를 반영하고 다시 빌드합니다. OpenCV, ONNX Runtime, nlohmann/json 의존성은 유지합니다. 함께 제공된 CMake 구성을 사용해도 됩니다.
3. 앱의 ONNX 내보내기에서 공식 이전 모델 분류 체크포인트를 다시 내보냅니다. CLI는 프로젝트 루트에서 다음과 같습니다.

```sh
python python/export_onnx.py --checkpoint best.pt --output deploy/이전 모델.onnx
```

4. 생성된 `이전 모델.onnx`와 `이전 모델.json`을 같은 폴더에 배치합니다. 새 JSON은 schema 4입니다. 과거 이전 모델 JSON에는 전처리 정보가 부족하므로 새 C++ 로더가 재내보내기를 요구합니다. JSON만 수동으로 고치지 않습니다.
5. 원본 OpenCV 이미지를 아래 API에 전달합니다. 이후 모델을 바꿀 때는 ONNX와 JSON 두 파일을 함께 교체하고 다시 초기화합니다. 모델별 C++ 호출 분기는 필요하지 않습니다.

```cpp
#include "vision_inference.h"
#include <iostream>

int main() {
    VisionInference engine;
    if (!engine.InitializeFromJson("deploy/이전 모델.json")) return 1;
    const cv::Mat image = cv::imread("test.png");
    if (image.empty()) return 2;
    try {
        const auto result = engine.Classify(image);
        std::cout << result.class_name << " " << result.confidence
                  << " " << result.inference_ms << " ms\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 3;
    }
}
```

`inference_ms`는 현재 전처리, 텐서 생성, ONNX 실행, 후처리를 포함합니다. 모델 로드와 `imread`는 포함하지 않습니다. 순수 ONNX 실행 시간만 나타내는 값은 아닙니다.

## 모델별 처리

| 모델 | 전처리 | 출력 처리 |
|---|---|---|
| 공식 이전 모델 분류 | 실제 Python 예측기의 Resize, CenterCrop, RGB 정규화 설정을 JSON에 저장하고 재현 | 이미 계산된 확률을 보존 |
| EfficientNet 및 Custom 분류 | 기존 저장된 중앙 크롭, Pillow resize, 정규화 유지 | logits에 Softmax 적용 |

사용자가 학습 시 설정한 원본 중앙 크롭은 모델 전처리보다 먼저 적용합니다. 공식 이전 모델의 Resize 이후 CenterCrop은 별도 단계이며 Python의 홀수 좌표 반올림 규칙까지 맞춥니다. 전처리는 ONNX 그래프 내부가 아니라 JSON을 읽은 C++ 로더에서 수행하므로 호출자가 중복해서 resize 또는 normalize하지 않습니다. 추론 입력 크기와 변환은 내보낸 계약을 따릅니다.

내보내기는 설치된 공식 예측기의 실제 변환을 읽습니다. 이해하지 못하는 변환 구성은 오류로 처리합니다. 기본 검증은 합성 이미지 세 장으로 Python 예측기 입력과 배포 입력 텐서를 비교하고, Python 모델과 ONNX의 클래스 확률을 비교합니다. 허용 오차는 입력 텐서 atol=1e-6, rtol=0, 확률 atol=1e-3, rtol=1e-3입니다. `--no-verify`에서도 전처리와 출력 형식 검사는 수행하지만 최종 확률 일치 검사는 생략하므로 JSON에 검증 생략을 기록합니다.

이번 추가 범위는 공식 이전 모델의 분류 태스크입니다. 공식 이전 모델 검출, 세그멘테이션, OBB는 제공 C++ 분류 로더의 지원 대상이 아닙니다. ONNX C++ Grad-CAM은 이번 변경에 포함되지 않습니다.

## 검증 기록

- 로컬 Python 회귀 검사 95개 중 93개 통과, 의존성 부족으로 2개 생략.
- 실제 C++17 컴파일 후 공식 이전 모델 전처리 24개 조합을 Pillow/NumPy 기준과 비교해 float32 입력 텐서가 정확히 일치했습니다. 가로/세로 이미지, 홀수 크롭 좌표, 짧은 변 기준 resize와 직사각형 resize를 포함합니다.
- 기존 RGB/L 리사이즈 100개, RGB 흑백 변환, B0/B1 정규화 검사도 통과했습니다. 비교 대상은 디코딩된 동일 픽셀이며 JPEG 디코더 차이까지 보장하지 않습니다.
- C++ 확률 보존 코드를 컴파일해 Softmax 중복 적용이 없고 비정상 확률을 거부하는지 검사했습니다.
- CI에 공식 CustomCSP 분류 모델의 ONNX 내보내기 비교와 C++ ONNX 실행 검사를 추가했습니다. C++ 검사는 가로/세로 이미지, 원본 ROI, 클래스 순서, 다른 모델로 재초기화를 포함합니다.

현재 로컬 환경에는 PyTorch/폐기된 외부 엔진/ONNX Runtime과 OpenCV 개발 의존성이 없어 실제 이전 모델 모델 내보내기 검사 및 전체 C++ ONNX 실행 검사를 수행하지 못했습니다. 생략된 Python 검사 2개는 기존 레이어 관찰 역전파 검사와 새 이전 모델 내보내기 비교 검사입니다. 합성 입력 검사는 학습 정확도나 사용자 데이터의 성능 검증을 대신하지 않습니다.
