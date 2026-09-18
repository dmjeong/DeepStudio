> 과거 릴리스 기록입니다. 폐기된 엔진 관련 항목은 현재 기능이 아닙니다. 현재 지원 범위는 README를 참고하세요.

# Deep Vision Studio 23.8

EfficientNet 중심 분류 및 배포 기능 릴리스. 기능 변경으로 22.8에서 23.8로 올린다.

## 이전 모델 지원 종료

- 새 분류 프로젝트는 EfficientNet B0 사전학습으로 시작한다. B1도 선택할 수 있다.
- 폐기된 외부 엔진 패키지를 설치 요구사항과 시작 검사에서 제외하고 EXE에도 포함하지 않는다.
- 데스크톱과 웹에서 이전 모델 학습 선택을 없애고 worker, 추론 로더, ONNX exporter, C++ 로더에서도 거부한다.
- 기존 이전 모델 프로젝트와 과거 학습 결과는 보존한다. 이전 모델 가중치를 EfficientNet 가중치로 자동 변환하지 않는다. 분류 엔진을 바꾸고 ImageNet 가중치에서 새 학습을 시작해야 한다.
- 기존 Custom CSP 및 PatchCore는 별도의 구현으로 유지한다. 이전 모델에 의존했던 OBB 학습은 제공하지 않는다.
- 현재 정리 버전에서는 폐기한 모델 어댑터와 전용 테스트를 저장소 및 배포 ZIP에서 제거했다. 기존 사용자 Python 환경의 패키지는 변경하지 않는다.

## EfficientNet 추론

- 학습 및 레이어 디버깅 구조는 유지하고, 추론용 독립 FP32 복사본에만 Conv-BatchNorm 융합을 적용한다.
- ONNX 검증은 융합 전 체크포인트 모델을 기준으로 한다. 융합 후 모델끼리만 비교하지 않는다.
- Python 일반 추론에는 `torch.inference_mode()`를 사용한다. Grad-CAM은 별도로 역전파를 수행한다.
- C++ 전처리는 OpenCV Mat에서 resize와 NCHW 정규화로 연결해 두 번의 uint8 벡터 복사를 제거한다. 원본 ROI의 stride도 OpenCV에서 처리한다.
- C++ `ClassifyResult`에 `preprocess_ms`, `model_ms`, `postprocess_ms`를 추가했다. 세 값의 합은 기존 `inference_ms`다. 전처리 시간에는 입력 텐서 준비도 포함하며, 파일 읽기는 포함하지 않는다.
- 기존 C++ 클래스명 `VisionInference`와 파일명은 호출부 호환을 위해 유지한다. 이 이름이 이전 모델 모델 지원이나 폐기된 외부 엔진 의존성을 의미하지 않는다.

Conv-BN 융합은 [PyTorch 공식 API](https://docs.pytorch.org/docs/2.9/generated/torch.nn.utils.fuse_conv_bn_eval.html)를 사용한다. 스레드별 성능은 장비와 동시 실행 수에 따라 달라지므로 [ONNX Runtime의 스레드 설정](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)을 실제 장비에서 비교한다. 현재 환경에서 속도 개선률을 측정한 결과는 없다. ONNX Runtime이 이미 같은 융합을 수행하는 장비에서는 ONNX 모델 실행 시간이 줄지 않을 수도 있다.

## 학습 품질

EfficientNet의 새 학습은 기본적으로 BatchNorm 파라미터와 bias에 weight decay를 적용하지 않는다. 백본과 분류기의 차등 학습률 및 동결 설정은 유지한다. UI에서 이 옵션을 해제해 이전 방식과 비교할 수 있다. 구형 `last.pt`의 정확한 재개에는 이전 optimizer 그룹을 복원한다.

이 변경은 정확도 개선을 위한 학습 옵션이며 실제 데이터의 정확도 상승을 보장하거나 측정한 결과가 아니다. 같은 데이터 분할에서 검증 정확도, macro F1 및 불량 클래스 recall을 비교해야 한다. 이미 학습된 가중치의 정확도가 이 옵션만으로 바뀌지는 않는다.

## ONNX 내보내기

- 기본 opset을 17로 맞추고 설치된 PyTorch의 지원 범위를 확인한다. Python, 데스크톱, 웹 기본값이 같다.
- legacy exporter를 명시하며 지원 버전에서는 external data 분리를 끈다. EfficientNet 복원 시 Custom CSP 모델 모듈을 불필요하게 불러오지 않는다.
- ONNX 및 ONNX Runtime 요구 버전을 각각 1.16 이상, 1.20.1 이상으로 맞춘다.
- 실패 시 출력 파일 옆 `model.export-error.json`에 실패 단계, 오류, traceback, 패키지 버전을 기록한다. 진단은 로컬에만 저장한다.
- 검증 실패 시 기존 ONNX 및 JSON을 교체하지 않는다. 성공한 JSON에 opset, FP32, 동적 배치 여부, 융합 정보를 기록한다.
- 사용자가 보고한 ONNX 오류 자체는 이 환경에서 재현하지 못했다. 이 릴리스는 호환성 처리와 진단을 개선했지만 해당 오류의 해결을 확정하지 않는다.

프로젝트 폴더에서 실행:

```bash
python -m pip install -r python/requirements.txt
python python/export_onnx.py --checkpoint runs/example/best.pt --output exports/model.onnx
```

## Python 배포

`python/onnx_classifier.py`, `python/opencv_preprocess.py`, `python/center_crop.py`와 내보낸 ONNX 및 JSON을 함께 둔다. 배포 환경은 NumPy, OpenCV, ONNX Runtime을 설치한다. 아래 경로에는 PyTorch나 폐기된 외부 엔진가 필요 없다.

```python
from onnx_classifier import OnnxClassifier

# 한 번 로드하고 여러 이미지에 재사용한다.
model = OnnxClassifier("model.json", num_threads=2)
result = model.predict_file("sample.png")
print(result["class_name"], result["confidence"])
print(result["decode_ms"], result["preprocess_ms"], result["model_ms"], result["postprocess_ms"])
```

크롭, 크기, 채널 및 정규화는 내보낸 JSON을 따른다. RGB 배열에는 `predict_rgb()`를 사용한다. OpenCV BGR 배열은 먼저 `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`로 변환한다. ONNX 및 C++ Grad-CAM은 제공하지 않으며 기존 PyTorch Grad-CAM을 사용한다.

## 동일 모델 비교

```bash
python tools/efficientnet_benchmark.py --checkpoint runs/example/best.pt --validation data/val --threads 1 2 4 --warmup 10 --runs 50 --output efficientnet-benchmark
```

`report.json`은 원본 PyTorch, 융합 PyTorch, ONNX의 p50/p95 시간과 각 모델의 confusion matrix, 정확도, macro F1, 클래스별 recall을 기록한다. 수치 오차가 허용 범위를 넘으면 실패한다. `--validation`을 생략하면 합성 입력으로 시간과 출력 일치만 검사하고 정확도는 기록하지 않는다. 검증 폴더는 학습에 쓰지 않은 `클래스명/이미지` 구조여야 한다.

`recommended_onnx_threads`는 해당 장비와 입력의 실측 추천값이다. 실제 장비에서 비교한 뒤 JSON의 최상위 `num_threads`에 설정하면 C++ 및 Python 분류기가 적용한다.

## 검증 범위

로컬 Python 회귀 검사 113개 중 96개 통과, 17개 의존성 누락으로 생략. 웹 로직 검사 11개 통과. Python 문법 및 git whitespace 검사 통과. OpenCV를 사용하지 않는 C++ 확률 계약 검사는 컴파일 후 통과했다.

실제 B0/B1 RGB/흑백 ONNX 내보내기, 동적 배치, 융합 출력, Grad-CAM, optimizer 그룹 검사를 추가하고 CI에 연결했다. 그러나 로컬에는 Torch, OpenCV, ONNX 및 ONNX Runtime이 없고 패키지 설치가 실패했다. 실제 학습, 실제 내보내기, OpenCV C++ 빌드 및 전체 UI 실행은 로컬에서 검증하지 못했다. 프런트엔드 전체 빌드도 Vite 다운로드 HTTP 403으로 실행하지 못했다. 위 테스트 수를 전체 기능 검증 완료로 해석하면 안 된다.
