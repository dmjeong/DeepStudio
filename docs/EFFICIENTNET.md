# EfficientNet B0/B1 분류와 레이어 디버깅

224×224 B0 CPU의 전처리+추론+후처리 8 ms 목표는
[CPU 벤치마크 및 OpenVINO 배포 안내](EFFICIENTNET_CPU_LATENCY.md)를 참고한다.
코드와 라이브러리 조건은 [라이선스 검토](EFFICIENTNET_CPU_LICENSES.md)에 정리했다.
[C++17 모델 로드·백그라운드 기능·빌드·검증 방법](EFFICIENTNET_CPP17.md)도 제공한다.

분류 프로젝트의 학습 엔진에서 EfficientNet을 선택한다. 24.8부터 새 프로젝트의 기본 입력은 1채널이며 새 EfficientNet은 첫 Conv도 1채널이다. B0의 기본 입력 크기는 224, B1은 240이며 프로젝트 입력 크기를 조정할 수 있다. 공식 ImageNet 가중치는 B0 IMAGENET1K_V1, B1 IMAGENET1K_V2로 고정한다. 최초 다운로드는 TLS와 SHA256을 검증하고 이후 로컬 캐시를 사용한다.

| 모드 | 초기화 | 학습 설정 |
|---|---|---|
| 사전학습 모델로 시작 | 공식 백본, 새 분류기 | 화면 설정 |
| 내 가중치로 추가 학습 | 공식 원본 또는 이 앱의 EfficientNet 체크포인트 | 새 optimizer와 화면 설정 |
| 중단한 학습 재개 | 이 앱의 미완료 last.pt | 저장된 설정, optimizer, scheduler, scaler, RNG와 Best 복원 |

추가 학습에서 클래스 이름이나 순서가 바뀌면 분류기를 초기화한다. B0와 B1 가중치는 교환하지 않는다. 완료한 모델이나 다른 엔진의 체크포인트는 재개할 수 없다. 재개는 동일 데이터 경로, 파일 크기, 수정 시각, 분할 설정과 클래스 순서를 확인한다. 파일 내용 자체의 해시는 계산하지 않는다. 실행 환경과 장치가 달라지면 비트 단위 재현성을 보장하지 않는다.

## 직접 디버깅

VS Code에서 `DeepVisionStudio` 폴더를 열고 기존 실행 환경을 Python 인터프리터로 선택한다. `python/efficientnet.py`의 원하는 줄에 중단점을 찍고 제공된 EfficientNet launch 구성을 실행한다. 프로젝트 파일과 B0/B1을 선택하면 실제 GUI와 같은 학습 루프를 현재 프로세스에서 실행한다.

```bash
python debug_train.py --project /path/project.dvproj --variant efficientnet_b0 --device auto --batches 1
```

`--weights /path/last.pt`로 로컬 가중치에서 새 디버그 학습을 시작할 수 있다. `auto`는 GPU를 우선 사용하고 `cuda`는 GPU가 없으면 오류를 낸다. 디버그 실행은 AMP를 끄고 DataLoader worker를 0으로 설정하며 원래 프로젝트 설정을 저장하지 않는다. 결과는 프로젝트의 `debug-runs` 하위에 저장한다. 배치를 제한한 디버그 체크포인트는 정식 학습 재개에 사용할 수 없다.

| 위치 | 확인할 값 |
|---|---|
| ConvNormActivation.forward | conv_output, normalized, activated |
| MBConv.forward | expanded, spatial, attended, projected, residual, output |
| SqueezeExcitation.forward | pooled, squeezed, gate, scaled |
| EfficientNet.forward | 각 stage 출력, pooled, flattened, logits |
| 학습 루프 backward 이후 | parameter.grad와 LayerInspector.records |

MBConv 중단점 조건에 `self.debug_name == "stage4.block1"`을 사용하면 원하는 블록에서만 멈춘다. 실제 모듈 경로는 `features.4.1`이다. `--layers "features.4.1*" "classifier.1"`로 관찰 대상을 좁힐 수 있다. `--keep-values`는 선택한 출력 일부를 CPU에 보관한다. 출력 shape, dtype, 유한값 여부, 범위와 gradient 통계는 `layer-trace.json`에 기록한다. 출력 통계를 계산하면 장치 동기화가 발생하므로 성능 측정에서는 디버그 관찰을 사용하지 않는다.

## 구현과 배포 계약

전체 모델은 로컬 `python/efficientnet.py`에서 구성한다. Conv, BatchNorm과 텐서 연산은 PyTorch를 사용하며 모델 생성에 torchvision factory를 사용하지 않는다. 3채널은 공식 state_dict의 이름과 크기를 유지한다. 새 1채널 모델은 첫 Conv만 `[32,1,3,3]`으로 구성하며, MBConv, SE, 잔차 및 나머지 백본은 B0/B1 구조를 유지한다. torchvision의 구조를 참고한 라이선스는 `python/EFFICIENTNET_NOTICE.txt`에 포함했다.

새 1채널 모델의 사전학습 초기화는 공식 첫 Conv의 RGB 가중치를 채널 방향으로 합산한다. `W_gray = W_R + W_G + W_B`이며 나머지 백본 가중치는 그대로 로드하고 사용자 분류기를 초기화한다. 이 합산은 동일한 정규화 값의 gray를 3면에 복제한 Conv와 동등하다. RGB 채널별 정규화까지 포함한 구형 모델 전체 출력과 같다는 의미는 아니다. 새 구조의 학습 정확도는 별도로 검증해야 한다.

전처리는 기존 중앙 crop, OpenCV `INTER_LINEAR_EXACT` resize, 1채널 normalize 순서다. 흑백 파일을 RGB로 확장해 다시 grayscale로 바꾸는 처리를 제거했다. 새 모델은 정규화된 `[N,1,H,W]`를 첫 Conv에 직접 전달한다. mean `[0.449]`, std `[0.226]`은 기존 프로젝트 기본값이며 체크포인트의 저장 값을 우선한다.

체크포인트와 배포 JSON에 `implementation_version`, `in_channels`, `stem_in_channels`, `input_adapter`를 기록한다. 입력 메타데이터와 실제 첫 Conv 가중치가 충돌하면 로딩을 거부한다. 버전 1의 흑백 체크포인트는 입력 1채널, 첫 Conv 3채널인 기존 RGB 확장 구조로 복원한다. 추론 화면과 학습 로그에 구형 구조를 표시한다. 기존 가중치 추가 학습 및 중단 재개도 해당 구조를 보존하며 자동으로 새 1채널 Conv로 바꾸지 않는다. 새 구조는 1채널을 선택하고 ImageNet 사전학습으로 학습을 시작한다.

CPU EfficientNet의 Studio 추론 버튼은 ONNX Runtime으로 판정하고 전체 클래스 확률을 표시한다. PyTorch Grad-CAM은 `features.8`을 사용하며 체크한 경우 별도로 계산한다. [실행 경로와 이미지별 시간 안내](EFFICIENTNET_GUI_ONNX.md). ONNX는 logits를 출력하고 배포 JSON에 softmax 축과 EfficientNet backend를 기록한다. C++ 분류 로더는 `CV_8UC1`을 직접 받는다. 화면의 컬러 오버레이 변환은 추론 이후 표시 용도로만 수행한다.

Python 배포 파일은 `onnx_classifier.py`, `opencv_preprocess.py`, `center_crop.py`, `efficientnet_contract.py`, 그리고 같은 이름으로 내보낸 ONNX와 JSON이다. NumPy, OpenCV, ONNX Runtime이 필요하다.

```python
from onnx_classifier import OnnxClassifier

model = OnnxClassifier("model.json")
result = model.predict_gray(gray_frame)  # uint8 [H,W] 또는 [H,W,1]
print(result["class_name"], result["confidence"])
```

```bash
python -m pytest tests/test_efficientnet.py tests/test_efficientnet_channels.py tests/test_efficientnet_deployment.py -v
python tools/efficientnet_pretrained_smoke.py --device cpu --output efficientnet-validation.json
```

첫 명령은 공식 구조와 stage 출력, 실제 backward, 학습 재개, 저장/로드, Grad-CAM, 흑백/RGB ONNX와 UI 설정을 검증한다. 두 번째는 실제 공식 B0/B1 가중치를 받아 모든 leaf layer와 최종 출력을 비교하고 한 배치의 학습 업데이트를 확인한다. GPU 검증은 동일 명령의 `--device cuda`로 실행한다. CI 결과와 실제 장치에서의 검증 결과는 구분한다.

- [선택한 ONNX Runtime 경로의 추가 최적화 / Windows 비교 도구](EFFICIENTNET_ONNX_OPTIMIZATION.md)
