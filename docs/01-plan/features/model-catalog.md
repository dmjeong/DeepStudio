# 기본 모델 카탈로그와 지원 판정 설계

2026-09-18 · 요구사항 개정 3 · 아래 값은 등록부의 구현 범위이며 실제 release-ready 판정은 별도다.

## 1. 기본 탑재 모델

세부 변형은 사용자가 지정한 모델군을 구체화하기 위한 초기 제안이다. 확정·검증한 변형만 installer에 포함한다.
모든 행은 학습/fit, 추론, ONNX 배포, C#/C++17 실행을 출시 필수 항목으로 가진다.

| 태스크/모델군 | 초기 기본 변형 제안 | 학습과 초기 가중치 | 배포 계약/중요 검증 |
|---|---|---|---|
| 분류 / EfficientNet | B0, B1 | TorchVision ImageNet 초기 가중치, fine-tune | 분류 logits; B0 224, B1 기본 240; B1 224는 별도 프로필 |
| 분류 / ResNet | ResNet18, ResNet50 | TorchVision ImageNet 초기 가중치 | 분류 logits, 클래스 순서·헤드 교체 |
| 분류 / ConvNeXt V1 | Tiny | TorchVision V1 초기 가중치; V2로 대체하지 않음 | 분류 logits, LayerNorm·레이아웃 export |
| 분류 / LibreYOLO | LibreMobileNetV4 Small | Libre 분류 어댑터와 해당 초기 가중치 | 224; Libre 프레임워크의 모든 분류 모델을 뜻하지 않음 |
| 이상 탐지 / PatchCore | Wide-ResNet50-2, 경량 ResNet18 | pretrained backbone + 정상 특징 bank/coreset 생성 | bank·kNN·map·score까지 전체 ONNX, threshold 보존 |
| 객체 탐지 / Re-DETR v4 | Small, Medium, Large | 세 변형을 고정된 제품 ID로 관리 | upstream·checkpoint·후처리별 ONNX 검증 필요 |
| 객체 탐지 / LibreYOLO | LibreYOLO9 Tiny | 검토된 기본 체크포인트, 검출 head fine-tune | raw output decode·NMS·원본 좌표 복원 |
| 분할 / SAM2 | SAM2.1 Hiera Tiny, Small, Base+, Large | 네 변형 모두 image/prompt/video 계약으로 관리 | image encoder + prompt decoder ONNX, 이미지 캐시·프롬프트·영상 state 계약 |
| 분할 / DeepLab V3+ | SMP DeepLabV3Plus, ResNet34 encoder | encoder ImageNet 초기화 + segmentation head 신규 학습 | semantic logits, binary/multiclass 구분 |
| 분할 / U-Net | SMP Unet, ResNet18 encoder | encoder ImageNet 초기화 + segmentation head 신규 학습 | semantic logits, ignore index·원본 크기 복원 |

ResNet50도 기본 배포 목록에 넣되 최소사양 인수는 ResNet18과 별도 기록한다.
카탈로그에 없는 ConvNeXt/SAM2/탐지 변형은 첫 목록에 암묵적으로 포함하지 않는다.

## 2. 분류 모델의 계약

입력 크기·crop/resize·보간·색순서·정규화는 checkpoint와 함께 저장한다.
기존 EfficientNet의 1채널 adapter는 보존한다. 새 RGB 모델의 1채널 입력은
`GRAY→RGB 복제` 또는 `학습 시 변환한 입력 Conv` 중 명시한 계약만 허용하고 자동 혼용하지 않는다.
B1을 B0와 같다는 이유로 무조건 224로 바꾸지 않는다.
[TorchVision B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html),
[B1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b1.html),
[ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html),
[ConvNeXt Tiny](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html)

LibreYOLO는 단일 아키텍처가 아니다. 분류 기본 후보는 MobileNetV4 Small이며,
학습을 지원하지 않는 라이브러리 내 다른 family까지 학습 가능하다고 표시하지 않는다.
freeze/unfreeze, 클래스별 weight, augmentation, resume도 실제 지원하는 옵션만 노출한다.
[Libre 분류](https://www.libreyolo.com/docs/tasks/image-classification),
[MobileNetV4](https://www.libreyolo.com/docs/models/mobilenetv4),
[해당 가중치](https://huggingface.co/LibreYOLO/LibreMobileNetV4s-cls)

## 3. Re-DETR v4와 객체 탐지

제품 ID는 `re_detr_v4_small`, `re_detr_v4_medium`, `re_detr_v4_large`로 고정한다.
세 변형은 등록부에서 각각 독립적인 입력 크기·runtime·checkpoint·후처리 계약을 갖는다.
네이티브 SDK는 `pred_boxes=[1,N,4]`와 `pred_logits=[1,N,C]`를 검증하고 공통 detection 결과로 변환한다.
실제 upstream 구현·checkpoint를 변형별로 연결하기 전에는 카탈로그 항목을 release-ready로 표시하지 않는다.

- class offset, query 선택, sigmoid/softmax, decode, resize 방식은 변형별 manifest에 기록한다.
- NMS가 없는 모델에는 임의 NMS를 넣지 않는다. Libre 탐지의 NMS 규칙은 별도 계약이다.
- 두 구현체의 class offset, query 선택, sigmoid/softmax, decode, resize 방식이 같다고 가정하지 않는다.
- NMS가 없는 모델에는 임의 NMS를 넣지 않는다. Libre 탐지의 NMS 규칙은 별도 계약이다.

[RT-DETR 공식](https://github.com/lyuwenyu/RT-DETR),
[RF-DETR 공식](https://github.com/roboflow/rf-detr),
[RF-DETR 학습](https://rfdetr.roboflow.com/learn/train/)

LibreYOLO9 Tiny의 export 설정은 고정 shape/batch1부터 검증한다.
NMS를 그래프 안에 넣을지 SDK에서 할지 export manifest에 기록하고 중복 적용하지 않는다.
[Libre YOLO9](https://www.libreyolo.com/docs/models/yolov9),
[Libre ONNX](https://www.libreyolo.com/docs/export/onnx),
[가중치 출처](https://huggingface.co/LibreYOLO/LibreYOLO9t)

## 4. PatchCore: 일반 gradient 학습과 다른 fit

기존 프로젝트의 특징 추출·coreset·bank 구성과 점수 계약을 보존한다.
현재 경로는 `kNN 평균 거리 → bilinear upsample → Gaussian sigma4 → map max`이며 기본 k=9다.
새 라이브러리의 다른 점수식으로 바꾼 뒤 기존 threshold를 그대로 쓰지 않는다.

최초 전체 ONNX 프로필은 batch1, 224, bank≤4096으로 제한한다.
backbone·bank·distance·TopK·mean·map·score를 그래프로 표현하고 bank는 initializer/external data로 포함한다.
feature extractor만 ONNX로 내보낸 것을 PatchCore 전체 ONNX 지원 완료로 표시하지 않는다.
큰 bank가 필요한 경우 `feature.onnx + bank + native kNN`을 별도 hybrid 프로필로 설계할 수 있지만
기본 전체 ONNX 인수를 대체하지 않는다. SDK가 어떤 프로필인지 명확히 표시한다.

정상 데이터 fit, validation threshold 보정, bank 추가/재생성, 동작 가능한 재개 방식을 UI에서 구분한다.
SGD optimizer resume와 같은 기능으로 포장하지 않는다. `k > bank`, zero distance, 메모리 상한,
raw/normalized score와 판정 동등성을 C#/C++까지 검사한다.
[PatchCore 원 구현](https://github.com/amazon-science/patchcore-inspection),
[ONNX 구현 참고](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html),
[ONNX external data](https://onnx.ai/onnx/repo-docs/ExternalData.html)

## 5. SAM2: 프롬프트 분할

기본 설계는 단일 이미지·batch1, 점(positive/negative)·박스·선택적 이전 mask logits를 받는
`segmentation.prompted.v1`이다. 출력은 mask(s), quality score, 원본 좌표 대응 정보와 반복 prompt용 low_res_mask_logits/선택 mask index다.
기본 1024 프로필의 SDK logits 형식은 이미지당 FP32 [M,256,256]이며, 이진 mask나 확대된 mask로 대체하지 않는다.
DeepLab/U-Net의 클래스별 semantic mask와 혼동하지 않는다.
사용자 클래스 자동 분류기가 없는 SAM2 mask에 임의 클래스 이름을 붙이지 않는다.

학습은 객체 mask에서 프롬프트를 샘플링하는 fine-tune으로 정의한다.
최소사양 실증은 Tiny, batch1, encoder freeze + decoder fine-tune부터 수행한다.
전체 encoder fine-tune은 별도 VRAM 프로필을 통과한 경우 제공한다.
영상 memory/state 학습·추적과 ONNX export는 사용자 필요 여부 확정 후 별도 계약·인수 항목으로 추가한다.
자동 전체 mask 생성도 grid prompt와 중복 제거가 필요한 별도 실행 모드다.

Meta는 Windows에서 WSL을 권장한다. 따라서 이 제품의 **Windows native 학습 worker**는
P0 실증이 필요하다. 선택적 CUDA extension 미사용 시 빠지는 작은 구멍/점 제거 후처리를
그대로 방치하지 않고 동일 의미의 native 후처리 구현 또는 명시적 모델 프로필로 고정한다.
기본 모델만 별도 수동 WSL 설치를 요구하는 상태로 출시하지 않는다.
[Meta SAM2](https://github.com/facebookresearch/sam2),
[설치 제약](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md),
[학습](https://github.com/facebookresearch/sam2/blob/main/training/README.md)

배포는 `image_encoder.onnx + image_decoder.onnx + manifest + SDK`를 기본으로 한다.
ORT의 SAM2 예제는 모델 버전·decoder batch 제약이 있으므로 SAM2.1 조합의 동작을 별도로 검증한다.
이미지 content hash·모델 hash·전처리 설정으로 embedding 캐시를 구분한다.
image_context는 image_embeddings와 decoder 입력인 image_features_0/1을 함께 보유한다.
같은 이미지의 반복 프롬프트에는 encoder를 다시 실행하지 않고, 이미지나 모델 변경 시 캐시를 무효화한다.
[ORT SAM2 내보내기 예제](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/sam2/README.md)

## 6. DeepLab V3+와 U-Net

SMP의 `DeepLabV3Plus`, `Unet`을 후보로 삼는다. TorchVision DeepLabV3를 V3+라고 이름만 바꾸지 않는다.
encoder 사전학습과 완성된 segmentation 가중치를 구분하여, UI에
`ImageNet encoder 초기화 / segmentation head 신규 학습`을 표시한다.
학습 초기화를 할 때 자동 다운로드 대신 번들 로컬 가중치를 사용한다.

초기 해상도는 512, 최소 메모리 프로필은 batch1이다. binary는 1채널 logits+sigmoid,
multiclass는 C채널 logits+argmax 규약을 고정한다. target mask는 nearest 보간이며
이미지 resize/letterbox, background/ignore index, loss, class order를 저장한다.
[SMP 모델 API](https://smp.readthedocs.io/en/latest/models.html),
[SMP 구현·라이선스](https://github.com/qubvel-org/segmentation_models.pytorch)

## 7. 자산·라이선스·지원 상태

모델마다 코드 버전, worker runtime, checkpoint revision/hash, 가중치 출처와 재배포 근거를 고정한다.
TorchVision/SMP 코드의 허용 라이선스가 encoder 가중치의 배포 권리까지 자동 확정하지 않는다.
SAM2는 공식 코드·가중치의 Apache-2.0 조건과 필요한 NOTICE를 따른다.
Libre 최상위 MIT만 확인하지 않고 실제 묶는 vendored 코드·가중치를 따로 확인한다.
비상업 가중치나 별도 제한 모델을 허용된 기본 모델과 함께 일괄 설치하지 않는다.

[가중치 별도 조건](https://github.com/pytorch/vision#pre-trained-model-license),
[Libre NOTICE](https://github.com/LibreYOLO/libreyolo/blob/release/NOTICE),
[SAM2 라이선스](https://github.com/facebookresearch/sam2#license)

카탈로그 상태는 `requested → scoped → runtime_verified → trained_verified → export_verified → sdk_verified → release_ready`다.
가중치 배포 조건은 별도 `redistribution_status`로 관리한다. 소스 문서에서 지원한다고 해도
우리 Windows 설치·학습·ONNX·C#/C++ 검증이 끝나기 전에는 `release_ready`가 아니다.

## 8. 모델별 인수 기록

모델 ID/변형, source revision, runtime ID, checkpoint hash, 입력 크기·채널, batch/정밀도,
학습 옵션, GPU/드라이버, 최대 RAM/VRAM, 학습 성공·재개/fit, export graphs, 수치 허용 기준,
태스크 지표, C#/C++ 실행 결과, 라이선스 검토 결과를 한 행에 기록한다.
실제 fine-tune 결과를 export했는지 확인하며 원래 pretrained를 잘못 export하는 회귀를 막는다.

관련: [제품 계획](model-packs-windows-distribution.plan.md),
[상세 설계](../../02-design/features/model-packs-windows-distribution.design.md).
