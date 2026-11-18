# 세그멘테이션 교사 학습

교사용 프로젝트에서 이미지와 정답 마스크를 준비합니다. 마스크에서 0은 배경, 255는 학습 제외입니다. 나머지 값은 프로젝트의 클래스 순서를 따릅니다. 학습과 검증에 같은 실물의 패치가 섞이지 않게 원본 단위로 나눕니다.

| 교사 | 기본 시작 방법 | 로컬 가중치 |
|---|---|---|
| Mask2Former Swin-Small | 설치본의 Microsoft Swin-Small ImageNet 백본과 새 분할 헤드 | 같은 구조의 HF Swin-Small 분류 백본, HF Mask2Former 분할 모델, DeepStudio 체크포인트 |
| SegFormer MiT-B2 | 처음부터 학습 | 사용 권한이 있는 HF MiT-B2 백본·분류 모델, HF SegFormer B2 분할 모델, DeepStudio 체크포인트 |
| DINOv2 ViT-B/14 | 설치본의 Meta LVD-142M 백본과 자체 분할 헤드 | 공식 ViT-B/14 백본 또는 DeepStudio 체크포인트 |

## 데스크톱에서 사용하는 순서

1. **세그멘테이션 프로젝트 → 학습 → 모델 카탈로그 → 교사 학습 모델 선택**에서 교사를 선택합니다. 이 버튼은 학습 설정을 준비합니다. 데이터·마스크를 확인하고 **학습 시작**을 누릅니다. 교사 자체를 학습할 때는 ‘분할 교사 증류’의 파일 칸을 비웁니다.
2. 완료된 실행의 **Best 체크포인트**를 보관합니다. 교사의 추론 화면에서 검증 이미지와 마스크를 겹쳐 성능을 확인합니다. 데이터가 적으면 기본 사전학습 백본이 있는 교사부터 비교합니다.
3. 학생 프로젝트의 학습 화면에서 **U-Net ResNet18** 등 기본 분할 모델을 선택합니다. **분할 교사 증류 → 교사 모델 → 찾아보기**에서 방금 학습한 교사의 Best 파일을 고른 뒤 학습을 시작합니다. 시작값은 교사 비중 0.3, 온도 2입니다. 학습 시 교사와 학생이 같은 증강 영상을 보고 교사의 가중치는 고정됩니다.
4. **내보내기**에서 학생의 Best 체크포인트를 ONNX로 변환합니다. 검사 PC에는 학생의 ONNX와 JSON을 배포합니다. 교사는 학습 PC에서만 실행됩니다.

교사와 학생의 입력 채널과 정규화 규칙은 일치해야 합니다. 전체 클래스 교사는 클래스 순서도 같아야 합니다. 한 결함만 지도할 때는 교사용 프로젝트에 배경과 해당 결함을 등록하고 마스크를 이진화합니다. 예를 들어 학생 클래스가 `background, crack, flip, tray_damage`이면 교사는 `background, tray_damage`를 사용합니다. 두 프로젝트의 이름을 정확히 맞추세요. 학생의 다른 결함과 255 픽셀에는 이 교사의 손실을 적용하지 않습니다.

## 기존 Python 설치 업데이트

사용하던 Python 환경에서 `gui\repair_model_runtime.bat`으로 모델 실행 패키지를 설치하고 `gui\update_segmentation_models.bat`으로 분할 가중치를 받습니다. 신규 설치의 `gui\build.bat`은 가중치를 준비합니다. 학습 중 외부 서버로 이미지를 전송하지 않습니다.

SegFormer와 Mask2Former의 외부 Hugging Face 가중치는 한 파일로 저장한 `.safetensors`, `.bin` 또는 PyTorch state dict를 **로컬 가중치로 추가 학습**에서 선택합니다. 외부 분할 모델의 마지막 클래스 예측층은 프로젝트 클래스 순서에 맞춰 새로 학습합니다. NVIDIA 공식 SegFormer 배포본에는 비상업 조건이 있습니다. 제품용 로컬 파일은 출처와 사용 조건을 확인하세요. [NVIDIA 조건](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) · [Microsoft Swin-Small 모델 카드](https://huggingface.co/microsoft/swin-small-patch4-window7-224) · [Meta DINOv2](https://github.com/facebookresearch/dinov2#license)
