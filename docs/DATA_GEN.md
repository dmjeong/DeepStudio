# 실제 불량 학습 Data Gen

Data Gen은 사용자 정의 학습 항목별로 실제 불량 이미지와 마스크를 학습한다.
스크래치 같은 예시 이름에 따른 고정 생성 함수는 사용하지 않는다.
기존 규칙 증강은 별도 메뉴에서 기존 기록과 함께 사용할 수 있다.

## 사용 순서

1. 불량 데이터에서 학습 항목을 추가하고 검사 클래스에 연결한다. 이름 변경은 ID와 기존 모델 연결을 유지한다. 보관한 항목도 과거 이력을 삭제하지 않는다.
2. 실제 불량 이미지 폴더를 가져온다. 개체/생산 로트 그룹과 Train 또는 Val을 지정한다. 검사 프로젝트의 기존 val/test는 사용하지 않는다. 생성 모델용 Val은 검사 학습용 데이터에서 별도로 확보한다.
3. 마스크 폴더를 함께 지정하거나 이미지에서 브러시, 사각형, 다각형으로 영역을 표시한다. 가져올 마스크는 이미지와 같은 상대 경로의 PNG다. 실제 정상 이미지도 같은 항목에 등록한다.
4. 마스크를 검수한 후 새 학습 버전으로 고정한다. Train과 Val이 모두 필요하다. 같은 그룹을 분할하거나 재인코딩한 동일 픽셀 이미지를 중복 등록할 수 없다.
5. 불량 학습에서 로컬 사전학습 Inpainting 모델 폴더와 제품/불량 외관 설명을 입력한다. 고급 설정에서 학습 Step, 검증 주기, 해상도, rank, 학습률, 정밀도를 설정한다. 설정 저장 버튼으로 프로젝트에 보존한다.
6. 학습한 모델과 실제 정상 이미지를 선택하고 생성 요청 영역을 그린다. 길이, 폭(px)과 방향으로 사각형 영역을 지정할 수도 있다. 허용 마스크를 별도로 지정하지 않으면 요청 영역만 변경을 허용한다.
7. 샘플을 생성하고 검수 탭에서 원본/결과 슬라이더로 비교한다. 요청 마스크는 라벨 초안이다. 실제 불량 정답을 확인하고 필요하면 수정한 후 승인한다.
8. 승인본은 원본을 보존하는 별도 train 버전 폴더로 저장한다. images, masks, manifest.json의 명시적 클래스 연결과 출처를 함께 보관한다. 현재 검사 프로젝트에 자동 병합하지 않는다. 검사 데이터셋 편집/가져오기에서 해당 학습 방식을 확인해 반영한다. PatchCore 정상 학습에는 합성 불량을 넣지 않는다.

## 전용 실행 환경

데스크톱과 React는 같은 작업 엔진과 데이터 형식을 사용한다. 학습과 생성만 별도 Python 환경으로 실행할 수 있다. 검사 모델의 기본 의존성에는 Diffusers를 추가하지 않았다.

Python 3.11 환경에서 아래 순서로 설치한다. NVIDIA 드라이버와 GPU가 해당 CUDA PyTorch 빌드를 지원해야 한다. 예시는 공식 CUDA 12.8 빌드다.

```powershell
py -3.11 -m venv .venv-datagen
.venv-datagen\Scripts\python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.venv-datagen\Scripts\python -m pip install -r requirements-datagen.txt
$env:DEEP_STUDIO_DATAGEN_PYTHON = (Resolve-Path .venv-datagen\Scripts\python.exe).Path
```

이 환경 변수를 설정한 터미널에서 데스크톱 또는 웹 앱을 실행한다. Linux에서는 전용 환경의 `bin/python` 절대 경로를 지정한다. 지정하지 않으면 앱의 현재 Python을 사용하며 의존성이나 CUDA가 없으면 명확한 실패 상태로 종료한다.

베이스 모델은 공식 SD 또는 SDXL Inpainting 파이프라인 전체 폴더를 준비한다. `model_index.json`과 UNet, VAE, tokenizer, text encoder 및 scheduler가 필요하다. Safetensors 가중치만 로드하며 9채널 입력/4채널 출력 UNet을 검사한다. 일반 text-to-image 가중치는 거부한다. 온라인 자동 다운로드는 하지 않는다. 가중치는 소스 배포에 포함하지 않는다.

- [SDXL Inpainting 모델](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)
- [Diffusers 0.35.1 Inpainting API](https://huggingface.co/docs/diffusers/v0.35.1/en/api/pipelines/stable_diffusion/inpaint)
- [Diffusers 0.35.1 LoRA](https://huggingface.co/docs/diffusers/v0.35.1/en/training/lora)
- [PyTorch 공식 이전 버전 설치](https://pytorch.org/get-started/previous-versions/)

## 학습과 Best 정책

마스크로 가린 실제 불량 이미지를 조건으로 원래 불량을 복원하도록 학습한다. UNet attention의 LoRA만 업데이트하고 베이스, VAE, 텍스트 인코더는 동결한다. SD와 SDXL의 텍스트 조건 구조를 구분한다. SDXL에는 pooled text와 time IDs를 함께 전달한다. scheduler에 맞는 epsilon 또는 velocity를 예측한다.

손실은 불량 영역과 배경 영역을 각각 면적으로 정규화한 뒤 가중 합한다. 기본 배경 가중치는 0.1이다. Val은 고정 전처리, VAE mode, 이미지별 세 고정 난수 표본으로 측정한다. 검증 중 가중치 갱신은 없다. 작은 불량이 잠재 마스크에서 사라지면 실패로 알려준다.

- `best_val_loss`: 유한한 실제 Val Loss의 최솟값. 동점은 먼저 저장한 모델 유지. 조기 종료 min_delta와 무관하게 작은 실제 개선도 보존한다.
- `best_quality`: 작업자가 저장된 체크포인트의 생성 품질을 확인하고 근거를 입력해 선정한다. 미검수 모델에는 품질 점수를 만들지 않는다.
- `last`: 최근 완료 저장. 어댑터, optimizer, LR scheduler, GradScaler, global step, epoch, 데이터 순서와 위치, CPU/CUDA/샘플러 난수 상태를 포함한다.

Val 데이터가 없으면 학습 시작을 거부하며 Train Loss를 Val Loss로 대체하지 않는다. NaN/Inf 검증은 기존 Best를 바꾸지 않는다. 서로 다른 데이터 버전의 Loss를 직접 비교하지 않는다. 저장 파일의 SHA-256을 확인하고 이어학습/생성 시 베이스 가중치 변경도 차단한다.

취소 시 완료된 후보와 체크포인트는 유지한다. 강제 종료 중인 `.partial` 폴더는 완료 후보로 노출하지 않는다. UI 작업 이력과 모델 상태를 함께 확인하며 오류 메시지가 있는 작업을 학습 성공으로 판단하지 않는다.

## 검증 범위

이 기능은 실제 데이터 기반 학습을 위한 실험 기능이다. 코드 배포가 실제 GPU 품질 검증 통과를 뜻하지 않는다.

자동 검증은 데이터 경계와 원자적 저장, Best 정책, 마스크/원본 보존, 두 UI 통합과 9채널 소형 Diffusers UNet의 실제 LoRA 역전파 및 Safetensors 재로드를 다룬다. 소형 모델의 CPU 계약 테스트는 사전학습 모델의 실제 GPU 학습과 생성 품질 검증을 대체하지 않는다.

사용자 GPU와 대표 실제 데이터가 준비되면 아래 도구로 학습과 새 프로세스 생성 경로를 검증할 수 있다. 먼저 UI에서 데이터 버전을 고정하고 정상 이미지를 등록한다. ID는 프로젝트 datagen/index.json 및 datasets 폴더에서 확인할 수 있다.

```powershell
.venv-datagen\Scripts\python tools/datagen_gpu_smoke.py --project PROJECT.json --dataset DATASET_ID --normal NORMAL_ID --mask REQUEST_MASK.png --base-model C:\models\inpainting --prompt "description of the product and defect"
```

생성 결과, 취소 후 Last 재개, 메모리 부족 후 다음 작업, 경계/미세 불량, 실제 검사 미검률과 과검률 비교는 실제 장비에서 추가 확인해야 한다. AI 생성은 8비트 RGB/흑백으로 제한한다. 16비트 높이 데이터는 기존 규칙 도구만 사용한다.

이번 버전은 국소 인페인팅 기본 경로이며 DefectFill 또는 SynSur 논문 전체 재현이 아니다. GPU 성능, 실제 외관 품질, 실제 검사 성능 개선은 아직 확인하지 않았다.
