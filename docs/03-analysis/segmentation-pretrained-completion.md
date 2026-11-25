# 13.85 사전학습 연결 검증

2026-11-11, macOS / Apple M4 / Python 3.12.13 / PyTorch 2.14.1 /
torchvision 0.29.1 / timm 1.0.31 / ONNX Runtime 1.29.0.
새로 연결한 두 파일을 제품의 HTTPS 다운로드 코드로 실제 받아 해시를 검사했다.
모든 검사는 공개 가중치·합성 영상으로 수행했고 개인 검사 데이터는 사용하지 않았다.

## 실제 가중치와 기능

| 모델 | 기본 사전학습 | 패치 학습/저장/FP32 ONNX | INT8 생성/실행 | C++ 결과/암호화 동일성 |
| --- | --- | --- | --- | --- |
| U-Net MobileNetV4 Conv-S | timm ImageNet | 통과 | 통과 | 통과 |
| U-Net ShuffleNetV2 x1.0 | torchvision ImageNet | 통과 | 통과 | 통과 |
| U-Net EfficientNet-B1 | timm RA4 ImageNet | 통과 | 통과 | 통과 |
| BiSeNetV2 | CoinCheung ImageNet backbone_v2 | 통과 | 통과 | 통과 |
| STDC1 | 저자 ImageNet STDCNet813M | 통과 | 통과 | 통과 |

`tests/test_segmentation_pretrained.py`: 18개 통과, 종료 코드 0.
출력 비교와 양자화 실행 성공은 검사 정확도 인증이 아니다. INT8 정확도는 실제 데이터로 비교한다.

## 전체 회귀 검사

학습·가중치 준비/복구·모델 선택·ONNX 양자화·Windows 패키징 계약·UI·CLI 관련
19개 테스트 파일에서 **239개 통과, 1개 건너뜀**, 종료 코드 0 (61.78초).
건너뛴 검사는 이번 5종 밖의 기존 U-Net MobileNetV2 공식 가중치 검사이며,
해당 파일이 이 환경에 없기 때문이다. 이번 5종은 실제 사전학습 파일과 C++ 실행기를
지정해 검사했고, 가중치 누락 시 실패하도록 설정했다.
수정 파일의 Ruff 검사, `git diff --check`, CI YAML 파싱과 배포 카탈로그 일치 검사도 통과했다.

## 640×640 확인

5종 모두 실제 ImageNet 백본, RGB·배치 1·3클래스로 학습 스텝 후 ONNX로 내보냈다.
PyTorch 기준과 동일 입력을 비교했으며 기존 검증 허용 오차를 변경하지 않았다.

| 모델 | 최대 절대 오차 | 학습/내보내기 |
| --- | ---: | --- |
| U-Net MobileNetV4 Conv-S | 1.1921e-7 | 통과 |
| U-Net ShuffleNetV2 x1.0 | 8.9407e-8 | 통과 |
| U-Net EfficientNet-B1 | 1.5498e-5 | 통과 |
| BiSeNetV2 | 1.9455e-4 | 통과 |
| STDC1 | 2.3604e-5 | 통과 |

## 리뷰 및 회귀 방지

- **STDC 전체 로드:** 원본 파일의 canonical 187개 텐서를 확인하고 중복 alias까지
  값이 같은지 검사했다. random alias가 이미 로드한 사전학습 값을 덮는 경로를 제거했다.
- **BiSeNetV2 전체 로드:** detail 48개 / segment 239개 / bga 32개 텐서를 모두 검사했다.
  부분 파일·다른 구조·비정상 값을 정상 사전학습으로 처리하지 않는다.
- **흑백:** 두 모델의 RGB 필터를 로드한 후 입력 채널을 합산한다. BiSeNetV2는 두 입력 경로를 모두 검사했다.
- **오프라인 복원:** 저장된 Studio .pt를 불러올 때 사전학습 파일을 다시 찾지 않는지 검사했다.
- **백본 고정:** 새 5종의 백본 파라미터와 BatchNorm 통계는 유지되고 새 헤드는 갱신되는지 검사했다.
- **재개:** 13.84의 단일 optimizer 그룹으로 만든 두 모델의 체크포인트를 재개해 다음 에폭을 완료했다.
- **배포:** forward/레이어/출력 형태와 SDK 계약은 유지했다. C++ 테스트는 Release 바이너리다.
- **무결성:** 두 새 파일의 전체 SHA256을 고정하고 `weights_only=True`로 로드한다.
  인증서 검사를 끄지 않으며 배치 실행은 설치된 PyTorch/CUDA를 교체하지 않는다.

Windows/Linux CI에서 실제 공식 파일 다운로드 후 이 검사를 실행하도록 연결했다.
Windows/CUDA 학습, 사내망 다운로드, 사용자 데이터의 정확도와 CPU 속도는 여기서 실측하지 않았다.
원본 로컬 로그/산출물: `runs/seg-pretrained-20261111/` (Git 제외).
