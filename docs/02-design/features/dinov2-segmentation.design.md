# DINOv2 세그멘테이션 설계

## 구조와 계약
- ID `dinov2_vits14`, 기본 입력 392×392, 1/3채널. 고정 공간 크기 + dynamic batch ONNX.
- Meta 기본 ViT-S/14와 같은 384차원/12 block/6 head/MLP4/LayerScale/LayerNorm eps1e-6.
  공식 state_dict 키를 그대로 strict load한다. xformers/torch.hub/새 패키지 없이
  표준 PyTorch 연산으로 구성하며 Apache-2.0 출처와 수정 사항을 명시한다.
- 입력을 14의 배수로 오른쪽/아래 replicate pad. patch 위치 임베딩은 공식 bicubic
  offset=0.1 규칙을 따른다. 3/6/9/12번째 block의 정규화 특징을 융합한다.
- decoder: 각 특징 32채널 projection → concat → 64채널 fusion. 원본 입력에서
  stride1의 16채널 detail 경로를 얻고 전체 크기에서 결합하여 클래스별 logits 출력.
  고해상도 특징을 유지하지만 정확도 향상을 사전에 보장하지 않는다.
- 흑백 입력은 기존 GRAY 정규화(0.449/0.226)를 역변환해 RGB 사전학습 정규화로
  정확히 변환한다. 3채널은 기존 RGB 정규화를 그대로 사용. 변환은 모델 안에 포함되어
  Python/C++/C#의 기존 전처리 계약이 바뀌지 않는다.
- 초기 가중치는 backbone만. detail/decoder는 학습 필요. backbone freeze 시 decoder와
  detail은 학습 가능. 전체 미세조정/로컬 전이/스크래치/재개 계약을 공통 trainer로 처리.

## 배포와 자산
- 공식 파일: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
- SHA256: b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9
  (공식 HTTPS 파일을 내려받아 계산해 버전 고정; 배포자 서명이라는 의미 아님).
- `gui/builtin_assets/dinov2/`에 준비. --group segment와 전체 build/setup에 포함.
- 일반 segmentation ONNX logits 계약과 암호화 wrapper를 사용. 이름/전처리/클래스와
  선택된 ORT 설정은 기존 JSON에 기록. 추론 결과 표시는 공통 segmentation renderer 사용.
- 앱 등록, 모드 문구(ImageNet만이라고 쓰지 않음), frozen hidden imports, installer catalog,
  라이선스 목록을 함께 갱신한다.

## 검증
- 공식 CPU backbone reference와 224/직사각형 입력 특징 일치 비교.
- 무다운로드 생성, pretrained strict 로드, 1/3ch와 비배수 크기, 작은 특징 경로 gradient.
- 합성 마스크 실제 train/save/reload/resume/전이, GUI 모델 선택 및 준비 안내.
- FP32 ONNX 출력 비교, dynamic batch1/2, 암호화 전후 완전 동일 출력 및 실제 loader.
- macOS CPU 측정은 장비/입력/스레드/워밍업/median/p95를 기록. Windows/CUDA 및
  실제 검사 불량 검출률은 미검증으로 명시. 기존 전체 회귀 검사 수행.

PDCA 도구는 호출 대화의 원래 cwd를 사용하므로 이 worktree 문서를 찾지 못한다.
설계 게이트와 상태 기록은 이 worktree의 docs/.pdca-status.json에서 관리한다.
