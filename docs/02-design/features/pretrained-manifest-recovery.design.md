# 사전학습 가중치 설치 복구 — 설계

- Torchvision 공식 URL은 `builtin_assets.ASSET_FILES`의 고정 파일명에서 구성한다. ResNet50은 torchvision의 V2와 동일한 `resnet50-11ad3fa6.pth`를 사용한다.
- 앱은 다운로드하거나 manifest를 자동 수정하지 않는다. manifest/해당 항목이 없을 때만 공식 Torchvision 파일명에 고정된 SHA256 접두사로 검증한다.
  기존 항목이 있으면 기록된 전체 SHA256을 비교하고, 손상된 manifest나 불일치는 우회하지 않는다.
- prepare는 선택 모델마다 검증·다운로드 후 전체 SHA256과 크기를 임시 JSON 파일에 저장하고 원자적으로 교체한다.
  모델 다운로드 실패를 수집하고 다음 모델을 계속 준비한다. 마지막에 실패 모델 목록과 원인을 내보내며 종료 코드는 실패다.
  manifest 저장 자체가 실패하면 즉시 중단하여 가중치 준비 완료를 잘못 보고하지 않는다.
- `--register-existing`은 다운로드 없이 검증된 로컬 파일만 등록한다. 공식 해시가 없는 모델은 기존 유효 manifest가 필요하다.
- `gui/prepare_resnet_weights.bat`은 ResNet18/34/50을 준비한다. `offline` 인자를 주면 수동 설치한 해당 파일만 등록한다.
- EXE 패키징에는 계속 완전한 manifest와 모든 기본 모델이 필요하다. 런타임의 오프라인 인식과 패키지 완전성 검증은 분리한다.
