# 9.45 — ResNet 수동 가중치와 중단된 다운로드 복구

버그 수정 +0.01. 다운로드된 모델 전체가 성공해야 manifest가 만들어지던 문제와,
올바른 공식 ResNet 파일을 수동 설치해도 manifest가 없다고 막히던 문제를 수정했습니다.

## 사용법

소스에서 실행한다면 기존 `gui/builtin_assets/torchvision/resnet18-f37072fd.pth`는 그대로 두면 됩니다.
9.45는 공식 SHA256 접두사가 맞는 파일을 manifest 없이도 읽습니다. 임의 파일 이름 변경이나 검증 해제는 필요 없습니다.

가중치만 준비하려면 `gui/prepare_resnet_weights.bat`을 실행하세요. ResNet18·34·50만 받으며 EXE 빌드는 하지 않습니다.
이미 있는 공식 파일은 검증하고 manifest에 등록합니다. 다른 다운로드가 실패해도 성공 항목을 유지합니다.

인터넷 없이 수동 파일을 등록하려면 `gui` 폴더의 명령 프롬프트에서 실행합니다.

```bat
prepare_resnet_weights.bat offline
```

ResNet18 하나만 등록하려면 같은 폴더에서 다음을 실행합니다. 이 명령은 9.45에서 제공하는 옵션입니다.

```bat
python ..\python\prepare_builtin_assets.py --output builtin_assets --only resnet18 --register-existing
```

등록한 파일이 없는 다른 모델은 실패 목록에 나타납니다. 성공한 ResNet18의 등록 정보는 그대로 남습니다.

## EXE 실행과 경로

`gui/builtin_assets`는 소스 실행/빌드에 사용하는 폴더입니다. 이미 만들어진 EXE는 자신의 설치 폴더에 포함된
`builtin_assets`를 사용합니다(현재 onedir 배포는 `dist/DeepVisionStudio/_internal/builtin_assets`).
오류 메시지에 표시되는 실제 파일 경로를 우선 확인하세요. EXE에 수동 파일을 추가했다면 아래 명령의 출력 경로를
해당 EXE의 `builtin_assets` 폴더로 지정하면 됩니다. 앱을 닫은 뒤 실행하고 다시 시작하세요.

```bat
python ..\python\prepare_builtin_assets.py --output "dist\DeepVisionStudio\_internal\builtin_assets" --only resnet18 --register-existing
```

9.44 이하 EXE는 여전히 manifest를 요구하지만, 위 도구로 등록한 ResNet18 파일을 사용할 수 있습니다.
9.45 프로그램 로직을 EXE에 반영하려면 EXE 업데이트/빌드는 한 번 필요합니다.

## 버전·검증 원칙

- ResNet18: `resnet18-f37072fd.pth`, ResNet34: `resnet34-b627a593.pth`.
- ResNet50: `resnet50-11ad3fa6.pth` (V2). 이전 준비 코드의 V1 파일 `resnet50-0676ba61.pth`를 이름만 바꾸면 안 됩니다.
- 기존 manifest의 해시와 다른 파일, 손상된 manifest, 비공식 PT 파일을 자동 승인하지 않습니다.
- 모든 기본 모델과 완전한 manifest가 없으면 EXE 전체 빌드는 계속 중단합니다.
- 모델마다 완료 항목을 임시 JSON에 기록한 뒤 원자적으로 교체합니다. 실패 로그에는 모델명과 원인이 표시됩니다.

[검증 보고서](03-analysis/pretrained-manifest-recovery.analysis.md)
