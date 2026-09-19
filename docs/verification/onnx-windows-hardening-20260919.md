# ONNX·Windows 배포 강화 검증 기록

검증일: 2026-09-19

## 적용 범위

- EfficientNet CPU ONNX 준비 단계가 fused 그래프의 수치 검증에 실패하면 원본 unfused 그래프를
  ONNX Runtime 최적화 `all`, `basic`, `disabled` 순서로 다시 검증한다. 통과한 그래프와 최적화
  설정은 추론 결과에 기록하며, 모든 조합이 실패하면 검증되지 않은 ONNX 세션을 사용하지 않는다.
- 기본 ResNet, ConvNeXt V1, DeepLab V3+, U-Net 모델에 학습·추론·Grad-CAM·ONNX 공통 런타임
  속성을 부여했다. UI 학습 설정의 optimizer, scheduler, augmentation, freeze, resume와 학습 이력을
  실제 기본 모델 학습기에 전달하고 checkpoint에 저장한다.
- SAM2 decoder의 point 입력 축을 동적으로 내보내고 1·2·3·8개 prompt로 ONNX Runtime 출력 비교를
  수행한다. C++ image context는 모델 재초기화 뒤 폐기되며, 자동 mask threshold와 단계별 시간을
  일관되게 반환한다.
- C# P/Invoke는 `SafeHandle`을 직접 전달해 native 호출 중 session과 SAM image context가 해제되지
  않도록 했다.
- Windows 앱 전용 WSL 경로와 marker 인코딩을 설치기와 런타임에서 일치시켰고, Windows 경로를
  owned WSL의 `wslpath`로 변환한 뒤 Docker bind mount에 사용한다.
- `.dvmodel`은 파일 SHA-256에 더해 Ed25519 서명을 검증한다. production catalog gate는 버전 1
  공개키 신뢰 저장소가 없거나 pack 서명·checksum·license가 하나라도 맞지 않으면 설치파일 생성을
  중단한다.

## 실행 결과

| 검사 | 결과 |
|---|---:|
| Python 전체 회귀 | 766 passed, 13 skipped, 0 failed |
| 최종 변경 관련 Python 재검증 | 73 passed, 0 failed |
| C++17 Release 빌드와 CTest | 4/4 passed |
| Web Node 테스트 | 11/11 passed |
| Web TypeScript·Vite production build | passed |
| `git diff --check`, Python compile | passed |

전체 Python 검사는 macOS ARM64, Python 3.11, PyTorch·ONNX·ONNX Runtime·PySide6가 설치된 격리
환경에서 실행했다. C++ 검사는 Release 구성에서 실제 ONNX Runtime fixture와 C ABI를 사용했다.

## Windows release gate

현재 개발 장비에는 `dotnet`, `pwsh`, `wix`가 없어 C# DLL과 WiX MSI/Burn EXE를 로컬에서 만들지
않았다. `deepstudio3` push 후 Windows workflow가 C# SDK, C++17 DLL, PyInstaller GUI, WiX 7.0.0
설치파일 계약을 실행한다. production 플래그는 별도 모델 payload와
`DEEPVISION_MODEL_PACK_TRUST_STORE`가 모두 제공될 때만 활성화할 수 있다.

카탈로그에서 `requested` 상태인 Re-DETR v4 Small/Medium/Large, SAM2.1 네 변형과 LibreYOLO 두
변형은 실제 upstream source revision, 모델 자산, 라이선스 고지, Windows Docker 실행 결과가 아직
없다. 이 항목은 컨테이너 모델 팩 프로토콜과 네이티브 ONNX 계약까지만 구현되어 있으며
`release_ready`로 표시하지 않는다. 검증되지 않은 placeholder를 기본 모델로 설치하는 대신 strict
release gate가 해당 설치파일 생성을 거부한다.
