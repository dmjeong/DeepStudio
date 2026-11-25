# 사전학습 가중치 준비·수동 파일 인식 검토

## 확인한 원인

1. `prepare_builtin_assets.prepare()`가 모든 다운로드가 끝난 뒤에만 manifest를 만들었다.
   중간 오류로 반환하면 앞에서 받은 파일의 등록도 없고 뒤의 모델 다운로드도 시작하지 않았다.
2. `builtin_asset_path()`는 파일이 있어도 manifest 또는 항목이 없으면 즉시 실패했다.
   수동으로 설치한 공식 ResNet 파일을 인증할 별도 경로가 없었다.
3. ResNet50은 학습 메타데이터가 V2를 주장하지만 준비 스크립트는 V1 파일을 받았다.

사용자가 확인한 경로는 `gui/builtin_assets/torchvision/resnet18-f37072fd.pth`다.
사용자 PC의 다운로드 자체가 중단된 원인(네트워크·인증서·선행 모델 실패 등)은 해당 로그 없이는 단정할 수 없다.

## 수정과 검토

- 성공한 모델마다 전체 SHA256·크기를 manifest에 원자적으로 기록. 다음 모델 실패/KeyboardInterrupt에도 완료 항목 유지.
- 개별 다운로드 실패는 모델별로 기록하고 나머지 모델을 계속 시도. 마지막 종료 코드는 실패이며 불완전한 EXE를 빌드하지 않음.
- 수동 파일은 manifest/항목이 없을 때 고정된 공식 Torchvision SHA256 접두사로 검사. 앱 실행 중 다운로드·manifest 쓰기를 하지 않음.
- 기존 항목의 해시·크기 불일치, 손상 JSON, 비공식 파일의 manifest 부재는 거부. 전체 패키지 완전성 검사는 계속 manifest 필수.
- ResNet50 V2 파일명과 torchvision 학습 레시피를 일치시킴. 기존 학습 체크포인트나 사용자 파일을 재작성하지 않음.
- 가중치 전용 Windows 배치파일 및 오프라인 등록 옵션 추가. 전체 EXE 빌드와 분리.

## 실행 검증

- 수정 전 새 회귀 테스트: 7 실패 / 5 통과로 수동 파일 거부·중단된 manifest·ResNet50 불일치 재현.
- 수정 후 관련 테스트: **78 통과**. 이 중 신규 18개는 실패 뒤 후속 모델 계속 준비, 증분 저장, atomic replace 실패,
  사용자 중단, 기존 항목 유지, 손상 파일 거부, CLI 실패 코드, 배치 경로 등을 포함한다.
- 공식 서버에서 ResNet18(V1), ResNet34(V1), ResNet50(V2)를 실제 다운로드하고 SHA256 확인.
- manifest 삭제 후 네트워크 경로를 실패하도록 바꾸고도 세 모델의 파일 인식 성공.
- 실제 ResNet18(1채널), ResNet50(3채널), U-Net ResNet18(1채널), DeepLab V3+ ResNet34(3채널)의
  사전학습 로드와 CPU 순전파 성공. 출력 형상·유한값 검사. 분류 224², 분할 64² 생성 입력.
- 수동 파일을 오프라인으로 일부만 등록하고 나머지를 추가해 manifest 병합 확인.
- 첫 전체 검사는 1,214개가 통과했으나 macOS PySide6의 종료 시 QObject/QPushButton 정리에서 종료 코드 139가 발생했다.
  테스트 공통 teardown에서 모든 fixture 종료 후 DeferredDelete를 처리하도록 보완했다(제품 코드 변경 아님).
- 최종 전체 테스트: **1,214 통과 / 12 건너뜀**, 154.82초, **종료 코드 0**.
  로그 `/tmp/dvs-pretrained-final.log`, JUnit `/tmp/dvs-pretrained-final.xml`.
- 변경 파일 Ruff E/F/W 및 `git diff --check` 통과.

[실행 결과·파일 해시](artifacts/pretrained-manifest-runtime.json)

검증 환경은 macOS/Python 3.12다. Windows 배치파일은 정적 계약 검사만 수행했으며 실제 cmd.exe와 EXE 빌드는 실행하지 않았다.
순전파 검증은 가중치 로드 연결 검사이며 검사 데이터 정확도를 주장하지 않는다.
