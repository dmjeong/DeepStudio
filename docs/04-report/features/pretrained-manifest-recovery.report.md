# 9.45 완료 보고 — ResNet 가중치 준비

- 다운로드 실패 전 완료된 가중치는 모델별 manifest 등록으로 보존된다. 후속 모델도 계속 준비하며 실패 원인을 모아 보고한다.
- 공식 ResNet 파일을 수동으로 넣은 경우 manifest가 없어도 SHA256 접두사를 검증해 오프라인으로 로드한다.
- 가중치만 준비하는 배치파일과 오프라인 등록 옵션을 제공한다. 전체 EXE 빌드의 완전성 검사는 유지한다.
- ResNet50의 실제 파일을 학습 설정의 V2와 일치시켰다. 기존 학습 체크포인트는 변경하지 않았다.
- 관련 검사 78 통과. 공식 ResNet18·34·50 다운로드, manifest 삭제 후 오프라인 로드,
  ResNet18·50/U-Net/DeepLab CPU 순전파 및 manifest 재등록 성공.
- 최종 전체 검사 1,214 통과 / 12 건너뜀, 종료 코드 0. 공통 Qt 테스트 정리를 보완해 종료 오류도 재검사했다.
- Windows 실제 배치 실행/EXE 빌드는 이 macOS 환경에서 수행하지 않았다. Windows 배치 정적 검사와 같은 Python 코드 실행을 검증했다.

[사용법](../../RELEASE_9.45.md) · [검증](../../03-analysis/pretrained-manifest-recovery.analysis.md)
