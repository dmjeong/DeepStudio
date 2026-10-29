# 9.57 — 기본 세그멘테이션 가중치 준비 복구

Python 실행 사용자는 저장소에서 아래 명령을 실행한다. EXE를 새로 만들 필요 없다.

```bat
git pull --ff-only
gui\build.bat weights segment
python gui\main.py
```

이 명령은 U-Net의 ResNet18, DeepLab의 ResNet34, SAM2 Tiny/Small/Base+/Large를 준비한다. **전체 기본 모델**은 `gui\prepare_model_weights.bat`을 실행한다. 이미 정상적으로 준비한 파일은 재사용한다. 화면에 표시된 `[assets]` 경로가 앱 검색 경로다. 모델마다 `[ready]` 또는 원인을 포함한 `[failed]`를 출력한다. 하나가 실패해도 나머지는 계속 준비하며 성공 파일을 보존한다.

다운로드 없이 검사: `gui\build.bat weights check`. 신규 Python 환경 전체 설치: `gui\build.bat setup gpu`. 둘 다 EXE 빌드는 하지 않는다.

SAM2 공식 가중치를 직접 넣은 경우에도 공개 SHA256이 일치하면 manifest 없이 인식한다. 다른 버전/손상 파일 또는 기존 manifest와 불일치하는 파일은 거부한다. SSL 검증을 끄지 않는다.

검증: 공식 SAM2 4종 다운로드·해시·모델 로드·이미지 연산 성공(macOS CPU). U-Net 공식 ResNet18 1/3채널 기능 검사 완료. 전체 테스트 1,258개 통과·12개 건너뜀. 이후 상대 경로/안내 보강 검사 46개 통과. 모든 프로세스 종료 코드 0.

Windows 네이티브 배치 실행은 미검증이다. SAM2 일반 semantic 추론 연결은 별도 미완료 항목이며 이 릴리스는 가중치 준비만 복구한다. 자세한 근거는 [분석](03-analysis/segmentation-weight-preparation.analysis.md) 참조.
