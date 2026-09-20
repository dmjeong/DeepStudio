# 4.65 — 기본 모델 가중치 번들화

`build.bat`은 PyInstaller 전에 모든 기본 모델의 사전학습 가중치를
`builtin_assets`에 준비하고 SHA-256 manifest를 생성한다. 이 폴더는 EXE에
포함되며, 실행 중 모델 화면·학습·추론에서 가중치를 내려받지 않는다.

SAM2 Hiera Tiny, Small, Base+, Large의 Settings 다운로드/프로젝트 적용 UI를 제거했다.
분할 프로젝트에서 SAM2 변형을 선택하면 ONNX Export가 번들 공식 checkpoint를 자동 사용한다.
SAM2 prompt mask 미세조정은 semantic class-mask 학습과 데이터 계약이 달라 여전히 학습 버튼에서
시작하지 않으며, 공식 checkpoint의 ONNX encoder/decoder export 경로를 사용한다.
