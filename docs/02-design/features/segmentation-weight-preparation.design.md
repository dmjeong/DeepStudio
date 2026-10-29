# 기본 가중치 준비 설계

- `prepare_builtin_assets.py`의 기본 경로를 `builtin_asset_root()`로 통일. `--group segment`, `--check`, 기존 `--only` 및 오프라인 등록 지원. 모델별 결과를 남기며 일부 실패에도 나머지 준비를 계속한다.
- `gui/prepare_model_weights.bat` 기본 동작은 전체 가중치 준비. `segment`는 ResNet18/34 + SAM2 Tiny/Small/Base+/Large, `check`는 다운로드 없는 검사. `build.bat weights`는 무거운 EXE 의존성/빌드 이전에 이 도구로 분기한다.
- SAM2 공식 URL과 공개 SHA256을 고정. TLS 검증을 유지하는 공통 다운로드기를 사용하고 완료·해시 검증 후에만 파일을 게시한다. 공식 원본과 같은 해시의 수동 파일은 manifest가 없어도 등록/사용 가능. 기존 manifest 손상은 우회하지 않는다.
- 학습 모드 설명에 선택 모델의 파일 유무와 복구 배치를 표시. 실행 전에는 해시까지 검증하고 원래 오류를 보인다. SAM2에 scratch를 권하지 않는다.
- 모델 구조/학습/추론/ONNX 계산은 변경하지 않는다. SAM2 일반 semantic 추론 연결은 별도 미완료 항목으로 유지한다.
