# Deep Vision Studio Desktop 11.60

PatchCore ONNX를 내보내도 JSON에 `input_name`이 빠져 C++ SDK가 로드하지 못하던 문제를 수정했습니다. 실제 ONNX 그래프에서 입력·출력 이름을 읽어 JSON에 기록합니다. 모델 연산과 판정 임계값은 바꾸지 않습니다.

기존 PatchCore 체크포인트를 다시 ONNX로 내보내면 적용됩니다. 재학습은 필요하지 않습니다.

회귀 검사는 실제 PatchCore 체크포인트를 내보내 C++ 로더로 열고, 원본/중앙 크롭 및 이전/현재 점수 방식에서 점수·이상 지도·판정을 비교합니다. C++ 지도는 같은 그래프를 실행하는 Python ONNX Runtime과 비교하며, 학습 외 입력에서는 PyTorch 지도와도 비교합니다. `ANOMALY_PROBE`에 `cpp` 빌드의 실행 파일 경로를 설정해 `tests/test_cpp_patchcore_export.py`를 실행합니다.

정상 이미지만 있는 경우 자동 임계값이 미설정으로 남는 정책과, 큰 영상의 작은 결함이 입력 리사이즈로 사라질 수 있는 제약은 이번 수정 범위에 포함되지 않습니다. 상세 결과는 `docs/03-analysis/patchcore-health-20261110.md`를 참고하세요.
