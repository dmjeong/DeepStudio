# 5.88 — EfficientNet ONNX 고정밀 대체 변환

## 신고에서 확인한 사실

`bugfix`의 5.87 JSON은 B1 구성(69개 BN), 1채널 100×100, zero 입력에서 실패한다.
기존 FP32/BN 후보는 이 입력에서 모두 실패했으며, PyTorch FP64와 원본 FP32의 최대
오차는 0.0005633856944451399로 기존 비교 기준을 통과했다. ORT 최적화 해제만으로는
해결되지 않았고, BN 계수 고정도 해결하지 못했다. 특정 연산이 원인이라고 확정하지 않는다.

## 수정

기존 후보가 실패할 때만 계산을 FP64로 유지하는 `portable_fp64`를 추가한다.
원본 가중치를 수정하지 않고 복사본만 변환한다. ORT CPU에서 FP64 Conv와
GlobalAveragePool이 지원되지 않아 Conv는 unfold와 그룹별 MatMul, pooling은
ReduceMean으로 변환한다. 나머지 모델도 FP64로 계산하며 입출력만 기존 FP32를 유지한다.
고정밀 모델 자체가 아니라 원본 FP32 모델과 비교하며, seeded/zero/dynamic batch 요청 시
batch 2가 기존 허용 오차 및 top-1 검사를 모두 통과해야 저장한다.

JSON의 `export.optimization.compute_precision`은 `float64`, `io_precision`은 `float32`,
`latency_optimized`는 false다. `onnxruntime.graph_optimization_level`은 `disabled`로 저장한다.
이는 ORT의 FP64 미지원 융합 연산으로 재변환되는 것을 방지한다. C++/C#에서는 기존과 같이
ONNX와 함께 생성된 JSON 설정을 적용해야 한다. 이 대체 경로는 더 느리고 메모리 사용이
증가할 수 있으며 CPU 8ms 목표를 보장하지 않는다.

오류 진단도 실패한 모든 후보와 JSON 경로를 표시한다. 최초 불일치 stage 내부에서는
누적 출력 비교와 각 연산에 같은 PyTorch 입력을 넣는 독립 비교를 구분한다.
원시 입력/중간 텐서는 진단 JSON에 기록하지 않는다.

## 검증 범위

실제 ONNX Runtime으로 FP64 Conv의 그룹/stride/dilation과 B1 100×100 배포·재로딩을
검사한다. 고정밀 후보가 원본과 다르면 기존 파일을 덮어쓰지 않는 회귀 검사도 포함한다.
공개 B1 사전학습 가중치로 별도 비교하며, 사용자 비공개 체크포인트 자체는 실행하지 않았다.
