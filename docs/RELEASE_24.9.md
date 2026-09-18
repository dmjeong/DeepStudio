# 24.9: 데스크톱 Grad-CAM 계산 선택 전달 수정

한 장당 18ms로 느려졌다는 보고를 조사하면서 데스크톱 추론의 설정 전달 오류 두 곳을 확인했다. 이 오류를 실제 모델 속도 저하의 원인으로 확정한 것은 아니다.

- 메모리에 로드한 모델을 비동기 워커로 전달할 때 Grad-CAM 체크 상태를 전달하지 않아, 체크를 꺼도 엔진 기본값 `True`로 계산했다. 이제 실행 시작 시 체크 상태를 전달한다.
- 단일 이미지 직접 실행 경로도 체크 상태를 계산 코드에 전달하지 않았다. 이제 매 실행마다 현재 체크 상태를 읽어 반영한다.
- 체크를 해제하면 다음 실행에서는 Grad-CAM용 추가 forward와 역전파를 요청하지 않는다. 체크하면 기존처럼 계산한다. 저장된 결과의 히트맵 표시 전환은 재추론하지 않는다.

수정 전에는 새 회귀 검사 2개가 실패했고 수정 후 통과했다. 검사는 원본 GUI 메서드를 실행하며 Qt 컨트롤과 실행 워커를 대체 객체로 제공해 설정 전달을 확인한다. 실제 Torch 모델의 속도 측정은 아니다.

관련 검사 66개 중 50개 통과, 16개는 Qt, Torch, OpenCV 또는 ONNX 실행 의존성이 없어 건너뛰었다. 실행 명령:

```sh
python -m unittest tests.test_inference_contracts tests.test_inference_timing tests.test_inference_cache tests.test_inference_region tests.test_efficientnet_channels tests.test_efficientnet_deployment -v
```

## 18ms 보고의 현재 진단 범위

| 표시 또는 경로 | 측정 범위 |
| --- | --- |
| Deep Studio 화면의 추론 시간 | 파일 읽기, 전처리, 모델 실행, 판정 후처리 |
| Deep Studio 화면의 Grad-CAM 시간 | 별도로 수행하는 설명 맵 계산 |
| 화면의 전체 처리 시간 | 추론, Grad-CAM, 미리보기 및 경로별 캐시 저장 등 |
| Python ONNX의 `model_ms` | ONNX Runtime 실행과 출력 유효성 확인 |
| Python ONNX의 `total_ms` | 전처리, 모델 실행, 판정 후처리 |
| Python ONNX의 `file_total_ms` | 파일 디코딩을 포함한 전체 처리 |

화면의 별도 `추론` 항목이 18ms라면 그 안에 Grad-CAM 시간이 포함된 것은 아니다. 이 수정은 Grad-CAM을 끄려던 데스크톱 실행의 불필요한 추가 계산을 제거한다. C++ 또는 Python ONNX의 모델 실행 시간이 18ms인 문제를 이 수정으로 해결했다고 주장하지 않는다.

24.8의 1채널 변경은 첫 Conv와 입력 처리 경로를 바꾼 것이다. 나머지 백본의 연산량을 크게 줄인 변경이 아니므로 전체 지연 감소를 보장할 수 없다. 기존 가중치와 정규화는 이번 수정에서 바꾸지 않는다. 입력 버퍼 재사용은 여전히 미구현이다.

현재 환경에서는 실제 모델 실행이 불가능하므로 보고된 18ms를 재현하거나 수정 전후 지연을 측정하지 못했다. ONNX의 세션별 스레드 수는 기존 `num_threads`로 지정할 수 있지만 실제 측정 없이 특정 값을 강제하지 않았다. 기존 `tools/efficientnet_benchmark.py`는 동일 체크포인트에 대해 워밍업 후 1, 2, 4스레드의 원본 PyTorch, Conv-BN 융합, ONNX 시간을 비교하는 용도다.

이 릴리스의 완료 범위는 Grad-CAM 설정 전달 오류 수정이다. 추론 속도 개선 및 정확도 유지의 실측 검증은 미완료다.
