# ONNX Runtime / C++17 추가 최적화

2026-09-18. 선택한 배포 경로는 **Windows x64 + C++17 + ONNX Runtime CPU**다.
OpenVINO는 비교용 선택 구성으로 남겨 두었고 기본 사용 안내는 ONNX Runtime으로 바꿨다.
정확한 i7 모델을 몰라도 아래 도구가 Windows의 CPU 이름을 읽고 설정을 비교한다.
Windows/i7에서 직접 실행한 결과는 아직 없다.

## 이번에 반영한 변경

- 분류 입력/출력 float 버퍼와 Ort::Value를 모델 초기화 때 만들고 반복 재사용한다.
- 전처리가 재사용 입력 버퍼에 직접 쓴다. OpenCV 색 변환/resize 버퍼도 재사용한다.
- 입력과 목표 크기가 같으면 identity resize와 중간 이미지 복사를 생략한다.
  행별 접근을 유지해 연속 메모리가 아닌 ROI도 처리한다.
- 전처리 정규화 계산 순서, 채널 변환, FP32, 224 크기는 유지한다.
- 백그라운드 제출 이미지 복사는 유지한다. 반환 직후 카메라 버퍼를 재사용해도 안전하다.
- JSON에 ONNX Runtime spinning/작업 분할 옵션을 추가했다. 지정하지 않으면 SDK의 기존
  동작을 유지한다. 설정 비교 도구가 실제 C++ 실행 파일로 측정하고 새 JSON을 만든다.

모델 인스턴스는 재사용 scratch를 가지므로 동시에 Classify하지 않는다.
`ClassificationWorker`는 한 worker에서 순차 호출하며, 각 결과는 독립된 확률 벡터를 소유한다.

## 얼마나 줄었나

Apple M4, ORT 1.30.0, C++17 Release, B0 FP32, 224×224 gray, 2 classes,
고정 시드 임의 가중치. 이전 커밋 `1d16b09`의 실제 C++ 실행 파일과 변경 실행 파일을
번갈아 측정했다. 각 조건 3회 × 1,000개, 회당 warmup 50개, 4 threads.
한 번에 한 요청이며 파일 디코딩/초기화는 제외했다.

| 항목 | 이전 | 변경 후 |
|---|---:|---:|
| 동기 전처리 평균(3회 평균) | 0.0401 ms | 0.0353 ms |
| 백그라운드 전처리 평균(3회 평균) | 0.0388 ms | 0.0341 ms |
| 동기 wall p95(3회 중앙값) | 5.423 ms | 5.461 ms |
| 백그라운드 wall p95(3회 중앙값) | 5.857 ms | 5.460 ms |
| 백그라운드 wall 최대(전체 3,000개) | 9.837 ms | 35.129 ms |

전처리에서 약 **0.005 ms** 절감했다. 전체 p95는 실행별 변동이 있으므로 큰 속도 향상이나
최대 지연 개선을 입증한 결과로 해석하지 않는다. 특히 변경 후에도 35 ms 이상인 외부
스케줄링 등을 포함한 지연 표본이 관측됐으며 원인은 프로파일링으로 확정하지 않았다.
모든 요청의 8 ms 마감은 달성하지 않았다.
[회차별 측정 통계와 환경](benchmarks/onnx-reuse-m4-20260918.json)

추가 스레드/작업 분할 탐색에서 M4의 6/8 threads 또는 dynamic_block_base=4가
4 threads보다 일관되게 빠르지 않았다. 이를 기본 최적값으로 강제하지 않았다.
i7의 P/E core와 OS 스케줄링 특성은 별도 측정해야 한다.

실제 C++ 비교 도구로 16개 설정을 각 2회 × 200개 측정한 결과도 4 threads,
기본 spinning/기본 block 설정을 선택했다. 새 프로세스에서 3,000개 재확인 결과는
p95 5.467 ms, 최대 7.712 ms였다. 이 확인 회차만 모두 8 ms 이내였으며 위 A/B 회차의
35.129 ms 관측을 없애거나 보장으로 바꾸는 근거는 아니다.

검증: OpenVINO OFF/ON 구성의 CTest 각 4개 통과, C++ 설정 비교/전처리 테스트 10개 통과,
실제 gray/RGB B0 export의 Python↔C++ 결과 비교 24개 조합 통과. 설정 비교 테스트는
선택 JSON을 다시 읽었을 때 thread 수를 유지하는 것과 원본 JSON 미수정도 확인한다.

## Windows에서 설정 자동 비교

[C++17 빌드 안내](EFFICIENTNET_CPP17.md)의 ONNX Runtime 전용 Release 빌드를 사용한다.
아래 비교 스크립트는 Python 표준 라이브러리만 필요하다. C++ 앱 실행에는 Python이 필요 없다.
프로젝트 `DeepVisionStudio`에서:

```bat
python tools\tune_efficientnet_cpp.py --executable build-cpp\Release\efficientnet_cpu_benchmark.exe --config deploy\model.json --image test.png --threads 1 2 4 6 8 --block-bases 0 4 --spinning default off on --rounds 2 --runs 200 --output tuned-onnx
```

- CPU 이름을 Windows 레지스트리에서 읽는다. 모르면 이름/OS를 수동 입력할 필요가 없다.
- thread 수, spinning, dynamic block 설정을 순차 측정한다. 동시 성능 테스트는 피한다.
- 기본 범위는 Submit의 복사부터 대기열, 전처리/추론/후처리, future 결과 수신까지다.
  `--direct`는 동기 Classify를 측정한다.
- 회차마다 순서를 섞고 **회차 중 가장 높은 p95**가 가장 작은 설정을 선택한다.
  최저 중앙값만 뽑아 최대 지연을 숨기지 않도록 각 회차 max/8 ms 비율도 보관한다.
- 테스트 이미지에 대한 클래스/확률 일치와 실제 적용 설정을 확인한다.
  한 이미지의 수치 비교는 제품 정확도 검증을 대신하지 않는다.
- `summary.json`, 각 회차 원시 시간 JSON, `recommended.json`을 만든다.
  원래 모델/JSON은 변경하지 않고 기존 출력 폴더를 덮어쓰지 않는다.

선택된 설정을 새 프로세스에서 길게 확인한다. 결과가 목표를 넘으면 exit 2다.

```bat
build-cpp\Release\efficientnet_cpu_benchmark.exe --config tuned-onnx\recommended.json --image test.png --background --runs 3000 --warmup 50 --require-target --output tuned-onnx\confirmation.json
```

`--threads`를 생략해야 JSON의 선택값이 유지된다. 별도 부하가 있는 실제 앱에서도 확인한다.
위 기준은 wall **p95 ≤ 8 ms**이며 모든 요청 ≤ 8 ms와 다르다.

앱에서는 선택된 JSON을 읽고, 스레드 인자를 **-1**로 지정한다:

```cpp
ClassificationWorker inference("tuned-onnx/recommended.json", "onnxruntime", -1, 2, 30);
// -1: JSON의 num_threads 사용. Ready/Submit/future/Stop은 기존 사용법과 동일.
```

recommended.json은 원본 ONNX를 가리킨다. 다른 PC로 배포할 때는 모델도 복사하고
JSON의 model_path를 새 상대 경로로 맞춘다.

선택적 JSON 설정 예시(이 값 자체가 권장 최적값이라는 뜻은 아니다):

```json
{
  "num_threads": 4,
  "onnxruntime": {"allow_spinning": true, "dynamic_block_base": 4}
}
```

spinning은 응답성과 CPU/전력 사용 사이의 선택이다. 생략하면 설치된 SDK 기본값을 따른다.
정확한 코어 번호를 모르는 상태에서 P-core affinity나 높은 프로세스 우선순위를 강제하지 않는다.
[ORT 스레드 설정](https://onnxruntime.ai/docs/performance/tune-performance/threading.html),
[작업 분할과 지연 변동](https://onnxruntime.ai/docs/performance/tune-performance/troubleshooting.html)

## 더 큰 개선 후보: static INT8 QDQ

이미 Conv–BN 융합, ORT_ENABLE_ALL, sequential 실행, CPU arena/memory pattern,
모델 1회 로딩, 워밍업을 적용했다. 반복 버퍼 절감 외에 큰 연산량 차이는 양자화에서
기대할 수 있지만, 실제 속도 이득은 CPU 명령어 지원/모델 연산에 따라 달라진다.
ORT는 CNN에 static quantization을 권장한다. EfficientNet은 대표 검사 이미지로
calibration한 뒤 FP32 대비 정확도·클래스별 recall·불량 누락과 C++ 지연을 함께 비교해야 한다.
실제 학습 모델과 검증 데이터 없이 임의 가중치 INT8 결과를 제품 개선으로 채택하지 않았다.
[ORT 양자화 공식 문서](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

가중치 권한 검토는 사용자 요청대로 제외했다. 새 최적화는 별도 라이브러리를 추가하지 않는다.
기존 ONNX Runtime/OpenCV의 고지와 배포 조건은 그대로 따른다.
