# 8.22 — C++ 동시 분류의 공유 버퍼 오류 수정

OpenMP 또는 여러 `std::thread`가 같은 분류기에서 추론하면 재사용하는 입력·출력
텐서와 전처리 좌표 캐시를 서로 덮어쓸 수 있었다. 암호화 여부와 관계없이 발생하는
호출부의 공유 상태 문제다. 기존 native C++ 분류기를 공개 합성 모델로 호출한
재현에서는 8,000회 중 3,965회의 결과가 단일 호출 기준과 달랐다.

## 수정

- `VisionInference::Classify()`는 전처리·모델 실행·출력 복사를 잠금으로 보호한다.
  ORT 재사용 텐서와 OpenVINO InferRequest도 이 잠금 안에서 사용한다.
  출력 복사 이후의 후처리는 호출별 데이터로 수행한다.
- eVision/BW8 `Classifier`의 기존 API는 전체 추론을 잠금으로 보호한다.
- 병렬 CPU 추론은 `CreateContext()`로 작업자별 전처리 캐시·입력·출력 텐서를
  시작할 때 한 번 만들어 `InferBW8(context, ...)` 또는
  `InferEvision(context, roi)`로 실행한다. 모델과 ORT 세션은 하나만 유지한다.
  같은 Context를 공유한 호출도 안전하게 순차 처리한다.
- Context는 다른 모델에 사용할 수 없고, 모델은 이동할 수 없다. 작업자 종료 후
  Context와 모델을 해제해야 한다. 입력 이미지와 ROI 위치도 실행 중 바꾸지 않는다.
- `example/cpp/with_evision/parallel.cpp`는 인자 없이 암호화 모델을 한 번 로드하고
  작업자별 준비 추론 후 OpenMP로 실행한다. OpenMP가 없는 환경에서는 std::thread를 쓴다.

ONNX 그래프, 전처리 수식, 정밀도, 암호화 형식과 검증된 실행 설정은 변경하지 않는다.
동시 호출의 버퍼 분리에는 OpenVINO 전환이나 dynamic batch export가 필요하지 않다.
여러 ROI를 배치 하나로 실행하는 것은 별도의 성능 실험이며 이 변경에 포함되지 않는다.

## 검증

로컬 Apple M4 / macOS / C++17 / ONNX Runtime 1.29.0에서 확인했다.

- native Release CTest 6개 통과. 8개 작업자 × 500회 × 평문·암호화로 총 8,000회
  클래스·confidence·확률 배열이 단일 호출 기준과 바이트 단위로 일치했다.
- BW8 Release CTest 5개 통과. 기본 호출·작업자별 Context·같은 Context 공유의
  8개 작업자 호출 총 9,600회에서 결과 불일치와 예외가 없었다.
- OpenMP를 켠 BW8 Debug/Release CTest 각각 5개도 통과했다. 위 검사에 OpenMP 3,200회를 더해
  총 12,800회의 동시 호출을 검사했다. 서로 다른 이미지 크기와 row pitch,
  잘못된 입력 이후 복구, 다른 모델의 Context 거부도 확인했다.
- EfficientNet B0 실제 구조의 기존 로컬 테스트 가중치에서도 ORT 내부 스레드
  1·4개 × 작업자 2·4개 × 평문·암호화 총 960회의 결과가 단일 호출과 일치했다.
  이는 공유 상태 회귀 검증이며 사용자 학습 모델의 정확도 평가가 아니다.
- B0 단일 호출 비교에서 기존/수정 헤더의 p50은 약 4.97~5.06ms 범위였다.
  평균은 실행 구간별 변동을 보였으므로 성능 차이가 0이라고 보장하지 않는다.
  여러 호출이 기본 API의 잠금을 기다리는 시간은 `inference_ms`에 포함되지 않는다.
  병렬 처리 성능은 Context API의 전체 처리 시간을 대상 PC에서 비교해야 한다.

VS2017 v141의 실제 Windows 실행과 eVision SDK 실장비는 로컬에서 확인하지 못했다.
기존 Windows CI는 새 실행 파일과 회귀 검사도 Release/Debug로 빌드·실행한다.
안내와 사용 코드는 `example/cpp/with_evision/README.md`에 있다.
