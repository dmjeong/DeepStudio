# Deep Vision Studio 7.01

`example/cpp/with_opencv`와 `example/cpp/with_evision`으로 예제를 분리했다.
OpenCV 경로는 기존 엔진을 유지한다. eVision 경로는 OpenCV와 기존 cpp 엔진에 의존하지
않는 BW8 전용 ONNX Runtime 분류 엔진이다. 신규 독립 추론 경로 추가로 +1.0을 적용했다.

eVision 앱에는 `classifier.h`, `bw8_preprocess.h`, ONNX Runtime SDK, nlohmann-json 헤더가
필요하다. `InferEvision(image)`는 SDK 객체의 BW8 형식을 확인하고 포인터·행 간격을 전달한다.
`InferBW8(pointer, width, height, row_pitch)`로도 호출할 수 있다. 실제 eVision SDK는
사용자의 기존 프로젝트/라이선스를 사용하며 여기에 포함하거나 재배포하지 않는다.

시작 시 모델 로드와 준비 추론 1회를 완료하며, 실패하면 생성자가 예외를 반환한다.
추론은 동기식이고 버퍼 수명은 호출자가 유지한다. 전처리는 JSON의 중앙 crop과
uint8 bilinear resize, 1/3채널 정규화 계약을 따른다. 합성 ONNX와 OpenCV 비교 테스트를
제공한다. 실장비 카메라 및 라이선스가 필요한 eVision SDK 실행 검증은 별도다.
