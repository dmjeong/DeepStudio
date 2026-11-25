# 분할·Dinomaly C++17 예제

명령행 인자를 쓰지 않는다. `main.cpp`의 `encrypted`, `threads`를 필요에 맞게 바꾼다.

1. `paths.example.bat`을 `paths.bat`으로 복사하고 이미 사용 중인 ONNX Runtime 폴더와 OpenCV 폴더를 적는다.
2. `build.bat`을 실행한다. Release/x64의 `dense_model_example.exe`가 생성된다.
3. EXE 옆에 `model.json`, JSON이 참조하는 `model.onnx`, `image.png`를 둔다.
4. 모델을 한 번 읽고 워밍업한 뒤 두 번째 추론의 전처리·모델·후처리·총 시간을 출력한다.

분할은 원본 좌표의 `classes.png`(픽셀=클래스 번호), 이상 탐지는 원시 점수 `anomaly_scores.f32`와 크기를 기록한 `anomaly_scores.json`을 저장한다.
점수 파일은 행 우선 순서의 little-endian float32이며 별도 영상 코덱이 필요 없다.
비교용 `original.png`는 입력 픽셀을 유지한다. 검정에 가까운 `classes.png`는 0,1,2 같은 클래스 번호이므로 정상이다.
JSON에 판정 기준이 있어야 분할 결과의 `defects` 목록이 생성된다.

암호화 모델은 `encrypted=true`로 빌드하고 `model.dvsenc`, 별도 키 `model.key`를 둔다.
메모리에서 복호화한다. 키 취급은 실제 제품의 라이선스/보안 정책에 맞춘다.
FP32와 INT8 사용법은 같다. `.onnx`만 교체하지 말고 대응하는 JSON을 함께 배포한다.

내 솔루션에서는 모델 객체를 멤버로 보관하고 시작 시 한 번 초기화한다. 반복 검사에서는 다음만 호출한다.

```cpp
// image는 현재 프레임의 BGR 또는 BW8 cv::Mat. JSON의 크롭/패치는 런타임이 수행한다.
const auto segmentation = model.Segment(image);
// Dinomaly 프로젝트라면:
const auto anomaly = model.Anomaly(image);
```

SDK 소스는 `cpp/include`, `cpp/src` 전체가 필요하다. 설치본에는 `../vision-runtime`으로 함께 제공된다.
헤더 하나만 복사하는 형태가 아니다. OpenCV·ONNX Runtime의 헤더/라이브러리/DLL도 필요하다.
컴파일러와 DLL 비트수는 x64로 맞춘다. VS2017은 저장소의 기존 호환 헤더 준비 절차를 적용한다.
