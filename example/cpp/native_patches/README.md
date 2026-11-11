# 패치 분류·탐지 예제 (C++17)

`main.cpp` 상단 설정만 바꿔 사용한다. 명령행 인자는 없다.

1. 최신 `cpp` SDK를 Release/x64로 빌드한다. ONNX Runtime, OpenCV, JSON 의존성 준비 방법은 `cpp/README.md`를 따른다.
2. `cmake --build build --config Release --target native_patch_example`로 예제를 빌드한다.
3. 실행 파일 옆에 `model.onnx`, `model.json`, `image.png`를 둔다. ONNX 이름은 JSON의 `model_path`와 같아야 한다.
4. 암호화 파일이면 코드의 `encrypted = true`로 바꾸고 `model.dvsenc`, `model.key`, `image.png`를 둔다.

초기화와 준비 추론은 시작할 때 한 번이다. 이후 `Classify(image)` 또는 `Detect(image)`만 호출한다.
JSON에 저장한 패치 배율·겹침·합치기 방식이 자동 적용된다. 박스는 원본 픽셀 좌표이고 분류는 원본 한 장의 확률이다.

**기존 `with_evision/classifier.h` 단독 예제는 schema 7 패치 모델을 지원하지 않는다.** 이 예제는 `vision_inference.h`와 최신 빌드 라이브러리를 사용한다. 헤더만 복사하거나 새 JSON에서 patches를 지우면 안 된다. C#도 최신 `vision_runtime` 라이브러리가 필요하다.

패치는 여러 번 추론하므로 전체 축소 추론보다 시간이 더 들 수 있다. 속도 수치는 사용 장비의 Release 빌드에서 측정한다.
