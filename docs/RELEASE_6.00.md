# Deep Vision Studio 6.00

C++17 분류 ONNX 예제의 명령행 인자 요구를 제거했다. `example/cpp/main.cpp`의
`model_json`, `image_file`, `runs`를 지정하고 실행한다. 기본값은 함께 제공되는 합성 테스트
모델이며, 상대 경로는 작업 폴더가 아닌 EXE 폴더 기준이다.

`example/cpp/classifier.h`는 프로그램 내부에서 사용하는 최소 래퍼다. 생성 시 JSON/ONNX를
한 번 로드하고 `Infer(cv::Mat)` 또는 `InferFile(path)`로 반복 추론한다. JSON의 전처리,
클래스, 검증된 런타임 설정을 유지한다. 한글 이미지 경로는 파일을 바이너리로 읽어 디코딩한다.

Windows 빌드는 ONNX Runtime DLL을 예제 EXE와 같은 디렉터리에 복사한다. PATH보다 먼저
검색되는 시스템 디렉터리의 ONNX Runtime과 혼동하지 않게 한다. OpenCV 및 종속 DLL도
EXE 옆에 함께 배포해야 한다(vcpkg 빌드 시 자동 복사).

공개 예제와 별도의 자동 테스트는 모두 인자 없이 실행된다. 테스트는 실제 ONNX 출력 확률,
저장된 런타임 설정, 한글 경로, GRAY/BGR/BGRA 카메라 ROI, 입력 변경 후 출력 갱신,
누락/잘못된 파일과 오류 후 정상 추론을 확인한다. 설치 패키지에도 새 소스를 포함한다.

이 예제는 분류용이다. 합성 테스트 통과는 사용자의 비공개 모델 정확도나 Windows/i7의
8ms 달성을 보증하지 않는다.
