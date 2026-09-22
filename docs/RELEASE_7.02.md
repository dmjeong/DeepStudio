# 7.02 — VS2017 ONNX Runtime 헤더 호환성

VS2017 15.9(v141 14.16), x64, C++17을 기준으로 한다. C++17 전환은 C API의
noexcept 함수 포인터 선언을 처리하지만, C++ wrapper의 Float16/BFloat16 constexpr
생성자 오류는 별도로 남는다. upstream 이슈: https://github.com/microsoft/onnxruntime/issues/20564

`cpp/include/ort_vs2017.cmake`는 사용 중인 SDK의 헤더를 빌드 폴더에 복사하고,
생성자 2개와 FromBits 2개의 constexpr 지정만 제거한다. 예상 선언이 없으면
조용히 패치하지 않고 구성 단계에서 실패한다. 원본 SDK, 함수 본문, 모델,
최적화 설정, 라이브러리와 DLL을 바꾸지 않는다. VS2019 이상은 보정하지 않는다.

OpenCV/eVision CMake 빌드는 v141에서 자동 적용한다. 수동 VS 프로젝트에서는
`example/cpp/with_evision/README.md`의 생성 명령과 include 경로 설정을 따른다.
CI는 VS2022 설치에 v141 도구를 추가해 실제 VS2017 컴파일러로 두 예제를
빌드하고 합성 ONNX 모델 추론·준비 추론·버퍼 계약 테스트를 실행한다.
실제 eVision SDK와 카메라 하드웨어 검증은 포함하지 않는다.

검증 결과: Windows CI에서 MSVC 19.16(v141)로 eVision/BW8와 OpenCV 예제의
빌드 및 ONNX 추론 테스트가 모두 통과했다. 같은 작업의 최신 MSVC 빌드와
전처리 비교도 통과했다. 로컬 패키징/버전 검사 33개도 통과했다.
검증 기록: https://github.com/dmjeong/DeepStudio/actions/runs/35694527725/job/106638372697
