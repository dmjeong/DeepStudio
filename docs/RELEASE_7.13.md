# 7.13 — C++ 예제 JSON 헤더 누락 수정

nlohmann/json 3.12.0의 공식 단일 헤더와 MIT 라이선스를 `example/cpp/nlohmann`에 포함했다.
OpenCV/eVision CMake 예제는 이 헤더를 사용하므로 별도 JSON 패키지 설치가 필요 없다.
설치본에도 폴더 전체가 포함되며, 수동 VS 프로젝트에는 이 폴더를 함께 복사한다.
