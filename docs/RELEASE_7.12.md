# 7.12 — VS2017 호환 헤더 생성 배치파일

`example/cpp/setup_vs2017.bat`을 더블클릭하고 ONNX Runtime SDK 폴더를 입력한다.
원본 SDK의 include 옆에 include-vs2017 폴더가 생성되며, VS에 설정할 경로가 표시된다.
Python 3.8 이상이 필요하다. CMake와 PowerShell은 필요하지 않다.
`setup_vs2017.py`를 직접 실행해도 같은 동작을 한다.

7.02에서 VS2017 빌드·추론을 확인한 네 가지 선언 보정만 적용하며, 원본 파일은 유지한다.
지원하지 않는 헤더 형식이면 출력 전에 실패한다. 설치본에도 두 파일을 함께 포함한다.
