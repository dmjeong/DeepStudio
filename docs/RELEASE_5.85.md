# 5.85 — 간단 Windows 설치파일

`gui\\build.bat installer`는 Windows용 `DeepVisionStudio-Setup-5.85.0-win-x64.exe`를 만든다.
설치물에는 `DeepVisionStudio.exe`를 포함한 앱 실행 디렉터리와 `Examples` 디렉터리만 들어간다.

`Examples/cpp`는 CMake 기반 C++17 ONNX Runtime 예제이고, `Examples/csharp`는 .NET 8 기반
C# 예제다. 두 예제는 설치 위치에서 바로 빌드할 수 있도록 각각 필요한 SDK source를
`vision-runtime` 하위에 포함한다.

이 설치본의 최상위에는 사용자 학습 데이터, Docker/WSL, 저장소 전체를 넣지 않는다. `app/`에는
앱 실행용 기본 자산이 포함된다. ONNX 모델은 앱에서 내보낸 결과 또는 사용자가 제공한 배포
번들을 예제의 실행 인수로 지정한다.
