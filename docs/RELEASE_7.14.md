# 7.14 — 수동 VS 프로젝트 ONNX 링크 설정

bugfix에 기록된 HVision Release x64 빌드는 컴파일 후 링크 단계에서
`LNK2001: OrtGetApiBase`, `LNK1120`으로 실패했다. 이는 ONNX Runtime의
import library 연결이 빠졌거나 적절히 연결되지 않은 상태다. C4819/C4244 등은 이 로그에서 경고다.
Debug 로그는 제공되지 않았으므로 같은 오류인지 직접 확인되지는 않았다.

두 예제의 classifier.h에 MSVC용 `#pragma comment(lib, "onnxruntime.lib")`를 추가했다.
기존 VS 프로젝트는 모든 구성(Debug/Release)의 링커 추가 라이브러리 디렉터리에
실제 x64 onnxruntime.lib가 들어 있는 폴더를 지정해야 한다. DLL만 복사하는 것으로는
링크되지 않는다. 헤더/lib/DLL은 동일한 SDK에서 가져온다.

HVision 소스와 프로젝트 파일은 저장소에 없어 해당 프로젝트 자체는 수정하거나 빌드하지 않았다.
