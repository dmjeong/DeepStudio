# Deep Vision Studio 개발 안내

애플리케이션 루트는 DeepVisionStudio입니다. 현재 아키텍처는 docs/ARCHITECTURE.md를
참고하세요. Python 자체 모델의 공개 클래스는 model.CustomCSP, CPU C++ 추론기는
VisionInference입니다. 모델 state_dict 키와 저장된 전처리 계약을 유지하세요.

분류는 EfficientNet B0/B1 또는 Custom CSP, 분할·박스 탐지·재구성은 Custom CSP,
이상 탐지의 기본 엔진은 PatchCore입니다. 회전 박스는 데이터 편집만 지원합니다.

Qt 객체를 계산 엔진에 전달하지 마세요. 계산은 작업 워커에서 실행하고 저장된 판정과
시간을 UI에서 다시 표시합니다. 새 의존성이나 런타임을 추가할 때 실제 실행 경로를
검증하세요. ONNX 자동 준비 실패 시 기존 PyTorch 추론은 유지되어야 합니다.

Python 회귀 검사, 웹 타입/빌드 검사, C++ 수정 시 Release/CTest를 수행합니다.
파일/폴더명 변경은 CI, 시작 스크립트, ZIP 루트, 문서의 경로도 함께 바꿉니다.
