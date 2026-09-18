# Studio 정리 완료 보고

현재 소스 루트: `DeepVisionStudio`.
Python 자체 모델: `CustomCSP`. C++ 공개 API: `VisionInference`,
`vision_inference.h/.cpp`, CMake 라이브러리 `vision_inference`, 실행 예제 `vision_demo`.
외부 C++ 호출부는 새 헤더/클래스/링크 타깃으로 수정해야 한다.
새 프로젝트 확장자는 `.dvproj`다. 기존 JSON 내용은 유지하며 파일 열기의 모든 파일
필터로 이전 프로젝트를 선택할 수 있다. 자체 모델의 state_dict 구조는 유지했다.

폐기한 외부 엔진 코드와 전용 설정·의존 연결·테스트를 제거하고, 문서·CI·ZIP 루트도
새 이름으로 맞췄다. 지원 태스크는 README 표를 기준으로 한다. 회전 박스 모델 학습은
제공하지 않고 편집/크롭 기능만 남긴다. 자동 CPU 추론의 수치 검증 실패 복구를 유지했다.

검증:
- Python 전체: 589 passed, 11 skipped (환경별 검사), 0 failed.
- 웹: 11개 테스트, TypeScript/Vite 빌드, FastAPI 자산/API smoke 통과.
- C++17 Release: ONNX 단독 및 OpenVINO 포함 구성 각각 CTest 3/3 통과.
- 배포 ZIP 생성, 필수 시작 파일·새 루트·Windows 배치 형식 검증 통과.
- 현재 Git 추적 파일/폴더 이름과 내용에 폐기 명칭이 남지 않도록 검사.

검증 장비는 macOS ARM64다. Windows 실행 및 i7에서 8 ms 달성은 이번 검증 결과에
포함하지 않는다. 비공개 가중치/이미지는 사용하거나 외부로 전송하지 않았다.
상세 항목은 [검증 기록](../03-analysis/studio-cleanup.analysis.md)을 참고한다.
