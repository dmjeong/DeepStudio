# Studio 정리 설계

- 루트 DeepVisionStudio, Python 자체 모델 CustomCSP, C++ VisionInference 및 vision_inference.h/.cpp.
- 외부 엔진 전용 학습/다운로드/전처리/모델 선정 모듈과 전용 테스트를 제거한다.
- 혼합 모듈에서는 폐기된 분기/상태 필드/옵션만 제거하고 공통 알고리즘과 테스트를 유지한다.
- 학습 모드는 custom 및 efficientnet 계열만 허용한다. 모르는 저장 모드는 명확히
  미지원으로 처리하여 의도하지 않은 새 학습을 시작하지 않는다.
- 데이터 라벨 형식은 정규화 박스/폴리곤으로 명명한다. 좌표 계약은 유지한다.
- 새 프로젝트 확장자는 .dvproj. JSON 로더의 기존 파일 내용 읽기 능력은 유지한다.
- 빌드 타깃, include, CI, ZIP 루트, 시작 스크립트와 문서를 함께 바꾼다.
- 모델 state_dict 키와 수치 연산을 보존한다. ONNX 실패 시 PyTorch 복구도 유지한다.
