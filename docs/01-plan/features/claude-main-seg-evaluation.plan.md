# Claude 최신 변경 통합 및 C++ 분할 평가

- 요청: Claude 최신 브랜치를 main에 반영하고 export한 segmentation 모델을 C++에서 평가한다.
- 기준: main fe6782e, Claude determined-turing-ssd181 04348c4. 이전 with-evision e49c7c2를 포함한다.
- 범위: 두 브랜치 기능 보존, 호환성 충돌 해결, 명령행 인자 없는 JSON 설정 기반 C++17 평가 예제.
- 평가: 원본 이미지와 클래스 ID 정답 PNG, 클래스별 IoU/Dice 및 배경 제외 평균, 준비 후 추론 시간, 결과 마스크와 JSON/CSV.
- 성공 기준: Python 관련 회귀, 실제 ONNX export 및 native 로딩/추론/평가 실행, 정답 계산 회귀, 정상 Git main fast-forward push.
- 주의: 원본 main 작업 폴더의 미커밋 작업 보존. Windows 빌드는 이 macOS 환경의 실행 검증과 구분한다.
