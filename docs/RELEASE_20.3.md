# Deep Vision Studio 20.3

20.2의 EfficientNet 기능을 검증하면서 발견한 문제를 수정했다. 큰 기능 추가 +1.0, 버그 수정 +0.1 규칙을 적용한다.

- 검출 화면의 사각형 핸들 변수 이름을 명확하게 바꾸어 정적 검사 오류를 수정했다.
- 기존 회귀 테스트 객체에 학습 확장 메서드와 콤보박스 조회 동작을 반영했다. 체크포인트 저장 실패 시 이전 파일 보존도 검증한다.
- deepstudio3의 데스크톱과 React 실행 방법을 README에 추가했다.

EfficientNet 사용법: [EFFICIENTNET.md](EFFICIENTNET.md)
