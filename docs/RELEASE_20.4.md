# Deep Vision Studio 20.4

EfficientNet에서 Best 선정 기준을 Val Loss로 설정할 때 데스크톱 결과
화면이 엔진을 Custom CSP로 분류해 예외를 발생시키는 오류를 수정합니다.
학습 시작 전 메트릭 카드 구성에서 예외가 발생하면 이전 결과가 남을 수
있었습니다. 공통 엔진 판별 함수를 사용하고 EfficientNet의 세 학습 모드를
회귀 검사합니다.

이 오류가 사용자가 보고한 동일 혼동 행렬의 원인이라고 확정한 것은
아닙니다. 별도 실제 이미지 학습 비교는 tools/training_comparison.py와
Classification training comparison 워크플로에서 실행합니다. 모델 정의와
가중치 로딩, 학습 알고리즘은 이 수정으로 변경하지 않습니다.
