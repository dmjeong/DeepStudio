# 3.64 — SAM2 가중치 프로젝트 적용과 학습 화면 검증 수정

Settings의 SAM2.1 다운로드는 이제 파일 캐시만 남기지 않는다. 분할 프로젝트를 연 상태에서
`현재 프로젝트에 적용`을 누르면 선택한 Hiera 변형과 공식 checkpoint의 절대 경로를 프로젝트에
저장한다. 이 선택은 export 화면으로 전달되어 SAM2 encoder/decoder ONNX exporter가 같은
checkpoint를 사용한다.

또한 EfficientNet 학습 화면의 실제 Qt 검증은 지원 범위 밖의 `custom` 모드 대신
`efficientnet_scratch` 모드를 검증한다. 이 오류 때문에 중단되던 이후의 CI 회귀 검증도 다시
실행된다.
