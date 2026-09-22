# 7.15 — EROIBW8 입력 예제 누락 수정

`example/cpp/with_evision/roi_example.h`의 `EvisionRoiExample::Inspect`는 실제
Open eVision EROIBW8 참조를 받는다. 모델은 시작 시 한 번 만들고 준비 추론 후 재사용한다.
기존 ROI를 그대로 전달하며 위치/크기를 변경하지 않는다.

`roi_example.cpp`에는 이미지 로드, ROI 연결 및 위치 지정, 추론과 결과 출력까지 담았다.
기존 MFC 앱은 main 대신 헤더의 클래스를 멤버로 보관하고 검사 시 Inspect를 호출한다.
라이선스가 필요한 실제 Open eVision SDK는 이 환경에 없어 SDK 빌드는 미검증이다.
자동 테스트는 원점이 아닌 부모 이미지 내부 ROI의 포인터와 행 간격을 모사하여 추론을 검사한다.
