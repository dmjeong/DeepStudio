# 8.19 — 전 태스크 ONNX 암호화 내보내기

새 기능 버전 규칙에 따라 7.19에서 8.19로 올렸다.

- GUI/Web ONNX 내보내기에 AES-256-GCM 암호화 옵션을 추가했다. ONNX와 배포 JSON을
  `.dvsenc` 한 파일로 보관하며 SAM2 encoder/decoder도 함께 넣는다.
- 기존 export와 출력 검증 후 암호화한다. 복원한 그래프 바이트와 설정이 원본과
  일치하는지 검사한 후에만 결과를 교체한다. 실패하면 기존 결과를 보존한다.
- C++17 OpenCV/eVision, C ABI, C#에 메모리 복호화 로드를 추가했다. 초기화 후 기존
  추론 함수를 재사용한다. 실행 인자 없는 준비 추론 예제와 공개 합성 테스트 파일을 제공한다.
- Windows는 OS의 BCrypt를 사용한다. 실제 키는 배포 예제나 로그에 넣지 않는다.
- 검증 결과와 미검증 환경은 [구현·검증 기록](ENCRYPTED_ONNX.md), 사용법은
  [암호화 예제](../example/ENCRYPTED_MODELS.md)에 기록했다.

초기 복호화 비용은 존재한다. Windows/i7 및 실제 eVision SDK 실행은 이 개발 환경에서
직접 측정하지 않았으며 Windows CI에 v141 Release/Debug와 암호화 계약 테스트를 추가했다.
