# Deep Vision Studio 11.5

Anomaly의 ImageNet 다운로드에서 Python이 인증서 체인을 확인하지 못해 실패하는 문제를 수정합니다.

- truststore를 통해 Windows 시스템 인증서 저장소를 사용하고 certifi 공개 CA 목록도 적용
- 운영자가 지정한 SSL_CERT_FILE 및 REQUESTS_CA_BUNDLE PEM 인증서 지원
- HTTPS 인증서와 호스트명 검증 유지. 전역 SSL 설정 변경 없음
- 공식 파일명의 SHA256 접두사 검증 후 캐시 반영. 다운로드 실패 시 임시 파일 정리
- 인증서 오류는 다운로드 URL, 캐시 위치, 회사 루트 인증서 확인 및 로컬 가중치 선택 방법 표시
- 기존 캐시는 오프라인으로 재사용. 사전학습 로드 실패 시 임의 가중치로 진행하지 않음
- 로컬 HTTPS에서 인증서 실패, 신뢰한 CA 적용 후 복구, 호스트명 불일치, 저장 무결성 검증
- ResNet18, ResNet50, 기본 Wide ResNet50-2의 새 다운로드부터 학습, 저장, 오프라인 재로드까지 CI 검증

React는 start_web.bat 실행 시 추가 의존성을 설치합니다.

인증서가 만료되었거나 회사 루트 인증서가 OS에도 등록되지 않은 환경은 관리자 확인이 필요합니다. 인증서 검증을 끄지 않습니다.

구현 참고: https://truststore.readthedocs.io/en/latest/
