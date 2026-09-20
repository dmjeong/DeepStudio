# 5.86 — LibreYOLO 기본 가중치 인증서 오류 수정

기본 가중치 준비 단계에서 LibreYOLO와 Re-DETR v4는 upstream `requests` downloader를 사용한다.
이 경로에도 `truststore`를 적용해 Windows 시스템 인증서 저장소를 사용한다.

사내 HTTPS 검사 프록시의 root CA가 Windows Trusted Root Certification Authorities에 설치된 PC에서는
인증서 검증을 끄지 않고 Hugging Face CDN의 가중치를 내려받을 수 있다. 조직의 root CA가 Windows에
설치되어 있지 않은 경우에는 IT 정책에 따라 해당 CA를 설치하거나 `REQUESTS_CA_BUNDLE`에 PEM 경로를
지정해야 한다.
