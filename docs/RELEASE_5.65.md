# 5.65 — SAM2 Windows native 학습

SAM2 Hiera Tiny, Small, Base+, Large는 이제 Windows 설치본에서 기본 제공
가중치로 학습할 수 있다. Studio의 semantic class-index mask에서 전경 클래스를
이진 객체 마스크와 양성 point prompt로 변환해 SAM2 prompt encoder와 mask decoder를
미세조정한다. Hiera image encoder는 고정하므로 RTX GPU에서 실용적인 메모리와
속도로 실행한다.

학습 결과의 Best/Last 체크포인트는 변경된 SAM2 모듈만 저장하고, 설치본의 검증된
기본 Hiera 가중치와 결합해 로드한다. 따라서 체크포인트가 원본 가중치를 중복하지
않으며, 선택한 동일 Hiera 변형에서만 로드된다.

SAM2 Export는 학습된 `best.pt`를 자동 선택해 encoder ONNX, decoder ONNX,
`sam2.json`을 생성한다. 학습 기록이 없는 새 프로젝트는 이전처럼 기본 제공 공식
가중치를 내보낸다.
