# 15.88 — 최신 분할·이상 탐지와 배포 연결

기본 모델에 EfficientViT B0/B1 + Detail, SCTNet-S + Detail, EoMT DINOv2-S + Detail,
Dinomaly DINOv2-S/14를 추가했다. Docker 설치 없이 선택할 수 있다.

- 사전학습 백본, RGB/BW8, 증강, 패치 학습과 재개를 지원한다.
- 분할 교사 증류는 동일한 증강 입력을 사용하며 배포에는 학생만 남긴다.
- 내보내기 화면에서 FP32/INT8을 선택하고, INT8에는 보정 이미지 폴더를 지정한다.
- 두 정밀도 모두 암호화할 수 있고 C++에서 로드한다. 고정 타일 배포는 JSON schema 8이다.
- 흑백 모델의 원본 비교에서 컬러 원본을 유지한다. 앱/C++ 마스크 크기 복원도 일치시켰다.
- 지표가 붙은 Best 파일로의 재개와 Dinomaly의 대시보드·CSV를 보완했다.

Python 실행자는 소스를 갱신한 뒤 **`gui/update_segmentation_models.bat`**을 실행하고 앱을 다시 연다.
EXE 배포는 새 빌드가 필요하다. 기존 체크포인트를 억지로 변환하지 않는다.
새 모델의 검사 헤드는 해당 모델로 학습해야 하며 기존 모델을 계속 쓰는 데 재학습은 필요 없다.

내 솔루션의 C++ 런타임도 `cpp/include`, `cpp/src` 전체로 갱신하고 다시 빌드한다.
새 `dense_tiles.h`를 빠뜨리지 않는다. JSON과 ONNX를 한 쌍으로 배포한다.
[인자 없는 예제](../example/cpp/dense_models/README.ko.md)는 시작 시 초기화·워밍업하고 구간별 시간을 표시한다.

검증: Python 634개 통과, Windows 배치 전용 1개 제외; native CTest 8개 통과.
새 모델별 기본 입력 크기의 실제 사전학습·학습·FP32/INT8·암호화·C++ 출력 대조를 포함한다.
실제 검사 정확도와 Windows/i7 속도는 현장 데이터로 확인해야 한다.
INT8 비교 결과를 정확도 불변으로 표시하지 않는다.

[상세 사용법](MODERN_DENSE_MODELS.md) · [검증 보고서](03-analysis/modern-dense-models.md) ·
[코드·가중치 출처](../licenses/modern-dense-models.md).
