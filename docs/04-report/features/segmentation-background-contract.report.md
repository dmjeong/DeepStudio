# 완료 보고: 9.43

첫 클래스와 배경의 번호 충돌을 수정하고, 미라벨 이미지를 기본 배경 학습에 포함했다. 기존 프로젝트 복구와 백업/롤백, SAM2 배경·ignore·ONNX 일관성, 자동 배경 통계를 구현했다.

- [사용 안내](../../SEGMENTATION_BACKGROUND.ko.md)
- [검증 상세 및 미검증 범위](../../03-analysis/segmentation-background-contract.analysis.md)
- [실제 모델 학습·ONNX 결과](../../03-analysis/artifacts/segmentation-background-runtime.json)

배포 대상은 `dmjeong/DeepStudio`의 `main`이다. 기존 DeepStudio 작업 폴더의 미커밋 변경은 보존했고, 배포는 Claude `2bfa569`에서 분리한 작업 폴더에서 준비했다. 이번 릴리스에 미완성 B1 실험은 포함하지 않았다.
