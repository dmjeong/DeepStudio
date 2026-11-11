# DINOv2 세그멘테이션 완료 보고서 — 10.57

## 사용자에게 보이는 변경

학습 화면의 Segmentation 모델 목록에서 **DINOv2 ViT-S/14 + Detail**을 선택한다.
기본 392×392, 흑백/컬러 지원. `LVD-142M 가중치로 시작`은 공식 사전학습 백본을
불러온다. 이미지 전체 클래스가 아닌 픽셀/영역별 정답 마스크로 분할 헤드를 학습한다.
기존 `.pt` 저장/추론, ONNX 및 암호화 내보내기 메뉴를 그대로 사용한다.

Python 사용자는 `gui/prepare_model_weights.bat segment` 또는 저장소 루트에서
`python python/prepare_builtin_assets.py --only dinov2_vits14`를 실행하면 된다.
전체 build/setup에도 포함된다. 앱 실행 중 다운로드하지 않는다.

## 확인한 결과

- 전체 회귀: **1,273 passed / 12 skipped / exit 0**, 164.04초.
  로그 `/tmp/dvs-1057-final.log`, JUnit `/tmp/dvs-1057-final.xml`.
- 공식 DINOv2 구현(`7764ea0f912e53c92e82eb78a2a1631e92725fc8`) 대비
  3개 입력 크기·4개 중간 특징 모두 최대 차이 0.
- 실제 공식 가중치에서 시작해 1/3채널 각각 합성 마스크 40에폭 학습,
  best 복원/앱 추론/ONNX 수치 비교/암호화 일치 검증 통과.
- 스크래치 전체 backbone 역전파, 동결 시 decoder/detail 학습, 재개 및
  클래스 이름 변경 전이, GUI 준비 안내, 홀수 크기 출력 계약 테스트 통과.
- C++17 Release 빌드 + ORT 1.29: 실제 1채널 모델을 일반 C++ 클래스와
  C ABI(C# 연결 계층)로 로드. 평문/암호화, 행 간격이 있는 BW8 ROI,
  각 3회 반복에서 Python ORT 마스크와 완전 일치. CTest **1/1 passed / exit 0**.
  Windows/VS2017 및 C# 관리 프로그램 자체를 실행한 결과는 아니다.
- 기본 자산 준비 CLI로 공식 파일 실제 다운로드 및 SHA256 검증, manifest 발행 확인.
  새 패키지 추가, torch.hub 실행, 런타임 네트워크 요청 불필요.

## 성능과 학습 한계

Apple M4 CPU, 4스레드, 392×392에서 전처리+ORT+argmax median은
흑백 **87.84ms**, 컬러 **85.82ms**였다. 같은 조건의 U-Net 참고값은 **39.18ms**.
이 모델을 CPU 속도 개선으로 권장하지 않으며 **원본 128칩 전체 20ms를 달성하지 않았다**.
카메라/디스크/UI 시간은 포함하지 않았고 Windows i7은 측정하지 않았다.

배경 편향도 확인했다. 합성 데이터 12에폭·가중치 보정 없음은 mIoU 0.3075였고,
balanced·40에폭은 1채널 0.9906 / 3채널 0.9967이었다. 이는 정답 패턴을 학습할 수
있는지 보는 기능 실험이다. 실제 공정에서의 검출률, 다른 모델 대비 우위 또는
학습 이미지 수의 충분성을 의미하지 않는다. 작은 결함에서는 입력 해상도/ROI,
클래스 가중치, 전경별 IoU를 함께 확인해야 한다.

## 재현

```bat
python tools/dinov2_segmentation_smoke.py --weights gui/builtin_assets/dinov2/dinov2_vits14_pretrain.pth --report dinov2-smoke.json --native-fixtures dinov2-native-fixtures
```

공식 소스 대조는 `--reference-repo <공식 dinov2 checkout>`을 추가한다.
U-Net 비교는 `--unet-weights <공식 resnet18-f37072fd.pth>`를 추가한다.
가중치 경로·공식 SHA256을 확인하고, 임시 학습 파일은 완료/실패 때 정리한다.
`--native-fixtures`는 새 폴더만 허용하며 합성 모델/이미지/테스트 키를 저장한다.
생성된 키는 테스트 전용이며 실제 배포 키로 재사용하지 않는다.

기존 C++ 빌드 설정에 `-DBUILD_TESTING=ON -DVISION_DINOV2_FIXTURES=<위 폴더 절대 경로>`를
추가하고 `dinov2_deployment` target을 빌드한다. `ctest -R dinov2_deployment --output-on-failure`로
실제 export 산출물을 검사한다. OpenCV/ORT/nlohmann/OpenSSL(Windows는 BCrypt)은 기존 SDK 요구사항이다.

## 관련 기록

- [계획](../01-plan/features/dinov2-segmentation.plan.md)
- [설계](../02-design/features/dinov2-segmentation.design.md)
- [분석과 한계](../03-analysis/dinov2-segmentation.analysis.md)
- [실측 JSON](../03-analysis/artifacts/dinov2-segmentation-runtime.json)
- [최초 짧은 학습 결과](../03-analysis/artifacts/dinov2-segmentation-initial-12epochs.json)

기존 SAM2 자동 semantic inference 문제는 이번 DINOv2 추가 범위와 별개로 남아 있다.
