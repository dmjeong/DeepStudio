# Claude 최신 통합 및 C++ 세그멘테이션 평가 결과

2026-11-10, 버전 11.57. main `fe6782e`에 최신 Claude `04348c4`를 병합했다. 이전 with-evision `e49c7c2` 변경도 포함한다. 기존 main 작업 폴더의 미커밋 변경을 건드리지 않고 별도 worktree에서 통합했다.

## 제공 내용

- Lite/MobileNetV2 U-Net, DINOv2 신규 분류·분할, 패치 학습과 전체 영역 배포, 결함 판정 규칙, 라벨링·학습 취소·가중치 가져오기 개선.
- 기존 DINOv2 ID/체크포인트, 배경 자동 등록과 미라벨 배경 학습, 원본 전경 표시 및 정답 IoU 보존.
- [C++17 평가 예제](../../example/cpp/segmentation_eval/README.ko.md): `build.bat`, `run.bat`, EXE 옆 `evaluation.json`. 클래스별 IoU/Dice·혼동행렬·시간·원본 크기 예측 마스크·전경 표시·CSV/JSON 저장.
- Windows VS2017(v141) Release/Debug 빌드와 실제 export→평가 테스트를 Windows SDK CI에 추가.

## 검증 결과

환경: macOS 26.5 arm64 / Apple M4, Python 3.12.13, PyTorch 2.14.1, ONNX Runtime 1.29.0, native OpenCV 4.12.0.

| 검사 | 결과 |
|---|---|
| 전체 Python 회귀(네이티브 실행 포함) | 1,415 통과, 14 건너뜀. 빌드 경로를 확인하는 문자열 검사 2건 실패 후 호환 표현 복원 |
| 위 잔여 2건 + 빌드 가중치 경로 실제 인자 검사 | 관련 47개 모두 통과, 프로세스 종료 0 |
| C++ export/평가와 판정 규칙 비교 | 12개 통과, 종료 0 |
| native SDK CTest | 6/6 통과: 배포, 암호화, C ABI, classification worker·동시성 포함 |
| 새 평가 지표 CTest | 1/1 통과: 배경/없는 클래스/오검출 클래스/ignore/검사 영역 |
| 변경 클래스·드로잉·가중치·패키징 회귀 | 81개 통과 |
| CI와 같은 Python lint | 통과 |
| 변경 Python 파일 F821/F811, diff 공백 검사 | 통과 |
| INT8 비교 도구 | Release 빌드, 동일 모델 smoke, 읽을 수 없는 이미지 종료 2 확인. INT8 정확도 승인 결과가 아님 |

평가 연결 테스트는 U-Net Lite, U-Net ResNet18, U-Net MobileNetV2, DeepLab V3+ ResNet34, 기존 DINOv2 ViT-S, 신규 DINOv2 ViT-S/ViT-B 분할 구조를 실제 ONNX로 내보내 실행했다. 무작위 fixture 가중치로 계약을 검사하며 정확도 증거로 사용하지 않는다. 정답 누락·잘못된 크기·클래스 번호·모두 ignore인 입력도 실패 처리했다.

## 학습된 합성 트레이 모델 확인

이전에 학습한 `lite_lr_1e3/training/best.pt`를 새 코드로 다시 내보냈다. 전체 입력 3504×1504, 내부 패딩 3520×1504. 시험 4장 + 크랙 형태 시험 2장 + 정상 시험 4장, 합계 10장을 평가했다.

- C++ 혼동행렬과 Python 원본 픽셀 합산 결과 일치.
- crack IoU 0.680653, FM IoU 0.813299, 전경 mIoU 0.746976.
- 4스레드, 영상별 워밍업 3회/측정 3회: p50 338.235ms, p95 350.862ms.
- 다른 검사도 진행하던 로컬 환경의 실행 확인값이다. Windows/i7 지연시간 또는 성능 향상 보장으로 사용하지 않는다.
- 합성 데이터 점수이며 실제 검사 정확도를 보장하지 않는다. 정상 칩 단위 오검출 0을 새로 인증한 결과도 아니다.

로컬 산출물은 `runs/seg-eval-tray/model.onnx`, `model.json`, `final_results/`에 있다. 큰 가중치·영상·실행 결과는 Git에 포함하지 않았다.

## 남은 환경 검증

Windows/VS2017 실행은 원격 CI에서 별도 확인해야 한다. 이 보고서는 Windows 빌드 성공을 주장하지 않는다. 실제 검사 PC에서는 같은 모델·정답 마스크로 `evaluation.json`의 경로를 지정하고 측정한다.

SAM2는 프롬프트 encoder/decoder 전용 SDK를 사용한다. 이 예제의 대상은 단일 semantic logits ONNX이며, 암호화 파일은 기존 SDK의 키 전달 초기화 API를 통해 사용해야 한다.
