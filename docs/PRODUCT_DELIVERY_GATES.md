# 제품 완료 기준

이 문서는 대화가 길어져도 바뀌지 않는 Deep Vision Studio의 제품 계약이다. 구현 중인
항목을 완료라고 말하거나 릴리스를 만들기 전에, 해당 항목의 실행 증거와 아래 게이트를
확인한다. 계획·mock·정적 검사만 통과한 상태는 완료가 아니다.

## 기본 제공 모델

아래 모델은 설치본에 포함한다. Docker 모델 팩은 이 목록을 대체하지 않으며, 사용자가
새 모델을 추가할 때만 사용한다.

| Task | 기본 모델 |
| --- | --- |
| Classification | EfficientNet B0/B1, ResNet, ConvNeXt V1, LibreMobileNetV4 |
| Anomaly Detection | PatchCore |
| Object Detection | Re-DETR v4 Small/Medium/Large, LibreYOLO9 |
| Segmentation | SAM2 전체 변형, DeepLab V3+, U-Net |

## 모델별 출시 게이트

각 기본 모델 변형은 다음을 모두 만족해야 한다.

1. Windows 설치본에서 모델 선택, 학습 시작, epoch 진행 정보, Best 선택 근거와 epoch
   시간이 화면에 갱신된다.
2. 사전학습·전이학습·스크래치가 모델의 실제 지원 범위와 함께 선택 가능하다. 지원하지
   않는 조합은 선택 전에 이유를 보여 주고, 지원한다고 표시하지 않는다.
3. 생성된 checkpoint를 Python GUI에서 다시 로드해 실제 이미지 한 장을 추론한다.
4. ONNX export는 실제 해당 모델 checkpoint로 수행한다. `onnx.checker`와 PyTorch 대
   ONNX Runtime 출력 비교를 같은 입력에서 실행하고, 사용한 입력·출력 계약·허용 오차를
   결과에 남긴다.
5. export가 성공한 ONNX와 JSON을 C++17 및 C# 예제로 각각 로드하여 이미지 한 장을
   실행한다. Python mock 또는 가짜 wrapper만으로 이 게이트를 충족할 수 없다.
6. 실패하면 export를 성공으로 표시하지 않는다. 실패한 backend, graph 최적화 설정,
   최대 오차, 허용 오차, 재현 방법을 화면과 로그에 남긴다.

## 공통 기능 게이트

| 영역 | 완료 조건 |
| --- | --- |
| 라벨링 | Detection/Segmentation에서 생성·선택·이동·삭제·Undo/Redo가 실제 캔버스에서 동작하고, erase는 추가가 아니라 삭제한다. |
| 설정 | Task 표기는 `Classification`, `Object Detection`, `Segmentation`, `Anomaly Detection`으로 통일한다. 상태 배지로 구현 여부를 과장하지 않는다. |
| 확장 | Settings의 Docker 모델 추가 화면은 외부 신규 모델 전용이며, pack의 라이선스·NOTICE·해시·작업 계약을 검사한다. |
| 배포 | Windows 단일 설치 EXE는 필요한 Python/Qt/ONNX Runtime/C++ runtime과 기본 모델 실행 모듈을 포함한다. 최소 사양은 README에 명시하고 설치 후 smoke test를 한다. |
| 성능 | 224 px EfficientNet B0의 시간은 이미지 전처리·추론·후처리 범위를 포함해 표시한다. 8 ms 목표는 특정 i7·Windows·runtime 버전·warm-up·스레드 수로 재현 가능한 실측값이 있을 때만 주장한다. |

## 작업 방식

- 새 요구사항은 이 문서 또는 연결된 이슈에 먼저 기록하고, 구현·실행 검증·문서의 세
  열로 나눠 추적한다.
- 변경 전 현재 게이트와 영향받는 모델을 확인하고, 변경 후 관련 테스트와 실제 실행
  증거를 함께 남긴다.
- CI가 실패하면 다음 기능을 완료라고 보고하거나 배포하지 않는다. 실패 원인을 고치거나
  실패가 현재 변경과 무관함을 재현 가능한 근거로 분리한다.
- 릴리스 노트는 구현됨, 자동 검증됨, 실제 실행 검증됨을 구분해 적는다.
