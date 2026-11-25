# 검증 결과 — 세그멘테이션 첫 클래스·자동 배경

## 근본 원인과 수정

기준 브랜치: `origin/claude/with-evision-refactor-optimize-pvffuh`, `2bfa569`.

1. `ProjectManager.create_new`와 `ClassManager.add`는 첫 사용자 이름을 ID 0에 저장했지만 편집기·마스크·학습기는 ID 0을 배경으로 해석했다. 새/빈 프로젝트에서 배경을 자동 예약하도록 수정했다. 기존 클래스 번호는 임의로 바꾸지 않고 복구 버튼으로 첫 이름을 새로운 전경 ID에 추가한다.
2. `89b6ae0`의 데이터 로더는 미라벨 이미지를 건너뛰었다. 사용자 요구에 맞춰 누락 마스크만 `None`으로 식별하고 메모리에서 0번 정답을 제공한다. 원본 파일은 변경하지 않는다.
3. SAM2 `_prompt_targets`는 배경 샘플을 버렸고 ignore 픽셀을 bool 변환으로 잃었다. 배경은 음성 point + 빈 target으로 학습하며, target에 255를 보존한다.
4. SAM2 학습은 single-mask token을 사용했지만 ONNX는 multimask를 사용했다. fine-tune 체크포인트에 한해 동일한 single-mask와 stability 설정을 사용한다. 기반 인코더 SHA256도 저장·재로드한다.
5. 복구는 기존 전경 ID, ignore 픽셀, 클래스 범위 지우개, 편집 이력과 원본 백업을 함께 처리한다. 평면 마스크의 배경 0만으로는 사라진 경계를 추측하지 않는다.

## 실행한 검증

- 수정 전 재현 4건 전부 실패: 첫 클래스 생성, 빈 프로젝트 추가, 미라벨 train/val 포함, SAM2 배경/ignore.
- 관련 회귀 검사: 277 passed / 2 skipped.
- 전체 저장소 검사: 1,200건 중 1,187 passed / 12 skipped / 릴리스 문서 누락 1 failed. 문서와 최종 버전을 9.43으로 맞춘 뒤 해당 파일의 두 검사 모두 통과했다. 코드·기능 실패는 남지 않았다.
- 추가 최종 회귀: `test_segmentation_background_contract.py` 28 passed. 기존 전체 검사 이후 추가한 클래스 범위 지우개 복구도 포함한다.
- 최초 전체 검사에서 발견한 테스트 대역의 `dataset`/`class_names` 인터페이스 불일치는 수정했고 재검사 및 두 번째 전체 검사에서 통과했다.
- CI와 동일한 GUI/core/web E/F/W lint, 수정 Python 파일 lint, compileall, git diff --check 통과.
- 실제 Qt 입력: 첫 클래스 사각형 그리기 → PNG 저장 → 재열기 → 다른 클래스 변경 → 첫 클래스로 변경; 빈 프로젝트의 작업실 클래스 추가; 잠금·이동·지우개·이전/다음 관련 기존 검사 통과.
- 실제 화면 캡처를 확인해 통계 표의 이름 잘림·설명 겹침을 수정하고 통계 패널을 스크롤 가능하게 만들었다.

전체 검사의 12개 skip은 테스트 환경에서 지원하지 않거나 선택 의존성이 필요한 기존 검사다. Windows/CUDA 통과로 간주하지 않는다. 검사 중 디스크 부족이 한 차례 발생했으며 이번 실행의 pytest 임시 체크포인트만 정리했다. 이후 성공한 임시 산출물을 자동 정리하는 옵션으로 재실행했다.

## 실제 모델 실행

생성한 64×64 이미지로 학습/검증 각 2장(마스크 1장 + 미라벨 정상 1장)을 구성했다. 모델 정확도를 측정하려는 데이터가 아니다.

| 모델 | 실제 학습 | 체크포인트·ONNX |
|---|---|---|
| U-Net ResNet18 | CPU, 배경 포함 2장, 1 epoch | 클래스 이름·자동 배경 개수 저장, ONNX 출력 검증 통과 |
| DeepLab V3+ ResNet34 | CPU, 배경 포함 2장, 1 epoch | 클래스 이름·자동 배경 개수 저장, ONNX 출력 검증 통과 |
| 공식 SAM2.1 Hiera Tiny | CPU 1024, 배경 포함 train 2장, val은 미라벨 배경 2장 | 기반 가중치 재로드 및 encoder/decoder ONNX 출력 검증 통과 |

[실행 결과 JSON](artifacts/segmentation-background-runtime.json) · [실제 Qt 화면](../02-design/assets/segmentation-background-dataset.png)

## 남는 검증 범위

Windows 설치본/CUDA, SAM2 Small·Base+·Large 실가중치 실행, 사용자 비공개 데이터 정확도는 이번 환경에서 측정하지 않았다. 마스크가 전혀 없는 프로젝트도 배경 학습은 가능하지만, 결함을 학습하려면 결함 영역 정답이 필요하다.
