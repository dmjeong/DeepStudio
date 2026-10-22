# 9.23 라벨링 작업실 구현 분석

작성일: 2026-10-22. 기준 커밋 `20d406f4b7e42af9f94eae90b2cefce4c96c73ab` 이후의 로컬 변경을 검사했다. 이 기록은 Windows 설치본 출시 승인이 아니다.

## 구현과 데이터 흐름

Detection과 Segmentation은 선택 모델과 관계없이 공통 작업실을 연다. OBB의 기존 회전 좌표 편집은 유지한다. 모델 학습·ONNX·C++ 추론의 연산이나 스레드 설정은 변경하지 않았다.

| 구성 | 역할 |
|---|---|
| `annotation_workspace.py` | 편집 명령, 다중 선택, 연결 부분 분리, 격자와 AI 제안 연결 |
| `annotation_workspace_ui.py` | 도구·이미지·클래스/객체/이력·탐색 배치 |
| `annotation_workspace_state.py` | Undo, 초안, 저장 상태와 이미지 이동 |
| `annotation_shortcuts.py` | 사용자 단축키와 텍스트 입력 보호 |
| `annotation_grid_dialog.py` | 8×16 후보와 기준점·행/열·범위 검증 |
| `annotation_io.py` | 비동기 저장과 최대 2개 재사용 스레드의 썸네일 IO |
| `mask_annotations.py` | 표준 PNG, v1/v2 편집 정보, 객체 소유 픽셀과 변경 구역 캐시 |
| `annotation_tools.py`, `annotation_journal.py` | 원자 파일 교체, 초안, 프로젝트 잠금과 중단 복구 |
| `annotation_assist.py` | 번들 SAM2 모델·임베딩 재사용, 외부 업로드 없는 제안 |

편집 정보는 학습용 TXT/PNG와 호환된다. 숨김은 표시만 바꾸며, 잠금은 이후 편집으로부터 보호한다. 지우기의 보호 대상은 해당 편집 명령이 생성될 때 기록한다. 나중에 잠금을 켜거나 껐다는 이유로 과거 학습 픽셀이 달라지지 않는다.

초안은 사용자 데이터 폴더에 별도로 저장한다. 명시적 정답 저장만 학습용 파일을 바꾸고, `검수 완료·다음`은 같은 저장에 검수 상태를 추가한다. 비동기 저장 중 생긴 새 편집은 저장 완료로 잘못 표시하지 않는다. 원본 이미지·라벨·클래스 revision이 달라지면 덮어쓰기를 거부한다.

## 시나리오별 확인

| 계획 항목 | 로컬 검증 내용 |
|---|---|
| 박스·선택·다중 편집 | 실제 Qt 포인터 이벤트, 확대 후 좌표, 클래스 일괄 변경, 잠금·숨김·삭제·복제 |
| 다각형 | 점 삽입·삭제·이동, 스냅, 자기 교차 거부, 진행 중 Undo와 Esc |
| 브러시·지우개 | 브러시 픽셀 선택, 같은 객체로 묶기, 선택/클래스/전체 범위, 원형·사각형, 지우기 미리보기 |
| 기존 PNG | 클래스 픽셀 선택, 연결 부분 분리 미리보기·취소·확정, 구멍·255 보존 |
| 저장 | 창 유지, 실패 후 재시도, 저장 도중 편집, 외부 변경 거부 |
| 복구 | 미완성 다각형 초안, 오래된 초안 차단, 실제 자식 프로세스의 `os._exit(23)` 후 저널 복구 |
| 격자 | 128개 생성, 3점/4점 좌표, 이미지에서 기준점 클릭, 범위 밖 제외, 잠긴 기준 영역 보존, 오래된 후보 거부, 한 번의 Undo |
| 표시·단축키 | 원본 1:1과 맞춤 구분, 작은 창, 입력란 보호, 키 재지정, 밝기·대비가 원본/정답을 바꾸지 않음 |
| AI | 포함/제외점·박스, 취소, 수동 편집·이미지 변경 후 오래된 응답 폐기, 처리 중 안전한 닫기 |
| 학습 연결 | 저장한 v2 마스크를 실제 `SegmentationDataset`에서 읽어 원본 클래스 픽셀과 비교 |
| 웹 호환 | v2의 최종 픽셀을 읽기 전용으로 제공, 구형 웹 편집기의 v2 덮어쓰기 차단. 기존 v1 편집은 유지 |

연결 부분 분리는 8방향 연결을 사용한다. 적용 전후 픽셀 비교를 통과해야 반영하며, 분리 전체를 한 번에 실행 취소한다. 원래 객체 경계가 없는 PNG에 임의의 객체 경계를 복원했다고 주장하지 않는다.

## 실행 결과

최종 자동 테스트·측정 결과다. [플랫폼·원시 수치·소스 SHA-256 기록](artifacts/annotation-workspace-local-validation.json)을 함께 보관한다.

| 검사 | 결과 |
|---|---|
| 관련 Python/Qt/웹 회귀 | **174개 통과**, 13.21초. 새 작업실 테스트 33개 포함 |
| Qt 125/150/200% | 각 **10개** 좌표·창·편집·실제 스타일 테스트 통과 |
| 웹 단위 테스트 | 14개 통과 |
| 웹 TypeScript·Vite 빌드 | 성공 |
| CI와 같은 E/F/W 검사 및 새 파일 추가 검사 | 통과. 포맷 검사와 `git diff --check`도 통과 |

245개 경고는 기존 Pillow·FastAPI·Qt의 사용 중단 예정 API 경고다. 테스트 실패로 숨기거나 집계에서 제외하지 않았다.

Python 회귀 명령:

```sh
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest \
  tests/test_annotation_workspace.py tests/test_mask_annotations.py \
  tests/test_detection_annotation.py tests/test_segmentation_annotation.py \
  tests/test_dataset_editor.py tests/test_dataset_filters.py \
  tests/test_class_management.py tests/test_project.py tests/test_project_persistence.py \
  tests/test_inference_review.py tests/test_webapp.py tests/test_web_package.py \
  tests/test_web_recovery.py tests/test_version_policy.py tests/test_gui_smoke.py \
  -q --tb=short
```

Qt 배율 검사는 `QT_SCALE_FACTOR=1.25`, `1.5`, `2` 각각에서 실제 1:1·앱 스타일 검사와 Detection/Segmentation 기존 이벤트 테스트를 실행했다. Windows 모니터 사이에서 창을 옮기는 검사를 대신하지 않는다.

## 편집 성능과 메모리

플랫폼: macOS 26.5 arm64, Python 3.12.13, PySide6 6.12, Qt offscreen. 합성 이미지로 측정했으며 사용자의 비공개 이미지나 모델 가중치를 사용하지 않았다.

`tools/benchmark_annotation.py`는 4096×4096, 객체 128개, 지우개 32px, 포인터 이동 159개를 보낸 뒤 Qt 이벤트와 그리기 완료까지 잰다. 첫 창 생성과 획 확정은 따로 측정한다. 원시 결과는 `local-artifacts/annotation-benchmark.json`에 저장한다.

| 합성 4096×4096 | 128개 영역 | 1,000개 영역 스트레스 |
|---|---:|---:|
| 첫 창 생성 | 331.64ms | 600.39ms |
| 이동 → 그리기 중앙값 | 1.36ms | 1.43ms |
| 이동 → 그리기 p95 | **1.81ms** | **1.98ms** |
| 최대값 | 5.83ms | 12.96ms |
| 100ms 초과 이동 | 0 / 159 | 0 / 159 |
| 획 확정 | 9.80ms | 22.19ms |

1,000개 스트레스는 같은 측정 함수에서 격자 생성기를 25×40/작은 칩으로 바꿔 실행했다. 예전 코드와의 직접 배수 비교나 사용자 작업 시간 개선폭으로 해석하지 않는다. 지우개 이동은 변경 구역만 갱신하고 획 확정도 전체 이력 재렌더링 대신 캐시를 갱신한다.

메모리 검사는 1536×2048/128개 영역의 이미지 50장을 반복해 **500회 열고 닫았다**. 남아 있는 편집창 weak reference는 0개다. Qt의 전역 아이콘 캐시 기본 예산은 10,240KiB다. 초기 RSS가 올라가다가 약 210회부터 수렴했고, 250회 237.84MiB, 500회 237.06MiB였다. 250~500회 범위는 236.84~238.34MiB다. 초기 증가를 문서 누수라고 단정하지 않고 썸네일 없는 대조군·1MiB 아이콘 캐시 대조군과 비교했다. 기존 앱 전체의 전역 아이콘 캐시 한도는 임의로 낮추지 않았다.

썸네일 IO는 최대 2개 재사용 스레드로 제한하고 작업 완료 시 문서 상태 콜백을 해제한다. 별도 회귀에서 40개 요청이 2개 이하의 스레드를 사용하고 상태 객체를 놓는지 확인했다. 이 두 IO 스레드는 모델 추론의 스레드 수를 제한하는 설정이 아니다.

이 수치는 라벨링 화면의 반응 시간이다. 칩 128개를 검사하는 모델 추론 20ms 목표나 Windows i7 성능의 측정값이 아니다.

## 실제 SAM2 통합

공식 SAM2.1 Hiera Tiny 체크포인트를 임시 로컬 자산 경로에 준비해 다음 흐름을 실행했다. 가중치를 저장소에 추가하지 않았고, 검증 도구는 다운로드하지 않는다.

```sh
QT_QPA_PLATFORM=offscreen \
DVS_BUILTIN_ASSETS_DIR=/tmp/dvs-annotation-sam2-assets \
.venv/bin/python tools/verify_annotation_sam2.py
```

합성 256×192 이미지 → GUI 포함점 입력 → 실제 CPU SAM2 제안 → 적용 → 정답 저장 → PNG 재열기를 확인했다. 제안 영역은 **6,461픽셀**, 저장·재열기 픽셀은 동일하며 저장 후 창은 유지됐다. 처리 중 GUI 20ms 타이머는 **87회** 실행됐고, 전체 흐름은 **1.81초**였다. 모델 준비·이미지 임베딩·제안·저장까지 포함한 시간이며 보장 지연시간이 아니다.

Hiera Small/Base+/Large는 기존 번들 자산 선택 경로를 연결했다. 이 세 가중치와 GPU의 실제 실행을 Tiny CPU 측정으로 대신 검증했다고 표시하지 않는다.

## 계획 조정과 출시 전 조건

| 항목 | 최종 처리 |
|---|---|
| 확정 정답·검수 상태 | 명시적 저장이 학습 정답 확정. 자동 초안은 학습 제외. 검수는 별도 이력 |
| 학습 스냅샷 | 기존 학습 중 편집 제한을 유지. 모든 이미지/정답의 물리 복사본을 만드는 별도 스냅샷은 도입하지 않음 |
| 이미지 탐색 | 현재 이미지 주변 최대 21장 썸네일. 전체 목록은 기존 데이터셋 화면. 인접 원본 이미지의 디코딩 캐시/Windows 200ms 수락은 별도 |
| OBB | 기존 회전 편집기와 저장 계약 유지. 새 축 정렬 박스를 OBB로 바꾸지 않음 |
| 웹 | v2 픽셀의 읽기·덮어쓰기 보호. 웹 작업실 전체 개편은 이번 범위가 아님 |
| Windows 배포 | 네이티브 설치본, 오프라인 실행, 다중 모니터 DPI, target i7 반응 시간은 이 환경에서 미실행 |
| 모델군 | 공통 라벨링 도구를 모델 선택과 분리. 모든 모델의 재학습·ONNX·C++ 전체 검증을 새로 실행한 작업은 아님 |
| 사용자 작업 시간 | 동일 작업자의 전후 라벨링 시간·오류율 비교는 아직 없음. 작업 시간 50% 개선을 실측 결과로 주장하지 않음 |

패키징·Windows 실기기·사용성 수락 조건을 남겨 두며, 로컬 테스트 통과를 제품 전체 무결점 보장으로 표시하지 않는다. `inference_review.py`의 작은 lambda→def 변경은 기존 CI E731 수정이며 정렬 기준은 유지했고 관련 13개 테스트를 실행했다.

사용법: [라벨링 작업실](../ANNOTATION_WORKSPACE.ko.md). 실제 화면: [Detection](../02-design/assets/annotation-workspace-implemented.png), [Segmentation](../02-design/assets/annotation-workspace-segmentation.png).
