# 기본 모델·학습 방식·ONNX·라벨링 검토

검토일: 2026-09-20

대상: `5b0a2944138f14ff5935733e06b84dae4c04e477`

범위: 현재 저장소 코드, 전체 Python 테스트, 웹 단위 테스트/빌드, Qt 조작 재현. 구현 완료 보고서가 아니다.

## 1. 요청한 다섯 항목의 판정

| 질문 | 판정 | 확인한 사실 |
|---|---|---|
| Docker로 신규 모델 추가가 간편한가? | 일부 기반만 구현 | Settings에서 `.dvmodel` 설치 가능. 일반 Docker 이미지를 곧바로 등록하는 기능은 아니다. 모델 작성자가 worker·manifest·ONNX 계약·서명 등을 구현해야 한다. |
| 모든 기본 모델이 사전학습/전이/처음부터 학습 가능한가? | 아니다 | 기본 18개 변형 중 9개는 native 학습 경로가 미구현이다. EfficientNet에는 동일 구조를 유지하는 전용 scratch 경로가 없다. PatchCore는 고정 특징 추출기와 메모리 뱅크 fit 방식이다. |
| 모든 모델의 ONNX export에 문제가 없는가? | 보장할 근거 없음 | 실제 모델 검증과 작은 대체 모델의 계약 검증이 섞여 있다. LibreYOLO export 연결이 없고, SAM2/RT-DETRv4 실모델의 전체 경로도 미완료다. |
| 지우개가 추가 동작을 하는가? | 일부 재현, 설계 결함 확인 | Qt에서 픽셀은 0으로 지워지지만 객체 목록에 새 브러시가 추가된다. 그 항목의 클래스를 바꾸면 다시 칠해진다. 드래그 미리보기도 삭제 결과와 다르다. |
| 개선 방향 | 기본 모델 실제 구현 + 검증 체계 + 편집 모델 교체 | [상세 설계](../02-design/features/model-readiness-hardening.design.md)에 작업 순서와 인수 조건을 정의했다. |

카탈로그에 이름이 있는 것, 실행 코드가 있는 것, 실제 Windows에서 학습과 C#/C++ 배포까지 통과한 것은 서로 다른 상태다. 이전에 기본 모델로 분류한 작업은 누락된 native 구현을 완성한 작업이 아니었다.

## 2. 모델별 현재 상태

`구현`은 해당 코드 경로가 존재한다는 뜻이다. 이번 검토에서 모든 사전학습 파일을 다운로드하고 모든 모드로 학습했다는 뜻이 아니다. `미구현`은 설치한 Docker 팩으로 대신 해결할 기본 모델이 아니라, 제품에 내장 구현해야 하는 누락이다.

| 태스크 / 모델 ID | 변형 수 | 사전학습 초기화 | 로컬 가중치 전이 | 동일 구조 scratch | ONNX 확인 범위 |
|---|---:|---|---|---|---|
| classify / `efficientnet_b0`, `efficientnet_b1` | 2 | 구현 | 구현 | 전용 경로 없음¹ | export·수치 검증·ORT 설정 복구 코드 있음. 모든 사용자 체크포인트/Windows 보장 아님 |
| classify / `resnet18`, `resnet50` | 2 | ImageNet 구현 | 구현 | 구현 | 실제 구조의 무작위 가중치 export/ORT 테스트 통과. ResNet18 합성 데이터 학습 후 export도 통과 |
| classify / `convnext_v1_tiny` | 1 | ImageNet 구현 | 구현 | 구현 | 실제 구조의 무작위 가중치 export/ORT 테스트 통과 |
| classify / `libreyolo_classify_mobilenetv4_small` | 1 | 미구현 | 미구현 | 미구현 | 제품 export adapter 미연결 |
| anomaly / `patchcore_wide_resnet50_2`, `patchcore_resnet18` | 2 | 고정 백본 + fit 구현 | 로컬 백본/체크포인트 경로 있음² | 일반 scratch 학습과 다름² | 작은 대체 백본·메모리 뱅크의 ONNX 테스트. 실제 두 백본+실규모 bank 검증 부족 |
| detect / `re_detr_v4_small`, `re_detr_v4_medium`, `re_detr_v4_large` | 3 | 미구현 | 미구현 | 미구현 | 두 출력 그래프의 계약 테스트. 실제 S/M/L 학습→export 증거 아님 |
| detect / `libreyolo_detect_9t` | 1 | 미구현 | 미구현 | 미구현 | 제품 export adapter 미연결 |
| segment / `deeplabv3plus_resnet34` | 1 | ImageNet **백본** 구현 | 구현 | 구현 | 실제 내장 구조의 무작위 가중치 export/ORT 테스트 통과 |
| segment / `unet_resnet18` | 1 | ImageNet **백본** 구현 | 구현 | 구현 | 실제 내장 구조의 무작위 가중치 export/ORT 테스트 통과 |
| segment / `sam2_hiera_tiny`, `small`, `base_plus`, `large` | 4 | 미구현 | 미구현 | 미구현 | 작은 encoder/decoder fixture의 계약 테스트. 실제 Hiera 네 변형 검증 아님 |

¹ EfficientNet 선택 중 학습 모드 `custom`은 화면에서도 `Custom CSP`로 표시되고, 실제로 CustomCSP를 만든다. 이를 EfficientNet scratch 지원으로 계산하면 안 된다. 선택된 `model_id`와 실제 구조가 달라지는 조합도 거부되지 않는다.

² PatchCore의 fit은 정상 이미지 특징을 모아 메모리 뱅크를 만드는 과정이다. 임의 초기화 백본을 생성할 수 있는 것과 백본을 scratch 학습시키는 것은 다르다. 일반 분류 모델과 동일한 세 가지 학습 모드를 약속해서는 안 된다.

근거: `gui/core/training_modes.py:25–37`, `gui/widgets/training_widget.py:88–113`, `gui/core/trainer.py:457`, `webapp/worker.py:87–100, 241–257`, `python/builtin_models.py`, `gui/core/patchcore_trainer.py`.

## 3. 우선 수정할 결함

### R1 — 기본 모델 9개 변형의 실제 실행 경로 누락 / 출시 차단

`PENDING_NATIVE_MODEL_IDS`에 LibreYOLO 2개, RT-DETRv4 3개, SAM2 4개가 명시되어 있다. `validate_training_options`가 native worker·ONNX 인수 미완료라는 오류로 실행을 차단한다. 엉뚱한 모델로 대체 실행하지 않는 차단은 유지해야 하지만, 차단 안내만 추가해서는 기본 제공 요구가 충족되지 않는다.

수정: 실제 upstream 구조의 native adapter, 초기화, 데이터 변환, 학습, 체크포인트, 추론, ONNX, C#/C++를 변형별로 연결한다. 기본 모델에 `.dvmodel` 설치를 요구하지 않는다.

### R2 — EfficientNet 선택과 실제 학습 모델이 달라질 수 있음 / 높음

재현: `model_id=efficientnet_b0`, `training_mode=custom`인 프로젝트를 검증하면 통과하고, `TrainWorker._build_model`은 `CustomCSP`를 반환한다. EfficientNet 전용 worker의 모드 집합은 finetune/transfer/resume 세 가지뿐이다.

수정: 모델 구조와 초기화 방식을 별도 필드로 분리한다. `scratch`도 동일한 EfficientNet factory에서 `weights=None`으로 생성해야 한다. 기존 모호한 설정은 저장된 checkpoint 구조를 확인하여 명시적으로 마이그레이션한다.

### R3 — 프로젝트 전환 후 이전 체크포인트가 ONNX 화면에 남음 / 높음

근거: `gui/widgets/export_widget.py:165–179`. 새 프로젝트에 run이 없으면 `ckpt_edit`를 비우지 않는다. 출력 폴더는 새 프로젝트로 바뀌므로 이전 모델을 새 프로젝트의 export로 오인할 수 있다.

재현: A 프로젝트의 `.pt` 경로 설정 → 학습 이력이 없는 segment B로 변경 → A 경로 유지, 출력은 `B/exports/model_segment.onnx`.

수정: 프로젝트 전환 시 입력/검증/완료 상태를 초기화하고, 내보내기 전에 project/run/model/task/checkpoint hash의 일치를 검사한다. 외부 checkpoint 가져오기는 별도 명시적 동작으로 둔다.

### R4 — 지우개를 편집 가능한 일반 브러시 객체로 저장 / 높음

근거: `segmentation_canvas.py:174–183`, `segmentation_annotation.py:142–167, 194–213`, `mask_annotations.py:70–85`.

실제 Qt 마우스 클릭 재현 결과:

| 단계 | 목록 행 수 | 클릭한 픽셀의 class ID |
|---|---:|---:|
| 클래스 1 다각형 생성 | 1 | 1 |
| 지우개 클릭 | 2 (`2 브러시` 추가) | 0 |
| 문서 payload 저장 형식으로 다시 읽기 | — | 0 |
| 추가된 행의 클래스 선택을 2로 변경 | 2 | 2 |

따라서 현 코드에서 “지우개 클릭 직후 저장 픽셀이 항상 전경으로 추가된다”까지 재현된 것은 아니다. **삭제가 객체 추가로 표현되고, 삭제 기록이 다시 칠하는 객체로 바뀔 수 있다는 문제는 재현했다.** 삭제 기록을 목록에서 지우면 예전 영역이 복원되는 구조도 사용자 기대와 다르다.

Qt의 드래그 미리보기는 흰색 선, 웹은 노란 선이다. 실제로 빠진 마스크를 즉시 보여주지 않는다. 도형 hit-test도 최종 픽셀 구멍이 아닌 원래 다각형 중심이다.

현재 detection 도구에는 지우개 버튼 자체가 없으며 선택 객체 Delete 방식이다. 동일한 사용 보고를 detection의 정상 toolbar에서 재현했다고 주장하지 않는다. 검출은 픽셀 지우개 대신 객체 클릭 삭제 동작을 명확히 제공해야 한다.

### R5 — 외부 팩이 기본 모델 ID를 덮어쓸 수 있음 / 높음

근거: `gui/core/model_registry.py:288–300`, `tests/test_model_registry.py:test_registry_activates_current_pack_over_catalog_without_loading_old_versions`.

임시 설치 디렉터리의 manifest를 `model_id=resnet18`, `runtimes=[container]`, `release_status=release_ready`로 만들어 registry를 읽으면 기본 ResNet18이 해당 정의로 바뀌며 오류가 없다. 이 재현은 registry 병합 단계 테스트이며 미서명 archive가 installer 서명 검사를 우회했다는 뜻은 아니다.

수정: 기본 모델 ID를 예약하고 사용자 팩은 `vendor.model` 네임스페이스만 허용한다. 팩이 주장하는 `release_ready`를 제품 인수 증거로 신뢰하지 않는다.

### R6 — 테스트 수준과 제품 지원 표시가 일치하지 않음 / 높음

- `test_builtin_model_adapters.py`: 실제 5개 내장 구조의 ONNX 검증은 유효하다. 다만 `cpp_supported=True` assertion은 C++ 실행 증거가 아니다.
- `test_sam2_export.py`: 작은 테스트용 encoder/decoder. Hiera T/S/B+/L 인수 증거가 아니다.
- `test_export_contracts.py`: `TinyReDetr`의 boxes/logits 계약. RT-DETRv4 S/M/L 인수 증거가 아니다.
- `test_patchcore_onnx_export.py`: `TinyBackbone`, 8×8 입력, 4×2 bank. 실제 PatchCore bank의 처리 시간·메모리를 보장하지 않는다.
- `web/tests/teaching-workflow.mjs`는 지우개 후 목록 행이 늘어나는 현재 동작을 성공 조건으로 둔다. 현재 `npm test`는 이 브라우저 파일을 실행하지 않는다.

수정: 계약 단위검사 / 실구조 검사 / 학습 결과 검사 / Windows SDK 검사 증거를 분리한다. 제품 지원 표시는 마지막 인수 조건으로 계산한다.

### R7 — Docker 설치 흐름과 학습 화면 통합 부족 / 중간

`model_manager_widget.py:_install_pack`는 archive 검증·설치를 GUI thread에서 직접 호출한다. 큰 팩은 화면을 멈추게 할 수 있다. 진행률·취소·실행환경 검사·self-test·복구 버전 선택이 하나의 설치 흐름으로 연결되어 있지 않다. 신뢰 키는 `PackInstaller` 인자/환경변수에 의존한다.

`packaging/model-pack-template/worker.py`의 train/infer/export는 `not_implemented`다. 추가 모델 개발이 Dockerfile 작성만으로 끝나지 않는다. `training_widget.py:_on_pack_completed`도 일반 학습과 동일한 RunRecord/지표 대신 결과 dict를 표시한다.

수정: 개발자 scaffold/검사 도구와 최종 사용자의 설치 마법사를 구분하고, native/container의 학습 이벤트와 산출물을 공통 계약으로 통합한다.

### R8 — 사전학습 사용 가능과 설치본에 파일 포함을 혼동 / 배포 전 해결

`default-model-catalog.json`의 배포 정책은 `weights_included=false`다. `python/builtin_models.py`는 사전학습 요청 시 캐시 또는 다운로드를 이용한다. 따라서 “모든 pretrained가 한 설치파일에 포함되어 오프라인 최초 실행 가능”이라는 상태가 아니다.

`packaging/windows/README.ko.md`는 첫 실행 오프라인 목표와 외부 가중치 미포함을 함께 적고 있어 제품 안내를 정리해야 한다. 이번 검토에서 가중치 재배포의 법적 적합성을 판정하지 않았다. 사용자가 가중치 이슈는 보류한 만큼 이 검토에서 재배포/다운로드를 새로 진행하지 않았다.

## 4. ONNX의 현재 복구 로직과 한계

EfficientNet은 최적화 그래프 실패 → 원본 FP32 → ORT all/basic/disabled 순으로 모든 probe를 재검사하고, 통과한 실행 설정을 schema 6 manifest에 넣는 코드가 있다 (`python/export_onnx.py:485–560`). 과거 “최적화를 끄면 통과” 문제를 무시한 구조는 현재 코드에서는 아니다.

다만 해당 fallback은 EfficientNet에 한정된다. 다른 모델도 같은 수준의 복구·진단을 갖췄다고 할 수 없다. 또한 사용자의 보안 데이터·가중치는 제공되지 않았으므로 그 체크포인트가 현재 코드에서 통과하는지는 검증하지 않았다.

개선 시 오차 허용값을 임의로 늘려 통과시키지 않는다. 모델 원본 대비 graph 변환 정확도와 선택한 ORT 설정의 정확도를 각각 검증한다. 통과한 설정을 C#/C++가 실제로 적용하는 것을 Windows에서 확인해야 한다. 정답 판정이 맞는지와 로그잇 수치가 가까운지도 별개로 기록한다.

## 5. 이번 실행 결과와 검증하지 못한 범위

환경: macOS 26.5 ARM64, Python 3.11.15, PyTorch 2.14.0, torchvision 0.29.0, ONNX 1.22.0, ONNX Runtime 1.29.0, PySide6 6.11.2.

| 검사 | 명령/방법 | 결과 |
|---|---|---|
| 전체 Python 테스트 | `QT_QPA_PLATFORM=offscreen python -m pytest -q` | 840 passed, 11 skipped, 185 warnings; 87.89초 |
| 웹 단위 테스트 | `cd web && npm test` | 13 passed |
| 웹 타입/배포 빌드 | `cd web && npm run build` | 성공 |
| 지우개 동작 | QApplication + QTest 실제 toolbar/클릭/클래스 dropdown | R4 재현 |
| ONNX 화면 프로젝트 전환 | ExportWidget에 A→B 순서로 set_project | R3 재현 |
| EfficientNet/custom 조합 | validate_training_options + 실제 model factory | 검증 통과, CustomCSP 생성 |
| 기본 ID 덮어쓰기 | 임시 설치 manifest + registry_with_installed_packs | R5 재현 |

11개 skip은 통과에 포함하지 않았다. 경고에는 ONNX legacy exporter 폐기 예정, 고정 shape tracing, 중복 archive 방어검사, Qt 그래프 한글 폰트 등이 있다. 이번 실행에서 실패는 없었지만 위의 제품 요구 누락을 검출하지 못하는 테스트 공백이 있다.

Windows 설치본·CUDA 학습·실제 i7 지연시간·18개 변형의 pretrained 학습·전체 실모델 C#/C++ 실행·웹 Playwright E2E는 이번에 실행하지 않았다. 특히 사용자 RTX4000에서의 성능이나 EfficientNet 8ms 목표 달성은 이 결과로 판단할 수 없다.

## 6. 외부 참고와 최종 방향

Roboflow 공식 설명의 Smart Polygon은 선택 영역의 추가/제거와 수정 후 저장 흐름을 제공한다. 여기서 참고할 점은 도구의 이름보다 **기존 영역을 편집하고 결과를 즉시 확인하는 동작**이다. 현재 앱이 Roboflow 동등 품질이라고 평가할 근거는 없다. [공식 사용 예](https://blog.roboflow.com/how-to-label-segmentation-data/), [영역 수정 설명](https://blog.roboflow.com/measuring-the-accuracy-of-drawn-circles-with-computer-vision/).

우선순위: 잘못된 모델/체크포인트 선택 방지 → 지우개와 저장 모델 수정 → 기본 native adapter 완성 → 실모델 ONNX/Windows SDK 인수 → Docker 확장 설치 경험과 오프라인 배포 완성. 상세 인수 조건은 설계 문서에 둔다.
