# 기본 모델 완성·ONNX 인수·라벨링 개선 설계

작성일: 2026-09-20

상태: 설계. 아래는 구현/인수 목표이며 현재 동작을 보장하지 않는다.

근거: [코드 검토 및 재현 결과](../../03-analysis/model-readiness-review-20260920.md)

## 1. 제품 불변 조건

1. 현재 카탈로그 18개 변형은 모두 Windows native 기본 제공 모델이다. 사용자가 Docker 팩을 추가해야 기본 모델이 실행되는 구조를 만들지 않는다.
2. Docker는 이후 추가하는 외부 모델의 실행 방법이다. 기본 모델과 같은 학습·지표·추론·내보내기 UI를 사용한다.
3. 선택한 모델 구조는 초기화 방식, 데이터셋, 실행 장치 변경으로 바뀌지 않는다.
4. 검증된 ONNX는 전처리/후처리/실행 설정과 함께 배포한다. 검증 실패 파일을 정상 배포물로 공개하지 않는다.
5. 편집 화면과 저장 라벨이 같아야 한다. 지우개는 라벨을 제거하고 새로운 객체를 만들지 않는다.
6. 사용자 이미지와 가중치를 외부로 업로드하지 않는다. 수치 진단과 실제 검증도 로컬에서 실행한다.

## 2. 모델 정의와 학습 방식 분리

### 2.1 공통 모델 계약

현재 `model_id`, 학습 모드 문자열, release_status가 섞인 구조를 아래 개념으로 나눈다.

```text
ModelDescriptor
  model_id, family, variant, task, architecture_revision
  origin: builtin | extension
  runtime: native | container
  input_contract, output_contract
  supported_initialization, supported_operations
  pretrained_assets: id, source, hash, scope(backbone/full), local_availability

TrainingRequest
  project_id, model_id, model_revision, dataset_revision
  initialization: pretrained | local_transfer | scratch
  weights_asset_id / local_checkpoint_hash
  resume_run_id (별도 동작: 저장된 설정/optimizer/rng 복원)
  effective_config, requested_device

ModelEvidence
  model_revision, operation, initialization, input_profile
  runtime/provider/version, OS/architecture, test_kind
  checkpoint_hash, artifact_hash, test_suite_revision, result
```

`resume`는 새로운 초기화 방식이 아니라 저장된 실행의 재개다. 새 class head로 전이할 때는 모델 구조 호환성을 검증하고 허용한 head만 교체한다. 클래스 순서·채널·crop/resize·정규화는 checkpoint에 보존한다. 로드 누락을 `strict=False`로 숨기지 않는다.

기본 모델 ID는 예약한다. 추가 팩은 `vendor.model` 형식을 사용하며 기본 모델 ID와 충돌하면 설치/등록 양쪽에서 거절한다. 제품 업데이트로 기본 모델을 갱신하는 경로와 외부 팩 설치 경로를 분리한다.

### 2.2 실제 실행 adapter

```text
adapter.build(initialization, weights, model_config)
adapter.dataset(label_contract, transforms)
adapter.train(request, events) -> RunArtifacts
adapter.load_checkpoint(path) -> ModelInstance
adapter.infer(instance, inputs) -> CanonicalPrediction
adapter.export(instance, profile) -> StagedDeployment
adapter.verify(staged, reference, inputs) -> VerificationReport
```

dispatcher는 `(model_id, model_revision)`으로 adapter를 결정한다. 초기화 모드 이름을 보고 다른 factory로 이동하지 않는다. 지원하지 않는 조합은 학습 시작 전 구체적인 이유로 거절한다.

### 2.3 모델군별 구현 범위

| 모델군 | 구현 작업 | 인수 시 확인 |
|---|---|---|
| EfficientNet B0/B1 | 동일 factory에 scratch 추가; pretrained/local/resume와 메타데이터 통일 | 네 경로에서 실제 architecture ID와 B0/B1 파라미터 구조 일치 |
| ResNet18/50, ConvNeXt Tiny | 기존 adapter 유지; 공통 초기화/재개/지표 연결 | pretrained/local/scratch 각각 학습→checkpoint 재로드 |
| DeepLabv3+, U-Net | 백본 사전학습 범위 표시; full segmentation checkpoint 전이 구분 | background/ignore index, head 교체, 원본 해상도 복원 |
| LibreYOLO 분류/검출 | 고정 upstream revision의 native builder·trainer·checkpoint adapter 구현 | 실제 지정 architecture 사용; generic CSP 대체 금지; 두 task 각각 export |
| RT-DETRv4 S/M/L | 세 모델별 native 설정/가중치 변환/학습 criterion/출력 연결 | bbox 좌표계, class activation, top-k/후처리와 S/M/L identity |
| SAM2 Hiera T/S/B+/L | 실제 encoder/prompt encoder/mask decoder, prompt 데이터 변환, fine-tune/scratch 연결 | 점/박스 prompt, 여러 객체, padding, high-resolution features, encoder/decoder 다중 graph |
| PatchCore 두 백본 | pretrained/local feature extractor + fit/rebuild bank 경로 정리 | bank/백본/threshold 결합 checkpoint; 정상/비정상 판정과 anomaly map |

PatchCore UI에는 일반 `스크래치 학습`을 노출하지 않는다. `정상 데이터로 메모리 뱅크 만들기`, `로컬 백본으로 만들기`, `저장 모델 불러오기/다시 구성`을 노출한다. 랜덤 백본은 연구 기능으로 별도 격리하고 기본 학습 지원으로 계산하지 않는다.

SAM2는 semantic class classifier와 다른 prompt/객체 마스크 모델이다. foreground mask를 임의의 제품 클래스처럼 해석하지 않는다. class 매핑은 사용자의 객체 라벨과 연결한다. 현재 네 Hiera 변형의 이미지/prompt 경로를 필수 인수 단위로 둔다. 기존 Windows README에 적힌 video 상태 계약도 별도로 명시하고, 이를 구현·검증하기 전에는 포괄적인 “SAM2 모든 기능 지원” 문구를 쓰지 않는다.

### 2.4 모든 실행 경로의 공통 화면

`epoch_finished` 이벤트에 epoch/total, train_loss, val_loss, task_metrics, epoch_seconds, elapsed_seconds, actual_device, effective_batch_size를 담는다. 정의되지 않는 지표는 `null`과 이유를 보낸다. PatchCore fit 단계에 가짜 epoch loss를 만들지 않는다.

RunRecord는 best metric 이름·방향·epoch·값, last checkpoint, best checkpoint, 선택 규칙을 모두 저장한다. 분류/검출/분할에서 val loss 계산이 지원되면 선택 목록에 동일하게 제공한다. UI는 지원되지 않는 옵션을 숨기거나 이유를 표시한다. native/container는 같은 결과 표와 시간 표시를 사용한다.

## 3. ONNX를 최우선 출시 조건으로 만들기

### 3.1 내보내기 흐름

```text
프로젝트 / run / checkpoint 선택
  -> identity·hash·task·class/preprocess 계약 검사
  -> 학습 checkpoint를 새 프로세스에서 복원 (eval, FP32)
  -> 임시 디렉터리에 graph 생성
  -> ONNX checker + 입출력 schema 검사
  -> 원본 모델 ↔ 변환 모델 ↔ ORT 출력 검증
  -> 실제 배포 설정으로 모든 검증 입력 재실행
  -> C++17 / C# 실행 인수 (릴리스 단계)
  -> report + manifest + graph를 원자적으로 공개
```

프로젝트를 바꾸면 이전 checkpoint·진행률·성공 배지를 모두 지운다. 외부 `.pt/.pth`는 명시적인 “외부 체크포인트 가져오기”에서 구조를 분석한 뒤 연결한다. 이미 실행 중인 job은 시작 당시 project/run/hash에 고정하고 완료 이벤트를 현재 다른 프로젝트에 적용하지 않는다.

현재 `검증` checkbox로 일반 배포 검증을 생략하는 흐름은 없앤다. 개발용 미검증 graph 생성은 별도 기능이며 `unverified`를 붙이고 정상 SDK 배포 bundle로 만들지 않는다.

### 3.2 검증 입력과 비교 규칙

- 빠른 계약 검사: 영/상수/seeded noise, 채널별 패턴, 극단값, 지원 shape/batch.
- 실제 모델 검사: 고정된 합성 라벨 데이터로 실제 adapter 학습 후 저장/새 프로세스 복원/export. variant와 초기화 방식별 실행.
- 정확도 인수: 배포 허용된 작은 고정 검증셋과 사용자가 로컬에서 선택한 이미지. 같은 decoded pixels에서 PyTorch와 ORT 전처리까지 비교한다. 사용자 데이터는 전송하지 않는다.
- C#/C++ 동일 입력: 파일 디코드, 채널 순서, resize/crop, 정규화, tensor, 출력/후처리를 단계별로 비교한다.

| 출력 | 필수 검사 |
|---|---|
| 분류 | logits 유한성·절대/상대 오차, 확률, top-1; 동률 근처 판정 변화도 별도 기록 |
| 의미 분할 | per-pixel logits 오차, argmax 차이, class IoU/경계 변화, 원본 좌표 복원 |
| 검출 | graph raw output 비교, class/score, 후보 수, bbox 좌표계, 매칭 IoU, 최종 threshold/top-k/NMS 동작 |
| PatchCore | score·threshold 양쪽 판정, anomaly map, bank 내용/크기, 실제 bank 실행 메모리 |
| SAM2 | 각 encoder feature, prompt 입력/출력, mask logits/IoU score, 다중 마스크 선택, 원본 크기 복원 |

허용 오차는 모델/정밀도/비교 대상별 profile로 버전 관리한다. 먼저 기존 FP32 기준을 사용하고 실제 검증셋 분포로 근거를 남긴다. 실패할 때마다 허용값을 키우지 않는다. 지표 보존과 수치 비교를 둘 다 통과해야 하며, 경계값 실패를 숨기지 않는다.

### 3.3 실행 최적화와 배포 계약

원본 FP32 기준 출력을 먼저 고정하고 Conv/BN fusion·constant folding·simplification 변환도 원본과 비교한다. ORT all/basic/disabled는 adapter가 허용하는 후보만 시도한다. 후보 하나가 통과하면 **전체 입력을 그 설정으로 재검사**한다. 그래프가 유효하지 않거나 수치가 비유한 경우 설정 재시도로 성공을 꾸미지 않는다.

manifest에는 model/task/variant/revision, graph hashes, checkpoint hash, preprocessing/postprocessing, input/output schema, opset, precision, ORT provider/version/optimization/threads, tolerances, probe/real-image counts, report hash를 기록한다. 스키마를 모르는 SDK나 필요한 옵션을 지원하지 않는 SDK는 명시적으로 거절한다.

`최적화 비활성화로 검증 통과`는 정상적인 검증 결과로 표시하되 실행 속도는 별도로 측정한다. 동일 Windows CPU에서 warm-up 후 전처리+추론+후처리의 median/p95, cold start, 모델 로딩, 큐 대기 시간을 구분한다. <8ms는 CPU 정확한 모델명·thread·입력·bundle이 고정된 실측으로만 판단한다.

### 3.4 실제 인수 표

18개 변형별로 native load → 해당 학습 모드 → checkpoint reload → inference → ONNX → ORT → Windows C++17 → Windows C# → 설치본 실행을 행렬로 관리한다. 작은 대체 모델은 `contract-fixture` 증거만 만든다. 학습 데이터 다운로드 성공, manifest 생성, cpp_supported 플래그는 SDK 실행 통과를 대체할 수 없다.

CI를 빠른 단위검사와 무거운 실모델/Windows 인수로 분리한다. 릴리스 대상 variant 행에 `skipped`, 누락, 증거 hash 불일치가 있으면 해당 전체 지원 릴리스를 차단한다. “기본 모델 모두 제공” 출시에서는 18개 중 일부 완료를 전체 완료로 표시하지 않는다.

## 4. 라벨링 편집 설계

### 4.1 사용자 동작

| 도구 | 기대 동작 |
|---|---|
| 선택 | 보이는 실제 객체 선택; 이동·크기·꼭짓점 편집; 겹친 객체 순환 선택 |
| 다각형/사각형 | 닫힌 영역 생성, class 지정, 저장 전 미리보기 |
| 브러시 | 현재 class의 영역 추가; 원본 픽셀 기준 크기와 cursor 표시 |
| 지우개 E (분할) | 선택 객체가 있으면 그 객체, 없으면 현재 class에서 픽셀 제거. 대상은 toolbar에 표시. `모든 클래스`는 명시적으로 선택 |
| 클릭 삭제 (검출) | hover 객체 강조 후 클릭으로 bbox 객체 삭제. 빈 곳 클릭은 아무 변화 없음 |
| Smart mask (후속) | 로컬 SAM2 양/음성 점 또는 박스로 후보 생성 → 기존 영역과 합치기/빼기 → 확인 후 반영 |

지우개 드래그 중에는 지워진 마스크를 실제로 표시한다. 흰색/노란색 새 도형을 겹쳐 보여주지 않는다. 빈 배경을 지워도 객체·변경 상태·undo 기록이 늘어나지 않는다. 한 번의 drag는 undo 한 번이고 pointer가 화면 밖에서 끝나거나 capture를 잃어도 그리기 상태가 남지 않는다. gesture 시작 시 tool/class/target을 고정한다.

검출 bbox를 일부 픽셀만 지우는 애매한 기능은 만들지 않는다. 클릭 삭제는 객체 단위이며, 영역 수정은 bbox handle로 한다. 지원되지 않는 tool 값은 draw로 fallthrough하지 않고 거절한다.

### 4.2 문서 v2와 실행 취소

현재의 `base + 모든 paint/erase stroke 리스트`를 그대로 객체 목록에 노출하는 구조를 변경한다.

```text
AnnotationDocument v2
  image_id, width, height, revision, task, class_schema
  semantic: authoritative class-index raster + object ownership map
  objects: stable ID, class, geometry/mask, locked/visible
  detection: stable ID, bbox, class

EditTransaction (문서 객체와 분리)
  tool, target_ids/class, dirty_tiles
  before/after pixel + ownership patches OR object change
```

semantic raster는 저장 PNG의 기준이다. semantic 객체 mask는 겹치지 않게 ownership을 관리한다. 새 칠하기가 기존 픽셀을 덮으면 이전 객체의 해당 소유 영역도 함께 제거한다. 지우개는 대상 mask와 raster를 동일 transaction으로 수정하고 background=0으로 바꾼다. 감춰진 이전 class가 다시 드러나지 않는다. 빈 객체는 제거하며 undo는 픽셀·소유권·목록을 함께 복원한다.

class 0은 background로 예약하고 255 ignore는 별도 편집 모드로 취급한다. 일반 지우개는 ignore/잠긴 객체를 보존하며, 사용자가 제외 영역 편집을 명시한 경우에만 변경한다. 객체 클래스 변경은 그 객체가 실제로 소유한 픽셀만 바꾸며 삭제 기록 자체를 색칠 객체로 만들지 않는다.

SAM2 instance mask는 별도 instance 문서 계약을 사용한다. 중첩 instance를 semantic raster로 변환할 때는 명시적 합성 규칙을 사용하고 원래 instance 문서를 보존한다. semantic class-index PNG만으로 겹친 instance를 복원하려 하지 않는다.

다각형에서 지운 구멍은 현재 mask의 일부로 보존한다. 지운 뒤 원래 polygon 좌표를 재사용하여 구멍을 덮어쓰지 않는다. 브러시/지우개로 수정된 객체는 mask 편집 상태로 전환하고, polygon 재변환은 별도 명시적 동작으로 둔다. hit-test도 현재 mask/geometry를 사용한다.

### 4.3 기존 데이터 마이그레이션

v1 payload를 기존 renderer로 한 번 렌더하여 최종 class raster를 얻는다. 편집 metadata의 checksum이 다르면 이미 수정된 PNG 픽셀을 우선한다. 도형 순서를 해석해 실제로 남은 픽셀만 객체 ownership으로 만든다. class 0 stroke는 최종 raster에 반영하고 새 객체로 만들지 않는다.

구멍/덮어쓰기로 원래 polygon을 보존하기 어렵다면 mask 객체로 변환한다. 마이그레이션 전/후 class PNG가 픽셀 단위로 같은지 검사하고 원본을 백업한다. 프로젝트의 학습 라벨 형식은 그대로 유지한다. 저장은 임시 파일+원자 교체, 낡은 revision 덮어쓰기 거절, 실패 시 현재 편집/선택 이미지 유지로 한다.

### 4.4 반응 속도와 공통 검증

pointer move마다 전체 문서를 PNG로 인코딩하거나 전체 작업 이력을 deep-copy하지 않는다. 변경 tile만 갱신하고 undo는 압축 patch를 사용한다. preview 목표는 기준 PC/4K 이미지에서 p95 33ms 이하로 두고 측정한다. memory budget을 넘으면 오래된 undo를 정리하되 현재 라벨은 보존한다.

Qt와 웹은 같은 semantic 계약과 golden operation fixture를 사용한다. 최소 인수:

- 단일 click/drag/확대/이동 상태에서도 대상 픽셀이 배경이 되고 목록 수가 늘지 않음.
- 빈 곳 지우기 no-op; class0/255, 잠금, 선택 class/전체 class 구분.
- 구멍/겹침/이미지 경계, 불규칙한 mask, 배경-only 라벨.
- undo/redo, 클래스 변경, 삭제, 저장→다시 열기 후 동일 픽셀.
- detection 클릭 삭제/빈 곳/겹친 박스 선택과 undo.
- Qt QTest와 실제 브라우저 pointer E2E 모두에서 preview/저장 결과 일치.
- 대형 이미지에서 latency/메모리, 저장 실패/다음 이미지 이동 취소.

Roboflow 공식 설명에서 참고한 것은 기존 영역의 추가/제거 및 수정 후 확정 흐름이다. UI 코드나 자산을 복제하지 않는다. [참고](https://blog.roboflow.com/how-to-label-segmentation-data/).

## 5. 추가 Docker 모델 경험

### 5.1 모델 작성자용

제안 CLI는 아직 미구현이다:

```text
dvmodel init --task detect --id vendor.detector
dvmodel validate ./model
dvmodel test ./model --train --infer --export --sdk-contract
dvmodel pack ./model --offline --sign <key-reference>
```

init은 동작하는 최소 worker·manifest·dataset/output 변환·시험 입력을 생성한다. 개발자가 실제 모델 코드를 채워야 함을 명시한다. train/infer/export 중 `not_implemented`가 있으면 지원 capability를 선언하지 못한다. 재현 가능한 dependency lock, image digest, weights/code notices, 지원 device/shape, resource budget을 팩에 포함한다.

### 5.2 사용자용 Settings → 모델 추가

파일 선택 → 이름/작성자/task/크기/지원 기능/장치/서명 표시 → 호환성 검사 → background 설치 → engine/image load 검사 → 작은 load/infer self-test → 원자적 활성화 순서로 제공한다. 진행률과 취소가 가능해야 하고 실패하면 기존 버전을 유지한다. 설치 완료와 실행 준비 완료를 구분한다.

사용자는 manifest나 환경변수를 편집하지 않는다. 신뢰 publisher 관리, 설치 버전/용량, 기본/추가 구분, 오류 상세, 버전 복구/제거를 Settings에서 제공한다. 외부 팩의 자체 “검증됨” 문구 대신 실제 이 PC self-test 결과와 제품 conformance 증거를 표시한다.

Docker/WSL이 없는 PC, virtualization 비활성화, GPU 미지원, 디스크 부족을 실행 전에 진단한다. 기본 모델은 이 확장 runtime 없이 계속 실행된다. 확장 runtime 설치/활성화는 설치 프로그램이 안내할 수 있지만 PC 정책·관리자 권한·재부팅이 필요한 조건을 “무조건 추가 단계 없음”으로 약속하지 않는다.

## 6. Windows 배포와 문서

기본 native 모델 dependency는 lock된 설치 payload로 공급한다. 추가 pip/conda/Node/Git 설치 없이 깨끗한 Windows VM에서 앱 시작, 라벨링, 학습, ONNX, C#/C++ 샘플을 실행한다. 사용자 PC에 사전 설치된 개발환경 덕분에 성공한 결과를 설치본 인수로 계산하지 않는다.

가중치 처리 보류와 오프라인 pretrained 최초 실행은 서로 다른 조건이다. 가중치 파일을 미포함한 배포는 그 사실을 정확히 표시한다. 향후 모든 pretrained를 포함하는 정식 설치본을 만들 때는 asset 목록/hash/출처/재배포 근거를 확정하고 별도로 인수한다. 현재 검토에서 가중치 배포 결정을 임의로 바꾸지 않는다.

README에는 검증한 Windows 버전, CPU 명령어, RAM/disk, 모델별 GPU VRAM과 지원 batch/input, driver/runtime, offline 지원 범위, 최초 로딩 시간, 추가 모델의 가상화 조건을 적는다. 현재 설계 사양을 실측 최소사양으로 표시하지 않는다.

## 7. 구현 순서와 완료 기준

| 순서 | 작업 | 완료 기준 |
|---|---|---|
| A | 선택 모델/초기화 identity, export 상태 초기화, 기본 ID 보호 | R2/R3/R5 재현을 회귀 테스트로 전환; 잘못된 조합을 실행 전에 차단 |
| B | 지우개 transaction, live subtract, 클래스 변경 제한, v1→v2, detection 클릭 삭제 | 삭제가 객체를 추가하지 않고 저장/재열기/undo의 픽셀 동일성 통과; Qt/웹 E2E |
| C | 공통 adapter/초기화/RunRecord와 기존 9개 실행 경로 정리 | 실제 구조가 유지되고 모든 가능한 모드의 학습·추론·ONNX 증거 저장 |
| D | LibreYOLO 2개·RT-DETRv4 3개·SAM2 4개 native 구현 | 각 변형에서 사전학습/전이/scratch·학습 결과 export까지 실행; 기본 모델 팩 요구 없음 |
| E | 18개 전체 ONNX 수치/후처리/Windows C#/C++ 인수 | 누락/skip 없이 variant별 결과 및 artifact hash 확보 |
| F | 추가 모델 scaffold/설치 wizard/rollback, 오프라인 설치본 | 빈 Windows 환경에서 기본 기능 + 새 외부 모델 전체 workflow 인수 |

E의 검증 기반은 A부터 준비하여 C/D 모델이 연결될 때마다 실행한다. ONNX를 마지막에 몰아서 처리하지 않는다. B는 모델 구현과 독립적으로 먼저 사용자 오류를 줄인다. Smart mask 같은 편의 기능은 지우개/저장 정확성 확보 뒤에 구현한다.

설계 문서 작성만으로 제품 버전을 올리지 않는다. 후속 구현은 사용자 규칙에 따라 버그 수정 +0.01, 간단한 기능 +0.1, 새 기능/새 모델 +1.0을 적용하고 실제 변경 범위로 릴리스 노트를 작성한다. 완료 보고는 코드 작성, 로컬 테스트, Windows 인수, Git push 여부를 각각 구분한다.
