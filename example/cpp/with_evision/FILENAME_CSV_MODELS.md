# 파일명 버전 + CSV로 여러 분류 모델 교체하기

사용자 `example/useage`의 `CSeVision`·`shared_ptr` 사용 방식에 맞춘 C++17 예제다.
**기존 클래스에 적용할 코드:** [sevision_csv_example.cpp](sevision_csv_example.cpp)
**관리 헤더:** [csv_model_manager.h](csv_model_manager.h)
**사용할 CSV:** [models.csv](models.csv)

## CSV와 파일 배치

```csv
index,camtype,task,name
0,TOP,classify,sevision_ai_261126_16H.dvsenc
1,SIDE,classify,sevision_side_261126_15H.dvsenc
```

```text
검사프로그램.exe
models.csv
sevision_ai.key
models/
  sevision_ai_261126_16H.dvsenc
  sevision_side_261126_15H.dvsenc
```

- `index`: 모델을 찾는 고유 번호. 0부터 연속일 필요가 없고 행 순서와 무관하다.
- `camtype`: 프로그램에서 쓰는 카메라 구분. 예제의 TOP/SIDE를 실제 값으로 바꾼다.
- `task`: **이 관리기는 BW8 분류용 `classify`만 지원한다.** 분할·탐지 값은 거부한다.
- `name`: 모델 폴더 안의 파일명. 경로나 하위 폴더는 적지 않는다.
- UTF-8 CSV이며 BOM/Windows 줄바꿈/따옴표를 지원한다. 열 순서는 바꿔도 된다.
- 날짜는 `YYMMDD`, 연도는 2000~2099다. `261126_16H`는 2026-11-26 16시다.
- 같은 시간에 여러 번 배포하려면 `sevision_ai_261126_163025.dvsenc`처럼 `HHMMSS`도 지원한다.
- 비교하는 것은 **파일명의 날짜·시간**이다. 같은 이름을 덮어써도 다시 읽지 않는다.
  같은 시각을 다른 표기(`16H`/`160000`)로 바꿔도 새 버전으로 보지 않는다.
- `.dvsenc` 내부 `model_version`은 내보내기 시각으로 남는다. 파일명과 같을 필요는 없으며
  파일명 변경은 가중치/입출력/학습 결과를 바꾸지 않는다. 재학습은 필요 없다.
- 같은 index의 camtype/task/파일명 앞부분(`sevision_ai`)은 고정한다. 다른 용도로
  재배치하거나 행을 삭제하려면 감시 중이 아닌 프로그램 시작 전에 CSV를 바꾼다.
  실행 중 새 index 추가와 기존 모델의 새 날짜 파일 적용은 지원한다.

## 내 CSeVision에 넣을 부분

기존 include 경로를 유지하고 새 헤더를 추가한다. `csv_model_manager.h`는 기존
`csv_model_watcher.h`, `classifier.h`, `bw8_preprocess.h`, `model_crypto.h`를 사용한다.
ONNX Runtime/nlohmann/Windows bcrypt 링크 설정은 기존 암호화 예제와 같다.

```cpp
// SeVision.h — 기존 클래스 안의 멤버
std::unique_ptr<dvs_bw8::CsvModelManager> models_;
```

```cpp
// SeVision.cpp — 검사 스레드를 시작하기 전에 한 번 실행
#include "csv_model_manager.h"
#include "example_paths.h"

void CSeVision::InitializeModel()
{
    if (models_) throw std::logic_error("Models already initialized");
    const auto folder = example::ExecutableDirectory();
    auto next = std::make_unique<dvs_bw8::CsvModelManager>(
        folder / "models.csv", folder / "models", folder / "sevision_ai.key");
    next->Start(); // 모든 최초 모델 로드·워밍업 성공 후 반환. 실패 시 예외.
    models_ = std::move(next);
}
```

`InitializeModel()`의 실패는 기존 초기화 오류 창/로그에서 처리하고 검사를 시작하지 않는다.
그 함수 안에서 만든 `models_`를 검사 중 다시 대입하거나 없애지 않는다.
위 멤버를 헤더에 선언할 때는 헤더에도 `#include "csv_model_manager.h"`를 둔다.

```cpp
// 한 프레임 검사 시작 시. 기존 thd_classifier 전역 변수 대신 지역 복사본 사용.
auto thd_classifier = models_->Snapshot(0, "TOP");
if (!thd_classifier)
    throw std::runtime_error("해당 index/camtype 모델이 준비되지 않았습니다.");

auto result = thd_classifier->InferEvision(roi); // roi는 기존 EROIBW8
```

다른 cpp에서는 예제의 `CSeVision::GetModel(index, camtype)`처럼 공개 함수를 통해 받는다.
한 이미지의 OMP 반복문은 **그 바깥에서 한 번 받은 동일한 shared_ptr**를 사용한다.

```cpp
auto model = GetModel(0, "TOP"); // 예제 GetModel은 null이면 예외를 던짐. OMP 진입 전 실행.
std::vector<dvs_bw8::Result> results(rois.size());
std::vector<std::exception_ptr> errors(rois.size());
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(rois.size()); ++i) {
    try { results[i] = model->InferEvision(*rois[i]); }
    catch (...) { errors[i] = std::current_exception(); }
}
// errors를 확인한다. 다음 프레임에서 GetModel을 새로 호출한다.
```

이전 모델의 `contexts`를 새 모델에 넘기지 않는다. 기본 `InferEvision(roi)`는
동시 호출마다 독립적인 작업 메모리를 빌려 쓰며, 필요할 때 늘리고 재사용한다.
ROI 객체/부모 영상은 검사 중 수정하지 않는다. 이 예제는 흑백 BW8 입력용이다.

## CSV 변경 시 실제 동작

1. **감시 스레드 하나**가 기본 1초마다 CSV 전체를 읽는다. 대기 중에는 쉰다.
2. 같은 파일명이나 과거 날짜면 모델 파일/키를 다시 읽지 않는다.
3. 새 파일이면 Package를 한 번 복호화하고 분류기 생성·워밍업을 마친다.
4. 로딩 중 CSV가 바뀌면 그 후보를 적용하지 않고 다음 주기에 다시 읽는다.
5. 준비된 모델의 shared_ptr를 교체한다. 진행 중인 검사는 자신이 받은 기존 모델로 끝난다.
6. 이전 모델의 마지막 검사 참조가 사라지면 감시 스레드에서 해제한다.

각 모델은 준비되는 대로 개별 교체된다. 여러 카메라 모델이 하나의 거래처럼 동시에
교체되는 API는 아니다. 한 모델의 로드 실패는 다른 정상 행의 갱신을 막지 않는다.
형식 오류·중복 index·기존 행 삭제/재배치는 CSV 전체를 거부하며 현재 모델을 유지한다.

누락 파일·잘못된 키·변조·입출력 불일치·워밍업 실패 시 기존 모델을 유지하고
`models_->LastStatus()`에 원인을 남긴다. **CSV가 그대로여도 다음 주기에 재시도**한다.
최초 Start는 모든 행이 준비되어야 성공하고, 하나라도 실패하면 아무 모델도 공개하지 않는다.

옛 모델을 쓰는 검사가 남아 있으면 해당 index의 다음 교체는 기다린다. 따라서 받아 둔
shared_ptr를 전역에 영구 보관하지 않는다. 다른 index의 갱신은 계속 진행한다.
교체할 때 기존/신규 모델이 동시에 메모리에 존재하며, 복호화·초기화는 CPU를 사용한다.
이 때문에 교체 중 지연시간까지 동일하다고 보장하지 않는다.

모델별 키가 다르면 파일 경로 대신 콜백을 전달한다.

```cpp
dvs_bw8::CsvModelManager models(csvPath, modelFolder,
    [&](int index) -> dvs_crypto::Key { return ReadMyModelKey(index); });
```

키는 CSV에 넣지 않는다. 콜백은 초기 Start와 감시 스레드에서 호출할 수 있어야 한다.
동봉 `example-only.key`와 작은 모델은 공개 테스트용이므로 실제 모델 보호에 사용하지 않는다.

## 배포·종료·빌드

새 `.dvsenc` 파일 복사를 먼저 완료한 뒤 CSV의 `name`을 새 이름으로 바꾼다.
CSV는 임시 파일로 완성하고 같은 디스크에서 교체한다. Windows에서는 `ReplaceFileW`나
적절한 `MoveFileExW` 교체를 사용하고, 파일 공유 충돌 시 재시도한다.
CSV를 쓰는 중간 내용을 읽지 않게 하기 위한 규칙이다. 수정 시각만 보지 않으므로
mtime이 같아도 CSV 내용 변경을 확인한다.

종료: 새 검사 접수 중단 → 검사 스레드 join → `models_->Stop()` → CSeVision 해제.
`Stop()`은 모델 로드가 진행 중이면 그 로드가 끝날 때까지 기다릴 수 있다.

기존 with_evision CMake 빌드에 `evision_sevision_csv_example`과
`evision_filename_csv_test`가 포함된다. 실행 인자는 필요 없다.
`sevision_csv_example.cpp`는 동봉된 작은 합성 모델과 모의 BW8 입력으로 실행한다.
실제 프로젝트는 자신의 EROIBW8을 그대로 전달한다.
Windows VS2017 15.9/C++17 빌드는 기존 네이티브 CI에 포함되며,
로컬 macOS 검증은 실제 Windows/eVision SDK 검증을 대신하지 않는다.
