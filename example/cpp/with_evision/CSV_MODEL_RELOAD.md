# CSV 감시로 암호화 모델 교체하기 — C++17

프로그램을 종료하지 않고 분류 모델을 바꾸는 예제다. 기존 `classifier.h`를 수정하지 않고
`csv_model_watcher.h`를 추가한다. `classifier.h`와 `model_crypto.h`에는 `ModelVersion()`이 있어야 한다.
OpenCV는 필요 없으며 기존 ONNX Runtime·암호화 링크 설정을 사용한다.

## 파일 배치

```text
검사프로그램.exe
models.csv
model.key
models/
  chip.dvsenc
```

`models.csv`는 UTF-8로 저장한다. BOM, Windows 줄바꿈, 쉼표가 들어간 경로의 따옴표도 지원한다.

```csv
slot,task,model_file,model_version
chip,classify,models/chip.dvsenc,20261126_153000_123456
```

- `slot`: 프로그램이 선택하는 검사 이름. 중복할 수 없다. 파일명과 독립적이다.
- `task`: 이 예제는 **분류용 `classify`**다. 분할·탐지용 로더를 구현한 예제가 아니다.
- `model_file`: 암호화된 `.dvsenc` 파일. 상대 경로는 **CSV가 있는 폴더 기준**이다.
- `model_version`: 암호화 파일 내부 버전과 정확하게 같아야 한다. 기존 파일명은 유지해도 된다.
- `model_name` 같은 추가 열은 둘 수 있지만 표시/관리용이며 로딩 동작에 영향을 주지 않는다.
- 입력 크기·정규화·클래스·ORT 설정은 모델 내부 정보를 사용한다. CSV의 추가 열로 덮어쓰지 않는다.
- 키를 CSV에 쓰지 않는다. 예제의 공개 `example-only.key`는 실제 모델에 사용하지 않는다.

## 기존 솔루션에서 호출

```cpp
#include "csv_model_watcher.h"
#include "example_paths.h" // 저장소 example/cpp에 있음

// 프로그램 시작 시 생성한다. 검사 스레드보다 오래 유지한다.
const auto exe = example::ExecutableDirectory();
dvs_bw8::CsvModelWatcher models(
    exe / "models.csv", "chip", exe / "model.key");
models.Start(); // 최초 모델 로드·준비 추론 완료 후 반환. 실패하면 예외.

// 기존 각 검사 스레드에서 호출한다.
auto thd_classifier = models.Snapshot(); // shared_ptr, 파일 읽기/재초기화 없음
auto result = thd_classifier->InferEvision(roi); // 기존 EROIBW8

// 종료 순서: 새 검사 접수 중단 → 검사 스레드 join → 감시 중단 → 객체 해제
models.Stop();
```

위 코드는 각 호출 위치를 보여준다. `models`를 초기화 함수의 지역 변수로 만들고 바로
함수를 끝내면 감시도 종료된다. 실제로는 프로그램/검사 관리자 클래스의 멤버로 보관한다.
`Start()`·`Stop()`은 그 객체를 소유한 스레드에서 호출한다.
`Snapshot()`·`LastStatus()`는 여러 스레드에서 호출해도 된다.
상태 문자열은 UI 타이머에서 `models.LastStatus()`로 읽을 수 있다. 감시 스레드는 UI를 직접 건드리지 않는다.
추론 실패는 기존 검사 코드에서 예외 처리한다. OMP 영역 밖으로 예외를 던지지 않는다.

한 이미지의 여러 ROI를 **모두 같은 모델 버전으로** 검사하려면 반복문 밖에서 한 번 읽는다.

```cpp
auto model = models.Snapshot();
std::vector<dvs_bw8::Result> results(rois.size());
std::vector<std::exception_ptr> errors(rois.size());
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(rois.size()); ++i) {
    try { results[i] = model->InferEvision(*rois[i]); }
    catch (...) { errors[i] = std::current_exception(); }
}
// results/errors를 확인한 뒤 다음 프레임에서 Snapshot()을 다시 받는다.
```

서로 다른 작업은 서로 다른 ROI 객체를 사용한다. ROI와 부모 이미지 버퍼는 검사 중 바꾸지 않는다.
Context는 기본 `InferEvision(roi)`가 관리한다. 이전 모델의 명시적 Context는 새 모델에 넘기지 않는다.

이미 키를 메모리나 라이선스 장치에서 받는 프로그램은 파일 대신 콜백을 전달할 수 있다.

```cpp
dvs_bw8::CsvModelWatcher models(csvPath, "chip", []() -> dvs_crypto::Key {
    return GetModelKey(); // 사용자가 이미 구현한 키 공급 함수
});
```

키 공급 콜백은 최초 `Start()` 호출과 이후 감시 스레드에서 실행된다. 그 스레드에서 호출할 수 있는
함수를 사용한다. 콜백은 외부 원본 키의 수명을 관리하고, 감시기는 받은 키 복사본을 사용 후 지운다.

## 변경 감지와 교체 순서

1. 1초마다 작은 CSV 내용을 읽는다. 수정 시각이 같아도 내용 변경을 감지한다.
2. 현재 버전과 같거나 이전 버전이면 모델 파일을 읽거나 복호화하지 않는다.
3. 새 버전이면 파일을 읽고 인증·복호화한다. CSV와 내부 버전의 일치를 확인한다.
4. 새 `Classifier`를 생성해 입출력 계약 확인·준비 추론을 마친다.
5. 그동안 CSV의 해당 행이 바뀌었으면 이 후보를 버리고 다음 주기에 최신 행을 다시 확인한다.
6. 준비가 완료된 `shared_ptr`로 교체한다. 진행 중인 추론은 기존 `shared_ptr`로 끝난다.
7. 이전 모델은 마지막 사용자가 놓은 뒤 감시 스레드에서 해제한다.

자동으로 이전 버전으로 되돌리지 않는다. 같은 가중치를 새 버전으로 배포하려면 다시 내보내면 된다.
날짜 문자열 비교는 내보내는 PC의 시간대가 같고 시계가 올바르다는 전제다.
모델 버전이 없는 구형 파일은 자동 교체 대상으로 받지 않는다. 기존 `.pt`를 다시 내보내면 되며 재학습은 필요 없다.

CSV 손상, 파일 누락, 잘못된 키, 암호문 변조, 버전 불일치, 모델 초기화 실패 시
현재 모델을 유지하고 `LastStatus()`에 원인을 기록한다. **같은 CSV 그대로여도 다음 주기에 재시도**한다.
최초 모델도 준비되지 않은 `Start()` 실패는 호출자에게 예외를 전달한다.

메모리 증가를 제한하려고 이전 모델을 사용하는 추론이 남아 있으면 다음 모델 교체를 기다린다.
사용자가 옛 `Snapshot()`을 전역에 계속 보관하면 다음 교체가 대기한다. 프레임/작업이 끝나면 놓아준다.
적재 중에는 기존·새 모델과 복호화 버퍼가 함께 필요하다. 모델 준비가 CPU 자원을 사용하므로
교체 중 지연시간까지 무조건 동일하다고 보장하지 않는다. 병렬 공간을 처음 확장하는 비용도 있을 수 있다.
`Stop()`은 진행 중인 모델 로드가 끝날 때까지 기다릴 수 있다.

## 파일을 전달하는 쪽의 규칙

고정 이름을 사용해도 된다. 먼저 `chip.dvsenc.tmp`에 복사를 끝내고 같은 디스크의
`chip.dvsenc`로 원자적으로 교체한다. Windows에서는 `ReplaceFileW` 또는 적절한
`MoveFileExW` 교체 동작을 사용한다. 파일을 연 상태로 조금씩 덮어쓰지 않는다.
Windows에서 다른 읽기 작업 때문에 공유 위반으로 교체가 실패하면 잠시 뒤 재시도한다.
모델 전달이 끝난 뒤에 **CSV를 마지막으로** 임시 파일에서 교체한다.
모델·CSV를 불일치하게 전달하면 현재 모델을 유지하며 재시도한다.

## 실행 예제와 검증

`csv_reload.cpp`는 명령행 인자 없이 실행하는 합성 BW8 예제다. `assets/reload/models.csv`를 읽는다.
실제 프로젝트에는 이 파일의 `main()` 대신 위 초기화·추론 코드를 넣는다.
`csv_reload_test.cpp`는 임시 폴더에서 고정 파일명 교체와 4개 스레드의 동시 추론을 검증한다.
동봉 모델은 작은 공개 테스트 그래프이며 실제 검사 모델이 아니다.

CMake의 `evision_csv_reload_example`, `evision_csv_reload_test` 타깃으로 빌드한다.
기존 Windows 네이티브 예제 CI의 v141/C++17 빌드에도 포함된다.
로컬 검증 환경은 macOS C++17·ONNX Runtime이며 Windows VS2017와 실제 eVision SDK 실행은 별도 검증 대상이다.
