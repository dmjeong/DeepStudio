# CSeVision: 초기화와 CSV 확인, 두 함수만 사용

**복사할 코드:** [sevision_csv_example.h](sevision_csv_example.h)

**실행 예제:** [sevision_csv_example.cpp](sevision_csv_example.cpp)

**샘플 CSV:** [models.csv](models.csv)

기존 CSeVision 클래스에 헤더의 멤버와 함수 두 개를 옮긴다.
이 예제는 감시 스레드를 만들지 않는다. 별도 Start/Stop/GetModel 호출도 없다.

## 1. 프로그램 시작 시 한 번

```cpp
std::string error;
const auto folder = example::ExecutableDirectory();
if (!vision.InitializeModel(folder / "models.csv", folder / "models",
                            folder / "sevision_ai.key", error)) {
    // error를 로그/화면에 표시하고 검사를 시작하지 않는다.
}
```

CSV에 있는 모든 모델을 읽고 워밍업한다. 모두 성공해야 true를 반환한다.
상대 경로는 CSV 폴더를 기준으로 해석한다. 성공 후 검사 스레드를 시작한다.

## 2. 내가 이미 쓰는 감시 스레드에서 반복 호출

```cpp
std::string error; // 이 스레드의 지역 변수
if (!vision.ChangeModel(error)) {
    // error를 로그에 표시. 기존 모델로 검사는 계속할 수 있다.
}
```

이 함수는 CSV를 **한 번 읽고 반환**한다. 호출 간격은 본인 스레드에서 정한다(예: 1초).
동일한 파일명은 키/모델을 다시 읽지 않는다. 더 최신인 파일명만 복호화·초기화·워밍업 후
shared_ptr로 교체한다. true는 확인 성공이며, 변경이 없어도 true다.
실패한 행은 이전 모델을 유지하고 false와 오류를 반환한다. 다른 정상 행은 갱신할 수 있다.
CSV가 그대로여도 다음 ChangeModel 호출에서 실패했던 모델을 다시 시도한다.

## 추론은 기존 atomic_load 방식

```cpp
// CSV의 index=0 모델. 한 프레임 검사 시작에 받고 OMP 루프에서도 이 복사본을 사용한다.
auto model = std::atomic_load(&vision.thd_classifier.at(0));
auto result = model->InferEvision(roi); // 실제 EROIBW8 전달
```

InitializeModel이 성공한 뒤, CSV에 등록된 index로만 접근한다.
`thd_classifier`는 index별 shared_ptr를 담는 map이다. 검사 중에는 map에 직접 항목을
추가/삭제하거나 포인터를 일반 대입하지 않는다. 읽기는 위처럼 atomic_load로만 한다.

같은 index의 camtype/task/파일명 앞부분은 처음 설정대로 유지한다.
카메라와 index를 연결하는 곳은 본인의 검사 코드다. 예: TOP 검사는 index=0,
SIDE 검사는 index=1. camtype을 읽어 카메라 영상을 자동 연결하는 기능은 아니다.

이전 모델에서 만든 Context를 새 모델에 넘기지 않는다. `InferEvision(roi)`가 호출별
독립 작업 메모리를 관리한다. 병렬 작업마다 ROI/부모 이미지 버퍼가 안정적으로 유지되어야 한다.
OMP 안의 추론 오류는 각 반복 안에서 잡고, 병렬 영역 밖에서 처리한다.

## CSV는 그대로 네 항목

```csv
index,camtype,task,name
0,TOP,classify,sevision_ai_261126_16H.dvsenc
1,SIDE,classify,sevision_side_261126_15H.dvsenc
```

- `index`는 고유 번호다. 행 순서는 무관하며 번호가 연속일 필요도 없다.
- `task=classify`: 현재 예제는 BW8 분류용이다. 분할/탐지를 이 분류기로 읽지 않는다.
- `name`은 모델 폴더 안의 `.dvsenc` 파일명이다. UTF-8 CSV, BOM/따옴표/CRLF를 지원한다.
- `261126_16H`는 2026-11-26 16시다. `261126_163025`처럼 분·초까지도 가능하다.
- 같은 이름을 덮어쓰면 변경으로 보지 않는다. 더 최신 날짜·시간의 이름을 사용한다.
- 파일명 날짜는 배포 버전이다. 암호화 파일 내부의 내보내기 시각과 같을 필요는 없다.
- index 추가/삭제 또는 camtype/task/모델명 앞부분 변경은 다음 프로그램 시작 때 반영한다.
  추론 중 map 구조가 바뀌지 않도록 실행 중에는 거부한다.

```text
검사프로그램.exe
models.csv
sevision_ai.key
models/
  sevision_ai_261126_16H.dvsenc
  sevision_side_261126_15H.dvsenc
```

모델 파일 복사를 먼저 끝내고 CSV의 name을 갱신한다. CSV는 임시 파일에 완성한 뒤
교체해 쓰는 도중의 내용을 읽지 않게 한다. 키는 CSV에 넣지 않는다.
재학습은 필요 없으며, 기존 암호화 모델을 날짜가 붙은 이름으로 배치하면 된다.

## 스레드와 파일

교체 전에 시작한 검사는 기존 shared_ptr로 끝난다. 이전 모델은 마지막 사용자가 놓은 뒤
다음 ChangeModel 호출에서 해제한다. 옛 모델을 아직 쓰면 그 index의 다음 교체는 기다린다.
받아 둔 shared_ptr를 전역에 영구 보관하지 말고 프레임/작업이 끝나면 놓는다.
종료 시 본인의 감시·검사 스레드를 먼저 종료/join한 뒤 CSeVision을 해제한다.
초기화와 교체에는 CPU/메모리가 필요하므로 교체 중 지연시간까지 동일하다고 보장하지 않는다.

새 헤더는 기존 csv_model_manager.h의 CSV 파서만 재사용한다. CsvModelManager 객체나
자동 감시 스레드는 만들지 않는다. 기존 classifier.h, bw8_preprocess.h, model_crypto.h,
CSV 관련 헤더, ONNX Runtime/nlohmann/bcrypt 설정을 함께 유지한다.

기존 CMake의 evision_sevision_csv_example은 인자 없이 실행된다. 작은 공개 모델과
모의 BW8 버퍼를 사용하며 실제 eVision SDK가 없어도 예제를 검증할 수 있다.
실제 솔루션에서는 모의 버퍼 대신 자신의 EROIBW8을 전달한다.
동봉 example-only.key는 공개 테스트 전용이며 실제 배포에 쓰지 않는다.

자동 스레드 방식이 필요한 기존 사용자는 csv_model_manager.h의 API를 계속 사용할 수 있다.
