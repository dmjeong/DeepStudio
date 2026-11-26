# 22.71 — 모델 버전 자동 기록

재학습하지 않아도 됩니다. 기존 `.pt`를 내보내면 버전이 자동 기록됩니다.
앱 버전이나 JSON `schema_version`과 다른, 배포 모델의 생성 시각입니다.

```json
{
  "model_version": "20261126_143052_123456",
  "model_exported_at": "2026-11-26T14:30:52.123456+09:00"
}
```

- 내보내는 PC의 현지 시간: 연월일_시분초_마이크로초. ISO 시각에는 UTC 오프셋도 남깁니다.
- 분류·탐지·분할·이상검출의 배포 JSON에 기록합니다. PatchCore, Dinomaly, SAM2의 별도 내보내기에도 적용합니다.
- FP16·INT8로 새 배포 모델을 만들면 새 버전이 부여됩니다. 지원되는 정밀도 범위는 기존과 같습니다.
- 암호화는 버전을 그대로 보존합니다. `.dvsenc` 내부 JSON에 들어가며 평문 JSON을 추가 배포할 필요는 없습니다.
- 일반 ONNX는 동반 JSON에서 읽습니다. ONNX 계산 그래프나 가중치에 버전 정보를 삽입하지 않습니다.
- 내보내기 완료 로그에도 버전과 시각이 표시됩니다. 추론 중에는 파일을 다시 읽지 않습니다.
- 기존 파일에 소급해서 버전을 채우지 않습니다. 구형 모델은 계속 열리며 C++ 접근자는 빈 문자열을 반환합니다.

## C++ eVision에서 읽기

`example/cpp/with_evision/classifier.h`를 갱신하면 일반 JSON과 암호화 모델에서 같은 함수를 사용합니다.

```cpp
dvs_bw8::Classifier model(modelPath, key); // .dvsenc, 기존 초기화 방식
std::string version = model.ModelVersion(); // 복호화/초기화 재실행 없음
```

세션을 만들기 전에 읽으려면 최신 `example/cpp/model_crypto.h` 또는 `cpp/include/model_crypto.h`를 사용합니다.
아래 코드는 이미 읽은 `key`가 있는 초기화 함수 안에 놓습니다.

```cpp
dvs_crypto::Package package(modelPath, key); // 메모리에서 한 번 복호화
std::string version = package.ModelVersion();
auto shadow = std::make_shared<dvs_bw8::Classifier>(package); // 같은 메모리로 초기화·워밍업
```

`package.Config().at("model_exported_at").get<std::string>()`으로 시간대가 포함된 시각도 읽을 수 있습니다.
구형 패키지는 이 필드가 없으므로 `value("model_exported_at", std::string())`을 사용합니다.
키를 지우는 기존 예외 안전 처리와 워밍업 성공 후 모델을 교체하는 절차는 유지합니다.
모델 교체 시 예전 `Context`는 새 모델에 넘기지 않습니다.

## 버전 비교의 범위

이 값은 배포 시각이며 내용 해시가 아닙니다. 같은 가중치라도 다시 내보내면 버전이 바뀌고,
파일명을 바꾸거나 기존 산출물을 암호화만 다시 하면 버전이 유지됩니다.
PC 시계가 틀리면 기록도 틀리므로 자동 시간 설정을 사용하세요.
여러 PC 사이의 전역 고유성이나 시계 역행 시 증가 순서는 보장하지 않습니다.
새 버전인지 비교할 때 빈 문자열 두 개를 같은 모델의 증거로 사용하면 안 됩니다.

## 검증 범위

타임스탬프/시간대 일치, 전체 태스크 JSON, FP16·INT8 변환 시 갱신,
암호화·재암호화 시 버전/그래프/추론 출력 보존과 기존 내보내기 회귀 테스트를 포함합니다.
Windows VS2017에서의 실기 빌드는 별도이며 변경된 C++ 접근자는 C++17 표준 문자열 읽기만 사용합니다.
macOS C++17에서 일반·암호화·구형 모델 로드와 동일 추론 출력을 확인했습니다.
초기화 후 파일을 삭제하고도 메모리에 저장된 버전을 읽고 추론할 수 있음을 확인했습니다.
