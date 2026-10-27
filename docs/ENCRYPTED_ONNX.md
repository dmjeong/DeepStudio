# 암호화 ONNX 내보내기

## 개발 목표와 검증 기준

이 문서는 이번 요청의 범위를 고정한다. 구현, 검증, 최종 검토 때 아래 목록을 다시 확인한다.

- EfficientNet에 제한하지 않는다. classification, object detection, segmentation,
  anomaly detection의 공통 export 결과를 감싸며 SAM2 encoder/decoder도 한 파일로 묶는다.
- ONNX와 JSON을 함께 AES-256-GCM으로 암호화한다. 키는 별도 32바이트 파일 또는
  호출자가 제공한 메모리에서 받는다. 저장소/로그/패키지에 실제 키를 넣지 않는다.
- 모델의 정밀도, 최적화 수준, 스레드 수, 입출력명, 전후처리를 변경하지 않는다.
- C++17/VS2017용 Windows CNG(BCrypt), 비 Windows OpenSSL을 사용한다.
- 복호화/인증/파싱은 초기화에만 있다. 반복 추론은 기존 코드 그대로 사용한다.
- C++ OpenCV/eVision, C# 예제는 인자 없이 실행하고 처음에 로드/워밍업한다.
- 잘못된 키, 변조, 잘림, 잘못된 크기/경로, 실패 시 기존 산출물 보존을 검사한다.
- 전 태스크의 원본/암호화 출력 일치, 다중 그래프, 외부 가중치, 기존 export 회귀를 검사한다.
- 성능은 초기화와 반복 추론을 분리한다. 초기 복호화 비용은 존재한다. 동일한 실행
  경로를 유지하며 반복 추론을 번갈아 측정한다. 대상 Windows/i7 미실측을 숨기지 않는다.

## 포맷 v1

확장자 `.dvsenc`. `DVSENC01` (8 bytes, AAD) + 무작위 nonce (12 bytes) +
AES-256-GCM ciphertext + tag (16 bytes). 매 암호화마다 새 nonce를 생성한다.
인증이 성공하기 전에는 내부 JSON/모델을 파싱하지 않는다.

평문 payload는 little-endian uint32 JSON 길이, UTF-8 JSON, 순서대로 연결한 ONNX
바이트이다. JSON에는 `format`, 원래의 `config`, `models: [{name, size}]`가 있다.
모델 이름은 상대 POSIX 경로만 허용하고 중복/상위 경로/절대 경로를 거부한다.
외부 tensor 파일이 있는 그래프는 tensor를 내부에 포함시켜 함께 보호한다. 연산과 값은
변경하지 않는다. v1은 전체 파일 2 GiB 미만이며 넘으면 명시적으로 거부한다.

키 파일은 정확히 32바이트 난수이다. 암호화 모델과 키를 함께 공개하면 보호가 사라진다.
이 기능은 라이선스/PC 인증 기능이 아니며 실행 중 메모리 분석을 완전히 막지 못한다.
학습 PC에서는 기존 exporter가 임시 그래프를 생성한 후 암호화하고 정리한다. 배포
로더는 복호화한 파일을 디스크에 쓰지 않는다. 임시파일 삭제는 디스크 보안 삭제가 아니다.

## 구현 결과

- GUI와 웹 내보내기에서 암호화와 새 키 생성을 제공한다. 프로젝트를 바꾸면 이전
  키 선택을 해제한다. 키 생성은 기존 파일을 덮어쓰지 않고 웹 응답에도 키 바이트를 반환하지 않는다.
- `python/model_crypto.py`가 기존 export 결과를 포장한다. EfficientNet, built-in,
  LibreYOLO, Re-DETR, PatchCore 및 공식 SAM2 export가 같은 경로를 사용한다.
- C++ `model_crypto.h` → 기존 `VisionInference` / `Sam2Inference` 메모리 초기화,
  C ABI `dv_create_session_encrypted`, C# `VisionSession.OpenEncrypted`로 연결했다.
- eVision/BW8와 OpenCV 예제는 프로그램 시작 시 한 번 로드·준비 추론하고, 기존
  추론 함수를 반복 호출한다. 키/파일 읽기/복호화가 추론 반복문 안에 없다.
- 예제, 필수 헤더, 공개 합성 테스트용 키와 모델, 사용 설명서를 설치파일의
  `Examples` 목록에 포함했다. 사용법: [ENCRYPTED_MODELS.md](../example/ENCRYPTED_MODELS.md).
- 실제 `.key` / `.dvsenc`는 Git에서 기본 제외한다. 공개 합성 예제 두 파일만 예외다.

## 2026-09-28 로컬 검증

아래는 최초 구현 시점의 기록이다. 이후 별도 재검토에서 외부 가중치·진단 파일·
Windows 경로·예제 빌드·웹 설정 유지 문제를 발견해 수정했다.
최신 결과와 미검증 범위는 [2026-09-29 재검토 보고서](ENCRYPTION_AUDIT_2026-09-29.md)를 참고한다.

macOS Apple M4, Python 3.11, ONNX Runtime 1.29.0, C++17 Clang, .NET 8 환경이다.

| 범위 | 결과 및 한계 |
|---|---|
| Python 통합 회귀 | 135개 통과, 종료 코드 0. export, 암호화, Qt/Web 연결, 설치 목록, 버전 정책 포함 |
| 실제 모델 구조 export | EfficientNet B0/B1 × 1/3채널, ResNet18/50, ConvNeXt Tiny, DeepLab V3+, U-Net. 사전학습 다운로드 없이 임의 가중치 사용. 원본/복원 그래프 바이트와 설정 동일, batch 1/2의 ORT 출력 동일 |
| 업스트림/다중 그래프 계약 | LibreYOLO 분류/YOLO9/Re-DETR, Re-DETR Small/Medium/Large, SAM2 네 변형의 encoder/decoder, PatchCore 두 백본 ID. 작은 합성 모듈로 실제 exporter와 ORT를 실행. 해당 대형 사전학습 가중치 자체의 전수 실험은 아님 |
| Native SDK | 5개 CTest: 모든 태스크의 평문/암호화 C ABI 출력, SAM2 point/box/mask, 잘못된 키·변조·잘림·크기 오류 등 |
| C++ 예제 | OpenCV 3개, BW8 Release 3개·Debug 3개, 설치본 폴더만 사용한 OpenCV 3개 CTest 통과 |
| C# 예제 | 저장소·설치본의 암호화 예제 빌드 및 반복 분류 실행 |
| 웹 | TypeScript/배포 빌드와 기존 Node 테스트 14개 통과 |
| Windows | BCrypt/v141 Release·Debug 및 전 태스크 native 계약 테스트를 CI에 연결. 이 로컬 환경에서 Windows 실행은 수행하지 않음 |

최초 19개 export 검사의 단독 실행은 모든 assertion 후 프로세스 종료에서 한 번
중단됐다. macOS crash report의 중단 스택은 ORT의
`Microsoft::Applications::Events::HttpResponseDecoder` → `DebugEventSource::DispatchEvent`
통계 전송 스레드였다. 암호화/추론 함수 스택이 아니었다. 이후 모델 계약 테스트는
`onnxruntime.disable_telemetry_events()`로 외부 전송을 비활성화하고 정상 종료를
확인했다. ORT 자체의 종료 경쟁 문제를 수정했다고 주장하지 않는다.

## 추론 성능 비교

C++ Release, B0 1×1×224×224, FP32, ORT 최적화 `all`, 같은 합성 입력으로 측정했다.
가중치는 임의 가중치이며 실제 검사 정확도 평가가 아니다. 30회 준비 후 400회씩,
평문/암호화 세션 순서를 번갈아 실행했다. 파일 읽기는 반복 측정에 포함하지 않고
전처리+모델+후처리를 측정했다. 모든 반복에서 확률 배열까지 완전히 같았다.

| 스레드 | 평문 평균 | 암호화 평균 | 평문 p50 / p95 | 암호화 p50 / p95 |
|---|---:|---:|---:|---:|
| 1 | 12.60694 ms | 12.60930 ms | 12.54275 / 12.95446 ms | 12.55592 / 12.95200 ms |
| 4 | 6.57847 ms | 6.58964 ms | 6.37979 / 8.17908 ms | 6.39046 / 8.25575 ms |

관측 평균 차이는 약 +0.02%, +0.17%다. 추론 함수에 추가 연산은 없지만 이 표만으로
모든 환경에서 정확히 0% 차이라고 단정할 수 없다. Windows/i7, 사용자 가중치,
EROIBW8 실장비 지연시간은 아직 측정하지 않았다. 초기 로드·복호화는 비용이 있으며
반복 추론 시간과 구분한다. 대형 모델은 암호화 중 원본·암호문·복원 검증 버퍼가
동시에 존재하므로 파일 크기의 여러 배에 해당하는 메모리가 필요할 수 있다.

`cpp/tests/encrypted_benchmark.cpp`는 실제 PC에서 같은 JSON/암호화 모델을 비교하는
도구다. 테스트 도구만 파일 경로 인자를 사용하고, 제공된 사용자용 예제는 인자가 없다.
암호화가 기존 export의 정밀도나 optimizer 설정을 바꾸지 않는지 함께 기록한다.

## 검증 실행

```bat
python -m pytest -q tests/test_model_crypto.py tests/test_encrypted_export_models.py tests/test_export_project_binding.py tests/test_export_contracts.py tests/test_webapp.py tests/test_windows_packaging_contracts.py tests/test_version_policy.py
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Windows/i7 수치는 위 벤치마크를 대상 PC에서 실행해야 채울 수 있다. 로컬 검증 결과를
Windows 인증, 모든 사용자 가중치의 정확도 검증, 완전한 역공학 방지로 표현하지 않는다.
