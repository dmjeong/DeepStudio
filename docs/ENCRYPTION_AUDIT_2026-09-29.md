# 암호화 내보내기 재검토 — 2026-09-29

기존 8.19 구현을 다시 검토해 실제 결함을 발견했다. 이전의 정상 export·추론 검사만으로
완료를 판단한 것은 부족했다. 아래 수정과 회귀 검사를 8.20에 포함했다.

## 발견한 문제와 수정

| 문제 | 수정 전 재현 | 수정 및 회귀 검증 |
|---|---|---|
| 원본 외부 가중치 덮어쓰기 | ONNX가 참조하는 `weights.dvsenc`를 암호화 출력으로 선택하면 그 가중치 파일을 덮어씀. 원본 모델의 실제 ORT 결과도 변경됨 | 모든 외부 tensor 의존 파일과 출력 경로의 충돌을 검사. 실패 후 원본 바이트 보존 확인 |
| 외부 sparse tensor 누락 | sparse initializer의 외부 데이터를 넣지 않고 암호화 성공으로 보고. 원본은 실행되지만 암호화 모델의 메모리 로드는 실패 | protobuf를 재귀 순회해 dense/sparse, Constant, 중첩 그래프, 로컬 함수까지 포함. 외부 파일을 삭제한 뒤 메모리 ORT 추론으로 4가지 위치 검사 |
| 오류 보고서가 키/체크포인트를 삭제·손상 | 키 파일 이름이 `model.export-error.json`이면 성공 시 키 삭제, 실패 시 진단 JSON으로 덮어씀 | 출력과 원본 경로 충돌 차단. 진단 경로가 원본/키/배포 파일과 겹치면 파일을 쓰거나 지우지 않음. 직접 경로·심볼릭 링크·하드 링크의 성공/실패 18개 검사 |
| Windows 하위 폴더 그래프 이름 오류 | 패키지의 `graphs/model.onnx`를 Windows 경로 정규화하면 `graphs\\model.onnx`가 되어 패키지 이름 검사에서 거부 | 패키지 내부 이름은 POSIX 문자열 그대로 사용. classification/SAM2 중첩 경로 fixture 추가. 로컬 실행은 macOS이며 Windows 실행은 CI에 연결 |
| Python/C++ 패키지 검사 불일치 | Python이 거부한 중복 JSON 키, 잘못된 contracts/graphs 타입을 C++가 허용. Python은 UTF-16/32와 비유한 숫자를 허용 | C++ 중복/타입 검사 추가. Python은 UTF-8·유한 숫자만 허용. 정상 13개와 비정상 17개 패키지를 두 로더에서 검사 |
| SDK 변경 후 이전 빌드 캐시 사용 | `ONNXRUNTIME_ROOT`를 바꿔도 include/lib가 이전 SDK로 남음. Windows에서 다른 버전의 DLL을 복사할 위험 | 명시한 SDK에서 헤더/라이브러리를 다시 찾고 DLL 탐색 캐시도 갱신. 실제 SDK 폴더를 바꿔 eVision/OpenCV CMake cache 경로 확인. 배치 빌드 폴더를 소스·SDK·환경별로 분리 |
| `with_opencv` 빌드 실패 | 상위 CMake 파일을 include할 때 새 `encrypted.cpp`를 `with_opencv`에서 찾음. 상위 `cpp` 진입 검사만으로는 발견 안 됨 | 소스 파일을 CMake 파일 기준 절대 경로로 지정. 설치본만 사용해 `with_opencv` configure/build와 3개 실행 검사 통과 |
| 웹 암호화 선택 초기화 | 암호화를 켜고 키/출력을 입력한 뒤 다른 탭을 갔다 오면 암호화가 풀리고 `.onnx`로 돌아감 | 선택·키 경로·출력을 프로젝트별 sessionStorage draft로 유지. 탭 이동·새로고침·명시적 해제·프로젝트 변경/복귀를 실제 Chrome에서 확인. 키 바이트는 저장하지 않음 |

추가로 C++ 예제에서 생성자가 예외를 던져도 임시 키를 지우도록 정리했다. 외부
가중치의 offset/length와 총 크기를 로드 전에 검사한다. 설치 목록에 암호화 빌드
보조 파일과 eVision 예제 소스가 빠지면 설치본 검증이 실패하게 했다.

외부 가중치는 검사한 범위만 직접 읽도록 바꿨다. 지원 범위의
[ONNX 1.16 소스](https://github.com/onnx/onnx/blob/v1.16.0/onnx/external_data_helper.py#L30)는
`length=0`을 파일 끝까지 읽는 것으로 처리하므로, 설치된 ONNX 버전의 읽기 함수에
의존하지 않는다. 0바이트 tensor와 사전 검사 후 잘린 파일에 대한 회귀 검사도 추가했다.

## 검증 결과

환경: macOS Apple M4, Python 3.11, ORT 1.29.0, C++17 Apple Clang, .NET 8.

- Python 통합 회귀 **178개 통과**(종료 코드 0). 모델 export, 암호화, 기존 ONNX fallback,
  Qt/Web worker 연결, 원본 보존, 설치 구성, 버전 정책을 포함한다.
- Native SDK **5/5 CTest 통과**. 13개 정상 평문/암호화 쌍과 17개 비정상 파일을 검사한다.
- Debug에 UndefinedBehaviorSanitizer를 적용하고 오류 즉시 중단 옵션으로 **5/5 CTest 통과**.
- C++ OpenCV 예제 **3/3**, BW8 Release **3/3**, BW8 Debug **3/3** 통과.
- 설치본 `Examples/cpp/with_opencv`를 별도 폴더에서 빌드해 **3/3** 통과.
  frozen GUI는 기존 시험용 파일로 대체한 설치 목록 검사이며, Windows 설치 EXE 실행 검사는 아니다.
- C# **13개 fixture × 3회 초기화/해제 × 5개 입력**. 분류·검출·분할·이상검출·SAM2의
  노출 결과를 시간 필드만 제외하고 비트 단위로 비교. SAM2 point/box/mask/automatic,
  다른 세션의 context 거부, 해제 후 호출 거부, 잘못된 키/파일 뒤 재시도도 통과.
- 웹 Node 테스트 **14개**, TypeScript/Vite 빌드 통과. Chrome의 실제 화면에서 위
  암호화 설정 유지 동작을 검증했다. 새 Playwright 회귀는 기존 전체 workflow에 연결했지만
  로컬에는 Playwright 패키지가 없어 전체 workflow 실행은 하지 않았다.

검증은 저장소 테스트로 재실행할 수 있다. Python:

```bat
python -m pytest -q tests/test_model_crypto.py tests/test_encrypted_export_models.py tests/test_encrypted_export_safety.py tests/test_export_project_binding.py tests/test_export_contracts.py tests/test_webapp.py tests/test_windows_packaging_contracts.py tests/test_version_policy.py tests/test_efficientnet_deployment.py tests/test_export_fallback.py
```

C#은 native CTest로 fixture를 만든 후 실행한다. Windows에서는 같은 빌드의
`vision_runtime.dll`과 종속 DLL을 실행 경로에 두거나 `DEEP_VISION_NATIVE_RUNTIME_DIR`로 지정한다.

```bat
dotnet run --project tests/csharp/EncryptedContracts/EncryptedContracts.csproj -c Release -- build/test-models
```

## 보장하지 않는 범위

모든 태스크의 입출력 계약은 검사했지만 모든 사전학습 가중치나 사용자 비공개
체크포인트를 전수 검사한 것은 아니다. 실제 모델과 합성 계약 fixture의 구분은
[초기 검증 기록](ENCRYPTED_ONNX.md)에 적었다.

Windows BCrypt/v141 Debug·Release, 실제 eVision SDK, 대상 i7에서의 실행·성능은
이 Mac에서 검증하지 못했다. Windows CI에 관련 빌드와 C#/native 검사를 연결했으며,
아직 실행 완료로 간주하지 않는다. 따라서 버그 0개 또는 Windows 최종 검증 완료로 표현하지 않는다.

이번 수정은 내보내기·초기화·화면 상태·빌드 경로에 한정했다. 반복 추론 함수와
정밀도·최적화·스레드 설정은 바꾸지 않았다. 기존 암호화 전후 성능 측정은
[성능 기록](ENCRYPTED_ONNX.md#추론-성능-비교)을 참조한다. 대상 PC 속도 보장은 별도 실측이 필요하다.

추가 AddressSanitizer 검사는 이 Mac의 sanitizer 초기화에서 main 진입 전에 멈췄다.
빈 `int main(){return 0;}` 프로그램에서도 같은 현상을 재현했다. 스택은
`AsanInitInternal → InitializeShadowMemory → get_dyld_hdr → malloc → AsanInitFromRtl`
초기화 재진입 대기였다. 이 검사는 통과로 계산하지 않는다.
