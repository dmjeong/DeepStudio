# 암호화 모델 내보내기와 C++ / C# 사용

학습툴의 **ONNX 내보내기 → 모델·설정 암호화**를 켜고 **새 키 생성**으로 키를
만든 다음 내보낸다. 결과는 `.dvsenc` 한 파일이다. ONNX와 전처리·클래스명·실행
설정 JSON이 모두 그 안에 들어간다. 키는 별도 보관한다. 재학습할 필요는 없다.
웹 화면에서도 새 키를 저장할 `.key` 절대 경로를 입력하고 **새 키 생성**을 누르면 된다.

분류, 객체 검출, 분할, 이상 검출의 공통 export 결과를 암호화한다. SAM2는
encoder와 decoder, 설정을 한 파일에 넣는다. 기존 출력 검증은 그대로 실행한다.
이미 내보낸 JSON/ONNX는 저장소의 `tools/model_encrypt.py`를 실행해 암호화할 수도
있다. 인자 없이 실행하면 파일 선택 창이 열린다. 기존 평문 파일을 자동 삭제하지 않는다.

## 필요한 파일

| 실행 방식 | 모델 파일 | 코드와 실행 환경 |
|---|---|---|
| C++ eVision/BW8 분류 | 내 모델 `.dvsenc` + 별도 키 | `classifier.h`, `bw8_preprocess.h`, 상위 폴더의 `model_crypto.h`, `nlohmann` 폴더, ONNX Runtime |
| C++ OpenCV 전 태스크 | 동일 | `cpp/encrypted.cpp`, 함께 제공된 native runtime 소스, ONNX Runtime, OpenCV |
| C# 전 태스크 | 동일 | `csharp/encrypted`, C# SDK, 이번 버전의 `vision_runtime.dll`과 ONNX Runtime/OpenCV DLL |

Windows 암호화 라이브러리는 OS에 포함된 BCrypt다. 추가 암호화 DLL이나 OpenSSL 설치가
필요 없다. C++ 링크에는 `bcrypt.lib`가 추가된다(CMake와 헤더에서 자동 지정).
VS2017은 15.9 / v141 14.16 / x64 / C++17을 사용한다. 기존
`cpp/setup_vs2017.bat`의 ONNX 헤더 호환 처리도 그대로 사용한다.
eVision SDK는 실제 EROIBW8을 사용하는 기존 프로젝트의 것을 사용한다.

**`assets/example-only.key`는 공개된 예제 전용 키다. 실제 모델에 쓰면 안 된다.**
키를 모델과 함께 누구에게나 배포하면 모델도 복호화할 수 있다. 아래 키 파일 읽기는
호출 방법을 보여주는 예제이며 제품에서는 별도 키 보관/전달 방식을 정해야 한다.
키를 잃으면 암호화 모델을 열 수 없으므로 안전한 별도 위치에 보관한다.

## C++ eVision: 초기화 한 번, EROIBW8 반복 추론

프로그램 시작 때 클래스 멤버를 만들고, 이후 검사에서는 `InferEvision()`만 호출한다.
`classifier`를 지역 변수로 매번 다시 만들지 않는다.

```cpp
#include "classifier.h"
#include <memory>

class Vision {
public:
    void Initialize(const std::filesystem::path& model,
                    const std::filesystem::path& key_file) {
        auto key = dvs_crypto::ReadKey(key_file);
        try {
            classifier = std::make_unique<dvs_bw8::Classifier>(model, key);
        } catch (...) {
            dvs_crypto::Wipe(key.data(), key.size());
            throw;
        }
        dvs_crypto::Wipe(key.data(), key.size());
        // 여기까지 복호화, ONNX 로드, 준비 추론이 모두 끝났다.
        // 이제 검사 버튼을 활성화한다.
    }

    auto Inspect(EROIBW8& roi) {
        if (!classifier) throw std::runtime_error("Initialize first");
        return classifier->InferEvision(roi);
    }
private:
    std::unique_ptr<dvs_bw8::Classifier> classifier;
};
```

실행 파일 옆에 모델을 둘 때는 `example_paths.h`의 `ExecutableDirectory()`와
합친 경로를 전달한다. 작업 폴더가 달라지는 Visual Studio F5 실행에서도 같다.
BW8 포인터는 기존처럼 `InferBW8(ptr, width, height, rowPitch)`로 넣는다.
기본 `InferEvision(roi)` 호출이 비어 있는 작업 공간을 자동으로 빌려 동시에 추론한다.
모두 사용 중이면 공간을 더 만들고, 완료된 공간은 다음 호출에서 재사용한다.
Context 배열이나 스레드 번호·개수 지정은 필요 없다. 모델과 복호화는 한 번만 수행한다.
작업 공간은 메모리이며 프로그램의 스레드를 생성하거나 점유하지 않는다.
인자 없는 OMP 예제는 `cpp/with_evision/parallel.cpp`이며 자세한 연결 방법은
[eVision 예제의 병렬 추론 안내](cpp/with_evision/README.md#openmp로-동시에-추론하기)를 따른다.

인자 없는 전체 예제는 `cpp/with_evision/encrypted.cpp`다. 기존 CMake 빌드에
`evision_encrypted_example`이 함께 추가된다. `cpp/build_encrypted.bat`을 실행하면
SDK 경로를 입력해 빌드·테스트할 수 있다. eVision SDK 없이도 합성 BW8 입력으로 실행한다.

## C++ OpenCV: 전 태스크

`cpp/encrypted.cpp`는 보호된 설정에서 태스크를 읽어 분류/검출/분할/이상 검출/SAM2로
분기한다. `dvs_crypto::Package`로 메모리에서 풀고 `InitializeFromPackage()`를 호출한다.
SAM2는 같은 패키지로 두 그래프를 초기화한다. 패키지 객체가 범위를 벗어나면 임시
평문 버퍼를 지우고, ONNX Runtime 세션은 계속 유지한다.

분류만 사용하면 `example::Classifier(package_path, key)` 생성자도 제공한다.
생성자가 준비 추론까지 수행한다. 전 태스크 예제는 해당 태스크의 준비 추론을 실행한
뒤 검사 반복문에 들어간다. 경로는 소스에서 수정하며 실행 인자가 없다.

이미 CMake를 설정한 빌드 폴더에서는 일반 명령 프롬프트에서 실행한다.

```bat
cmake --build example/build --config Release --target onnx_encrypted_example
example\build\Release\onnx_encrypted_example.exe
```

## C#: 로드 한 번, 세션 재사용

```csharp
byte[] key = File.ReadAllBytes(keyFile);
VisionSession model;
try { model = VisionSession.OpenEncrypted(modelFile, key); }
finally { System.Security.Cryptography.CryptographicOperations.ZeroMemory(key); }
// model을 창/검사 객체의 멤버로 유지한다. 처음 한 번 준비 추론한 뒤 사용한다.
var result = model.InferClassification(pixels, width, height, channels);
// 종료할 때 model.Dispose();
```

`csharp/encrypted/Program.cs`에 전 태스크 호출 예제가 있다. `task` 상수를 모델의
태스크로 바꾸고 이미지 배열과 크기를 지정한다. SAM2의 `EncodeSam`/`SegmentSam`/
`AutomaticSam`도 같은 세션을 사용한다. .NET 8 SDK에서 빌드하며 구형 native DLL과
섞지 않는다. Windows 실행 폴더에 이번에 빌드한 `vision_runtime.dll` 및 종속 DLL을 둔다.

```bat
rem C++ 예제를 설정한 폴더에서 C#용 native DLL도 빌드한다.
cmake --build example/build --config Release --target vision_runtime
dotnet build example/csharp/encrypted/EncryptedExample.csproj -c Release -o example/build-encrypted-csharp
rem vision_runtime.dll 및 ONNX Runtime/OpenCV DLL을 위 출력 폴더에 복사한 뒤 실행한다.
example\build-encrypted-csharp\EncryptedExample.exe
```

## 속도와 보호 범위

복호화·인증·JSON 읽기는 초기화 때 한 번만 실행된다. 이후 전처리, ONNX 실행,
후처리는 기존 함수 그대로다. 정밀도·최적화 수준·스레드 수를 변경하지 않는다.
초기화 시간이 복호화만큼 늘 수 있으므로 추론 시간과 별도로 측정해야 한다.
모든 PC에서 측정 시간이 단 1 ns도 늘지 않는다고 보장할 수는 없다.

배포 로더는 평문 ONNX/JSON을 디스크에 쓰지 않는다. 파일을 Netron으로 바로 열거나
키 없이 재사용하는 것을 막지만, 실행 중 메모리 분석까지 차단하지는 못한다.
학습 PC의 exporter는 임시 ONNX를 만들고 암호화 후 삭제한다. 이는 보안 삭제가 아니다.
v1은 전체 파일 2 GiB 미만, ONNX Runtime 실행용이다. OpenVINO의 별도 IR 변환이나
프로파일링은 암호화 로더에서 허용하지 않는다.

검증 범위와 결과는 저장소의 `docs/ENCRYPTED_ONNX.md`에 기록한다.
