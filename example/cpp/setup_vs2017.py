"""Create VS2017-compatible ONNX headers without CMake or PowerShell."""
from pathlib import Path
import shutil
import sys


DECLARATIONS = (
    "constexpr explicit Float16_t(uint16_t v)",
    "constexpr static Float16_t FromBits(uint16_t v)",
    "constexpr explicit BFloat16_t(uint16_t v)",
    "static constexpr BFloat16_t FromBits(uint16_t v)",
)


def prepare_headers(sdk_path):
    source = Path(str(sdk_path).strip().strip('"')).expanduser().resolve()
    if not (source / "onnxruntime_cxx_api.h").is_file():
        source = source / "include"
    header = source / "onnxruntime_cxx_api.h"
    if not header.is_file():
        raise ValueError("ONNX Runtime SDK 폴더 또는 include 폴더를 지정하세요.")
    # Validate everything before writing; preserve original SDK bytes.
    original = header.read_bytes()
    patched = original
    for declaration in DECLARATIONS:
        token = declaration.encode("ascii")
        if patched.count(token) != 1:
            raise ValueError("지원하지 않는 헤더 형식입니다. 원본 ONNX Runtime SDK를 지정하세요.")
        patched = patched.replace(token, token.replace(b"constexpr ", b""))
    destination = source.parent / "include-vs2017"
    if destination.is_symlink() or destination.resolve() == source:
        raise ValueError("출력 폴더가 원본을 가리킵니다. include-vs2017 폴더를 확인하세요.")
    # Never follow stale output symlinks back into the SDK or elsewhere.
    if destination.exists() and any(p.is_symlink() for p in destination.rglob("*")):
        raise ValueError("출력 폴더에 바로가기가 있습니다. 별도의 SDK 폴더에서 실행하세요.")
    shutil.copytree(source, destination, dirs_exist_ok=True)
    (destination / header.name).write_bytes(patched)
    return destination


def main():
    try:
        sdk = sys.argv[1] if len(sys.argv) > 1 else input(
            "ONNX Runtime SDK 폴더 경로를 붙여넣고 Enter를 누르세요:\n> ")
        if not sdk.strip():
            raise ValueError("폴더 경로를 입력하지 않았습니다.")
        output = prepare_headers(sdk)
        print("\n완료. VS의 [C/C++ > 일반 > 추가 포함 디렉터리]에 아래 경로를 넣으세요:")
        print(output)
        print("기존 ONNX include 경로보다 앞에 넣고 솔루션을 다시 빌드하세요.")
        print("VS2017 15.9 / x64 / C++17 / /Zc:noexceptTypes 설정을 사용하세요.")
        print("onnxruntime.lib와 DLL은 기존 SDK 파일을 사용합니다.")
        return 0
    except (OSError, ValueError, EOFError) as error:
        print("실패: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
