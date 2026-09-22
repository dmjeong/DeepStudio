"""Stage the small Windows installer layout used for application delivery.

This is deliberately separate from the full offline release payload.  The
result contains only the frozen desktop application and the two public ONNX
SDK examples.  The frozen ``app`` directory is opaque because it contains the
runtime and built-in assets required to launch the application.  Outside that
directory, the payload cannot accidentally include a repository checkout,
training data, Docker/WSL assets, or build output.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy


_stage_api = runpy.run_path(str(Path(__file__).with_name("stage_payload.py")))
stage_payload = _stage_api["stage_payload"]
PayloadStageError = _stage_api["PayloadStageError"]


_TOP_LEVEL = frozenset({"app", "Examples", "THIRD_PARTY_NOTICES.md"})
_REQUIRED_FILES = frozenset({
    "app/DeepVisionStudio.exe",
    "Examples/README.md",
    "Examples/assets/test.onnx",
    "Examples/cpp/CMakeLists.txt",
    "Examples/cpp/main.cpp",
    "Examples/cpp/classifier.h",
    "Examples/cpp/example_paths.h",
    "Examples/cpp/self_test.cpp",
    "Examples/cpp/setup_vs2017.bat",
    "Examples/cpp/setup_vs2017.py",
    "Examples/cpp/with_opencv/CMakeLists.txt",
    "Examples/cpp/with_evision/classifier.h",
    "Examples/cpp/with_evision/bw8_preprocess.h",
    "Examples/cpp/with_evision/CMakeLists.txt",
    "Examples/cpp/vision-runtime/CMakeLists.txt",
    "Examples/cpp/vision-runtime/include/vision_inference.h",
    "Examples/cpp/vision-runtime/include/ort_vs2017.cmake",
    "Examples/cpp/vision-runtime/src/vision_inference.cpp",
    "Examples/csharp/OnnxExample.csproj",
    "Examples/csharp/Program.cs",
    "Examples/csharp/vision-runtime/VisionRuntime.csproj",
    "Examples/csharp/vision-runtime/VisionRuntime.cs",
    "THIRD_PARTY_NOTICES.md",
})


def validate_simple_payload(root: str | Path) -> Path:
    """Reject any installer layout other than app + C++/C# examples."""
    payload = Path(root).expanduser()
    if payload.is_symlink():
        raise PayloadStageError("simple payload root symlink is not allowed")
    payload = payload.resolve()
    if not payload.is_dir():
        raise PayloadStageError(f"simple payload root does not exist: {payload}")
    actual = {entry.name for entry in payload.iterdir()}
    unexpected = actual - _TOP_LEVEL
    missing = _TOP_LEVEL - actual
    if missing or unexpected:
        raise PayloadStageError(
            "simple payload top level must contain only app, Examples, and "
            f"THIRD_PARTY_NOTICES.md (missing={sorted(missing)}, unexpected={sorted(unexpected)})"
        )
    missing_files = sorted(path for path in _REQUIRED_FILES if not (payload / path).is_file())
    if missing_files:
        raise PayloadStageError("simple payload is missing required files: " + ", ".join(missing_files))
    return payload


def stage_simple_payload(
    output: str | Path,
    *,
    app: str | Path,
    example_root: str | Path,
    cpp_runtime_root: str | Path,
    csharp_runtime_root: str | Path,
    notice: str | Path,
) -> Path:
    """Copy a reviewed allow-list into a payload suitable for the simple Setup."""
    examples = Path(example_root)
    cpp = Path(cpp_runtime_root)
    csharp = Path(csharp_runtime_root)
    root = stage_payload(
        output,
        [
            ("app", app),
            ("Examples/README.md", examples / "README.md"),
            ("Examples/assets", examples / "assets"),
            ("Examples/cpp/CMakeLists.txt", examples / "cpp" / "CMakeLists.txt"),
            ("Examples/cpp/main.cpp", examples / "cpp" / "main.cpp"),
            ("Examples/cpp/classifier.h", examples / "cpp" / "classifier.h"),
            ("Examples/cpp/example_paths.h", examples / "cpp" / "example_paths.h"),
            ("Examples/cpp/self_test.cpp", examples / "cpp" / "self_test.cpp"),
            ("Examples/cpp/setup_vs2017.bat", examples / "cpp" / "setup_vs2017.bat"),
            ("Examples/cpp/setup_vs2017.py", examples / "cpp" / "setup_vs2017.py"),
            ("Examples/cpp/with_opencv", examples / "cpp" / "with_opencv"),
            ("Examples/cpp/with_evision", examples / "cpp" / "with_evision"),
            ("Examples/cpp/vision-runtime/CMakeLists.txt", cpp / "CMakeLists.txt"),
            ("Examples/cpp/vision-runtime/include", cpp / "include"),
            ("Examples/cpp/vision-runtime/src", cpp / "src"),
            ("Examples/csharp/OnnxExample.csproj", examples / "csharp" / "OnnxExample.csproj"),
            ("Examples/csharp/Program.cs", examples / "csharp" / "Program.cs"),
            ("Examples/csharp/vision-runtime/VisionRuntime.csproj", csharp / "VisionRuntime.csproj"),
            ("Examples/csharp/vision-runtime/VisionRuntime.cs", csharp / "VisionRuntime.cs"),
        ],
        notice=notice,
    )
    return validate_simple_payload(root)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Stage the minimal Windows application installer")
    parser.add_argument("--validate", type=Path,
                        help="validate an already staged simple payload")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--app", type=Path)
    parser.add_argument("--example-root", type=Path)
    parser.add_argument("--cpp-runtime-root", type=Path)
    parser.add_argument("--csharp-runtime-root", type=Path)
    parser.add_argument("--notice", type=Path)
    args = parser.parse_args(argv)
    if args.validate:
        if any(value is not None for value in (
            args.output, args.app, args.example_root, args.cpp_runtime_root,
            args.csharp_runtime_root, args.notice,
        )):
            parser.error("--validate cannot be combined with staging arguments")
        print(validate_simple_payload(args.validate))
        return 0
    missing = [name for name in (
        "output", "app", "example_root", "cpp_runtime_root", "csharp_runtime_root", "notice",
    ) if getattr(args, name) is None]
    if missing:
        parser.error("staging requires " + ", ".join("--" + name.replace("_", "-") for name in missing))
    print(stage_simple_payload(
        args.output, app=args.app, example_root=args.example_root,
        cpp_runtime_root=args.cpp_runtime_root, csharp_runtime_root=args.csharp_runtime_root,
        notice=args.notice,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
