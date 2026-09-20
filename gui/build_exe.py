"""
Deep Vision Studio — EXE 빌드 스크립트

PyInstaller를 사용하여 독립 실행 파일(.exe) 생성

사용법:
    cd gui
    python build_exe.py

선택적 네이티브 SDK 포함:
    set VISION_NATIVE_RUNTIME_DIR=C:\\build\\vision-runtime
    python build_exe.py

빌드 결과:
    dist/DeepVisionStudio/DeepVisionStudio.exe

빌드 구조:
┌──────────────────────────────────────────────────────────┐
│  build_exe.py                                            │
│  └─► PyInstaller                                         │
│      ├── main.py (엔트리포인트)                           │
│      ├── app/ (메인 윈도우)                                │
│      ├── widgets/ (UI 위젯)                               │
│      ├── core/ (프로젝트, 학습 엔진, 시그널)               │
│      ├── resources/styles/ (다크 테마 QSS)                │
│      ├── resources/icons/  (셰브론 SVG)                   │
│      └── ../python/ (모델, 데이터셋)                       │
│                                                          │
│  결과: dist/DeepVisionStudio/                          │
│        ├── DeepVisionStudio.exe  ← 더블클릭 실행       │
│        ├── PySide6 런타임 DLL                              │
│        ├── PyTorch CUDA 런타임                              │
│        ├── model_sdk/schemas/ 모델 팩 계약                 │
│        ├── vision_runtime.dll 등 (환경변수 지정 시)         │
│        └── 기타 의존성                                     │
└──────────────────────────────────────────────────────────┘

필수 패키지:
    pip install pyinstaller
"""

import os
from pathlib import Path
import sys
import subprocess
import shutil
from importlib.util import find_spec

# 경로 설정
GUI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(GUI_DIR)
PYTHON_DIR = os.path.join(PROJECT_ROOT, "python")
MODEL_CATALOG_DIR = os.path.join(PROJECT_ROOT, "packaging", "windows", "models")
RESOURCES_DIR = os.path.join(GUI_DIR, "resources")
BUILTIN_ASSETS_DIR = os.path.join(GUI_DIR, "builtin_assets")
DIST_DIR = os.path.join(GUI_DIR, "dist")
BUILD_DIR = os.path.join(GUI_DIR, "build")

APP_NAME = "DeepVisionStudio"

# PyInstaller는 설치되지 않은 hidden import도 경고만 남기고 EXE 생성을 계속할
# 수 있다. 그 경우 EXE는 만들어져도 시작 시 ``Failed to execute main``으로
# 끝난다. 빌드 전에 실제 데스크톱 런타임을 모두 확인한다.
REQUIRED_RUNTIME_MODULES = {
    "PySide6": "PySide6",
    "matplotlib": "matplotlib",
    "Pillow": "PIL",
    "PyTorch": "torch",
    "torchvision": "torchvision",
    "NumPy": "numpy",
    "ONNX": "onnx",
    "ONNX Runtime": "onnxruntime",
    "psutil": "psutil",
    "OpenCV": "cv2",
    "cryptography": "cryptography",
    "SAM2": "sam2",
    "Hugging Face Hub": "huggingface_hub",
}


def _native_runtime_arguments() -> list[str]:
    """Return optional native SDK DLLs staged into the frozen application.

    The release job builds ``vision_runtime.dll`` and its ONNX Runtime/OpenCV
    dependencies before invoking this script.  Development builds may omit
    ``VISION_NATIVE_RUNTIME_DIR``; the GUI still builds, while the installer
    payload validator later requires the native runtime for SDK deployment.
    """
    value = os.environ.get("VISION_NATIVE_RUNTIME_DIR", "").strip()
    if not value:
        return []
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"VISION_NATIVE_RUNTIME_DIR is not a directory: {root}")
    suffixes = {".dll"} if sys.platform == "win32" else {".dylib", ".so"}
    files = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in suffixes)
    if not files:
        raise RuntimeError(f"VISION_NATIVE_RUNTIME_DIR contains no native runtime libraries: {root}")
    arguments: list[str] = []
    for path in files:
        arguments.extend(["--add-binary", f"{path}{os.pathsep}."])
    return arguments


def check_pyinstaller():
    """PyInstaller 설치 확인"""
    try:
        import PyInstaller
        print(f"PyInstaller {PyInstaller.__version__} 감지")
        return True
    except ImportError:
        print("PyInstaller가 설치되어 있지 않습니다.")
        print("  pip install pyinstaller 로 설치하세요.")
        return False


def check_runtime_dependencies() -> bool:
    """Fail before packaging when a module required by the desktop app is absent."""
    missing = [label for label, module in REQUIRED_RUNTIME_MODULES.items()
               if find_spec(module) is None]
    if not missing:
        return True
    print("데스크톱 실행 의존성이 누락되었습니다: " + ", ".join(missing))
    print("  gui 폴더에서 python -m pip install -r requirements.txt 를 실행하세요.")
    return False


def build():
    """EXE 빌드 실행"""
    # Windows에서 리다이렉트된 한글 빌드 로그도 UTF-8로 출력한다.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    if not check_pyinstaller():
        sys.exit(1)
    if not check_runtime_dependencies():
        sys.exit(1)
    if PYTHON_DIR not in sys.path:
        sys.path.insert(0, PYTHON_DIR)
    from builtin_assets import builtin_assets_ready
    if not builtin_assets_ready():
        raise RuntimeError("기본 모델 가중치 묶음이 없거나 손상되었습니다. build.bat으로 빌드를 시작하세요.")

    print("=" * 60)
    print(f"{APP_NAME} 빌드 시작")
    print("=" * 60)

    # ── PyInstaller 명령어 구성 ──
    cmd = [
        sys.executable, "-m", "PyInstaller",

        # 기본 설정
        "--name", APP_NAME,
        "--windowed",                    # 콘솔 창 없이 실행 (GUI)
        "--noconfirm",                   # 기존 빌드 덮어쓰기
        "--additional-hooks-dir", os.path.join(GUI_DIR, "pyinstaller_hooks"),

        # 아이콘 (있으면)
        # "--icon", os.path.join(RESOURCES_DIR, "icon.ico"),

        # 데이터 파일 포함
        "--add-data", f"{RESOURCES_DIR}{os.pathsep}resources",
        "--add-data", f"{BUILTIN_ASSETS_DIR}{os.pathsep}builtin_assets",
        "--add-data", f"{PYTHON_DIR}{os.pathsep}python",
        "--add-data", f"{os.path.join(PROJECT_ROOT, 'model_sdk', 'schemas')}{os.pathsep}model_sdk/schemas",
        "--add-data", f"{MODEL_CATALOG_DIR}{os.pathsep}packaging/windows/models",

        # 숨겨진 임포트 (PyInstaller가 자동 감지 못하는 것)
        "--hidden-import", "PySide6.QtWidgets",
        "--hidden-import", "PySide6.QtCore",
        "--hidden-import", "PySide6.QtGui",
        # PyInstaller의 Qt hook에만 의존하지 않고 Qt DLL·플러그인과 shiboken을
        # 함께 수집한다. Windows EXE가 시작 시 PySide6를 못 찾는 회귀를 막는다.
        "--collect-all", "PySide6",
        "--collect-all", "shiboken6",
        # Shipped native model families are imported lazily from the worker.
        # Collect their package and package data so the installed EXE does not
        # fail only when a user selects LibreYOLO/RT-DETRv4.
        "--collect-all", "libreyolo",
        "--collect-all", "sam2",
        "--collect-all", "hydra",
        "--collect-all", "omegaconf",
        "--collect-all", "iopath",
        "--hidden-import", "huggingface_hub",
        "--hidden-import", "matplotlib",
        "--hidden-import", "matplotlib.backends.backend_qtagg",
        "--hidden-import", "PIL",
        "--hidden-import", "torch",
        "--hidden-import", "torchvision",
        "--hidden-import", "torchvision.transforms",
        "--hidden-import", "torchvision.models.resnet",
        "--hidden-import", "numpy",
        "--hidden-import", "onnx",
        "--hidden-import", "onnxruntime",
        "--hidden-import", "psutil",
        "--collect-submodules", "truststore",
        "--hidden-import", "certifi",
        "--collect-data", "certifi",

        # 경로 유틸리티 (PyInstaller frozen 환경 대응)
        "--hidden-import", "core.paths",
        "--hidden-import", "core.version",

        # GPU/디바이스 관리 (DeviceManager 싱글톤)
        "--hidden-import", "core.device_manager",
        "--hidden-import", "core.model_registry",
        "--hidden-import", "core.container_worker",
        "--hidden-import", "core.model_pack_worker",
        # Model-pack workers are loaded by a manifest entrypoint at runtime;
        # keep every protocol/lifecycle module in the frozen release.
        "--hidden-import", "model_runtime.assets",
        "--hidden-import", "model_runtime.container_entrypoint",
        "--hidden-import", "model_runtime.deployment_bundle",
        "--hidden-import", "model_runtime.managed_wsl",
        "--hidden-import", "model_runtime.manager",
        "--hidden-import", "model_runtime.pack_builder",
        "--hidden-import", "model_runtime.pack_installer",
        "--hidden-import", "model_runtime.special_contracts",
        "--hidden-import", "model_runtime.worker_manager",
        "--hidden-import", "model_runtime.worker_protocol",
        "--hidden-import", "model_runtime.worker_server",
        "--hidden-import", "model_runtime.windows_worker",

        # Grad-CAM 시각화 모듈
        "--hidden-import", "core.gradcam",

        # 합성 불량 생성 엔진
        "--hidden-import", "core.defect_generator",

        # OpenCV (합성 불량 + 이미지 전처리)
        "--hidden-import", "cv2",

        "--paths", PROJECT_ROOT,
        "--hidden-import", "webapp.worker",
        "--hidden-import", "webapp.jobs",
        "--hidden-import", "webapp.storage",
        "--hidden-import", "webapp.locking",
        "--hidden-import", "tools.desktop_package_smoke",

        # torchvision ops (NMS 등 C++ 확장)
        "--hidden-import", "torchvision.ops",
        "--hidden-import", "torchvision.ops.boxes",

        # 불필요한 모듈 제외 (용량 절감)
        "--exclude-module", "tkinter",
        "--exclude-module", "PyQt5",
        "--exclude-module", "PyQt6",
        "--exclude-module", "IPython",
        "--exclude-module", "jupyter",
        "--exclude-module", "notebook",

        # 엔트리포인트
        os.path.join(GUI_DIR, "main.py"),
    ]
    cmd[cmd.index(os.path.join(GUI_DIR, "main.py")):cmd.index(os.path.join(GUI_DIR, "main.py"))] = _native_runtime_arguments()

    print(f"\n 빌드 명령어:")
    print(f"  {' '.join(cmd[:6])} ...")

    # ── 빌드 실행 ──
    try:
        result = subprocess.run(
            cmd,
            cwd=GUI_DIR,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
            capture_output=False,
            text=True,
        )

        if result.returncode == 0:
            executable = f"{APP_NAME}.exe" if sys.platform == "win32" else APP_NAME
            exe_path = os.path.join(DIST_DIR, APP_NAME, executable)
            if not os.path.isfile(exe_path):
                print(f"빌드 결과 파일 없음: {exe_path}")
                return 1
            print("\n" + "=" * 60)
            print(f"빌드 성공!")
            print(f"  실행 파일: {exe_path}")
            if os.path.exists(exe_path):
                size_mb = os.path.getsize(exe_path) / (1024 * 1024)
                print(f"  크기: {size_mb:.1f} MB")
            print("=" * 60)
        else:
            print(f"\n 빌드 실패 (exit code: {result.returncode})")
            return result.returncode

    except Exception as e:
        print(f"\n 빌드 오류: {e}")
        return 1
    return 0


def clean():
    """빌드 산출물 삭제"""
    for d in [BUILD_DIR, DIST_DIR]:
        if os.path.isdir(d):
            shutil.rmtree(d)
            print(f"삭제: {d}")

    spec_file = os.path.join(GUI_DIR, f"{APP_NAME}.spec")
    if os.path.isfile(spec_file):
        os.remove(spec_file)
        print(f"삭제: {spec_file}")

    print("정리 완료")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean()
    else:
        sys.exit(build())
