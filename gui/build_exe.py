"""
Deep Vision Studio — EXE 빌드 스크립트

PyInstaller를 사용하여 독립 실행 파일(.exe) 생성

사용법:
    cd DeepVisionStudio/gui
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

# 경로 설정
GUI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(GUI_DIR)
PYTHON_DIR = os.path.join(PROJECT_ROOT, "python")
MODEL_CATALOG_DIR = os.path.join(PROJECT_ROOT, "packaging", "windows", "models")
RESOURCES_DIR = os.path.join(GUI_DIR, "resources")
DIST_DIR = os.path.join(GUI_DIR, "dist")
BUILD_DIR = os.path.join(GUI_DIR, "build")

APP_NAME = "DeepVisionStudio"


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


def build():
    """EXE 빌드 실행"""
    # Windows에서 리다이렉트된 한글 빌드 로그도 UTF-8로 출력한다.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    if not check_pyinstaller():
        sys.exit(1)

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
        "--add-data", f"{PYTHON_DIR}{os.pathsep}python",
        "--add-data", f"{os.path.join(PROJECT_ROOT, 'model_sdk', 'schemas')}{os.pathsep}model_sdk/schemas",
        "--add-data", f"{MODEL_CATALOG_DIR}{os.pathsep}packaging/windows/models",

        # 숨겨진 임포트 (PyInstaller가 자동 감지 못하는 것)
        "--hidden-import", "PySide6.QtWidgets",
        "--hidden-import", "PySide6.QtCore",
        "--hidden-import", "PySide6.QtGui",
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
