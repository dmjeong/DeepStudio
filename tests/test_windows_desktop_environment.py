import runpy
from pathlib import Path


def test_frozen_app_cannot_borrow_python_qt_or_native_sdk_paths():
    clean = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/windows_desktop_smoke.py"))["clean_app_environment"]
    source = {"Path": r"C:\Python;C:\onnx-sdk\bin", "PYTHONPATH": r"C:\repo\python",
              "PYTHONHOME": r"C:\Python", "PYTHONUTF8": "1", "QT_PLUGIN_PATH": r"C:\PySide6",
              "QML2_IMPORT_PATH": r"C:\Qt", "DEEP_STUDIO_PROJECT": r"C:\repo",
              "VIRTUAL_ENV": r"C:\venv", "CUDA_PATH_V13_0": r"C:\CUDA",
              "VISION_NATIVE_RUNTIME_DIR": r"C:\native-sdk", "ONNXRUNTIME_ROOT": r"C:\onnx-sdk",
              "SystemRoot": r"C:\Windows", "TEMP": r"C:\Temp", "USERPROFILE": r"C:\Users\test"}
    result = clean(source)
    assert result == {"SystemRoot": r"C:\Windows", "TEMP": r"C:\Temp", "USERPROFILE": r"C:\Users\test",
                      "PATH": r"C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem",
                      "QT_QPA_PLATFORM": "offscreen"}
    assert source["PYTHONPATH"] == r"C:\repo\python"
