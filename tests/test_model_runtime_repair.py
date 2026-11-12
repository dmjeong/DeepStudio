"""Runtime repair uses the app interpreter and cannot replace the CUDA build."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import model_dependencies as model_runtime
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repair_model_runtime", ROOT / "gui/repair_model_runtime.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


@pytest.mark.parametrize("model_id", sorted(model_runtime.TIMM_MODELS))
def test_each_timm_model_gives_same_interpreter_recovery(model_id, monkeypatch):
    def unavailable(name):
        raise ModuleNotFoundError("No module named timm", name="timm")
    monkeypatch.setattr(model_runtime.importlib, "import_module", unavailable)
    from builtin_models import build_builtin_model
    with pytest.raises(RuntimeError) as caught:
        build_builtin_model(model_id, 2)
    message = str(caught.value)
    assert "repair_model_runtime.bat" in message
    assert sys.executable in message and "재학습할 필요가 없습니다" in message


def test_missing_transitive_dependency_is_explained(monkeypatch):
    def unavailable(name):
        raise ModuleNotFoundError("No module named safetensors", name="safetensors")
    monkeypatch.setattr(model_runtime.importlib, "import_module", unavailable)
    with pytest.raises(RuntimeError, match="safetensors"):
        model_runtime.require_model_runtime("efficientvit_b0_seg")


def test_unrelated_models_do_not_import_timm(monkeypatch):
    def forbidden(name):
        raise AssertionError(name)
    monkeypatch.setattr(model_runtime.importlib, "import_module", forbidden)
    for model_id in ("sctnet_s", "unet_lite", "stdc1", "bisenetv2", "unet_resnet18"):
        model_runtime.require_model_runtime(model_id)


def test_frozen_error_does_not_instruct_pip_on_exe(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    message = model_runtime.dependency_message("efficientvit_b0_seg", ModuleNotFoundError(name="timm"))
    assert "배포판" in message and "repair_model_runtime" not in message


def test_repair_preserves_full_cuda_versions_and_installs_transitive_deps(monkeypatch):
    versions = {"torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128", "numpy": "2.2.6"}
    monkeypatch.setattr(repair.importlib.metadata, "version", versions.__getitem__)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["check"] is True
        assert command[0] == sys.executable
        if "install" in command:
            assert "--no-deps" not in command
            constraints = Path(command[command.index("--constraint") + 1]).read_text()
            assert "torch==2.10.0+cu128\n" in constraints
            assert "torchvision==0.25.0+cu128\n" in constraints
            assert "numpy==2.2.6\n" in constraints
            assert "requirements-model-runtime.txt" in command[command.index("-r") + 1]
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(repair.subprocess, "run", run)
    repair.repair_runtime()
    assert len(calls) == 2 and "-c" in calls[1]
    assert all("prepare_builtin_assets" not in str(call) for call in calls)


def test_failed_install_is_not_reported_as_success(monkeypatch):
    monkeypatch.setattr(repair, "protected_versions", lambda: {"torch": "2.10.0", "torchvision": "0.25.0"})
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "pip")
    monkeypatch.setattr(repair.subprocess, "run", fail)
    assert repair.main([]) == 1


def test_model_runtime_pins_agree_with_all_installers():
    minimal = (ROOT / "gui/requirements-model-runtime.txt").read_text()
    for filename in ("gui/requirements.txt", "python/requirements.txt", "webapp/requirements.txt"):
        content = (ROOT / filename).read_text()
        for requirement in ("timm==1.0.31", "scipy>=1.13,<2"):
            assert requirement in content and requirement in minimal
