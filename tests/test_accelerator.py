"""CPU 전용 설치가 GPU 요청을 숨기는 회귀를 장치 없이 재현한다."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gui"))
from core.accelerator import cuda_index, nvidia_devices


class AcceleratorTests(unittest.TestCase):
    def test_driver_selects_supported_cuda_build(self):
        self.assertEqual(cuda_index([{"driver": "591.44"}]), "cu130")
        self.assertEqual(cuda_index([{"driver": "560.94"}]), "cu126")
        with self.assertRaisesRegex(RuntimeError, "업데이트"):
            cuda_index([{"driver": "470.42"}])
        with self.assertRaisesRegex(RuntimeError, "감지 실패"):
            cuda_index([])

    def test_nvidia_detection_does_not_depend_on_cpu_torch(self):
        response = subprocess.CompletedProcess([], 0, 'NVIDIA RTX 4000 Ada, 591.44\n', '')
        with patch("core.accelerator.shutil.which", return_value="nvidia-smi"), \
             patch("core.accelerator.subprocess.run", return_value=response):
            self.assertEqual(nvidia_devices(), [{"name": "NVIDIA RTX 4000 Ada", "driver": "591.44"}])

    def manager(self, available):
        def device(value):
            name, _, index = value.partition(":")
            return SimpleNamespace(type=name, index=int(index) if index else None)
        spec = importlib.util.spec_from_file_location("studio_test_device", ROOT / "gui/core/device_manager.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": SimpleNamespace(device=device), spec.name: module}):
            spec.loader.exec_module(module)
        manager = module.DeviceManager.__new__(module.DeviceManager)
        manager._cuda_available = available
        manager._gpu_list = [None] if available else []
        return manager

    def test_explicit_gpu_never_silently_becomes_cpu(self):
        manager = self.manager(False)
        with self.assertRaisesRegex(RuntimeError, "CPU로 전환하지"):
            manager.get_device("cuda:0")
        self.assertEqual(manager.get_device("cpu").type, "cpu")

    def test_auto_reports_broken_gpu_install_but_cpu_machine_still_works(self):
        manager = self.manager(False)
        with patch("core.accelerator.nvidia_devices", return_value=[{"name": "RTX", "driver": "591"}]):
            with self.assertRaisesRegex(RuntimeError, "NVIDIA GPU"):
                manager.get_device("auto")
            self.assertIn("auto", [value for _, value in manager.get_combo_items()])
            self.assertIn("CUDA 실행 불가", manager.get_status_text())
        with patch("core.accelerator.nvidia_devices", return_value=[]):
            self.assertEqual(manager.get_device("auto").type, "cpu")

    def test_selected_gpu_index_is_validated(self):
        manager = self.manager(True)
        self.assertEqual(manager.get_device("auto").index, 0)
        self.assertEqual(manager.get_device("cuda").index, 0)
        with self.assertRaisesRegex(ValueError, "GPU 1 없음"):
            manager.get_device("cuda:1")
        with self.assertRaises(ValueError):
            manager.get_device("unrecognized")

    def test_desktop_builder_upgrades_cpu_torch_and_verifies_cuda_backward(self):
        spec = importlib.util.spec_from_file_location(
            "studio_prepare_torch", ROOT / "gui/prepare_torch_runtime.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        installed = [
            {"versions": {"torch": "2.14.0+cpu", "torchvision": "0.29.0+cpu"}},
            {"versions": {"torch": "2.14.0+cu130", "torchvision": "0.29.0+cu130"}},
        ]
        reports = [
            {"cuda_available": False},
            {"cuda_available": True, "cuda_runtime": "13.0", "name": "RTX",
             "backward_checked": True, "error": ""},
        ]
        runner = MagicMock()
        with patch.object(module, "installed_state", side_effect=installed), \
             patch.object(module, "inspect_runtime", side_effect=reports), \
             patch.object(module, "nvidia_devices",
                          return_value=[{"name": "RTX 4000", "driver": "591.44"}]), \
             patch.object(module.subprocess, "run", runner):
            result = module.prepare_runtime("auto", python=ROOT / "python.exe")
        command = runner.call_args.args[0]
        self.assertIn("torch==2.14.0+cu130", command)
        self.assertIn("torchvision==0.29.0+cu130", command)
        self.assertEqual(result["name"], "RTX")


if __name__ == "__main__":
    unittest.main()
