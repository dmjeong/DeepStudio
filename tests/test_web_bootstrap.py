"""외부 다운로드 없이 실제 venv와 pip로 자동 설치 수명 주기를 검증한다."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from webapp.bootstrap import ensure_environment, environment_path, python_path, runtime_plan
from webapp.locking import exclusive_file


def fixture_wheel(directory, version):
    name = "studio_bootstrap_fixture"
    metadata = f"{name}-{version}.dist-info"
    path = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(f"{name}/__init__.py", f"VERSION = '{version}'\n")
        wheel.writestr(f"{metadata}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        wheel.writestr(f"{metadata}/WHEEL", "Wheel-Version: 1.0\nGenerator: studio-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        wheel.writestr(f"{metadata}/RECORD", "")


class WebBootstrapTests(unittest.TestCase):
    def test_actual_offline_install_reuse_repair_and_failed_upgrade(self):
        with tempfile.TemporaryDirectory(prefix="studio setup ") as folder:
            root = Path(folder) / "검사 환경"
            (root / "webapp").mkdir(parents=True)
            requirements = root / "webapp" / "requirements.txt"
            requirements.write_text("studio-bootstrap-fixture==1.0\n", encoding="utf-8")
            wheels = root / "wheels"
            wheels.mkdir()
            fixture_wheel(wheels, "1.0")
            fixture_wheel(wheels, "2.0")
            environment = root / ".venv"
            state = root / "state"
            marker = environment / ".deep-studio-web.json"
            with patch("webapp.bootstrap.environment_path", return_value=environment), patch.dict(os.environ, {
                "PIP_NO_INDEX": "1", "PIP_FIND_LINKS": wheels.as_uri(), "PIP_CONFIG_FILE": os.devnull,
            }):
                def setup():
                    return ensure_environment(root, state, runtime_modules=("studio_bootstrap_fixture",))

                python = setup()
                self.assertEqual(python, python_path(environment))
                self.assertTrue(marker.is_file())
                result = subprocess.check_output([str(python), "-c", "import studio_bootstrap_fixture as f; print(f.VERSION)"], text=True)
                self.assertEqual(result.strip(), "1.0")
                with patch("webapp.bootstrap.run_logged", side_effect=AssertionError("재실행에서 설치 호출 금지")):
                    self.assertEqual(setup(), python)

                subprocess.run([str(python), "-m", "pip", "uninstall", "-y", "studio-bootstrap-fixture"],
                               check=True, capture_output=True)
                setup()
                self.assertTrue(marker.is_file())

                requirements.write_text("studio-bootstrap-fixture==2.0\n", encoding="utf-8")
                with exclusive_file(state / "server.lock"):
                    with self.assertRaisesRegex(RuntimeError, "실행 중인 스튜디오"):
                        setup()
                setup()
                self.assertEqual(json.loads(marker.read_text())["versions"]["studio-bootstrap-fixture"], "2.0")

                requirements.write_text("studio-bootstrap-fixture==3.0\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "설치 또는 환경 검사 실패"):
                    setup()
                self.assertFalse(marker.exists())
                self.assertIn("ERROR", (environment / "deep-studio-setup.log").read_text(encoding="utf-8"))
                requirements.write_text("studio-bootstrap-fixture==2.0\n", encoding="utf-8")
                setup()
                self.assertTrue(marker.is_file())

    def test_active_environment_is_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("sys.prefix", folder), patch("sys.base_prefix", folder + "-base"):
                self.assertEqual(environment_path(Path(folder) / "project"), Path(folder).resolve())

    def test_conda_uses_its_existing_interpreter_location(self):
        with tempfile.TemporaryDirectory() as folder:
            executable = str(Path(folder) / "python.exe")
            with patch("sys.prefix", folder), patch("sys.base_prefix", folder), \
                 patch("sys.executable", executable), patch.dict(os.environ, {"CONDA_PREFIX": folder}):
                environment = environment_path(Path(folder) / "project")
                self.assertEqual(environment, Path(folder).resolve())
                self.assertEqual(python_path(environment), Path(executable))

    def test_cpu_install_is_not_forced_on_existing_cuda_packages(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "webapp").mkdir()
            (root / "webapp" / "requirements.txt").write_text("torch>=2.0\ntorchvision>=0.15\n", encoding="utf-8")
            environment = root / ".venv"
            python = python_path(environment)
            python.parent.mkdir(parents=True)
            python.touch()
            installed = {"python": [3, 11, 0], "versions": {"torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128"}}
            with patch("webapp.bootstrap.environment_path", return_value=environment), \
                 patch("webapp.bootstrap.installed_state", return_value=installed), \
                 patch("webapp.bootstrap.nvidia_devices", return_value=[{"name": "RTX", "driver": "591.44"}]), \
                 patch("webapp.bootstrap.inspect_runtime", return_value={"cuda_available": True, "cuda_runtime": "12.8", "backward_checked": True}), \
                 patch("webapp.bootstrap.subprocess.run", return_value=subprocess.CompletedProcess([], 0)), \
                 patch("webapp.bootstrap.run_logged") as logged:
                ensure_environment(root, root / "state")
                commands = [call.args[0] for call in logged.call_args_list]
                self.assertFalse(any("--index-url" in command for command in commands))
                self.assertTrue(any("-r" in command for command in commands))

    def test_cpu_environment_is_upgraded_on_nvidia_machine(self):
        versions = {"torch": "2.14.0+cpu", "torchvision": "0.29.0+cpu"}
        devices = [{"name": "RTX 4000 Ada", "driver": "591.44"}]
        self.assertEqual(runtime_plan(versions, {"cuda_available": False}, devices, "auto"), (True, "cu130"))
        self.assertEqual(runtime_plan(versions, {"cuda_available": False}, devices, "cpu"), (False, None))

    def test_cuda_request_without_gpu_does_not_install_cpu(self):
        with self.assertRaisesRegex(RuntimeError, "GPU 감지 실패"):
            runtime_plan({}, {}, [], "cuda")

    def test_new_gpu_install_and_failed_validation_leave_no_ready_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "webapp").mkdir()
            (root / "webapp/requirements.txt").write_text("torch>=2.0\ntorchvision>=0.15\n")
            environment = root / ".venv"
            python = python_path(environment)
            python.parent.mkdir(parents=True)
            python.touch()
            installed = {"python": [3, 11, 0], "versions": {"torch": "2.14.0+cpu", "torchvision": "0.29.0+cpu"}}
            marker = environment / ".deep-studio-web.json"
            marker.write_text(json.dumps({"requirements": "old", **installed}))
            with patch("webapp.bootstrap.environment_path", return_value=environment), \
                 patch("webapp.bootstrap.installed_state", return_value=installed), \
                 patch("webapp.bootstrap.nvidia_devices", return_value=[{"name": "RTX", "driver": "591.44"}]), \
                 patch("webapp.bootstrap.inspect_runtime", return_value={"cuda_available": False}), \
                 patch("webapp.bootstrap.subprocess.run", return_value=subprocess.CompletedProcess([], 0)), \
                 patch("webapp.bootstrap.run_logged") as logged:
                with self.assertRaisesRegex(RuntimeError, "GPU 실행 환경 검사 실패"):
                    ensure_environment(root, root / "state")
                commands = [call.args[0] for call in logged.call_args_list]
                self.assertTrue(any("torch==2.14.0+cu130" in command for command in commands))
                self.assertFalse(any("https://download.pytorch.org/whl/cpu" in command for command in commands))
                self.assertFalse(marker.exists())

    @unittest.skipUnless(os.name == "nt", "Windows 배치 실행 계약")
    def test_windows_batch_forwards_options_and_exit_status(self):
        with tempfile.TemporaryDirectory(prefix="studio launcher ") as folder:
            root = Path(folder) / "한글 경로"
            root.mkdir()
            shutil.copyfile(ROOT / "start_web.bat", root / "start_web.bat")
            (root / "start_web.py").write_text(
                "import json, pathlib, sys\npathlib.Path('args.json').write_text(json.dumps(sys.argv[1:]))\nsys.exit(7 if '--fail' in sys.argv else 0)\n",
                encoding="utf-8")
            # 가상환경 실행기 선택 분기를 실제 Python으로 실행한다.
            environment = {**os.environ, "CONDA_PREFIX": str(Path(sys.executable).parent), "VIRTUAL_ENV": ""}
            for arguments, expected in ((["--port", "8770", "--no-browser"], 0), (["--fail"], 7)):
                result = subprocess.run(["cmd", "/d", "/c", "start_web.bat", *arguments], cwd=root,
                                        env=environment, input="", capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", timeout=30)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                self.assertEqual(json.loads((root / "args.json").read_text()), arguments)


if __name__ == "__main__":
    unittest.main()
