"""Owned, offline-first WSL2/Docker Engine lifecycle helpers.

The installer supplies local WSL and engine artifacts.  This module only
executes explicit ``wsl.exe``/``docker`` argv lists; it never enables Windows
features, downloads distributions, or talks to a TCP Docker API implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Mapping, Sequence
import uuid


class ManagedWslError(RuntimeError):
    """Invalid owned-distro state or a failed WSL/Docker command."""


DISTRO_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{1,63}$")


def _absolute_file(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ManagedWslError(f"{name} must be an absolute regular file")
    return path.resolve()


def _absolute_dir(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ManagedWslError(f"{name} must be an absolute directory")
    return path.resolve()


@dataclass(frozen=True)
class WslCommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass
class ManagedWsl:
    distro: str
    state_dir: Path
    wsl_executable: str = "wsl.exe"
    runner: object = subprocess.run

    def __post_init__(self) -> None:
        if not DISTRO_RE.fullmatch(self.distro):
            raise ManagedWslError("invalid app-owned WSL distro name")
        self.state_dir = _absolute_dir(self.state_dir, "WSL state_dir")
        self.marker = self.state_dir / "owned-distro.json"

    def _run(self, argv: Sequence[str], *, timeout: float = 60.0,
             check: bool = True) -> WslCommandResult:
        command = tuple(str(value) for value in argv)
        try:
            result = self.runner(command, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=timeout,
                                 check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ManagedWslError(f"WSL command failed to start: {exc}") from exc
        result_value = WslCommandResult(command, result.returncode, result.stdout or "", result.stderr or "")
        if check and result.returncode != 0:
            detail = result_value.stderr.strip() or result_value.stdout.strip()
            raise ManagedWslError(f"WSL command failed ({result.returncode}): {detail}")
        return result_value

    def probe(self) -> WslCommandResult:
        """Read WSL state without enabling features or downloading anything."""
        return self._run((self.wsl_executable, "--status"), check=False)

    def list_distros(self) -> tuple[str, ...]:
        result = self._run((self.wsl_executable, "--list", "--quiet"), check=False)
        if result.returncode != 0:
            return ()
        return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())

    def is_owned(self) -> bool:
        if self.marker.is_symlink() or not self.marker.is_file():
            return False
        try:
            value = json.loads(self.marker.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return False
        return (isinstance(value, dict) and value.get("schema_version") == 1 and
                value.get("distro") == self.distro and value.get("owned") is True)

    def import_offline(self, tarball: str | Path, install_dir: str | Path, *, version: str) -> None:
        """Import a bundled distro tarball; ``--web-download`` is forbidden."""
        archive = _absolute_file(tarball, "WSL distro tarball")
        target = _absolute_dir(install_dir, "WSL install_dir")
        if self.distro in self.list_distros():
            if not self.is_owned():
                raise ManagedWslError("existing WSL distro is not owned by this application")
            return
        try:
            if any(target.iterdir()):
                raise ManagedWslError("WSL install_dir must be empty before import")
        except OSError as exc:
            raise ManagedWslError(f"cannot inspect WSL install_dir: {target}") from exc
        import_attempted = False
        imported = False
        temporary = self.marker.with_name(f".{self.marker.name}.{uuid.uuid4().hex}.pending")
        try:
            import_attempted = True
            self._run((self.wsl_executable, "--import", self.distro, str(target), str(archive), "--version", "2"))
            imported = True
            temporary.write_text(json.dumps({"schema_version": 1, "owned": True,
                                             "distro": self.distro, "version": version},
                                            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.marker)
        except Exception as exc:
            if import_attempted:
                try:
                    rollback = self._run((self.wsl_executable, "--unregister", self.distro), check=False)
                    if imported and rollback.returncode != 0:
                        raise ManagedWslError(
                            f"failed to rollback unmarked WSL distro {self.distro}: {rollback.returncode}") from exc
                except ManagedWslError:
                    raise
                except Exception as rollback_exc:
                    raise ManagedWslError(
                        f"failed to rollback unmarked WSL distro {self.distro}") from rollback_exc
            if temporary.exists():
                temporary.unlink(missing_ok=True)
            if isinstance(exc, ManagedWslError):
                raise
            raise ManagedWslError("owned WSL marker commit failed") from exc

    def run(self, argv: Sequence[str], *, timeout: float = 60.0,
            check: bool = True) -> WslCommandResult:
        if not self.is_owned():
            raise ManagedWslError("owned WSL distro is not initialized")
        command = tuple(str(value) for value in argv)
        if not command or any("\x00" in value for value in command):
            raise ManagedWslError("invalid WSL command")
        return self._run((self.wsl_executable, "-d", self.distro, "--", *command),
                         timeout=timeout, check=check)

    def docker(self, argv: Sequence[str], *, timeout: float = 60.0,
               check: bool = True) -> WslCommandResult:
        """Run Docker inside the owned distro; no host TCP socket is used."""
        return self.run(("docker", *tuple(argv)), timeout=timeout, check=check)

    def docker_argv(self) -> tuple[str, ...]:
        """Return the shell-free argv prefix used for Docker subprocesses.

        The prefix is intentionally usable by callers that need to attach a
        stdin/stdout stream (for example ``docker run -i``).  It performs the
        same ownership check as :meth:`run`, so a configured but uninitialized
        distro never silently falls back to a host Docker daemon.
        """
        if not self.is_owned():
            raise ManagedWslError("owned WSL distro is not initialized")
        return (self.wsl_executable, "-d", self.distro, "--", "docker")

    def shutdown(self) -> None:
        self._run((self.wsl_executable, "--terminate", self.distro), check=False)

    def unregister(self, *, remove_storage: bool = False) -> None:
        if not self.is_owned():
            raise ManagedWslError("refusing to unregister an unowned WSL distro")
        self._run((self.wsl_executable, "--unregister", self.distro), check=True)
        self.marker.unlink(missing_ok=True)
        if remove_storage:
            # The distro's VHDX is removed by WSL unregister.  Keep the app
            # state directory itself so logs and user packs remain intact.
            return


def configured_docker_command(*, environ: Mapping[str, str] | None = None,
                             state_dir: str | Path | None = None) -> tuple[str, ...]:
    """Resolve the app's Docker argv prefix without opening a shell.

    The Windows installer/launcher may set ``DEEPVISION_WSL_DISTRO`` after the
    owned distro has been initialized.  If it is absent, the helper also reads
    the installer-owned marker from the WSL state directory.  With no marker,
    the normal local ``docker`` executable remains the development/runtime
    default.  Once a distro is selected, ownership is mandatory and a missing
    marker raises instead of silently using a user's Docker Desktop daemon.
    """
    env = os.environ if environ is None else environ
    configured_state = state_dir or env.get("DEEPVISION_WSL_STATE_DIR")
    if configured_state is None:
        app_state = env.get("DEEP_STUDIO_STATE_DIR")
        if app_state:
            configured_state = Path(app_state) / "wsl"
        elif env.get("LOCALAPPDATA"):
            configured_state = Path(env["LOCALAPPDATA"]) / "DeepVisionStudio" / "wsl"
        else:
            configured_state = Path.home() / ".deep-vision-studio-react" / "wsl"
    configured_state = Path(configured_state).expanduser()
    distro = str(env.get("DEEPVISION_WSL_DISTRO", "")).strip()
    marker = configured_state / "owned-distro.json"
    if marker.is_symlink():
        raise ManagedWslError("owned WSL distro marker cannot be a symlink")
    if not distro and marker.is_file():
        try:
            marker_value = json.loads(marker.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise ManagedWslError("owned WSL distro marker is invalid") from exc
        if (not isinstance(marker_value, dict) or marker_value.get("schema_version") != 1 or
                marker_value.get("owned") is not True or
                not isinstance(marker_value.get("distro"), str)):
            raise ManagedWslError("owned WSL distro marker is invalid")
        distro = marker_value["distro"].strip()
    if not distro:
        return ("docker",)
    manager = ManagedWsl(distro, configured_state)
    return manager.docker_argv()
