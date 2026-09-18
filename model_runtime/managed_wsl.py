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
        if not self.marker.is_file():
            return False
        try:
            value = json.loads(self.marker.read_text(encoding="utf-8"))
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
        self._run((self.wsl_executable, "--import", self.distro, str(target), str(archive), "--version", "2"))
        self.marker.write_text(json.dumps({"schema_version": 1, "owned": True,
                                           "distro": self.distro, "version": version},
                                          ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

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
