"""Execute PowerShell parameter binding/cleanup, without pretending to run WiX."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PWSH = os.environ.get("DVS_PWSH") or shutil.which("pwsh")
pytestmark = pytest.mark.skipif(not PWSH, reason="PowerShell executable required")


def invoke(path, *arguments):
    return subprocess.run([PWSH, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(path),
                           *map(str, arguments)], text=True, encoding="utf-8", errors="replace",
                          capture_output=True, timeout=45)


def test_installer_passes_named_arguments_to_release_script(tmp_path):
    folder = tmp_path / "project" / "packaging" / "windows"
    folder.mkdir(parents=True)
    shutil.copyfile(ROOT / "packaging/windows/build_simple_installer.ps1", folder / "build_simple_installer.ps1")
    # The native compiler is a test double; PowerShell argument binding is real.
    (folder / "build_release.ps1").write_text('''
[CmdletBinding()]
param([string] $PayloadRoot, [string] $Version, [string] $OutputDirectory,
      [string] $WixVersion, [switch] $Simple, [string] $CertificatePath,
      [string] $CertificatePassword, [string] $SignToolPath, [string] $TimestampUrl,
      [switch] $RequireSignature)
$PSBoundParameters | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $OutputDirectory "binding.json")
''', encoding="utf-8")
    app, output = tmp_path / "한글 앱 폴더", tmp_path / "한글 설치 결과"
    app.mkdir()
    (app / "DeepVisionStudio.exe").write_bytes(b"fixture")
    wrapper = tmp_path / "invoke.ps1"
    wrapper.write_text('''
param([string] $Builder, [string] $App, [string] $Output)
$ErrorActionPreference = "Stop"
$env:OS = "Windows_NT"
function python { $global:LASTEXITCODE = 0 }
& $Builder -AppRoot $App -OutputDirectory $Output -Version "17.90" -RequireSignature `
    -CertificatePath "fixture certificate.pfx" -CertificatePassword "fixture only"
''', encoding="utf-8")
    result = invoke(wrapper, "-Builder", folder / "build_simple_installer.ps1", "-App", app, "-Output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    bound = json.loads((output / "binding.json").read_text(encoding="utf-8-sig"))
    assert Path(bound["PayloadRoot"]) == output / ".simple-payload-17.90.0"
    assert bound["Version"] == "17.90.0"
    assert Path(bound["OutputDirectory"]) == output
    assert bound["Simple"] and bound["RequireSignature"]
    assert bound["CertificatePath"] == "fixture certificate.pfx"
    assert bound["CertificatePassword"] == "fixture only"


def test_early_release_error_is_not_hidden_by_cleanup(tmp_path):
    wrapper = tmp_path / "failure.ps1"
    wrapper.write_text('''
param([string] $Builder, [string] $Missing, [string] $Output)
$ErrorActionPreference = "Stop"
$env:OS = "Windows_NT"
try {
    & $Builder -PayloadRoot $Missing -Version "17.90.0" -OutputDirectory $Output
} catch {
    Write-Output $_.FullyQualifiedErrorId
    Write-Output $_.Exception.Message
    exit 9
}
exit 2
''', encoding="utf-8")
    result = invoke(wrapper, "-Builder", ROOT / "packaging/windows/build_release.ps1",
                    "-Missing", tmp_path / "not-created", "-Output", tmp_path / "output")
    assert result.returncode == 9, result.stdout + result.stderr
    assert "PathNotFound" in result.stdout
    assert "not-created" in result.stdout
    assert "null or an empty" not in result.stdout


def test_release_collects_and_validates_payload_without_github_environment(tmp_path):
    import runpy
    stage = runpy.run_path(str(ROOT / "packaging/windows/stage_simple_payload.py"))["stage_simple_payload"]
    app = tmp_path / "app-source"
    app.mkdir()
    (app / "DeepVisionStudio.exe").write_bytes(b"fixture")
    payload = stage(tmp_path / "payload", app=app, example_root=ROOT / "example",
                    cpp_runtime_root=ROOT / "cpp", csharp_runtime_root=ROOT / "sdk/csharp",
                    notice=ROOT / "THIRD_PARTY_NOTICES.md")
    compiler = tmp_path / "compiler.ps1"
    compiler.write_text('''
if ($args[0] -eq "--version") {
    Write-Output "7.0.0"
} else {
    for ($i = 0; $i -lt $args.Count - 1; $i++) {
        if ($args[$i] -eq "-o") { Set-Content -LiteralPath $args[$i+1] -Value "compiler fixture" }
    }
}
$global:LASTEXITCODE = 0
''', encoding="utf-8")
    wrapper = tmp_path / "release.ps1"
    wrapper.write_text('''
param([string] $Builder, [string] $Payload, [string] $Output, [string] $Python, [string] $Compiler)
$ErrorActionPreference = "Stop"
$env:OS = "Windows_NT"
$env:GITHUB_SHA = ""
function Get-Command { [pscustomobject]@{ Source = $Compiler } }
function python { & $Python @args; $global:LASTEXITCODE = $LASTEXITCODE }
& $Builder -PayloadRoot $Payload -Version "17.90.0" -OutputDirectory $Output -Simple
''', encoding="utf-8")
    output = tmp_path / "release"
    result = invoke(wrapper, "-Builder", ROOT / "packaging/windows/build_release.ps1",
                    "-Payload", payload, "-Output", output, "-Python", sys.executable, "-Compiler", compiler)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "release-manifest.json").is_file()
    assert (output / "DeepVisionStudio-17.90.0.msi").read_text().strip() == "compiler fixture"
    assert (output / "DeepVisionStudio-Setup-17.90.0-win-x64.exe").is_file()
