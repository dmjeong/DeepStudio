"""Static contracts for the Windows-only WiX release entry point."""

from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "packaging" / "windows"


def test_wix_sources_are_well_formed_and_use_payload_contract():
    msi = (WINDOWS / "bootstrapper" / "DeepVisionStudio.msi.wxs").read_text(encoding="utf-8")
    bundle = (WINDOWS / "bootstrapper" / "DeepVisionStudio.bundle.wxs").read_text(encoding="utf-8")
    ET.fromstring(msi)
    ET.fromstring(bundle)
    assert 'Files Include="$(var.PayloadRoot)\\**"' in msi
    assert "MsiPackage SourceFile=\"$(var.MsiPath)\"" in bundle
    assert "Condition=\"VersionNT64\"" in bundle


def test_release_script_verifies_payload_before_wix_build():
    script = (WINDOWS / "build_release.ps1").read_text(encoding="utf-8")
    assert "--manifest" in script and "--verify" in script
    assert "WixToolset.Bal.wixext" in script
    assert "THIRD_PARTY_NOTICES.md" in script
    assert "external_downloads" not in script
