"""Version policy regression checks."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]


def test_current_version_is_reset_and_policy_is_documented():
    version = runpy.run_path(str(ROOT / "gui" / "core" / "version.py"))
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert version["APP_VERSION"] == "0.0"
    for increment in ("+0.01", "+0.1", "+1.0"):
        assert increment in changelog


def test_ci_skips_duplicate_release_tags_until_the_version_changes():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "skip until APP_VERSION is increased" in workflow
    assert "skip publishing duplicate assets" in workflow
