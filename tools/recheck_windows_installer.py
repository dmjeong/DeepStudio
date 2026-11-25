"""Re-test a completed CI installer on a separate runner without training libraries."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

from windows_desktop_smoke import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repository) or not args.run_id.isdigit():
        parser.error("Expected owner/repository and numeric run ID")
    endpoint = f"repos/{args.repository}/actions/runs/{args.run_id}/artifacts"
    metadata = json.loads(subprocess.check_output(["gh", "api", endpoint]))
    matches = [item for item in metadata["artifacts"]
               if item["name"] == "deepvisionstudio-wix-contract" and not item["expired"]]
    if len(matches) != 1:
        raise RuntimeError("Expected one completed Windows installer artifact")
    with tempfile.TemporaryDirectory(prefix="dvs-installer-recheck-") as temporary:
        folder = Path(temporary)
        archive_path = folder / "artifact.zip"
        with archive_path.open("wb") as output:
            subprocess.run(["gh", "api", f"repos/{args.repository}/actions/artifacts/{matches[0]['id']}/zip"],
                           stdout=output, check=True)
        # Extract only the simple MSI and its cabinets, not duplicate Setup/ZIP
        # or the separate advanced SDK payload. Basenames prevent Zip Slip.
        media = folder / "media"
        media.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            for entry in archive.infolist():
                path = Path(entry.filename.replace("\\", "/"))
                if "studio-simple-release" not in path.parts or path.suffix not in {".msi", ".cab"}:
                    continue
                target = media / path.name
                if target.exists():
                    raise RuntimeError(f"Duplicate installer media: {path.name}")
                with archive.open(entry) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
        archive_path.unlink()
        packages = list(media.glob("*.msi"))
        if len(packages) != 1:
            raise RuntimeError("Simple installer MSI not present in the artifact")
        run(args.output, msi=packages[0])
        report = json.loads(args.output.read_text(encoding="utf-8"))
        report["installer_source_run"] = args.run_id
        report["installer_artifact_id"] = matches[0]["id"]
        report["isolated_system_path"] = True
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
