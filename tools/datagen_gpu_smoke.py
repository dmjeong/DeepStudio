"""Run real pretrained training and reload/generation in separate GPU processes.

Requires the user's frozen real dataset and normal image. It reports execution
integrity, not defect quality or downstream inspection improvement.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import webapp  # noqa: E402,F401
from core.project import ProjectManager
from webapp.storage import project_view, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--normal", required=True, help="Registered normal image ID")
    parser.add_argument("--mask", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    project = project_view(ProjectManager.load(args.project))
    with tempfile.TemporaryDirectory(prefix="datagen-gpu-") as directory:
        root = Path(directory)
        def execute(name, kind, values):
            request = root / name / "request.json"
            write_json(request, {"kind": kind, "payload": {"project": project, **values}})
            command = [sys.executable, "-c", "from webapp.worker import run_job; import sys; run_job(sys.argv[1], monitor_parent=False)", str(request)]
            subprocess.run(command, cwd=ROOT, check=True)
            result = read_json(request.parent / "result.json")
            if result["status"] != "completed":
                raise RuntimeError(result)
            return result
        trained = execute("train", "datagen_train", {"dataset_id": args.dataset,
            "config": {"base_model": args.base_model, "prompt": args.prompt, "steps": 2, "validate_every": 1}})
        generated = execute("reload-generate", "datagen_generate", {"model_id": trained["output"]["model_id"],
            "image_id": args.normal, "requested": args.mask, "count": 1, "seed": 42})
        report = {"training": trained, "new_process_generation": generated, "quality_validated": False}
        output = Path(project["project_dir"]) / "datagen" / "gpu-smoke-report.json"
        write_json(output, report)
        print(json.dumps({"report": str(output), "status": "execution_passed", "quality_validated": False}))


if __name__ == "__main__":
    main()
