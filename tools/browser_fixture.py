"""실제 배포 소스를 사용하는 브라우저 편집/설정 검증 서버."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import webapp  # noqa: E402,F401
from core.project import ProjectManager
from webapp.storage import ProjectStore
from webapp.server import create_app
from PIL import Image
import uvicorn

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=True)
project = ProjectManager.create_new("검사 워크스페이스", "classify", str(root / "project"), ["good", "scratch", "empty"])
ProjectManager.save(project)
ProjectStore(root / "state").select(project)
image = root / "sample.png"
Image.new("RGB", (100, 64), (25, 60, 180)).save(image)
defect_folder = root / "defect_sources"
defect_folder.mkdir()
Image.new("RGB", (100, 64), (100, 100, 100)).save(defect_folder / "normal.png")
# A saved, deterministic inference job exercises the real paginated API and preview route.
# Fixture scores are synthetic test inputs, not model-quality measurements.
import numpy as np
from dataclasses import asdict, replace
from core.inference_types import make_anomaly_result
from patchcore_scores import score_normalization, normalize_score, normalized_details
from webapp.storage import write_json, project_view
from webapp.jobs import timestamp
job_id = "ab" * 16
job_dir = root / "state" / "jobs" / job_id
results_dir = job_dir / "results"
results_dir.mkdir(parents=True)
write_json(job_dir / "job.json", {"id": job_id, "kind": "infer", "status": "completed", "created_at": timestamp(),
    "project_path": ProjectManager.get_active_filepath(project), "output": {}, "error": ""})
write_json(job_dir / "request.json", {"payload": {"project": project_view(project)}})
for index in range(75):
    name = "good" if index % 2 == 0 else "scratch"
    path = root / "review_images" / name / f"image-{index:03}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    original = np.full((64, 100, 3), index + 50, np.uint8)
    Image.fromarray(original).save(path)
    spec = score_normalization(3)
    result = replace(make_anomaly_result(str(path), normalize_score(index / 10, spec), .5), source_class=name,
                     details=normalized_details(index / 10, 3, spec),
                     inference_sec=.01 + index / 1000, inference_status="completed", elapsed_sec=.02 + index / 1000)
    write_json(results_dir / f"{index}.json", {**asdict(result), "index": index, "cache_ready": True})
    np.savez(results_dir / f"{index}.npz", original=original, preview=original)
crop_json = root / "crop.json"
write_json(crop_json, {"preprocessing": {"center_crop": {"width": 80, "height": 48}}})
(root / "fixture.json").write_text(json.dumps({"image": str(image), "defect_folder": str(defect_folder),
    "project": ProjectManager.get_active_filepath(project), "crop_json": str(crop_json)}))
uvicorn.run(create_app(root / "state"), host="127.0.0.1", port=8766, access_log=False)
