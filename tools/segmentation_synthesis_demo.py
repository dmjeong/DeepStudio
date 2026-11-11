"""Exercise five defect classes at native tray size, using only invented data.

python tools/segmentation_synthesis_demo.py --output runs/seg-synthesis-demo
Creates a ready-to-open project, candidates, close-up montage and timings.
This is an engineering fixture, not evidence of inspection accuracy.
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT)]
from core.project import ProjectManager
from core.segmentation_synthesis_store import SynthesisStore

CLASSES = ["background", "crack", "foreign material", "shift", "double", "flip"]


def tray(index, defects):
    h, w = 1504, 3504
    image = np.full((h, w, 3), (48, 56, 61), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    image[:5, :30] = index + 10
    mask[:5] = 255
    sites = {}
    for row in range(8):
        for col in range(16):
            x, y = 85 + 210 * col, 55 + 175 * row
            cv2.rectangle(image, (x - 12, y - 8), (x + 142, y + 118), (70, 80, 88), -1)
            cv2.rectangle(image, (x, y), (x + 130, y + 110), (130, 142, 155), -1)
            for k in range(4):
                cv2.rectangle(
                    image,
                    (x - 10, y + 12 + k * 24),
                    (x + 4, y + 22 + k * 24),
                    (190, 180, 160),
                    -1,
                )
                cv2.rectangle(
                    image,
                    (x + 126, y + 12 + k * 24),
                    (x + 140, y + 22 + k * 24),
                    (190, 180, 160),
                    -1,
                )
            cv2.circle(image, (x + 20, y + 18), 5, (72, 80, 88), -1)
            sites[row, col] = x, y
    if defects:
        for cid, site in enumerate(((1, 2), (2, 5), (4, 3), (5, 9), (3, 13)), 1):
            x, y = sites[site]
            if cid == 1:
                points = np.array(
                    [
                        (x + 30, y + 20),
                        (x + 44, y + 35),
                        (x + 40, y + 58),
                        (x + 65, y + 75),
                        (x + 70, y + 95),
                    ],
                    np.int32,
                )
                cv2.polylines(image, [points], False, (25, 30, 34), 2)
                cv2.polylines(mask, [points], False, cid, 2)
            elif cid == 2:
                points = np.array(
                    [
                        (x + 30, y + 30),
                        (x + 50, y + 24),
                        (x + 65, y + 48),
                        (x + 40, y + 62),
                    ],
                    np.int32,
                )
                cv2.fillPoly(image, [points], (80, 50, 24))
                cv2.fillPoly(mask, [points], cid)
            elif cid == 3:
                image[y : y + 111, x : x + 131] = (70, 80, 88)
                cv2.rectangle(
                    image, (x + 24, y + 14), (x + 154, y + 124), (130, 142, 155), -1
                )
                cv2.circle(image, (x + 44, y + 32), 5, (72, 80, 88), -1)
                mask[y : y + 125, x : x + 155] = cid
            elif cid == 4:
                cv2.rectangle(
                    image, (x + 26, y + 18), (x + 156, y + 128), (152, 162, 173), -1
                )
                cv2.circle(image, (x + 46, y + 36), 5, (72, 80, 88), -1)
                mask[y + 18 : y + 129, x + 26 : x + 157] = cid
            else:
                cv2.rectangle(image, (x, y), (x + 130, y + 110), (95, 100, 115), -1)
                cv2.rectangle(
                    image, (x + 35, y + 25), (x + 95, y + 85), (170, 165, 148), -1
                )
                mask[y : y + 111, x : x + 131] = cid
    return image, mask


def run(output):
    if (output / "project").exists():
        output = output / ("run-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=True)
    project = ProjectManager.create_new(
        "Synthetic-tray-demo", "segment", str(output / "project"), CLASSES
    )
    for split_index, split in enumerate(("train", "val", "test")):
        for i in range(2):
            image, mask = tray(split_index * 2 + i, i == 0)
            Image.fromarray(image).save(
                Path(getattr(project.data, split + "_dir")) / f"{i}.png"
            )
            Image.fromarray(mask).save(
                Path(project.data.root) / "masks" / split / f"{i}.png"
            )
    ProjectManager.save(project)
    store = SynthesisStore(project)
    start = time.perf_counter()
    profile = store.analyze()
    analysis_time = time.perf_counter() - start
    start = time.perf_counter()
    result = store.generate(
        profile["profile_id"], {str(i): 1 for i in range(1, 6)}, seed=7, tolerance=0.12
    )
    generation_time = time.perf_counter() - start
    batch = store.load("batches", result["batch_id"])
    directory = store.directory("batches", batch["id"])
    montage = Image.new("RGB", (1200, 5 * 260 + 40), (28, 30, 34))
    draw = ImageDraw.Draw(montage)
    draw.text(
        (12, 10),
        "SYNTHETIC ENGINEERING FIXTURE | native 3504 x 1504 | source / target / generated / semantic mask",
        fill="white",
    )
    for index, row in enumerate(batch["samples"]):
        x, y, w, h = row["recipe"]["box"]
        rect = (
            max(0, x - 8),
            max(0, y - 8),
            min(3504, x + w + 8),
            min(1504, y + h + 8),
        )
        for col, name in enumerate(("reference", "original", "image", "mask")):
            image = Image.open(store.file(directory, row[name]))
            if name != "reference":
                image = image.crop(rect)
            if name == "mask":
                pixels = np.asarray(image)
                colors = np.array(
                    [
                        [0, 0, 0],
                        [0, 220, 220],
                        [230, 160, 30],
                        [160, 100, 230],
                        [210, 70, 150],
                        [100, 200, 60],
                    ],
                    np.uint8,
                )
                image = Image.fromarray(colors[np.minimum(pixels, 5)])
            image = image.convert("RGB")
            image.thumbnail((290, 220))
            montage.paste(
                image, (col * 300 + (300 - image.width) // 2, 55 + index * 260)
            )
        draw.text(
            (12, 285 + index * 260),
            f"{CLASSES[row['class_id']]} | context error {row['recipe']['context_rmse']:.4f} | {row['recipe']['box']}",
            fill="white",
        )
    montage.save(output / "comparison.png")
    report = {
        "fixture": "invented data; not an accuracy benchmark",
        "output": str(output),
        "size": [3504, 1504],
        "analyze_seconds": analysis_time,
        "generate_seconds": generation_time,
        **result,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return project


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/seg-synthesis-demo"))
    run(parser.parse_args().output.resolve())
