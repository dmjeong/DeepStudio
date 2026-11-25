"""Image-level prediction using saved tile geometry and aggregation."""
from types import SimpleNamespace

import numpy as np
import torch
from native_patches import detection_nms, scaled_image, tiles, validate_patches


def predict_patches(wrapper, image, contract, *, conf=.25, iou=.5):
    contract = validate_patches(contract)
    original = np.asarray(image.convert("RGB"))
    scaled = scaled_image(original, contract)
    h, w = scaled.shape[:2]
    if contract["task"] == "classify":
        from libreyolo_classification import FullFrameTransform
        transform = FullFrameTransform(contract["size"])
        all_logits, geometry = [], []
        batch, positions = [], []
        network = wrapper.model.eval()
        device = next(network.parameters()).device
        def run():
            with torch.inference_mode():
                all_logits.append(network(torch.stack(batch).to(device)).float().cpu())
            geometry.extend(positions)
            batch.clear()
            positions.clear()
        for tile, position in tiles(scaled, contract):
            batch.append(transform(tile)); positions.append(position)
            if len(batch) == contract["batch"]:
                run()
        if batch:
            run()
        logits = torch.cat(all_logits)
        if not torch.isfinite(logits).all():
            raise ValueError("패치 분류 출력에 NaN/Inf가 있습니다")
        combined, winners = logits.max(dim=0)
        top = int(combined.argmax())
        winner = geometry[int(winners[top])]
        return SimpleNamespace(probs=SimpleNamespace(data=combined.softmax(dim=0)),
                               studio_tiles=len(geometry), studio_winner=winner, studio_scaled=scaled)
    rows = []
    count = 0
    from PIL import Image
    for tile, (x, y, pw, ph) in tiles(scaled, contract):
        result = wrapper.predict(Image.fromarray(tile), imgsz=contract["size"], conf=conf, iou=iou,
                                 device=str(wrapper.device))
        result = result[0] if isinstance(result, (tuple, list)) else result
        count += 1
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        labels = result.boxes.cls.detach().cpu().numpy()
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = np.clip(box, 0, [pw, ph, pw, ph])
            if x2 <= x1 or y2 <= y1:
                continue
            rows.append({"class_id": int(label), "confidence": float(score),
                         "bbox": [(x1+x)/w, (y1+y)/h, (x2+x)/w, (y2+y)/h]})
    rows = detection_nms(rows, iou)
    return SimpleNamespace(boxes=SimpleNamespace(
        xyxyn=np.asarray([r["bbox"] for r in rows], np.float32).reshape(-1, 4),
        conf=np.asarray([r["confidence"] for r in rows]), cls=np.asarray([r["class_id"] for r in rows])),
        studio_tiles=count)
