"""공식 ImageNet 가중치 다운로드부터 PatchCore 저장/오프라인 복원까지 검사."""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from patchcore import PatchCore
from patchcore_data import PatchCoreDataset


def check_backbone(backbone_name):
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory, patch("torch.hub.get_dir") as hub:
        root = Path(directory)
        hub.return_value = str(root / "fresh-torch-cache")
        # The app now requires bundled weights. Prepare a verified temporary
        # bundle here; runtime inference must still work without downloads.
        from torchvision import models
        from builtin_assets import ASSET_FILES
        from model_download import cached_imagenet_weights
        from patchcore_weights import BACKBONES
        weights = getattr(models, BACKBONES[backbone_name][2]).IMAGENET1K_V1
        relative = ASSET_FILES[backbone_name]
        path = cached_imagenet_weights(weights.url, (root / relative).parent)
        manifest = {"schema_version": 1, "files": {relative: {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}}}
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        images = []
        rng = np.random.default_rng(19)
        for index in range(3):
            path = root / f"normal_{index}.png"
            Image.fromarray(rng.integers(50, 180, (64, 64, 3), dtype=np.uint8)).save(path)
            images.append(str(path))
        with patch.dict(os.environ, {"DVS_BUILTIN_ASSETS_DIR": str(root)}):
            pc = PatchCore(backbone_name=backbone_name, device="cpu", pretrained=True,
                           input_size=64, max_candidates=128, max_memory_bank=16, sampling_ratio=.1)
        pc.fit(DataLoader(PatchCoreDataset(images, 64), batch_size=2))
        assert pc.weight_source["kind"] == "imagenet"
        assert pc.memory_bank.shape == (16, pc.backbone.feature_dim)
        scores, maps = pc.predict_from_path(images[0])
        assert np.isfinite(scores).all() and np.isfinite(maps).all()
        pc.save(root / "best.pt")
        with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("offline restore")), \
                patch("model_download.download_context", side_effect=AssertionError("offline restore")):
            loaded = PatchCore.load(root / "best.pt", device="cpu")
        restored = loaded.predict_from_path(images[0])
        np.testing.assert_allclose(restored[0], scores, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(restored[1], maps, atol=1e-5, rtol=1e-5)
        print(f"PASS: official {backbone_name} ImageNet weights -> verified native TLS download -> fit -> predict -> save -> offline load -> identical results")


def main():
    for name in ("resnet18", "resnet50", "wide_resnet50_2"):
        check_backbone(name)


if __name__ == "__main__":
    main()
