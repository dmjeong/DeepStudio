"""Check real ImageNet initialization, a training step, offline restore and ONNX parity."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "gui")]


def main():
    import torch
    from builtin_models import BUILTIN_MODEL_SPECS, build_builtin_model, load_builtin_checkpoint, make_builtin_checkpoint
    from export_onnx import export_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cache:
        torch.hub.set_dir(str(args.cache))
    torch.set_num_threads(2)
    reports = []
    for model_id, spec in BUILTIN_MODEL_SPECS.items():
        torch.manual_seed(61)
        model = build_builtin_model(model_id, 2, 3, pretrained=True)
        assert model.weight_provenance["source"] == "imagenet"
        first = next(model.parameters())
        before = first.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        images = torch.randn(2, 3, 64, 64)
        labels = torch.tensor([0, 1]) if spec.task == "classify" else torch.zeros(2, 64, 64, dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(model(images), labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        assert not torch.equal(first, before)
        model.eval()
        checkpoint = make_builtin_checkpoint(model_id, model, num_classes=2, class_names=["ok", "ng"])
        restored = load_builtin_checkpoint(checkpoint)
        with torch.inference_mode():
            torch.testing.assert_close(model(images[:1]), restored(images[:1]), rtol=0, atol=0)
        source = args.output / f"{model_id}.pt"
        torch.save(checkpoint, source)
        result = export_checkpoint(source, source.with_suffix(".onnx"), verify=True, log=lambda _: None)
        assert result["verification"] == "passed"
        reports.append({"model_id": model_id, "pretrained": model.weight_provenance,
                        "training_loss": float(loss.detach()), "backbone_updated": True,
                        "checkpoint_restore": "passed", "onnx": result["verification"],
                        "input_size": list(spec.default_size)})
        print(f"PASS {model_id}: ImageNet, train, restore, ONNX", flush=True)
        del model, restored, checkpoint, optimizer, images, first, loss, before
    (args.output / "report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
