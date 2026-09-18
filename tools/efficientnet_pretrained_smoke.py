"""실제 공식 ImageNet 가중치로 B0/B1 레이어와 학습 경로를 검증한다."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "gui")]


def verify(architecture, device):
    import torch
    from torchvision import models
    from efficientnet import EfficientNet, VARIANTS, load_imagenet
    from model_download import cached_imagenet_weights
    spec = VARIANTS[architecture]
    path = cached_imagenet_weights(spec.url, Path(torch.hub.get_dir()) / "checkpoints")
    state = torch.load(path, map_location="cpu", weights_only=True)
    reference = getattr(models, architecture)(weights=None).to(device).eval()
    model = EfficientNet(architecture, in_channels=3).to(device).eval()
    reference.load_state_dict(state, strict=True)
    model.load_state_dict(state, strict=True)
    expected, actual, handles = {}, {}, []
    for network, target in [(reference, expected), (model, actual)]:
        for name, module in network.named_modules():
            if name.startswith("features.") and not list(module.children()):
                def capture(_module, _arguments, output, key=name, table=target):
                    table[key] = output.detach().cpu().clone()
                handles.append(module.register_forward_hook(capture))
    try:
        generator = torch.Generator().manual_seed(53)
        image = torch.randn(1, 3, spec.input_size, spec.input_size, generator=generator).to(device)
        with torch.no_grad():
            expected_logits, actual_logits = reference(image), model(image)
        if set(actual) != set(expected):
            raise AssertionError("공식 모델과 레이어 이름 불일치")
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-4, atol=1e-5, msg=lambda message: name + ": " + message)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-4, atol=1e-5)
        max_error = max((actual[name] - expected[name]).abs().max().item() for name in actual)
    finally:
        for handle in handles:
            handle.remove()
    classifier = EfficientNet(architecture, num_classes=2, in_channels=3).to(device).train()
    provenance = load_imagenet(classifier)
    before = classifier.features[0][0].weight.detach().clone()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=.001)
    torch.manual_seed(71)
    samples = torch.randn(2, 3, 64, 64, device=device)
    loss = torch.nn.functional.cross_entropy(classifier(samples), torch.tensor([0, 1], device=device))
    if not torch.isfinite(loss):
        raise AssertionError("사전학습 모델의 학습 손실 오류")
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in classifier.parameters()):
        raise AssertionError("유한하지 않은 gradient")
    optimizer.step()
    if torch.equal(before, classifier.features[0][0].weight):
        raise AssertionError("사전학습 백본 가중치가 갱신되지 않음")
    gray_report = verify_gray(architecture, device, path)
    return {"architecture": architecture, "weight_id": provenance["weight_id"],
            "state_tensors": len(state), "compared_layers": len(actual),
            "max_layer_absolute_error": max_error,
            "max_logit_absolute_error": (actual_logits - expected_logits).abs().max().item(),
            "backbone_updated": True, "loss": loss.item(), "native_gray": gray_report}


def verify_gray(architecture, device, weights_path):
    import torch
    from torchvision import models
    from efficientnet import EfficientNet, load_imagenet
    model = EfficientNet(architecture, num_classes=2, in_channels=1).to(device).eval()
    provenance = load_imagenet(model, weights_path=weights_path)
    if model.features[0][0].in_channels != 1 or hasattr(model, "rgb_mean"):
        raise AssertionError("새 흑백 모델이 내부 RGB 경로 사용")
    reference = getattr(models, architecture)(weights=None, num_classes=2)
    reference.features[0][0] = torch.nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False)
    reference.load_state_dict(model.state_dict(), strict=True)
    reference = reference.to(device).eval()
    gray = torch.randn(2, 1, 65, 79, device=device)
    with torch.no_grad():
        expected = reference(gray)
        actual = model(gray)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    model.train()
    before = model.features[0][0].weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    loss = torch.nn.functional.cross_entropy(model(gray), torch.tensor([0, 1], device=device))
    if not torch.isfinite(loss):
        raise AssertionError("1채널 사전학습 모델의 손실 오류")
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise AssertionError("1채널 모델의 유한하지 않은 gradient")
    optimizer.step()
    if torch.equal(before, model.features[0][0].weight):
        raise AssertionError("1채널 첫 Conv가 갱신되지 않음")
    return {"input_channels": 1, "stem_in_channels": 1, "stem_adaptation": provenance["stem_adaptation"],
            "reference": "torchvision_with_matching_one_channel_stem",
            "max_logit_absolute_error": float((actual - expected).abs().max()),
            "backbone_updated": True, "loss": float(loss.detach())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="efficientnet-validation.json")
    args = parser.parse_args()
    import torch
    import torchvision
    torch.set_num_threads(2)
    results = [verify(name, torch.device(args.device)) for name in ["efficientnet_b0", "efficientnet_b1"]]
    report = {"device": args.device, "torch": torch.__version__, "torchvision": torchvision.__version__, "results": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
