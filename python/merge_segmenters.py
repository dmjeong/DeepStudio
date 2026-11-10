"""두 분할 모델(칩 상태 A + 표면 결함 B)을 ONNX 파일 하나로 합친다. 설비는 Segment를 한 번만 부른다.

    판정 영역(원본) ─┬─▶ 평균 풀링(배율 s_A) ─▶ 32 배수로 0 채움 ─▶ A ─▶ 확률 ─▶ 자르고 nearest 확대 ─┐
                     └─▶ 평균 풀링(배율 s_B) ─▶ 32 배수로 0 채움 ─▶ B ─▶ 확률 ─▶ 자르고 nearest 확대 ─┤
                                                                                       ▼
                     B가 결함이라고 본 픽셀은 B의 확률, 나머지 픽셀은 A의 확률 ─▶ log ─▶ 출력

출력 클래스 = [배경, A의 결함 클래스…, B의 결함 클래스…]. softmax(log p) = p 이므로
    · 픽셀의 클래스 = 표면 결함이 있으면 B의 판정, 없으면 A의 판정 (표면 결함이 칩 상태보다 우선)
    · 덩어리 점수 = 그 판정을 낸 모델의 확률 그대로 → 모델마다 정한 판정 기준을 그대로 옮겨 쓴다.
두 모델 모두 패치 학습(판정 영역 한 번에) 모델이어야 하고, 배율은 1, 1/2, 1/4 …(정수배 축소)여야 한다.
모델마다 데스크톱 엔진과 같은 크기로 채운다(줄인 영역을 32의 배수로). 채움 크기가 다르면 BN 뒤 가장자리 값이 달라진다.
정수배 평균 풀링은 OpenCV INTER_AREA와 같은 평균이다(INTER_AREA는 결과를 8비트로 반올림해 최대 0.5/255 차이).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

EPSILON = 1e-30  # log(0) 대신 (-69): 다른 모델 쪽 클래스가 절대 이기지 못한다


def _load(path):
    import torch
    from export_onnx import load_custom_model, resolve_checkpoint_spec
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    spec = resolve_checkpoint_spec(checkpoint)
    if spec["task"] != "segment":
        raise ValueError(f"분할 모델만 합칠 수 있습니다: {path}")
    tiling = spec["preprocessing"].get("tiling")
    if not tiling:
        raise ValueError(f"패치 학습(판정 영역 한 번에) 모델만 합칠 수 있습니다: {path}")
    model = load_custom_model(checkpoint, spec).cpu().eval()
    if getattr(model, "global_attention", False):
        raise ValueError(f"ViT 패치 모델은 타일 추론이 필요해 합칠 수 없습니다: {path}")
    scale = float(tiling.get("scale", 1.0))
    factor = round(1 / scale)
    if not math.isclose(factor * scale, 1.0, abs_tol=1e-6):
        raise ValueError(f"배율 {scale:.0%}은 정수배 축소가 아니라 합칠 수 없습니다 (100%, 50%, 25%만): {path}")
    return {"path": Path(path), "model": model, "spec": spec, "factor": factor, "region": tiling.get("region")}


class MergedSegmenter:
    """torch.nn.Module을 늦게 만든다 (이 모듈을 불러오기만 할 때 torch가 필요 없도록)."""

    def __new__(cls, model_a, factor_a, model_b, factor_b, region):
        import torch
        from torch import nn
        import torch.nn.functional as F

        class _Merged(nn.Module):
            def __init__(self):
                super().__init__()
                self.a, self.b = model_a, model_b
                self.factors = (int(factor_a), int(factor_b))
                self.region = (int(region[0]), int(region[1]))

            def _probabilities(self, model, x, factor):
                width, height = self.region[0] // factor, self.region[1] // factor
                if factor > 1:
                    x = F.avg_pool2d(x, factor)  # 정수배 INTER_AREA
                # 데스크톱 엔진과 같은 채움: 줄인 판정 영역을 32의 배수로 (오른쪽·아래 0)
                x = F.pad(x, (0, -(-width // 32) * 32 - width, 0, -(-height // 32) * 32 - height))
                p = torch.softmax(model(x), dim=1)[..., :height, :width]
                if factor > 1:
                    p = p.repeat_interleave(factor, dim=2).repeat_interleave(factor, dim=3)  # 정수배 nearest
                return p

            def forward(self, x):
                pa = self._probabilities(self.a, x, self.factors[0])
                pb = self._probabilities(self.b, x, self.factors[1])
                surface = pb.argmax(dim=1, keepdim=True) > 0          # B가 결함이라고 본 픽셀
                zeros_a = torch.zeros_like(pa[:, 1:])
                zeros_b = torch.zeros_like(pb[:, 1:])
                from_a = torch.cat((pa[:, :1], pa[:, 1:], zeros_b), dim=1)
                from_b = torch.cat((pb[:, :1], zeros_a, pb[:, 1:]), dim=1)
                return torch.log(torch.where(surface, from_b, from_a).clamp_min(EPSILON))

        return _Merged().eval()


def _merged_rules(parts, class_names):
    """두 모델 옆 판정 기준(best.rules.json)을 합친 모델의 클래스 번호로 옮긴다. 둘 다 없으면 None."""
    from export_onnx import deployment_defect_rules
    found = [deployment_defect_rules(part["path"], list(part["spec"]["class_names"])) for part in parts]
    if not any(found):
        return None
    classes = []
    for part, rules in zip(parts, found):
        names = list(part["spec"]["class_names"])
        base = rules or {"min_area": 1, "min_score": 0.0, "classes": []}
        explicit = {c["class_name"]: c for c in base["classes"]}
        for name in names[1:]:  # 모델마다 기본 기준이 다를 수 있어 클래스마다 명시한다
            rule = explicit.get(name, base)
            classes.append({"class_id": class_names.index(name), "class_name": name,
                            "min_area": int(rule["min_area"]), "min_score": float(rule["min_score"])})
    return {"min_area": 1, "min_score": 0.0, "classes": classes}


def export_merged(checkpoint_a, checkpoint_b, output_path, *, region=None, opset_version=17, verify=True, log=print):
    """A(칩 상태) + B(표면 결함)를 ONNX 하나와 배포 JSON으로. B의 결함이 A보다 우선한다."""
    import numpy as np
    import onnx
    import torch
    from export_onnx import create_inference_config, export_to_onnx
    output = Path(output_path).resolve()
    if output.suffix.lower() != ".onnx":
        raise ValueError("출력 파일 확장자는 .onnx여야 합니다.")
    a, b = _load(checkpoint_a), _load(checkpoint_b)
    parts = (a, b)
    spec_a, spec_b = a["spec"], b["spec"]
    for key in ("in_channels",):
        if spec_a[key] != spec_b[key]:
            raise ValueError("두 모델의 입력 채널이 다릅니다 (둘 다 컬러 또는 둘 다 흑백으로 학습하세요).")
    norm = [(p["spec"]["preprocessing"].get("normalize_mean", p["spec"]["preprocessing"].get("mean")),
             p["spec"]["preprocessing"].get("normalize_std", p["spec"]["preprocessing"].get("std"))) for p in parts]
    if norm[0] != norm[1]:
        raise ValueError("두 모델의 입력 정규화가 다릅니다.")
    names_a, names_b = list(spec_a["class_names"]), list(spec_b["class_names"])
    if not names_a or not names_b:
        raise ValueError("두 모델 모두 클래스 이름이 있어야 합니다.")
    class_names = [names_a[0], *names_a[1:], *names_b[1:]]
    if len(set(class_names)) != len(class_names):
        raise ValueError("두 모델의 결함 클래스 이름이 겹칩니다. 이름을 다르게 학습하세요: "
                         + ", ".join(sorted({n for n in class_names if class_names.count(n) > 1})))
    region = region or a["region"] or b["region"]
    if not region:
        raise ValueError("판정 영역 크기(가로×세로 px)를 정하세요.")
    width, height = (int(v) for v in region)
    factor = max(a["factor"], b["factor"])
    if width % factor or height % factor:
        raise ValueError(f"판정 영역 {width}×{height}가 축소 배수 {factor}로 나누어떨어져야 합니다.")
    input_h, input_w = height, width  # 채움은 그래프 안에서 모델마다 한다 (입력·출력 = 판정 영역)
    model = MergedSegmenter(a["model"], a["factor"], b["model"], b["factor"], (width, height))
    channels = spec_a["in_channels"]
    generator = torch.Generator().manual_seed(42)
    dummy = torch.randn(1, channels, input_h, input_w, generator=generator)
    log(f"합치기: A {Path(checkpoint_a).name} (배율 1/{a['factor']}) + B {Path(checkpoint_b).name} (배율 1/{b['factor']})"
        f" → 클래스 {class_names}, 고정 입력 {input_w}×{input_h} (판정 영역 {width}×{height})")
    output.parent.mkdir(parents=True, exist_ok=True)
    export_to_onnx(model, dummy, output, opset_version, False, "segment")
    onnx.checker.check_model(str(output))
    if verify:
        import onnxruntime as ort
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            expected = model(dummy).numpy()
        actual = session.run(None, {session.get_inputs()[0].name: dummy.numpy()})[0]
        same = expected.argmax(1) == actual.argmax(1)
        agreement = float(same.mean())
        close = np.abs(np.exp(expected) - np.exp(actual))[np.repeat(same[:, None], expected.shape[1], 1)].max()
        log(f"ONNX 검증: 픽셀 판정 일치 {agreement:.4%}, 확률 최대 차이 {close:.2e}")
        if agreement < 0.9999 or close > 1e-3:
            output.unlink(missing_ok=True)
            raise ValueError("합친 ONNX가 PyTorch 결과와 다릅니다.")
    preprocessing = {key: value for key, value in spec_a["preprocessing"].items() if key != "tiling"}
    if norm[0][0] is not None and norm[0][1] is not None:
        preprocessing["normalize_mean"], preprocessing["normalize_std"] = list(norm[0][0]), list(norm[0][1])
    preprocessing["whole_image"] = {"region": [width, height], "scaled": [width, height],
                                    "interpolation": "INTER_LINEAR_EXACT", "pad": "zero_after_normalize",
                                    "input": [input_h, input_w]}
    config_path = Path(create_inference_config(output.parent, "segment", len(class_names), [input_h, input_w],
                                               channels, output.name, class_names, preprocessing, backend="custom",
                                               verification="passed" if verify else "skipped",
                                               config_filename=output.with_suffix(".json").name))
    manifest = json.loads(config_path.read_text(encoding="utf-8"))
    manifest["merged"] = {"surface_over_state": True,
                          "models": [{"checkpoint": str(Path(p["path"]).name), "classes": list(p["spec"]["class_names"]),
                                      "scale": 1 / p["factor"]} for p in parts]}
    rules = _merged_rules(parts, class_names)
    if rules is not None:
        manifest.setdefault("postprocessing", {})["defect_rules"] = rules
        log("판정 기준 포함: " + " | ".join(f"{c['class_name']} 면적 ≥ {c['min_area']}px · 점수 ≥ {c['min_score']:.2f}"
                                          for c in rules["classes"]))
    config_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output_path": str(output), "config_path": str(config_path), "class_names": class_names,
            "input_size": [input_h, input_w], "rules": rules}
