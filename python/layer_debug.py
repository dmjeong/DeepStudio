"""선택한 레이어의 순전파와 역전파 관찰. 일반 학습에는 설치하지 않는다."""

from fnmatch import fnmatchcase
import torch


class LayerInspector:
    def __init__(self, patterns=("features.*",), keep_values=False, max_elements=262144, max_layers=None):
        self.patterns = tuple(patterns)
        self.max_layers = max_layers
        self.gradient_scale = 1.0
        self.keep_values = keep_values
        self.max_elements = max_elements
        self.records = {}
        self.values = {}
        self._handles = []
        self._gradient_handles = {}

    @staticmethod
    def summary(tensor):
        detached = tensor.detach().float()
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "device": str(tensor.device), "requires_grad": tensor.requires_grad,
                "finite": bool(torch.isfinite(detached).all().item()),
                "min": detached.min().item() if detached.numel() else None,
                "max": detached.max().item() if detached.numel() else None,
                "mean": detached.mean().item() if detached.numel() else None}

    def _capture(self, name, output):
        if not isinstance(output, torch.Tensor):
            return
        record = self.records.setdefault(name, {})
        if output.requires_grad:
            record["output"] = self.summary(output)
            record.pop("gradient", None)
        else:
            record["eval_output"] = self.summary(output)
        if self.keep_values and (output.requires_grad or name not in self.values):
            self.values[name] = output.detach().flatten()[:self.max_elements].cpu().clone()
        previous = self._gradient_handles.pop(name, None)
        if previous is not None:
            previous.remove()
        if output.requires_grad:
            def gradient_hook(gradient):
                self.records[name]["gradient"] = self.summary(gradient.detach().float() / self.gradient_scale)
            self._gradient_handles[name] = output.register_hook(gradient_hook)

    def attach(self, model):
        self.close()
        self.records.clear()
        self.values.clear()
        selected = [(name, module) for name, module in model.named_modules()
                    if name and any(fnmatchcase(name, pattern) for pattern in self.patterns)]
        if self.max_layers is not None and len(selected) > self.max_layers:
            raise ValueError(f"관찰 레이어 {len(selected)}개: {self.max_layers}개 이하로 패턴을 좁혀 주세요")
        for name, module in selected:
            if name:
                def forward_hook(_module, _inputs, output, layer_name=name):
                    self._capture(layer_name, output)
                    if layer_name in self.records:
                        self.records[layer_name]["input_shapes"] = [list(value.shape) for value in _inputs if isinstance(value, torch.Tensor)]
                self._handles.append(module.register_forward_hook(forward_hook))
        if not self._handles:
            raise ValueError("관찰할 레이어 이름 없음: patterns 확인 필요")
        return self

    def close(self):
        for handle in self._handles:
            handle.remove()
        for handle in self._gradient_handles.values():
            handle.remove()
        self._handles.clear()
        self._gradient_handles.clear()
