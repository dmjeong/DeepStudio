"""Portable high precision fallback for numerically sensitive EfficientNet exports.

ORT CPU does not implement double Conv/GlobalAveragePool. Lower them to standard
ONNX unfold/MatMul and ReduceMean operations instead. This is a last resort, not
a latency optimization. The original FP32 checkpoint remains the acceptance
reference; input/output are FP32 for the existing native SDK contract.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F


class _DoubleConv(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.weight = conv.weight
        self.bias = conv.bias
        self.kernel, self.stride = conv.kernel_size, conv.stride
        self.padding, self.dilation = conv.padding, conv.dilation
        self.groups, self.out_channels = conv.groups, conv.out_channels
        self.in_channels = conv.in_channels

    def forward(self, x):
        columns = F.unfold(x, self.kernel, self.dilation, self.padding, self.stride)
        columns = columns.reshape(x.shape[0], self.groups, -1, columns.shape[-1])
        weights = self.weight.reshape(1, self.groups, self.out_channels // self.groups, -1)
        result = torch.matmul(weights, columns)
        height = (x.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel[0] - 1) - 1) // self.stride[0] + 1
        width = (x.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel[1] - 1) - 1) // self.stride[1] + 1
        result = result.reshape(x.shape[0], self.out_channels, height, width)
        return result if self.bias is None else result + self.bias.reshape(1, -1, 1, 1)


class _DoubleMean(nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3), keepdim=True)


class _PrecisionExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.inference_optimization = {
            "fallback": "portable_fp64", "compute_precision": "float64",
            "io_precision": "float32", "conv_bn_fused": 0,
            "constant_folding": False, "simplified": False,
            "latency_optimized": False,
        }

    def checkpoint_config(self):
        return self.model.checkpoint_config()

    def forward(self, images):
        return self.model(images.double()).float()


def prepare_precision_export(model):
    from efficientnet import EfficientNet
    if not isinstance(model, EfficientNet):
        raise TypeError("EfficientNet 모델 필요")
    candidate = copy.deepcopy(model).cpu().double().eval()
    for module in list(candidate.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                setattr(module, name, _DoubleConv(child))
            elif isinstance(child, nn.AdaptiveAvgPool2d):
                if child.output_size not in (1, (1, 1)):
                    raise ValueError("고정밀 내보내기는 global average pooling만 지원합니다")
                setattr(module, name, _DoubleMean())
    return _PrecisionExport(candidate).eval()
