"""Portable high precision fallback for numerically sensitive EfficientNet exports.

ORT CPU does not implement double Conv/GlobalAveragePool. Lower them to standard
ONNX MatMul, Gather, and reduction operations instead. Specialized pointwise and
depthwise paths avoid the most expensive im2col work, but FP64 remains a last
resort. The original FP32 checkpoint remains the acceptance reference;
input/output are FP32 for the existing native SDK contract.
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
        if self.groups == self.in_channels == self.out_channels:
            return self._depthwise(x)
        # A pointwise convolution only mixes channels. im2col/unfold would copy
        # exactly the same pixels, generating hundreds of unnecessary shape and
        # Gather nodes across EfficientNet's expansion/projection/SE layers.
        if self.kernel == (1, 1) and self.padding == (0, 0):
            sampled = x[:, :, ::self.stride[0], ::self.stride[1]]
            columns = sampled.reshape(x.shape[0], self.in_channels, -1)
        else:
            columns = F.unfold(x, self.kernel, self.dilation, self.padding, self.stride)
        columns = columns.reshape(x.shape[0], self.groups, -1, columns.shape[-1])
        weights = self.weight.reshape(1, self.groups, self.out_channels // self.groups, -1)
        result = torch.matmul(weights, columns)
        height = (x.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel[0] - 1) - 1) // self.stride[0] + 1
        width = (x.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel[1] - 1) - 1) // self.stride[1] + 1
        if torch.onnx.is_in_onnx_export():
            # The deployment contract permits dynamic batch only; image height
            # and width are fixed. Bake these reshape dimensions into the graph.
            height, width = int(height), int(width)
        result = result.reshape(x.shape[0], self.out_channels, height, width)
        return result if self.bias is None else result + self.bias.reshape(1, -1, 1, 1)

    def _depthwise(self, x):
        # Depthwise convolution has no channel reduction. One Gather on the
        # flattened image avoids unfold's two Gather passes and full transpose;
        # a reduction replaces thousands of tiny per-channel MatMul calls.
        height = (x.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel[0] - 1) - 1) // self.stride[0] + 1
        width = (x.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel[1] - 1) - 1) // self.stride[1] + 1
        # Export supports dynamic batch with fixed spatial dimensions. These
        # integer indices are shape constants, never input data or batch indices.
        height, width = int(height), int(width)
        padded = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        padded_width = int(x.shape[3]) + 2 * self.padding[1]
        indices = []
        for row in range(self.kernel[0]):
            start_y = row * self.dilation[0]
            for col in range(self.kernel[1]):
                start_x = col * self.dilation[1]
                indices.append([(start_y + out_y * self.stride[0]) * padded_width + start_x + out_x * self.stride[1]
                    for out_y in range(height) for out_x in range(width)])
        indices = torch.tensor(indices, dtype=torch.int64, device=x.device)
        columns = padded.flatten(2)[:, :, indices]
        weights = self.weight.reshape(1, self.in_channels, -1, 1)
        result = (columns * weights).sum(dim=2).reshape(x.shape[0], self.out_channels, height, width)
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
            "convolution_lowering": "pointwise_matmul_depthwise_gather_reduce",
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
