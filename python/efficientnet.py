"""레이어별 디버깅을 위한 EfficientNet V1 B0/B1.

Torchvision의 공개 B0/B1 구조와 state_dict 이름을 따른다. 모델 생성에는
torchvision 모델 팩터리를 사용하지 않는다. Conv, BN, 활성화, SE와 잔차의
중간값은 각 forward의 지역 변수에서 확인할 수 있다.
참조: https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/efficientnet.py
"""

from dataclasses import dataclass
import copy
import math
from pathlib import Path

import torch
from torch import nn
from efficientnet_contract import NATIVE_INPUT, LEGACY_GRAY_INPUT, IMPLEMENTATION_VERSION, adapt_rgb_stem


@dataclass(frozen=True)
class Variant:
    depth: float
    input_size: int
    weight_id: str
    url: str


VARIANTS = {
    "efficientnet_b0": Variant(1.0, 224, "IMAGENET1K_V1",
        "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth"),
    "efficientnet_b1": Variant(1.1, 240, "IMAGENET1K_V2",
        "https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth"),
}
# 확장 배수, 커널, 첫 블록 stride, 입력 채널, 출력 채널, B0 반복 횟수
STAGES = ((1, 3, 1, 32, 16, 1), (6, 3, 2, 16, 24, 2),
          (6, 5, 2, 24, 40, 2), (6, 3, 2, 40, 80, 3),
          (6, 5, 1, 80, 112, 3), (6, 5, 2, 112, 192, 4),
          (6, 3, 1, 192, 320, 1))


class ConvNormActivation(nn.Sequential):
    """숫자 모듈 이름은 공식 가중치 호환을 위해 유지한다."""

    def __init__(self, in_channels, out_channels, kernel=3, stride=1,
                 groups=1, activate=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel, stride,
                           padding=kernel // 2, groups=groups, bias=False),
                  nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)]
        if activate:
            layers.append(nn.SiLU(inplace=False))
        super().__init__(*layers)

    def forward(self, x):
        conv_output = self[0](x)
        normalized = self[1](conv_output)
        activated = self[2](normalized) if len(self) == 3 else normalized
        return activated


class SqueezeExcitation(nn.Module):
    """공간 평균에서 채널별 게이트를 계산한다."""

    def __init__(self, channels, squeeze_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, channels, 1)
        self.activation = nn.SiLU(inplace=False)
        self.scale_activation = nn.Sigmoid()

    def forward(self, x):
        pooled = self.avgpool(x)
        squeezed = self.fc1(pooled)
        activated = self.activation(squeezed)
        expanded = self.fc2(activated)
        gate = self.scale_activation(expanded)
        scaled = x * gate
        return scaled


class StochasticDepth(nn.Module):
    """학습 중 표본별 잔차 경로를 확률적으로 생략한다."""

    def __init__(self, probability):
        super().__init__()
        if not 0 <= probability <= 1:
            raise ValueError("Stochastic depth 확률 범위 오류")
        self.p = float(probability)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        survival = 1.0 - self.p
        mask = torch.empty([x.shape[0]] + [1] * (x.ndim - 1),
                           dtype=x.dtype, device=x.device)
        mask.bernoulli_(survival)
        if survival > 0:
            mask.div_(survival)
        return x * mask


class MBConv(nn.Module):
    """확장, depthwise, SE, projection과 잔차를 개별 줄로 실행한다."""

    def __init__(self, in_channels, out_channels, expand_ratio, kernel, stride,
                 drop_probability, debug_name):
        super().__init__()
        expanded_channels = in_channels * expand_ratio
        self.has_expansion = expanded_channels != in_channels
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.debug_name = debug_name
        layers = []
        if self.has_expansion:
            layers.append(ConvNormActivation(in_channels, expanded_channels, 1))
        layers.extend([
            ConvNormActivation(expanded_channels, expanded_channels, kernel,
                               stride, groups=expanded_channels),
            SqueezeExcitation(expanded_channels, max(1, in_channels // 4)),
            ConvNormActivation(expanded_channels, out_channels, 1, activate=False),
        ])
        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(drop_probability)

    def forward(self, x):
        residual = x
        offset = int(self.has_expansion)
        expanded = self.block[0](x) if self.has_expansion else x
        spatial = self.block[offset](expanded)
        attended = self.block[offset + 1](spatial)
        projected = self.block[offset + 2](attended)
        if self.use_res_connect:
            dropped = self.stochastic_depth(projected)
            output = residual + dropped
        else:
            output = projected
        return output


class Stage(nn.Sequential):
    def forward(self, x):
        for block_index, block in enumerate(self):
            # block.debug_name으로 조건부 중단점을 지정한다.
            x = block(x)
        return x


class EfficientNet(nn.Module):
    """입력은 NCHW이며 출력은 softmax 적용 전 클래스 점수다."""

    task = "classify"
    gradcam_target_layer = "features.8"

    def __init__(self, architecture="efficientnet_b0", num_classes=1000,
                 in_channels=1, dropout=0.2, stochastic_depth_prob=0.2,
                 *, input_adapter=NATIVE_INPUT):
        super().__init__()
        if architecture not in VARIANTS:
            raise ValueError(f"미지원 EfficientNet 모델: {architecture}")
        if type(in_channels) is not int or in_channels not in (1, 3) or type(num_classes) is not int or num_classes < 1:
            raise ValueError("입력 채널은 1/3, 클래스 수는 양수 필요")
        if input_adapter not in (NATIVE_INPUT, LEGACY_GRAY_INPUT) or (
                input_adapter == LEGACY_GRAY_INPUT and in_channels != 1):
            raise ValueError("EfficientNet 입력 변환 방식 오류")
        self.architecture = architecture
        self.in_channels = in_channels
        self.input_adapter = input_adapter
        self.dropout = dropout
        self.num_classes = num_classes
        spec = VARIANTS[architecture]
        repeats = [math.ceil(stage[-1] * spec.depth) for stage in STAGES]
        total_blocks = sum(repeats)
        block_id = 0
        stem_channels = 3 if input_adapter == LEGACY_GRAY_INPUT else in_channels
        features = [ConvNormActivation(stem_channels, 32, 3, 2)]
        for stage_index, (definition, repeat) in enumerate(zip(STAGES, repeats)):
            expansion, kernel, stride, in_ch, out_ch, _ = definition
            blocks = []
            for index in range(repeat):
                blocks.append(MBConv(in_ch if index == 0 else out_ch, out_ch,
                    expansion, kernel, stride if index == 0 else 1,
                    stochastic_depth_prob * block_id / total_blocks,
                    f"stage{stage_index + 1}.block{index}"))
                block_id += 1
            features.append(Stage(*blocks))
        features.append(ConvNormActivation(320, 1280, 1))
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
        # 버전 1 체크포인트만 이전 forward를 유지한다. 새 1채널 모델은 확장하지 않는다.
        if input_adapter == LEGACY_GRAY_INPUT:
            self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), persistent=False)
            self.register_buffer("rgb_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1), persistent=False)
        self._initialize()

    def checkpoint_config(self):
        return {"architecture": self.architecture, "dropout": self.dropout,
                "implementation_version": 1 if self.input_adapter == LEGACY_GRAY_INPUT else IMPLEMENTATION_VERSION,
                "in_channels": self.in_channels, "input_adapter": self.input_adapter,
                "stem_in_channels": self.features[0][0].in_channels}

    @property
    def backbone(self):
        return self.features

    @property
    def head(self):
        return self.classifier

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                bound = 1 / math.sqrt(module.out_features)
                nn.init.uniform_(module.weight, -bound, bound)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if self.input_adapter == LEGACY_GRAY_INPUT:
            grayscale = x * .226 + .449
            x = (grayscale - self.rgb_mean) / self.rgb_std
        stem_output = self.features[0](x)
        stage_output = stem_output
        for stage_index in range(1, 8):
            stage_output = self.features[stage_index](stage_output)
        final_features = self.features[8](stage_output)
        pooled = self.avgpool(final_features)
        flattened = torch.flatten(pooled, 1)
        regularized = self.classifier[0](flattened)
        logits = self.classifier[1](regularized)
        return logits

    def get_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        return {"total": total,
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "backbone": sum(p.numel() for p in self.backbone.parameters()),
                "head": sum(p.numel() for p in self.head.parameters()),
                "total_MB": total * 4 / 1024 ** 2}


def load_imagenet(model, *, weights_path=None):
    """1000개 클래스 원형에 엄격히 로드한 후 사용자 분류기만 새로 초기화한다."""
    from model_download import cached_imagenet_weights
    spec = VARIANTS[model.architecture]
    path = Path(weights_path) if weights_path else cached_imagenet_weights(
        spec.url, Path(torch.hub.get_dir()) / "checkpoints")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("ImageNet state_dict 형식 오류")
    # 원형을 별도로 구성해 분류기까지 빠짐없이 검증한다.
    reference_shape = EfficientNet(model.architecture, num_classes=1000, in_channels=3)
    reference_shape.load_state_dict(state, strict=True)
    target_state = dict(state)
    if model.features[0][0].in_channels == 1:
        # 동일하게 정규화한 gray를 3면에 복제한 Conv와 동등한 초기 가중치다.
        # 구형 모델의 채널별 ImageNet 정규화를 포함한 forward와는 구분한다.
        target_state["features.0.0.weight"] = adapt_rgb_stem(state["features.0.0.weight"].float())
    # 사용자의 클래스 의미가 ImageNet과 다르므로 수가 같아도 새 헤드를 유지한다.
    target_state["classifier.1.weight"] = model.classifier[1].weight.detach().cpu()
    target_state["classifier.1.bias"] = model.classifier[1].bias.detach().cpu()
    model.load_state_dict(target_state, strict=True)
    return {"architecture": model.architecture, "weight_id": "local_state_dict" if weights_path else spec.weight_id,
            "weight_url": None if weights_path else spec.url, "weight_path": str(path),
            "loaded_backbone_tensors": len(state) - 2, "classifier_reinitialized": True,
            "input_channels": model.in_channels, "stem_in_channels": model.features[0][0].in_channels,
            "stem_adaptation": "rgb_weight_sum" if model.features[0][0].in_channels == 1 else "unchanged"}


def prepare_for_inference(model, *, channels_last=False):
    """독립된 FP32 평가 복사본만 최적화한다. 학습 모델과 저장 가중치는 보존한다."""
    if not isinstance(model, EfficientNet):
        raise TypeError("EfficientNet 모델 필요")
    optimized = copy.deepcopy(model).float().eval()
    fused = 0
    for module in optimized.modules():
        if isinstance(module, ConvNormActivation) and isinstance(module[1], nn.BatchNorm2d):
            module[0] = torch.nn.utils.fuse_conv_bn_eval(module[0], module[1])
            module[1] = nn.Identity()
            fused += 1
    if channels_last:
        optimized.to(memory_format=torch.channels_last)
    optimized.inference_optimization = {"conv_bn_fused": fused, "channels_last": channels_last}
    return optimized
