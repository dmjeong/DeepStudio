"""DINOv2 ViT 백본 (Meta, Apache-2.0). 공식 체크포인트와 같은 구조·같은 키 이름이라 그대로 strict 로드한다.

공식 저장소(facebookresearch/dinov2)의 hub 백본과 같은 설정:
    패치 14px · LayerScale · LayerNorm eps 1e-6 · GELU MLP(4배) · 위치 임베딩 37×37(+CLS, 518px 학습)
    다른 입력 크기의 위치 임베딩은 bicubic 배율 보간(interpolate_offset 0.1, antialias 없음)

                 ┌────────────── x_norm(CLS) ──▶ 분류
    이미지 ─▶ 14×14 패치 ─▶ +위치 ─▶ 블록 ×12 ─▶ LayerNorm
                 └── 중간 블록 출력(LayerNorm) ──▶ 분할 디코더

외부 패키지(timm·transformers·xformers)나 torch.hub 코드 내려받기 없이 동작한다(오프라인 설치본).
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

PATCH_SIZE = 14
PRETRAIN_GRID = 37  # 518 / 14: 공식 위치 임베딩 격자
INTERPOLATE_OFFSET = 0.1  # 공식 hub 백본 기본값 (부동소수 오차 회피용 옛 보정, 수치를 맞추려면 같아야 한다)

# 변형별 구조. 가중치 파일은 builtin_assets의 dinov2_vits14 / dinov2_vitb14.
VARIANTS = {
    "vits14": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "vitb14": {"embed_dim": 768, "depth": 12, "num_heads": 12},
}


class _PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, H0·W0, D), 행 우선


class _Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads)
        q, k, v = qkv.unbind(2)
        # PyTorch SDPA: CPU·GPU 모두 메모리 효율 커널 (토큰 수² 주의 행렬을 한꺼번에 만들지 않음)
        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.proj(x.transpose(1, 2).reshape(batch, tokens, dim))


class _Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _LayerScale(nn.Module):
    def __init__(self, dim, value=1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(value)))

    def forward(self, x):
        return x * self.gamma


class _Block(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, num_heads)
        self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, dim * 4)
        self.ls2 = _LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoV2ViT(nn.Module):
    """DINOv2 ViT. forward는 (CLS 토큰, 패치 토큰, (H0, W0), 중간 블록 패치 토큰 목록)을 돌려준다.

    입력이 14의 배수가 아니면 오른쪽·아래를 0으로 채운다(정규화 뒤 0 = 데이터 평균 밝기). H0, W0는 채운 뒤 격자.
    """

    def __init__(self, embed_dim, depth, num_heads, in_channels=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = _PatchEmbed(in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + PRETRAIN_GRID * PRETRAIN_GRID, embed_dim))
        # 공식 체크포인트에 있는 마스크 토큰(자기지도 학습용). 추론·미세조정에서는 쓰지 않지만 strict 로드를 위해 둔다.
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.blocks = nn.ModuleList(_Block(embed_dim, num_heads) for _ in range(depth))
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self._init_random()

    def _init_random(self):
        """가중치 없이 시작할 때(처음부터 학습)의 초기값. 공식 체크포인트를 불러오면 모두 덮인다."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def _position(self, rows, cols, dtype):
        """(1, 1 + rows·cols, D) 위치 임베딩. 공식 interpolate_pos_encoding과 같은 계산."""
        if rows == cols == PRETRAIN_GRID:
            return self.pos_embed  # 공식 구현도 학습 격자 그대로면 보간하지 않는다
        embed = self.pos_embed.float()
        patches = embed[:, 1:].reshape(1, PRETRAIN_GRID, PRETRAIN_GRID, self.embed_dim).permute(0, 3, 1, 2)
        patches = F.interpolate(
            patches, mode="bicubic", antialias=False,
            scale_factor=((rows + INTERPOLATE_OFFSET) / PRETRAIN_GRID, (cols + INTERPOLATE_OFFSET) / PRETRAIN_GRID))
        patches = patches.permute(0, 2, 3, 1).reshape(1, rows * cols, self.embed_dim)
        return torch.cat((embed[:, :1], patches), dim=1).to(dtype)

    def forward(self, x, layers=()):
        # 정수로 고정: ONNX 추출(트레이싱) 때도 위치 임베딩 보간이 상수 배율이 된다 (배포 그래프는 고정 입력 크기)
        height, width = int(x.shape[-2]), int(x.shape[-1])
        pad_bottom, pad_right = (-height) % PATCH_SIZE, (-width) % PATCH_SIZE
        if pad_bottom or pad_right:
            x = F.pad(x, (0, pad_right, 0, pad_bottom))
        rows, cols = (height + pad_bottom) // PATCH_SIZE, (width + pad_right) // PATCH_SIZE
        tokens = self.patch_embed(x)
        tokens = torch.cat((self.cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)
        tokens = tokens + self._position(rows, cols, tokens.dtype)
        intermediate = []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index in layers:
                intermediate.append(self.norm(tokens)[:, 1:])  # 공식 get_intermediate_layers(norm=True)
        tokens = self.norm(tokens)
        return tokens[:, 0], tokens[:, 1:], (rows, cols), intermediate


def tokens_to_map(tokens, grid):
    """(B, H0·W0, D) 패치 토큰 → (B, D, H0, W0) 특징 지도."""
    rows, cols = grid
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], rows, cols)


def load_pretrained(backbone, path):
    """공식 체크포인트(또는 같은 키의 state_dict)를 strict로 불러온다. 구조가 다르면 ValueError."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "teacher"):
            if isinstance(payload.get(key), dict) and "pos_embed" not in payload:
                state = payload[key]
                break
    try:
        backbone.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"DINOv2 백본과 가중치 구조가 일치하지 않습니다(변형 ViT-S/14·ViT-B/14 확인): {path}") from exc
    if not math.isfinite(float(backbone.pos_embed.detach().abs().sum())):
        raise ValueError(f"DINOv2 가중치에 유한하지 않은 값이 있습니다: {path}")
