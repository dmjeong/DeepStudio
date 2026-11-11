# Modern dense models: provenance and adaptations

The model names identify Studio deployment variants, not reproduced paper benchmarks.
Code licenses and pretrained-weight provenance are recorded separately. ImageNet data
terms are not replaced by a repository's code license. Release distributors must retain
these notices and review the exact upstream weight/model-card terms.

| Component | Pinned source | Code license | Studio changes |
|---|---|---|---|
| EfficientViT | `mit-han-lab/efficientvit@de7d7733cc0329f391b33f1f459271562ec27bd5`; timm 1.0.31 `efficientvit_mit` | Apache-2.0 | timm native backbone; multiscale additive/MBConv head; stride-one detail branch; gray adapter; CPU ONNX path |
| SCTNet | `xzz777/SCTNet@d4bd6d8073a59831d69f303fc5b39c70023e2719` | MIT | native PyTorch backbone, no mmcv registry; BatchNorm in place of SyncBN; Studio segmentation/detail and auxiliary head; pixel teacher distillation instead of the paper's feature distillation |
| EoMT | `tue-mps/eomt@aa017c34afbe023de9462b17ae6067b52fef9867` | MIT | DINOv2-S without registers, queries in final four blocks, mask-guided attention; full-pixel Hungarian matching instead of sampled points; semantic class masks, stride-one detail branch; no official mask-annealing schedule |
| Dinomaly | `guojiajeremy/Dinomaly@1f252be03a918789b19848f0ca37166e7c28dada` | Apache-2.0 | small non-register DINOv2 encoder, frozen layers 2..9, dropout bottleneck, eight linear-attention reconstruction blocks, grouped cosine maps and hard-example weighting; no Gaussian map smoothing; validation-normal maximum threshold |
| DINOv2 | Existing Studio backbone from `facebookresearch/dinov2` | Apache-2.0 | Standard PyTorch attention; right/bottom patch padding; RGB/BW8 input adaptation; strict official state dictionary loading |
| SciPy | `scipy.optimize.linear_sum_assignment` | BSD-3-Clause | Hungarian matching during EoMT training only; no C++ runtime dependency |

Complete source license texts are alongside this file: `EfficientViT-LICENSE.txt`,
`SCTNet-LICENSE.txt`, `EoMT-LICENSE.txt`, `Dinomaly-LICENSE.txt`, `timm-LICENSE.txt`,
and `DINOv2-APACHE-2.0.txt`. SciPy's installed distribution license must accompany a frozen release.

## Pretrained assets

Weights are downloaded only by the existing explicit build/preparation flow. They are not committed.
Each file is pinned by SHA256 in `python/builtin_assets.py`, checked before use, and its provenance
is included in the training checkpoint. Loading failure never silently substitutes random weights.

| Asset | Exact source | SHA256 |
|---|---|---|
| EfficientViT B0 | [timm model](https://huggingface.co/timm/efficientvit_b0.r224_in1k/tree/4844777adc2285905eeb022672ba81620b52bd0c), `pytorch_model.bin` | `92b1a783baee19335e0dbe5af941ad91bfce392b5e55f2dbbae1e4dbea0b6b4e` |
| EfficientViT B1 | [timm model](https://huggingface.co/timm/efficientvit_b1.r224_in1k/tree/cebc58510738e196c3cce3c3077dc8737973b7ba), `pytorch_model.bin` | `fcd84d10cb51762639cba7e5e3d936fb391777b654ab6e61701f14219d3d4a0f` |
| SCTNet-S backbone | Author's `SCT-S_Pretrain.pth`, Google Drive ID `1rCHGRBCUOOq8frQvMELzfvA-gk-wbedE`, linked from the MIT repository | `6711235bef62dedb3b76853bc278878027b66ca35b9f8afb9419708ced36b65a` |
| DINOv2-S (EoMT/Dinomaly shared) | [Meta original file](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |

EfficientViT's timm model cards declare Apache-2.0. Meta publishes DINOv2 code/weights under Apache-2.0.
SCTNet publishes the backbone via its MIT repository; the Google Drive binary has no independent embedded license.
Do not present this provenance record as a guarantee about every dataset right or downstream commercial use.

The segmentation heads, detail branches and Dinomaly reconstruction decoder are trained locally.
No claim of official pretrained defect heads, ImageNet benchmark reproduction, or unconditional CPU speedup is made.
