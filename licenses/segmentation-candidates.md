# Segmentation candidates (13.84)

- timm 1.0.31: Apache-2.0, license copied as `timm-LICENSE.txt`.
  https://github.com/huggingface/pytorch-image-models
- MobileNetV4 Conv-S weights: timm/mobilenetv4_conv_small.e2400_r224_in1k,
  revision 331fb803779522b685cf942e15f914fb6741c1eb. Model card declares Apache-2.0.
  https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k
- EfficientNet-B1 weights: timm/efficientnet_b1.ra4_e3600_r240_in1k,
  revision 4e4a9c08eee4bdb118d69d27f87ace8fdf3fc743. Model card declares Apache-2.0.
  https://huggingface.co/timm/efficientnet_b1.ra4_e3600_r240_in1k
- ShuffleNetV2: torchvision BSD-3-Clause implementation, official IMAGENET1K_V1 file
  https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth.
  Torchvision's pretrained-model notice remains applicable:
  https://github.com/pytorch/vision#pre-trained-model-license
- BiSeNetV2 source: https://github.com/CoinCheung/BiSeNet/blob/master/lib/models/bisenetv2.py,
  MIT, copyright 2018 CoinCheung. Complete notice in `bisenet-LICENSE.txt`.
  Vendored original SHA256 is recorded below; automatic backbone download removed.
- STDC source: https://github.com/MichaelFan01/STDC-Seg (`models/model_stages.py`, `nets/stdcnet.py`),
  MIT. Complete notice in `stdc-LICENSE.txt`. Replaced CUDA-specific InPlaceABNSync with
  standard BatchNorm and global pooling with equivalent CPU/ONNX-compatible operations.

BiSeNetV2/STDC pretrained weights are **not bundled**. No claim is made that the repository's
code license automatically licenses all externally hosted Cityscapes/ImageNet checkpoints.
Users can train from scratch and transfer their own matching Studio checkpoints.

The STDC adapter uses BCE + Dice boundary supervision with class-neighbour disagreement
targets instead of a numeric-class-ID Laplacian. It ignores excluded pixels and their edges.
This is an explicit training-recipe adaptation, not a claim to reproduce published Cityscapes scores.

## Original source SHA256 (before local adaptation)

- `bisenetv2.py`: `c67cb04995daedee1de09dcacd432d0666962f4b201ee4b78b359856b41f2a44`
- `stdc_model.py`: `5decdf9f98c3192d66c7e9ad638c2f8ab5257b51d015010e15694e73bc28703a`
- `stdcnet.py`: `0bdbd2fe4341ce04893b4148b860864dbac1685f063d220c3d7d6f998f80ca44`
