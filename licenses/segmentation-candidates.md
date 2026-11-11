# Segmentation candidates (13.84, pretrained completion in 13.85)

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

## ImageNet backbones added in 13.85

- BiSeNetV2: `backbone_v2.pth`, published by CoinCheung in the MIT project's release 0.0.0.
  https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth
  SHA256: `c63b0eb92c8503232238c29c5b5e5091752860592ab1287d444612c4fe5d3d60`.
  The maintainer identifies ImageNet as its training dataset:
  https://github.com/CoinCheung/BiSeNet/issues/192#issuecomment-962935276 and
  https://github.com/CoinCheung/BiSeNet/issues/173#issuecomment-899207366.
  It contains detail/segment/bga backbone branches, not a Cityscapes segmentation head.
  This is the implementation maintainer's checkpoint, not an assertion that it is a checkpoint
  published by the original BiSeNetV2 paper authors.
- STDC1: `STDCNet813M_73.91.tar`, the paper authors' ImageNet checkpoint, linked by their
  MIT repository's README under training instructions.
  https://github.com/MichaelFan01/STDC-Seg#1-train-stdc-seg
  Official public Drive folder:
  https://drive.google.com/drive/folders/1wROFwRt8qWHD4jSo8Zu1gp1d6oYJ3ns1
  File ID: `1DFoXcV42zy-apUcMh5P8WhsXMRJofgl8`.
  SHA256: `5354c20179bfca3bd7c5f62b39fdeaecae90e913fd2ff0fa08a927fe3a3393b0`.

The build downloads these original files and includes the upstream MIT notices. Weight files
are not committed to Git. The archives do not contain a separate weight-specific license;
the above records their publishers, repository terms and dataset provenance, not a blanket
legal clearance of ImageNet data. No Cityscapes segmentation checkpoint is downloaded.

The STDC adapter uses BCE + Dice boundary supervision with class-neighbour disagreement
targets instead of a numeric-class-ID Laplacian. It ignores excluded pixels and their edges.
This is an explicit training-recipe adaptation, not a claim to reproduce published Cityscapes scores.

## Original source SHA256 (before local adaptation)

- `bisenetv2.py`: `c67cb04995daedee1de09dcacd432d0666962f4b201ee4b78b359856b41f2a44`
- `stdc_model.py`: `5decdf9f98c3192d66c7e9ad638c2f8ab5257b51d015010e15694e73bc28703a`
- `stdcnet.py`: `0bdbd2fe4341ce04893b4148b860864dbac1685f063d220c3d7d6f998f80ca44`
