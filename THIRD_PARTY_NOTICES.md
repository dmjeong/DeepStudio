# Third-party notices

Deep Vision Studio does not commit pretrained model weights to this source
repository. The Windows build fetches the declared basic-model weights into
the release payload. A release payload must include the exact source license
and notice for every dependency and weight it actually bundles; this file is
the source list for the payload review and is not a substitute for a
release-specific SBOM.

| Component | Use | Upstream license / source |
|---|---|---|
| PyTorch / torchvision | Python training and built-in model adapters | BSD-style license; https://github.com/pytorch/pytorch/blob/main/LICENSE |
| OpenCV | image conversion and C++ preprocessing | Apache-2.0; https://github.com/opencv/opencv/blob/4.x/LICENSE |
| ONNX Runtime | Python and C++ CPU inference | MIT; https://github.com/microsoft/onnxruntime/blob/main/LICENSE |
| nlohmann/json | C++ manifest parsing | MIT; https://github.com/nlohmann/json/blob/develop/LICENSE.MIT |
| WiX Toolset | Windows MSI/Burn build tool | MIT; https://github.com/wixtoolset/wix/blob/main/LICENSE |
| LibreYOLO 1.5.0 | 기본 LibreMobileNetV4 / LibreYOLO9 / Re-DETR v4 native runtime | MIT for the library; https://github.com/LibreYOLO/libreyolo/blob/release/LICENSE and the bundled third-party list at https://github.com/LibreYOLO/libreyolo/blob/release/NOTICE |
| RT-DETRv4 upstream (제품 표기 Re-DETR v4) | 기본 Re-DETR v4 소스 후보 | Apache-2.0; https://github.com/RT-DETRs/RT-DETRv4/blob/main/LICENSE |
| SAM2 1.0 (`2b90b9f`) | 기본 SAM2.1 Hiera runtime | Apache-2.0; https://github.com/facebookresearch/sam2/blob/main/LICENSE |
| Meta DINOv2 ViT-S/14, ViT-B/14 | Adapted backbone (`python/dinov2.py`) and official LVD-142M pretrained weights; classification and segmentation heads are trained locally | Apache-2.0; https://github.com/facebookresearch/dinov2/blob/main/LICENSE; bundled text: `licenses/DINOv2-APACHE-2.0.txt` |
| cryptography | Python AES-GCM model containers and model-pack signatures | Apache-2.0 OR BSD-3-Clause; https://github.com/pyca/cryptography/blob/main/LICENSE |
| OpenSSL 3 | Native encrypted-model loader on non-Windows builds | Apache-2.0; https://github.com/openssl/openssl/blob/openssl-3.0/LICENSE.txt |

The Windows native encrypted-model loader uses the OS-provided BCrypt API and
does not ship an OpenSSL DLL. Python cryptography wheels can contain OpenSSL;
the bundled wheel's applicable licenses must accompany a frozen Python release.

The LibreYOLO 1.5.0 and SAM2 1.0 runtimes are shipped dependencies of the Windows build.
Its distribution LICENSE and applicable NOTICE entries must be copied into the
installer payload with this notice. Re-DETRv4 remains a source candidate until
its source revision, license/NOTICE, Windows worker and ONNX acceptance tests
are all present. SAM2 pretrained weights are obtained only from Meta's published
`facebook/sam2.1-*` repositories during the controlled Windows build; they are
not committed to this repository. A user-added `.dvmodel` must carry its own `license` object, source
revision, asset hashes, and required notices before release signing.
Every bundled pretrained weight requires review against its exact model card
before redistribution.

## DINOv2 adaptation

Copyright (c) Meta Platforms, Inc. and affiliates. `python/dinov2.py` re-implements
the non-register ViT-S/14 and ViT-B/14 backbones of facebookresearch/dinov2 with
standard PyTorch operations and the same parameter names, so the official state
dictionaries load strictly. Changes: no xformers, training-only machinery or remote
loading; right/bottom padding to the 14-pixel patch grid; a gray-input adapter. The
classification head (`DinoV2Classifier`) and the segmentation decoder with a
convolutional detail path (`DinoV2Segmenter`) in `python/builtin_models.py` are
Deep Vision Studio additions. Fixed official weights:
https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
(SHA256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`) and
https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
(SHA256 `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`).
This applies to the basic DINOv2 models, not the separately licensed XRay-DINO or
Cell-DINO models. Include `licenses/DINOv2-APACHE-2.0.txt` with redistributed builds.

## C++ example bundled JSON header

`example/cpp/nlohmann/json.hpp`: nlohmann/json v3.12.0, unmodified single header.
Copyright (c) 2013-2025 Niels Lohmann. MIT license included at
`example/cpp/nlohmann/LICENSE.MIT`. Source: https://github.com/nlohmann/json/tree/v3.12.0
