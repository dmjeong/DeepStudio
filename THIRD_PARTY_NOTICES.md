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

## C++ example bundled JSON header

`example/cpp/nlohmann/json.hpp`: nlohmann/json v3.12.0, unmodified single header.
Copyright (c) 2013-2025 Niels Lohmann. MIT license included at
`example/cpp/nlohmann/LICENSE.MIT`. Source: https://github.com/nlohmann/json/tree/v3.12.0
