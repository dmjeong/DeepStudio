# Third-party notices

Deep Vision Studio does not commit pretrained model weights. A release payload
must include the exact source license and notice for every dependency it
actually bundles; this file is the source list for the payload review and is
not a substitute for a release-specific SBOM.

| Component | Use | Upstream license / source |
|---|---|---|
| PyTorch / torchvision | Python training and built-in model adapters | BSD-style license; https://github.com/pytorch/pytorch/blob/main/LICENSE |
| OpenCV | image conversion and C++ preprocessing | Apache-2.0; https://github.com/opencv/opencv/blob/4.x/LICENSE |
| ONNX Runtime | Python and C++ CPU inference | MIT; https://github.com/microsoft/onnxruntime/blob/main/LICENSE |
| nlohmann/json | C++ manifest parsing | MIT; https://github.com/nlohmann/json/blob/develop/LICENSE.MIT |
| WiX Toolset | Windows MSI/Burn build tool | MIT; https://github.com/wixtoolset/wix/blob/main/LICENSE |
| RT-DETRv4 upstream (제품 표기 Re-DETR v4) | Optional Re-DETR v4 Docker model pack source | Apache-2.0; https://github.com/RT-DETRs/RT-DETRv4/blob/main/LICENSE |
| SAM2 upstream | Optional SAM2 Docker model pack source | Apache-2.0; https://github.com/facebookresearch/sam2/blob/main/LICENSE |

The Re-DETRv4 and SAM2 rows describe optional upstream source references only.
They are not copied into this repository and their checkpoints are not part of
the default installer. A `.dvmodel` pack must carry its own `license` object,
source revision, asset hashes, and required notices before release signing.
