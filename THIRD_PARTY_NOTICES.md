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
| LibreYOLO | 기본 LibreMobileNetV4 / LibreYOLO9 소스 후보 | MIT for the library; https://github.com/LibreYOLO/libreyolo/blob/release/LICENSE and the bundled third-party list at https://github.com/LibreYOLO/libreyolo/blob/release/NOTICE |
| RT-DETRv4 upstream (제품 표기 Re-DETR v4) | 기본 Re-DETR v4 소스 후보 | Apache-2.0; https://github.com/RT-DETRs/RT-DETRv4/blob/main/LICENSE |
| SAM2 upstream | 기본 SAM2 소스 후보 | Apache-2.0; https://github.com/facebookresearch/sam2/blob/main/LICENSE |

The LibreYOLO, Re-DETRv4 and SAM2 rows are the source candidates for the
shipped basic-model implementations. Until source revision, license/NOTICE,
Windows worker and ONNX acceptance tests are all present, the production
installer gate keeps their catalog status below `release_ready`. LibreYOLO's
NOTICE lists additional vendored components and their individual terms; a
release that bundles those components must copy the applicable notices and
licenses. A user-added `.dvmodel` must carry its own `license` object, source
revision, asset hashes, and required notices before release signing.
Pretrained weights are separate assets and require review against the exact
model card before redistribution.
