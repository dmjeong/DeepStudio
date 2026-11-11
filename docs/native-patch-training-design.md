# Native detector and LibreYOLO classification training — 12.84 design

User request: expose usable patch training and editable augmentation for detection and LibreYOLO classification; add commercially usable model alternatives.

- Detection: split source train/val independently, crop boxes with each tile, keep background tiles, overlap 25%. Inference restores coordinates and merges duplicate boxes with class-aware NMS. Validation mAP is explicitly patch-level.
- Classification: one original image remains one labelled bag. All tiles participate; max class logits aggregate into an image prediction. No per-patch pseudo-labels. Source image batch is one, configured batch is tile microbatch. Tile gradients remain part of the bag graph; memory use grows with source size.
- Keep original data immutable. Save augmentation, crop, tile size/scale/aggregation in checkpoints and CSV. Resume must preserve these contracts.
- Native augmentations use the three editable controls (horizontal flip, rotation, color jitter); implicit mosaic/crops/scale jitter are disabled. Inference and ONNX use matching resize/normalization.
- Add D-FINE Nano, COCO-only Apache-2.0 weights (LibreYOLO/LibreDFINEn); exclude Objects365 variants. Add MobileNetV4 Medium, Apache-2.0 (LibreYOLO/LibreMobileNetV4m-cls). These are alternatives, not promises of improved user accuracy or latency.
- Export patch contract in schema 7. Generic C++/C# runtime implements tiling and aggregation; older loaders must reject the contract. Plain-image existing contracts continue working.
- Checks: coordinate/augmentation unit tests; actual native train/save/load/export for both tasks/new families; Qt control persistence; Python/ORT/C++ numerical and box parity; existing regression suite. Windows/GPU verification remains CI/target-machine work.
