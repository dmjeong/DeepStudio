# 5.75 — Best 체크포인트 파일명

모든 내장 학습 엔진의 최종 Best 체크포인트는 선택 기준과 값을 파일명에 기록한다.
예를 들어 정확도 기준이면 `best_accuracy_0.912500_epoch_12.pt`, 검증 손실 기준이면
`best_val_loss_0.183210_epoch_12.pt`가 생성된다.

적용 대상은 Custom CSP, EfficientNet B0/B1, ResNet, ConvNeXt, DeepLab V3+, U-Net,
LibreYOLO, Re-DETR v4, SAM2, PatchCore다. 학습 중에는 평가와 재개를 위해 `best.pt`를
임시로 사용하지만, 완료 후 RunRecord·추론·ONNX Export가 모두 지표가 포함된 파일을
가리키도록 바꾼다. `last.pt`는 중단 재개 호환성을 위해 그대로 둔다.
