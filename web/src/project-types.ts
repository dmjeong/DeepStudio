// core.project의 프로젝트 설정 계약. 런타임 검증은 API에서 수행합니다.
export interface DataConfig {
  root: string;
  train_dir: string;
  val_dir: string;
  test_dir: string;
  class_names: string[];
  num_classes: number;
  val_split: number;
  image_count: number;
}
export interface ModelConfig {
  model_id?: string;
  pack_path?: string;
  backbone_channels: number[];
  csp_depth: number[];
  dropout: number;
  pretrained_weights: string;
  freeze_backbone: boolean;
  backbone_lr_mult: number;
}
export interface AugmentationConfig {
  horizontal_flip: number;
  vertical_flip: number;
  rotation: number;
  color_jitter: number;
  scale_range: number[];
  mixup_alpha: number;
  mosaic: boolean;
}
export interface TrainingConfig {
  epochs: number;
  batch_size: number;
  input_size: number;
  in_channels: number;
  learning_rate: number;
  weight_decay: number;
  optimizer: string;
  scheduler: string;
  warmup_epochs: number;
  early_stop_patience: number;
  selection_metric: string;
  label_smoothing: number;
  class_weights: string;
  device: string;
  use_amp: boolean;
  training_mode: string;
  efficientnet_model: string;
  efficientnet_no_decay?: boolean;
  layer_debug_enabled: boolean;
  layer_debug_patterns: string;
  layer_debug_batches: number;
  augmentation: AugmentationConfig;
  anomaly_method: string;
  patchcore_sampling_ratio: number;
  patchcore_n_neighbors: number;
  patchcore_backbone: string;
  patchcore_weight_source: string;
  patchcore_weights: string;
  patchcore_append: boolean;
  patchcore_max_candidates: number;
  patchcore_max_memory_bank: number;
  patchcore_seed: number;
  patchcore_crop_enabled: boolean;
  patchcore_crop_width: number;
  patchcore_crop_height: number;
}
