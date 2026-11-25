import type { Event, Job, Run } from "./api";

export const CURRENT_RUN = -1;

const nonMetrics = new Set(["epoch", "best_epoch", "is_best", "learning_rate", "lr", "selected_value",
  "selection_value", "sample_count", "num_classes", "num_samples", "threshold", "optimal_threshold", "anomaly_threshold", "total_seconds"]);
const detectionAliases = {
  mAP_50: ["metrics/mAP50", "metrics/mAP50(B)", "mAP50"],
  mAP_50_95: ["metrics/mAP50-95", "metrics/mAP50-95(B)", "mAP50_95"],
  AR100: ["metrics/AR100", "metrics/ar_100", "metrics/recall", "metrics/recall(B)"],
};
const detectionLabels: Record<string, string> = {
  mAP_50: "mAP@0.50", mAP_50_95: "mAP@0.50–0.95", AR100: "AR@100",
  precision: "Precision", recall: "Recall", detection_tp: "정탐 박스 (TP)",
  detection_fp: "오검출 박스 (FP)", detection_fn: "누락 박스 (FN)", detection_support: "정답 박스 수",
};
export function trainingMetricLabel(key: string, task: string) {
  if (task !== "detect") return key;
  const canonical = Object.entries(detectionAliases).find(([, aliases]) => aliases.includes(key))?.[0] || key;
  return detectionLabels[canonical] || key;
}

export function scalarMetrics(values: Record<string, unknown>) {
  return Object.fromEntries(Object.entries(values).filter(([key, value]) =>
    !nonMetrics.has(key) && !/(_sec|_seconds|_hms)$/.test(key) && (value === null || typeof value === "number"))
    .map(([key, value]) => [key, typeof value === "number" && Number.isFinite(value) ? value : null])) as Record<string, number | null>;
}

export function displayMetrics(values: Record<string, unknown>, task?: string) {
  const metrics = scalarMetrics(values);
  if (task !== "detect") return metrics;
  const aliasesPresent = ["metrics/precision", "metrics/precision(B)", "metrics/recall", "metrics/recall(B)"]
    .some(key => key in metrics);
  for (const [key, aliases] of Object.entries(detectionAliases)) {
    const alias = aliases.find(name => name in metrics);
    if (!(key in metrics) && alias) metrics[key] = metrics[alias];
    if (metrics[key] !== undefined && metrics[key] !== null && metrics[key] < 0) metrics[key] = null;
  }
  if (aliasesPresent) {
    if (!("precision" in metrics)) metrics.precision = null;
    if (!("recall" in metrics)) metrics.recall = null;
  }
  return Object.fromEntries([...Object.keys(detectionLabels), "train_loss", "val_loss"]
    .filter(key => key in metrics).map(key => [key, metrics[key]]));
}

export function savedBestMetrics(run: Run,
  task = run.eval_results?.task || run.config_snapshot.task || run.config_snapshot.best_selection?.task) {
  if (run.best_epoch < 1) return {};
  const history = run.metrics_history || {};
  const epochs = history.epoch || (run.config_snapshot.training?.training_mode === "upstream_resume" ? [] :
    Array.from({ length: Math.max(0, ...Object.values(history).map(v => v.length)) }, (_, i) => i + 1));
  const index = epochs.indexOf(run.best_epoch);
  const saved = run.config_snapshot.best_selection || {};
  const name = run.best_metric_name;
  const legacyDetection = task === "detect" && detectionAliases.mAP_50_95.includes(name);
  const mapKeys = ["mAP_50_95", ...detectionAliases.mAP_50_95];
  const savedMetrics = saved.epoch === run.best_epoch ? { ...saved.metrics } : {};
  if (legacyDetection) {
    for (const alias of mapKeys) delete savedMetrics[alias];
  }
  const metrics = {
    ...(run.eval_results || {}),
    ...Object.fromEntries(Object.entries(history).filter(([, values]) => index >= 0 && index < values.length).map(([key, values]) => [key, values[index]])),
    ...savedMetrics,
  };
  if (legacyDetection) {
    const result = run.config_snapshot.upstream_result;
    const score = mapKeys.map(key => metrics[key]).find(validDetectionScore)
      ?? (result?.best_epoch === run.best_epoch ? result.best_mAP50_95 : null);
    metrics.mAP_50_95 = validDetectionScore(score) ? score : null;
  } else if (name && name !== "unavailable") {
    metrics[name] = run.best_metric;
  }
  return displayMetrics(metrics, task);
}

function validDetectionScore(value: unknown): value is number {
  return typeof value === "number" && value >= 0 && value <= 1;
}

export function trainingSelectionValue(run: Run, task: string) {
  return task === "detect" && detectionAliases.mAP_50_95.includes(run.best_metric_name)
    ? savedBestMetrics(run, task).mAP_50_95 ?? null : run.best_metric;
}

type DetectionEvaluation = { unit?: "patch" | "image"; protocol?: string; confidence?: number; iou?: number };

export function detectionEvaluationText(details: DetectionEvaluation = {}) {
  const area = details.unit === "patch" ? "패치별" : details.unit === "image" ? "이미지별" : "검증 단위 미기록";
  const protocol = details.protocol === "coco" ? " · COCO AP" : details.protocol === "voc11" ? " · VOC 11점 AP" : "";
  const { confidence, iou } = details;
  const thresholds = typeof confidence === "number" && typeof iou === "number"
    ? ` · Precision/Recall 기준: confidence ≥ ${confidence.toFixed(2)}, IoU ≥ ${iou.toFixed(2)}` : "";
  return `검증 데이터(Val) · ${area} 평가${protocol}${thresholds}` +
    (details.unit === "patch" ? " · 추론 화면에서는 패치 결과를 원본 영상에 합칩니다." : "");
}

export function initialRunIndex(runCount: number, hasLiveJob: boolean, hasJob = false) {
  return hasLiveJob || hasJob ? CURRENT_RUN : Math.max(0, runCount - 1);
}

export function trainingView(
  runs: Run[],
  requestedIndex: number,
  hasLiveJob: boolean,
  events: Event[],
  job?: Job | null,
  task?: string,
) {
  const current = requestedIndex === CURRENT_RUN;
  const runIndex = requestedIndex;
  const resultIds = new Set((job?.output?.runs || []).map((r: Run) => r.run_id));
  const selectedRun = current ? runs.find(r =>
    (job?.id && r.config_snapshot.job_id === job.id) || resultIds.has(r.run_id)) : runs[runIndex];
  const live = current && hasLiveJob && !selectedRun;
  const epochs = events.filter((e) => e.event === "epoch_finished");
  const bestEvent = events.filter((e) => e.event === "best_epoch_updated").at(-1)?.args;
  const history = selectedRun?.metrics_history || {};
  const train = live ? epochs.map((e) => e.args[1]) : history.train_loss || [];
  const val = live ? epochs.map((e) => e.args[2]) : history.val_loss || [];
  const best = live ? bestEvent?.[0] : selectedRun?.best_epoch;
  const epochNumbers = live ? epochs.map((e) => e.args[0]) : history.epoch;
  const bestMetrics = live ? displayMetrics({ ...(bestEvent?.[3] || {}), train_loss: bestEvent?.[1], val_loss: bestEvent?.[2] }, task)
    : selectedRun ? savedBestMetrics(selectedRun, task) : {};
  const selectedConfig = selectedRun?.config_snapshot || {};
  const savedTraining = selectedConfig.training || {};
  const previousNative = selectedConfig.engine === "upstream" || String(savedTraining.training_mode).startsWith("upstream_");
  const evaluation = live ? bestEvent?.[3]?.detection_evaluation : selectedRun?.eval_results?.detection_evaluation;
  const detectionEvaluation: DetectionEvaluation | undefined = evaluation || (previousNative ? {
    unit: savedTraining.patch_training === undefined ? undefined : savedTraining.patch_training ? "patch" : "image",
    protocol: "coco",
  } : undefined);
  const detectionClasses = live ? bestEvent?.[3]?.detection_per_class : selectedRun?.eval_results?.per_class;
  return {
    runIndex, selectedRun, train, val, best, epochNumbers, bestMetrics, detectionEvaluation, detectionClasses,
    missingResult: current && !hasLiveJob && !selectedRun,
    bestTrain: bestMetrics.train_loss ?? undefined,
    bestVal: bestMetrics.val_loss ?? undefined,
  };
}

export function runModel(run: Run) {
  const config = run.config_snapshot;
  const training = config.training || {};
  if (config.engine === "builtin" || String(training.training_mode).startsWith("builtin"))
    return config.model_id || config.model?.model_id || "기본 모델";
  if (config.engine === "upstream" || String(training.training_mode).startsWith("upstream_"))
    return config.model_id || config.model?.model_id || "Native 모델";
  if (String(training.training_mode).startsWith("efficientnet")) return training.efficientnet_model || "EfficientNet";
  if (config.task === "anomaly" && training.anomaly_method === "patchcore") return `PatchCore / ${training.patchcore_backbone || ""}`;
  return training.training_mode === "custom" || config.task === "anomaly" ? "Custom CSP" : "모델 미기록";
}
