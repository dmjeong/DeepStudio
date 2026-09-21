import type { Event, Job, Run } from "./api";

export const CURRENT_RUN = -1;

const nonMetrics = new Set(["epoch", "best_epoch", "is_best", "learning_rate", "lr", "selected_value",
  "selection_value", "sample_count", "num_classes", "num_samples", "threshold", "optimal_threshold", "anomaly_threshold", "total_seconds"]);
export function scalarMetrics(values: Record<string, unknown>) {
  return Object.fromEntries(Object.entries(values).filter(([key, value]) =>
    !nonMetrics.has(key) && !/(_sec|_seconds|_hms)$/.test(key) && (value === null || typeof value === "number"))
    .map(([key, value]) => [key, typeof value === "number" && Number.isFinite(value) ? value : null])) as Record<string, number | null>;
}

export function savedBestMetrics(run: Run) {
  if (run.best_epoch < 1) return {};
  const history = run.metrics_history || {};
  const epochs = history.epoch || (run.config_snapshot.training?.training_mode === "upstream_resume" ? [] :
    Array.from({ length: Math.max(0, ...Object.values(history).map(v => v.length)) }, (_, i) => i + 1));
  const index = epochs.indexOf(run.best_epoch);
  const saved = run.config_snapshot.best_selection || {};
  return scalarMetrics({
    ...run.eval_results,
    ...Object.fromEntries(Object.entries(history).filter(([, values]) => index >= 0 && index < values.length).map(([key, values]) => [key, values[index]])),
    ...(saved.epoch === run.best_epoch ? saved.metrics || {} : {}),
    ...(run.best_metric_name && run.best_metric_name !== "unavailable" ? { [run.best_metric_name]: run.best_metric } : {}),
  });
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
  const bestMetrics = live ? scalarMetrics({ ...(bestEvent?.[3] || {}), train_loss: bestEvent?.[1], val_loss: bestEvent?.[2] })
    : selectedRun ? savedBestMetrics(selectedRun) : {};
  return {
    runIndex, selectedRun, train, val, best, epochNumbers, bestMetrics,
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
  if (String(training.training_mode).startsWith("efficientnet")) return training.efficientnet_model || "EfficientNet";
  if (config.task === "anomaly" && training.anomaly_method === "patchcore") return `PatchCore / ${training.patchcore_backbone || ""}`;
  return training.training_mode === "custom" || config.task === "anomaly" ? "Custom CSP" : "모델 미기록";
}
