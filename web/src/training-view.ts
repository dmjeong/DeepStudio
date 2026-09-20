import type { Event, Job, Run } from "./api";

export const CURRENT_RUN = -1;

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
  const bestIndex = epochNumbers ? epochNumbers.indexOf(best || 0) : (best || 0) - 1;
  return {
    runIndex, selectedRun, train, val, best, epochNumbers,
    missingResult: current && !hasLiveJob && !selectedRun,
    bestTrain: live ? bestEvent?.[1] : history.train_loss?.[bestIndex],
    bestVal: live ? bestEvent?.[2] : history.val_loss?.[bestIndex],
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
