import type { DataConfig, ModelConfig, TrainingConfig } from "./project-types";
export type Config = Record<string, any>;
export interface Run {
  run_id: string;
  status: string;
  best_epoch: number;
  best_metric: number | null;
  best_metric_name: string;
  checkpoint_path: string;
  metrics_history: Record<string, (number | null)[]>;
  eval_results: Config;
  config_snapshot: Config;
}
export interface Project {
  name: string;
  task: string;
  filepath: string;
  project_dir: string;
  data: DataConfig;
  model: ModelConfig;
  training: TrainingConfig;
  runs: Run[];
  modified: string;
}
export interface Job {
  id: string;
  kind: string;
  status: string;
  created_at: string;
  finished_at: string | null;
  error: string;
  output: Config;
  duration_sec?: number;
  project_path?: string | null;
  project_error?: string;
}
export interface StudioState {
  version: string;
  project_summary?: { filepath: string; revision: string } | null;
  warnings?: string[];
  project: Project | null;
  recent: string[];
  active_job: string | null;
  jobs: Job[];
  home: string;
  error: string;
  model_runtime_available: boolean;
}
export interface Event {
  event: string;
  args: any[];
}
export interface JobView {
  job: Job;
  events: Event[];
  offset: number;
  console: string;
}
export const terminal = (job?: Job) =>
  !!job &&
  ["completed", "failed", "cancelled", "interrupted"].includes(job.status);
export async function api<T = Config>(
  path: string,
  method = "GET",
  body?: unknown,
): Promise<T> {
  const response = await fetch("/api" + path, {
    method,
    headers: { "Content-Type": "application/json", "X-Studio-Request": "1" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail ?? data),
    );
  return data;
}
export const filename = (path: string) => path.split(/[\\/]/).pop() || path;
export const join = (parent: string, child: string) =>
  parent.replace(/[\\/]$/, "") + "/" + child;
export const number = (value: unknown, digits = 4) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";
export const duration = (seconds?: number | null) =>
  seconds == null
    ? "—"
    : [
        Math.floor(seconds / 3600),
        Math.floor(seconds / 60) % 60,
        Math.floor(seconds) % 60,
      ]
        .map((v) => String(v).padStart(2, "0"))
        .join(":");
export const timing = (seconds?: number | null) =>
  seconds == null
    ? "미측정"
    : seconds < 1
      ? `${(seconds * 1000).toFixed(1)} ms`
      : `${seconds.toFixed(2)} 초`;
