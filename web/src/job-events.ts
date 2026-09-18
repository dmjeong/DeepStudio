import type { Event } from "./api";

export function retainJobEvents(previous: Event[], incoming: Event[]) {
  const all = [...previous, ...incoming].filter(e => e.event !== "inference_result");
  const latest = new Map<string, number>();
  const sampled = new Set(["progress_updated", "batch_finished", "lr_updated", "layer_debug"]);
  let logCount = 0;
  all.forEach((e, index) => {
    if (sampled.has(e.event)) latest.set(e.event, index);
    if (e.event === "log_message") logCount++;
  });
  let logsToDrop = Math.max(0, logCount - 500);
  return all.filter((e, index) => {
    if (e.event === "log_message" && logsToDrop > 0) { logsToDrop--; return false; }
    return !sampled.has(e.event) || latest.get(e.event) === index;
  });
}
