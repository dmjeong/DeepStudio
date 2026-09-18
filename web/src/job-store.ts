import { useSyncExternalStore } from "react";
import { api, terminal, type Event, type JobView } from "./api";
import { retainJobEvents } from "./job-events";
interface Snapshot { view: JobView | null; events: Event[]; error: string }
const empty: Snapshot = { view: null, events: [], error: "" };
interface Entry { snapshot: Snapshot; listeners: Set<() => void>; offset: number; timer?: ReturnType<typeof setTimeout>; running: boolean }
const entries = new Map<string, Entry>();
function entry(id: string): Entry {
  if (!entries.has(id) && entries.size >= 32) {
    for (const [key, value] of entries) { if (!value.listeners.size && !value.running) { entries.delete(key); break; } }
  }
  if (!entries.has(id)) entries.set(id, { snapshot: empty, listeners: new Set(), offset: 0, running: false });
  return entries.get(id)!;
}
async function poll(id: string) {
  const e = entry(id);
  if (!e.listeners.size || e.running) return;
  e.running = true;
  let delay = 800;
  try {
    const view = await api<JobView>(`/jobs/${id}?offset=${e.offset}`);
    e.offset = view.offset;
    e.snapshot = { view, error: "", events: retainJobEvents(e.snapshot.events, view.events) };
    if (terminal(view.job) && view.events.length < 250) delay = 0;
    else if (view.events.length === 250) delay = 20;
  } catch (error) { e.snapshot = { ...e.snapshot, error: (error as Error).message }; delay = 2500; }
  e.running = false;
  e.listeners.forEach(listener => listener());
  if (delay && e.listeners.size) e.timer = setTimeout(() => { e.timer = undefined; void poll(id); }, delay);
}
function subscribe(id: string | null, listener: () => void) {
  if (!id) return () => {};
  const e = entry(id);
  e.listeners.add(listener);
  if (!e.running && !e.timer && !terminal(e.snapshot.view?.job)) void poll(id);
  return () => { e.listeners.delete(listener); if (!e.listeners.size) { clearTimeout(e.timer); e.timer = undefined; } };
}
export function useJob(id: string | null): Snapshot {
  return useSyncExternalStore(listener => subscribe(id, listener), () => id ? entry(id).snapshot : empty);
}
