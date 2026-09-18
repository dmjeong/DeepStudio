import assert from "node:assert/strict";
import test from "node:test";
import { retainJobEvents } from "../src/job-events.ts";
import type { Event } from "../src/api.ts";

test("배치 진행을 유지하면서 로그와 빈번한 이벤트 수를 제한한다", () => {
  const incoming: Event[] = [];
  for (let i = 0; i < 600; i++) incoming.push(
    { event: "log_message", args: [String(i)] },
    { event: "batch_finished", args: [11, i + 1, 600, .5] });
  incoming.push({ event: "epoch_finished", args: [11, .5, .6] });
  const kept = retainJobEvents([], incoming);
  assert.equal(kept.filter(e => e.event === "log_message").length, 500);
  assert.deepEqual(kept.find(e => e.event === "batch_finished")?.args, [11, 600, 600, .5]);
  assert.equal(kept.at(-1)?.event, "epoch_finished");
  const next = retainJobEvents(kept, [{ event: "batch_finished", args: [12, 1, 600, .4] }]);
  assert.equal(next.filter(e => e.event === "batch_finished").length, 1);
  assert.equal(next.filter(e => e.event === "epoch_finished").length, 1);
});


test("레이어 관찰은 최신 스냅샷 하나만 보관한다", () => {
  const first: Event = { event: "layer_debug", args: [{ run_id: "run-A", epoch: 1, batch: 1, layers: {} }] };
  const latest: Event = { event: "layer_debug", args: [{ run_id: "run-A", epoch: 1, batch: 2, layers: { head: {} } }] };
  const kept = retainJobEvents([first], [latest]);
  assert.deepEqual(kept, [latest]);
});
