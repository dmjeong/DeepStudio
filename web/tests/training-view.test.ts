import assert from "node:assert/strict";
import test from "node:test";
import { CURRENT_RUN, initialRunIndex, runModel, trainingView } from "../src/training-view.ts";
import type { Event, Job, Run } from "../src/api.ts";

const saved: Run = {
  run_id: "previous", status: "completed", best_epoch: 12,
  best_metric: 0.8, best_metric_name: "accuracy", checkpoint_path: "best.pth",
  metrics_history: { epoch: [11, 12], train_loss: [0.4, 0.3], val_loss: [0.6, 0.5] },
  eval_results: {}, config_snapshot: {},
};
const events: Event[] = [
  { event: "epoch_finished", args: [1, 0.9, 0.8] },
  { event: "best_epoch_updated", args: [1, 0.9, 0.8] },
];
const job: Job = { id: "job-new", kind: "train", status: "completed", created_at: "", finished_at: "", error: "", output: {} };

test("기본 모델 실행 기록은 저장된 모델 ID를 표시한다", () => {
  assert.equal(runModel({ ...saved, config_snapshot: { engine: "builtin", model_id: "resnet50" } }), "resnet50");
});

test("과거 기록 선택 시 진행 중인 학습 이벤트가 그래프와 Best를 바꾸지 않는다", () => {
  const view = trainingView([saved], 0, true, events);
  assert.deepEqual(view.train, [0.4, 0.3]);
  assert.deepEqual(view.val, [0.6, 0.5]);
  assert.equal(view.best, 12);
  assert.equal(view.bestTrain, 0.3);
  assert.equal(view.bestVal, 0.5);
  assert.equal(view.selectedRun?.run_id, "previous");
});

test("현재 학습 선택과 새로고침은 실시간 에폭을 사용한다", () => {
  const index = initialRunIndex(1, true);
  const view = trainingView([saved], index, true, events);
  assert.equal(index, CURRENT_RUN);
  assert.equal(view.selectedRun, undefined);
  assert.deepEqual(view.epochNumbers, [1]);
  assert.equal(view.bestTrain, 0.9);
  assert.equal(view.bestVal, 0.8);
  assert.equal(initialRunIndex(1, false), 0);
});

test("작업 종료 후 해당 job의 Run만 표시하고 기록 없는 실패는 빈 결과로 유지한다", () => {
  const finished = { ...saved, run_id: "new", best_epoch: 1,
    config_snapshot: { job_id: job.id },
    metrics_history: { epoch: [1], train_loss: [0.9], val_loss: [0.8] } };
  assert.equal(trainingView([saved, finished], CURRENT_RUN, false, events, job).selectedRun?.run_id, "new");
  const failed = trainingView([saved], CURRENT_RUN, false, events, { ...job, status: "failed" });
  assert.equal(failed.selectedRun, undefined);
  assert.equal(failed.missingResult, true);
  assert.deepEqual(failed.train, []);
  assert.equal(failed.best, undefined);
  assert.equal(trainingView([], 0, false, events).bestTrain, undefined);
});

test("실패 후 새로고침과 다른 Run 추가도 이전 결과를 자동 선택하지 않는다", () => {
  const index = initialRunIndex(2, false, true);
  assert.equal(index, CURRENT_RUN);
  const other = { ...saved, run_id: "unrelated", config_snapshot: { job_id: "other-job" } };
  const view = trainingView([saved, other], index, false, events, job);
  assert.equal(view.selectedRun, undefined);
  assert.equal(view.bestTrain, undefined);
  assert.equal(trainingView([saved, other], 0, false, events, job).selectedRun, saved);
});

test("기존 job 출력의 Run ID로도 연결하되 배열 위치로 추정하지 않는다", () => {
  const oldJob = { ...job, output: { runs: [{ run_id: saved.run_id }] } };
  assert.equal(trainingView([saved], CURRENT_RUN, false, [], oldJob).selectedRun, saved);
  assert.equal(trainingView([saved], 10, false, [], job).selectedRun, undefined);
});

test("모델명은 요청 설정이 아니라 저장된 실제 실행 설정에서 읽는다", () => {
  const resumed = { ...saved, config_snapshot: {
    training: { training_mode: "efficientnet_resume", efficientnet_model: "efficientnet_b1" },
    requested_config: { training: { efficientnet_model: "efficientnet_b0" } },
  } };
  assert.equal(runModel(resumed), "efficientnet_b1");
});

test("이어학습 이전 Best의 손실이 없으면 다른 에폭 손실로 대체하지 않는다", () => {
  const view = trainingView([{ ...saved, best_epoch: 3 }], 0, false, events);
  assert.equal(view.best, 3);
  assert.equal(view.bestTrain, undefined);
  assert.equal(view.bestVal, undefined);
});
