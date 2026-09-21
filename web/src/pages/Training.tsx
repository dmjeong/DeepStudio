import { useEffect, useState } from "react";
import { useDraft } from "../drafts";
import type { TrainingConfig, ModelConfig, AugmentationConfig } from "../project-types";
import { api, number, terminal, type Config, type Project } from "../api";
import { Check, Empty, Field, JobMonitor, LossChart, Panel, PathField, Stat, useJob } from "../components";
import type { PageProps } from "../page-context";
import { CURRENT_RUN, initialRunIndex, runModel, trainingView } from "../training-view";
import { builtinAdapters, modelModes } from "../model-options";

type CatalogModel = { model_id: string; family: string; variant: string; input_size: number[]; input_channels: number[] };

export function Training({
  state,
  project,
  act,
  refresh,
  run,
  busy,
}: PageProps & { project: Project }) {
  const [cfg, setCfg] = useDraft<TrainingConfig>(project.filepath, "training", structuredClone(project.training));
  const [model, setModel] = useDraft<ModelConfig>(project.filepath, "model", structuredClone(project.model));
  const [dirty, setDirty] = useDraft(project.filepath, "training-dirty", false);
  const [optionError, setOptionError] = useState("");
  const [options, setOptions] = useState<Config>({ metrics: [], models: [], capabilities: {} });
  const [catalog, setCatalog] = useState<CatalogModel[]>([]);
  const [jobId, setJobId] = useState<string | null>(
    () =>
      state.jobs.find(
        (j) => j.kind === "train" && j.project_path === project.filepath,
      )?.id || null,
  );
  const { events, view: jobView } = useJob(jobId);
  const job = jobView?.job || state.jobs.find(j => j.id === jobId);
  const hasLiveJob = job ? !terminal(job) : !!jobId && state.active_job === jobId;
  const [runIndex, setRunIndex] = useState(
    () => initialRunIndex(project.runs.length, hasLiveJob, !!jobId),
  );
  const patch =
    project.task === "anomaly" && cfg.anomaly_method === "patchcore";
  const modelId = model.model_id || "";
  const selectedModel = catalog.find(item => item.model_id === modelId);
  const modelName = selectedModel ? `${selectedModel.family} ${selectedModel.variant}` : modelId;
  const modes = modelModes(project.task, modelId, modelName);
  const builtin = builtinAdapters.has(modelId);
  const unsupported = project.task !== "anomaly" && !modes.some(([value]) => value === cfg.training_mode);
  const efficientnet = project.task === "classify" && cfg.training_mode.startsWith("efficientnet");
  const resume = cfg.training_mode === "efficientnet_resume";
  const set = (key: keyof TrainingConfig, value: unknown) => {
    setCfg({ ...cfg, [key]: value });
    setDirty(true);
  };
  const setM = (key: keyof ModelConfig, value: unknown) => {
    setModel({ ...model, [key]: value });
    setDirty(true);
  };
  useEffect(() => {
    let active = true;
    setCatalog([]);
    if (project.task !== "obb") api(`/models?task=${project.task}`)
      .then(data => active && setCatalog(data.models))
      .catch(error => active && setOptionError(error.message));
    return () => { active = false; };
  }, [project.task]);
  useEffect(() => {
    let active = true;
    setOptionError("");
    setOptions({ metrics: [], models: [], capabilities: {} });
    api(
      `/options?task=${project.task}&engine=${efficientnet ? "efficientnet" : cfg.training_mode.startsWith("builtin") ? "builtin" : "custom"}&mode=${project.task === "anomaly" ? "custom" : cfg.training_mode}&anomaly_method=${cfg.anomaly_method}&model_id=${encodeURIComponent(modelId)}`,
    )
      .then((d) => active && setOptions(d))
      .catch((e) => active && setOptionError(e.message));
    return () => {
      active = false;
    };
  }, [project.task, unsupported, efficientnet, cfg.training_mode, cfg.anomaly_method, modelId]);
  const save = async () => {
    await api("/project", "PUT", { training: cfg, model });
    setDirty(false);
    await refresh();
  };
  const numeric = (
    label: string,
    key: keyof TrainingConfig,
    min: number,
    max: number,
    step = 1,
  ) => (
    <Field label={label} key={key}>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={Number(cfg[key] ?? project.training[key])}
        onChange={(e) => set(key, Number(e.target.value))}
      />
    </Field>
  );
  const { runIndex: selectedIndex, selectedRun, train, val, best, epochNumbers, bestMetrics, missingResult } =
    trainingView(project.runs, runIndex, hasLiveJob, events, job);
  return (
    <>
      <div className="training-layout">
        <Panel
          title="학습 설정"
          actions={
            <span className="muted">
              {dirty ? "저장하지 않은 변경" : "저장됨"}
            </span>
          }
        >
          {optionError && <p className="error">{optionError}</p>}
          <fieldset disabled={busy}>
            {project.task !== "obb" && <Field label="모델 카탈로그">
              <select value={modelId} onChange={event => {
                const id = event.target.value;
                const item = catalog.find(value => value.model_id === id);
                setModel({ ...model, model_id: id, pack_path: "", pretrained_weights: "" });
                setCfg({ ...cfg, training_mode: id ? modelModes(project.task, id, id)[0][0] : "custom",
                  efficientnet_model: id.startsWith("efficientnet_") ? id : cfg.efficientnet_model,
                  patchcore_backbone: id.startsWith("patchcore_") ? id.slice(10) : cfg.patchcore_backbone,
                  input_size: item?.input_size[0] ?? cfg.input_size,
                  in_channels: item && !item.input_channels.includes(cfg.in_channels) ? item.input_channels[0] : cfg.in_channels,
                  layer_debug_enabled: false, selection_metric: "engine_default" });
                setDirty(true);
              }}>
                <option value="">Custom CSP / 구형 프로젝트</option>
                {modelId && !selectedModel && <option value={modelId}>{modelId}</option>}
                {catalog.map(item => <option key={item.model_id} value={item.model_id}>{item.family} {item.variant}</option>)}
              </select>
            </Field>}
            <div className="form-grid">
              {project.task === "anomaly" ? (
                <Field label="이상 탐지 모델">
                  <select
                    value={cfg.anomaly_method}
                    onChange={(e) => set("anomaly_method", e.target.value)}
                  >
                    <option value="patchcore">PatchCore</option>
                    <option value="reconstruction">Custom CSP 재구성</option>
                  </select>
                </Field>
              ) : (
                <Field label="학습 엔진">
                  <select
                    value={cfg.training_mode}
                    onChange={(e) => {
                      setCfg({
                        ...cfg,
                        training_mode: e.target.value,
                        layer_debug_enabled: false,
                        selection_metric: "engine_default",
                      });
                      if (["efficientnet_finetune", "builtin_finetune"].includes(e.target.value)) setModel({ ...model, pretrained_weights: "" });
                      setDirty(true);
                    }}
                  >
                    {unsupported && <option value={cfg.training_mode} disabled>지원하지 않는 학습 모드 - 현재 엔진 선택 필요</option>}
                    {project.task !== "obb" && modes.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                  </select>
                </Field>
              )}
              <Field label="연산 장치">
                <select
                  value={cfg.device}
                  onChange={(e) => set("device", e.target.value)}
                >
                  {["auto", "cpu", "cuda", "cuda:0", "cuda:1"].map((v) => (
                    <option key={v} value={v}>{v === "auto" ? "자동 (GPU 우선)" : v === "cpu" ? "CPU" : `${v} (GPU 필수)`}</option>
                  ))}
                </select>
              </Field>
            </div>
            {efficientnet && !resume && <Field label="분류 모델" hint="B0 기본 224 px, B1 기본 240 px. 입력 크기는 아래에서 조정할 수 있습니다.">
              <select disabled={resume} value={cfg.efficientnet_model || "efficientnet_b0"}
                onChange={(e) => { setCfg({ ...cfg, efficientnet_model: e.target.value,
                  input_size: e.target.value === "efficientnet_b1" ? 240 : 224 });
                  setModel({ ...model, model_id: e.target.value, pretrained_weights: "" }); setDirty(true); }}>
                <option value="efficientnet_b0">EfficientNet B0</option>
                <option value="efficientnet_b1">EfficientNet B1</option>
              </select>
            </Field>}
            {!patch && !unsupported && !["efficientnet_finetune", "builtin_finetune"].includes(cfg.training_mode) && (
              <PathField
                label={
                  resume
                    ? "중단 시 저장된 last.pt (필수)"
                    : efficientnet || cfg.training_mode === "builtin_transfer" ? "추가 학습할 가중치 (필수)" : "초기 가중치 (선택)"
                }
                value={model.pretrained_weights}
                onChange={(v) => setM("pretrained_weights", v)}
                home={state.home}
              />
            )}
            {unsupported && <p className="learning-mode-help">선택 모델에 맞는 학습 모드를 다시 선택하세요.</p>}
            {modelId.startsWith("libreyolo_") && <p className="learning-mode-help">
              LibreYOLO 완성 .dvmodel 팩은 현재 기본 설치본과 저장소에 포함되지 않습니다. 모델 이름만으로 학습할 수 있는 상태는 아닙니다.
              .pt의 확장자를 바꾸는 방식이 아니라 학습 코드·가중치·Docker 이미지를 묶은 팩이 필요합니다.
              {" "}<a href="https://github.com/dmjeong/DeepStudio/blob/main/docs/LIBREYOLO_MODEL_PACKS.ko.md" target="_blank" rel="noreferrer">팩 구성·설치 방법</a>
            </p>}
            {builtin && <p className="learning-mode-help">{modelName}: ImageNet 가중치는 최초 1회 다운로드하며 캐시 또는 로컬 .pth로 오프라인 학습할 수 있습니다.
              {project.task === "segment" && " 분할 모델은 백본만 사전학습되며 분할 헤드는 새로 학습합니다."}</p>}
            {efficientnet && <p className="learning-mode-help">{resume
              ? "last.pt에서 모델과 optimizer, 스케줄러, 난수 상태, 학습 설정을 복원합니다. 데이터가 바뀌면 추가 학습을 선택하세요."
              : cfg.training_mode === "efficientnet_transfer"
                ? "내 가중치로 새 학습을 시작합니다. 클래스 구성이 달라지면 마지막 분류기를 새로 학습합니다."
                : "ImageNet 사전학습 가중치를 검증하고 로드합니다. 내 클래스 분류기를 새로 학습하며 전체 백본도 조정할 수 있습니다."}</p>}
            <fieldset disabled={resume}>
            {patch ? (
              <>
                <Field label="사전학습 백본">
                  <select
                    value={cfg.patchcore_backbone}
                    onChange={(e) => set("patchcore_backbone", e.target.value)}
                  >
                    {["wide_resnet50_2", "resnet50", "resnet18"].map((v) => (
                      <option key={v}>{v}</option>
                    ))}
                  </select>
                </Field>
                <Field label="가중치 출처">
                  <select
                    value={cfg.patchcore_weight_source}
                    onChange={(e) =>
                      set("patchcore_weight_source", e.target.value)
                    }
                  >
                    <option value="imagenet">ImageNet 사전학습</option>
                    <option value="backbone">로컬 백본 가중치</option>
                    <option value="patchcore">기존 PatchCore</option>
                  </select>
                </Field>
                <PathField
                  label="로컬 가중치"
                  value={cfg.patchcore_weights}
                  onChange={(v) => set("patchcore_weights", v)}
                  home={state.home}
                />
                <p className="muted">
                  PatchCore는 고정된 백본으로 특징을 추출하고 메모리 뱅크를
                  구성합니다. 에폭별 역전파 학습과 Best loss는 해당하지
                  않습니다.
                </p>
                <div className="form-grid">
                  {numeric(
                    "코어셋 비율",
                    "patchcore_sampling_ratio",
                    0.001,
                    1,
                    0.001,
                  )}
                  {numeric("이웃 수", "patchcore_n_neighbors", 1, 100)}
                  {numeric(
                    "최대 후보 패치",
                    "patchcore_max_candidates",
                    100,
                    1000000,
                  )}
                  {numeric(
                    "최대 메모리 뱅크",
                    "patchcore_max_memory_bank",
                    1,
                    100000,
                  )}
                  {numeric("시드", "patchcore_seed", 0, 2147483647)}
                </div>
                <Check
                  label="기존 메모리 뱅크에 추가"
                  value={cfg.patchcore_append}
                  onChange={(v) => set("patchcore_append", v)}
                />
              </>
            ) : (
              <>
                <div className="form-grid">
                  {numeric("에폭", "epochs", 1, 10000)}
                  {numeric("학습률", "learning_rate", 0.0000001, 1, 0.0001)}
                  <Field label="Best 선정 지표">
                    <select
                      value={cfg.selection_metric}
                      onChange={(e) => set("selection_metric", e.target.value)}
                    >
                      {options.metrics.map((v: Config) => (
                        <option value={v.value} key={v.value}>
                          {v.label}
                        </option>
                      ))}
                    </select>
                  </Field>
                  {numeric(
                    "조기 종료 대기 에폭",
                    "early_stop_patience",
                    0,
                    10000,
                  )}
                </div>
                <Check
                  label="백본 고정 (해제하면 전체 학습)"
                  value={model.freeze_backbone}
                  onChange={(v) => setM("freeze_backbone", v)}
                />
                {efficientnet && <Check
                  label="BatchNorm과 bias의 weight decay 제외"
                  value={cfg.efficientnet_no_decay ?? true}
                  onChange={(v) => set("efficientnet_no_decay", v)}
                />}
                <Check
                  label="GPU 혼합 정밀도 (AMP)"
                  value={cfg.use_amp}
                  onChange={(v) => set("use_amp", v)}
                />
              </>
            )}
            <div className="form-grid">
              {numeric("입력 크기 (px)", "input_size", 32, 4096)}
              {numeric("배치 크기", "batch_size", 1, 1024)}
            </div>
            <Check label="중앙 크롭 사용" value={Boolean(cfg.patchcore_crop_enabled)}
              onChange={(v) => set("patchcore_crop_enabled", v)} />
            {cfg.patchcore_crop_enabled && <>
              <div className="form-grid">
                {numeric("크롭 가로 (px)", "patchcore_crop_width", 1, 65536)}
                {numeric("크롭 세로 (px)", "patchcore_crop_height", 1, 65536)}
              </div>
              <p className="muted">원본 중앙 {cfg.patchcore_crop_width ?? project.training.patchcore_crop_width}×{cfg.patchcore_crop_height ?? project.training.patchcore_crop_height} px → 모델 입력 {cfg.input_size}×{cfg.input_size} px.
                모든 태스크의 학습과 추론에 적용하며 박스와 마스크도 함께 자릅니다. 원본보다 큰 크롭은 사용할 수 없습니다. 분류와 이상 탐지는 크롭 안의 내용에 맞는 이미지 라벨이 필요합니다.</p>
            </>}
            {!patch && (
              <details>
                <summary>최적화 및 데이터 증강</summary>
                <div className="form-grid">
                  <Field label="Optimizer">
                    <select
                      value={cfg.optimizer}
                      onChange={(e) => set("optimizer", e.target.value)}
                    >
                      {["adamw", "adam", "sgd"].map((v) => (
                        <option key={v}>{v}</option>
                      ))}
                    </select>
                  </Field>
                  <Field label="Scheduler">
                    <select
                      value={cfg.scheduler}
                      onChange={(e) => set("scheduler", e.target.value)}
                    >
                      {["cosine", "step", "none"].map((v) => (
                        <option key={v}>{v}</option>
                      ))}
                    </select>
                  </Field>
                  {numeric("Weight decay", "weight_decay", 0, 1, 0.0001)}
                  {numeric("Warmup 에폭", "warmup_epochs", 0, 1000)}
                  {project.task === "classify" && (
                    <Field label="클래스 가중치">
                      <select
                        value={cfg.class_weights}
                        onChange={(e) => set("class_weights", e.target.value)}
                      >
                        {["none", "balanced", "sqrt"].map((v) => (
                          <option key={v}>{v}</option>
                        ))}
                      </select>
                    </Field>
                  )}
                  {options.capabilities.input_channels?.length > 1 && (
                    <Field label="입력 채널" hint={efficientnet ? "새 사전학습: 1채널 선택 시 첫 Conv도 1채널입니다. 기존 가중치의 추가 학습과 재개는 저장된 구조를 유지합니다." : undefined}>
                      <select
                        value={cfg.in_channels}
                        onChange={(e) =>
                          set("in_channels", Number(e.target.value))
                        }
                      >
                        <option value="1">Grayscale (1)</option>
                        <option value="3">RGB (3)</option>
                      </select>
                    </Field>
                  )}
                </div>
                <div className="form-grid">
                  {[
                    ["수평 반전 확률", "horizontal_flip", 1],
                    ["수직 반전 확률", "vertical_flip", 1],
                    ["회전 범위", "rotation", 180],
                    ["색상 변화", "color_jitter", 1],
                    ["Mixup alpha", "mixup_alpha", 1],
                  ].filter(([, key]) => options.capabilities.augmentation?.includes(String(key))).map(([label, key, max]) => (
                    <Field label={String(label)} key={key}>
                      <input
                        type="number"
                        min="0"
                        max={max}
                        step="0.05"
                        value={Number(cfg.augmentation[key as keyof AugmentationConfig])}
                        onChange={(e) =>
                          set("augmentation", {
                            ...cfg.augmentation,
                            [key]: Number(e.target.value),
                          })
                        }
                      />
                    </Field>
                  ))}
                </div>
                {!unsupported && (
                  <Field label="백본 학습률 배수">
                    <input
                      type="number"
                      min="0.001"
                      max="1"
                      step="0.01"
                      value={model.backbone_lr_mult}
                      onChange={(e) =>
                        setM("backbone_lr_mult", Number(e.target.value))
                      }
                    />
                  </Field>
                )}
              </details>
            )}
            </fieldset>
            {(!!cfg.augmentation.vertical_flip || !!cfg.augmentation.mixup_alpha) && <p className="error">
              저장된 수직 반전과 Mixup은 현재 학습 코드에서 지원하지 않습니다.
              <button onClick={() => set("augmentation", { ...cfg.augmentation, vertical_flip: 0, mixup_alpha: 0 })}>미지원 두 값을 0으로 변경</button>
            </p>}
            <details>
              <summary>선택 레이어 관찰</summary>
              <p className="muted">{options.capabilities.layer_debug ? "초기 1~10개 배치의 출력과 Gradient 통계만 저장하고 전체 학습을 계속합니다. 최대 128개 레이어." : options.capabilities.layer_debug_reason}</p>
              <label><input type="checkbox" checked={cfg.layer_debug_enabled ?? false}
                disabled={!options.capabilities.layer_debug && !cfg.layer_debug_enabled}
                onChange={e => set("layer_debug_enabled", e.target.checked)} />레이어 관찰 활성화</label>
              <Field label="레이어 패턴 (쉼표 구분, * 와 ? 지원)">
                <input value={cfg.layer_debug_patterns ?? "features.0,features.1.*,classifier.1"} maxLength={1024}
                  onChange={e => set("layer_debug_patterns", e.target.value)} />
              </Field>
              <button disabled={!options.capabilities.layer_debug} onClick={() => set("layer_debug_patterns", options.capabilities.default_layer_patterns)}>현재 모델의 추천 패턴</button>
              {numeric("관찰 배치 수", "layer_debug_batches", 1, 10)}
            </details>
            <div className="row">
              <button onClick={() => act(save)}>설정 저장</button>
              <button
                className="primary"
                onClick={() =>
                  act(async () => {
                    await save();
                    const j = await run("/jobs/train");
                    setJobId(j.id);
                    setRunIndex(CURRENT_RUN);
                  })
                }
              >
                학습 시작
              </button>
            </div>
          </fieldset>
        </Panel>
        <div>
          <Panel
            title="학습 결과"
            actions={
              <select
                aria-label="학습 기록 선택"
                value={selectedIndex}
                onChange={(e) => setRunIndex(Number(e.target.value))}
              >
                {project.runs.map((r, i) => (
                  <option key={r.run_id} value={i}>
                    {r.run_id}
                  </option>
                ))}
                {jobId && (
                  <option value={CURRENT_RUN}>{hasLiveJob ? "현재 학습" : "최근 작업 결과"}</option>
                )}
              </select>
            }
          >
            {missingResult && <p role="status">이 작업의 저장된 학습 결과가 없습니다. 작업 로그에서 종료 상태를 확인하세요.</p>}
            {selectedRun && <p className="muted" style={{ overflowWrap: "anywhere" }}>
              Run: {selectedRun.run_id} | 모델: {runModel(selectedRun)}<br />
              체크포인트 SHA-256: {selectedRun.config_snapshot.checkpoint_sha256 || "미기록"}
            </p>}
            <div className="stats">
              <Stat label="Best epoch" value={best || "—"} />
              {Object.entries(bestMetrics).map(([key, value]) =>
                <Stat key={key} label={`Best ${key}`} value={number(value)} />)}
            </div>
            <LossChart
              train={train}
              val={val}
              best={best}
              epochs={epochNumbers as number[] | undefined}
            />
            {selectedRun && (
              <p className="muted">
                선정 지표: {selectedRun.best_metric_name} ={" "}
                {number(selectedRun.best_metric)}
              </p>
            )}
            <h3>종합 평가 지표 — Best epoch {best || "—"}</h3>
            {Object.keys(bestMetrics).length ? <table aria-label="Best epoch 종합 평가 지표">
              <thead><tr><th>저장 지표</th><th>값</th></tr></thead>
              <tbody>{Object.entries(bestMetrics).map(([key, value]) =>
                <tr key={key}><td>{key}</td><td>{number(value)}</td></tr>)}</tbody>
            </table> : <p className="muted">저장된 Best 지표가 없습니다.</p>}
          </Panel>
          <LayerDebug snapshot={selectedIndex === CURRENT_RUN
            ? [...events].reverse().find(e => e.event === "layer_debug")?.args[0] ?? selectedRun?.config_snapshot.layer_debug?.latest
            : selectedRun?.config_snapshot.layer_debug?.latest} />
          <JobMonitor id={jobId} />
        </div>
      </div>
      <Panel title="학습 기록 비교">
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Run</th>
                <th>상태</th>
                <th>Best epoch</th>
                <th>선정 지표</th>
                <th>Best 값</th>
                <th>가중치</th>
                <th>CSV</th>
              </tr>
            </thead>
            <tbody>
              {project.runs.map((r, i) => (
                <tr key={r.run_id}>
                  <td>
                    <button
                      className="text-button"
                      onClick={() => setRunIndex(i)}
                    >
                      {r.run_id}
                    </button>
                  </td>
                  <td>{r.status}</td>
                  <td>{r.best_epoch || "—"}</td>
                  <td>{r.best_metric_name}</td>
                  <td>{number(r.best_metric)}</td>
                  <td>
                    {r.checkpoint_path && (
                      <a
                        href={
                          "/api/download?path=" +
                          encodeURIComponent(r.checkpoint_path)
                        }
                      >
                        다운로드
                      </a>
                    )}
                  </td>
                  <td>
                    <a href={"/api/runs/" + i + "/csv"}>CSV</a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!project.runs.length && (
          <Empty>
            학습을 시작하면 실행 기록과 Best 결과가 여기에 저장됩니다.
          </Empty>
        )}
      </Panel>
    </>
  );
}

function LayerDebug({ snapshot }: { snapshot?: Config }) {
  if (!snapshot?.layers) return null;
  return <Panel title="선택 레이어 관찰">
    <p className="muted">Run: {snapshot.run_id} | epoch {snapshot.epoch}, batch {snapshot.batch}. Gradient는 AMP 배율 제거 후 통계입니다.</p>
    <div className="table-scroll"><table><thead><tr>
      <th>레이어</th><th>입력 shape</th><th>출력 shape</th><th>최소</th><th>최대</th><th>평균</th><th>출력 유한</th><th>Gradient 평균</th><th>Gradient 유한</th>
    </tr></thead><tbody>{Object.entries(snapshot.layers as Record<string, Config>).map(([name, row]) => {
      const output = row.output ?? row.eval_output ?? {};
      return <tr key={name}><td>{name}</td><td>{JSON.stringify(row.input_shapes)}</td><td>{JSON.stringify(output.shape)}</td>
        <td>{number(output.min)}</td><td>{number(output.max)}</td><td>{number(output.mean)}</td><td>{String(output.finite ?? "N/A")}</td>
        <td>{number(row.gradient?.mean)}</td><td>{String(row.gradient?.finite ?? "N/A")}</td></tr>;
    })}</tbody></table></div>
  </Panel>;
}
