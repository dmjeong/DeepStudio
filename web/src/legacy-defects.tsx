import { useEffect, useRef, useState } from "react";
import { api, filename, type Job } from "./api";
import { Check, Field, JobMonitor, Panel, PathField } from "./components";
import type { PageProps } from "./pages";

type Settings = {
  folder: string; output: string; roi: string; texture: string; per_image: number;
  params: { types: string[]; intensity: number; size_ratio: number; count: number;
    seed: number | null; feather: number; roughness: number; mix_types: boolean };
};
type Sample = { id: string; source: string; seed: number; changed_fraction: number; generation_sec: number };
const names: Record<string, string> = { scratch: "스크래치", stain: "얼룩", cutout: "패치 결손", noise: "노이즈", texture: "텍스처 교체", elastic: "탄성 변형", color: "색상 이상" };

export function LegacyDefects({ state, run, act, busy }: PageProps) {
  const project = state.project;
  const draftKey = `studio-defects-draft:${project?.filepath}`;
  const [cfg, setCfg] = useState<Settings | null>(null);
  const [settingsMessage, setSettingsMessage] = useState("");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [monitorId, setMonitorId] = useState<string | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [current, setCurrent] = useState<Sample | null>(null);
  const [error, setError] = useState("");
  const [zoom, setZoom] = useState(false);
  const latest = useRef<Settings | null>(null);
  const saved = useRef("");
  const saveQueue = useRef<Promise<unknown>>(Promise.resolve());

  useEffect(() => {
    let live = true;
    api<Settings>("/defects/settings").then(value => {
      if (!live) return;
      saved.current = JSON.stringify(value);
      try {
        const draft = localStorage.getItem(draftKey);
        if (draft) {
          const parsed = JSON.parse(draft);
          if (parsed && typeof parsed === "object" && parsed.params && Array.isArray(parsed.params.types)
              && ["folder", "output", "roi", "texture"].every(key => typeof parsed[key] === "string")) value = parsed;
          else localStorage.removeItem(draftKey);
        }
      } catch { localStorage.removeItem(draftKey); }
      latest.current = value;
      setCfg(value);
    }).catch(e => { if (live) setError(e.message); });
    return () => { live = false; };
  }, [draftKey]);

  const change = (value: Settings) => {
    latest.current = value;
    setCfg(value);
    try { localStorage.setItem(draftKey, JSON.stringify(value)); }
    catch { setSettingsMessage("임시 설정 보관 실패. 설정 저장 버튼 사용 필요"); }
  };
  const saveSettings = async (value: Settings) => {
    const task = saveQueue.current.catch(() => {}).then(() =>
      api("/defects/settings", "PUT", { project_path: project!.filepath, settings: value }));
    saveQueue.current = task;
    await task;
    saved.current = JSON.stringify(value);
    if (JSON.stringify(latest.current) === saved.current) {
      localStorage.removeItem(draftKey);
      setSettingsMessage("설정 저장 완료");
    }
  };
  useEffect(() => {
    if (!cfg || busy || JSON.stringify(cfg) === saved.current) return;
    const timer = setTimeout(() => {
      void saveSettings(cfg).catch(e => setSettingsMessage(`설정 저장 대기: ${e.message}`));
    }, 600);
    return () => clearTimeout(timer);
  }, [cfg, busy]);

  useEffect(() => {
    let live = true;
    api<Job[]>("/defects/jobs").then(value => {
      if (!live) return;
      setJobs(value);
      setJobId(old => old && value.some(job => job.id === old) ? old : value[0]?.id || null);
    }).catch(e => { if (live) setError(e.message); });
    return () => { live = false; };
  }, [project?.filepath, state.active_job, state.jobs.length]);
  useEffect(() => {
    setSelected(new Set()); setPage(0); setCurrent(null); setSamples([]);
  }, [jobId]);
  useEffect(() => {
    if (!jobId) return;
    let live = true;
    let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      try {
        const data = await api<{total: number; samples: Sample[]}>(`/defects/${jobId}/candidates?offset=${page * 48}&limit=48`);
        if (live) {
          setSamples(data.samples); setTotal(data.total);
          setCurrent(old => data.samples.find(sample => sample.id === old?.id) || data.samples[0] || null);
          setError("");
        }
      } catch (e) { if (live) setError((e as Error).message); }
      if (live && state.active_job === jobId) timer = setTimeout(load, 1000);
    };
    void load();
    return () => { live = false; clearTimeout(timer); };
  }, [jobId, page, state.active_job]);

  if (!project || !cfg) return <Panel title="Defect Gen"><p>{error || "설정 불러오는 중"}</p></Panel>;
  const start = (preview: boolean) => act(async () => {
    await saveSettings(cfg);
    const job = await run("/jobs/defects", { ...cfg, preview, project_path: project.filepath });
    setJobId(job.id); setMonitorId(job.id); setPage(0);
  });
  const toggle = (id: string, checked: boolean) => setSelected(old => {
    const next = new Set(old); if (checked) next.add(id); else next.delete(id); return next;
  });
  const output = state.jobs.find(job => job.id === monitorId)?.output;
  return <>
    <Panel title="Defect Gen - 생성과 검수">
      <p className="muted">규칙 기반 표면 합성입니다. 원본과 결과를 비교하고 선택한 후보를 그대로 저장하세요. 변경 마스크는 실제 픽셀 변화이며 불량 정답 라벨은 별도 검수가 필요합니다.</p>
      <fieldset disabled={busy}>
        <div className="columns">
          <PathField label="정상 이미지 폴더" directory value={cfg.folder} onChange={folder => change({...cfg, folder})} home={state.home}/>
          <Field label="검수 결과 저장 폴더의 절대 경로"><input value={cfg.output} onChange={e => change({...cfg, output: e.target.value})}/></Field>
        </div>
        <div className="check-grid">{Object.entries(names).map(([key, label]) =>
          <Check key={key} label={label} value={cfg.params.types.includes(key)} onChange={checked => change({...cfg, params: {...cfg.params, types: checked ? [...cfg.params.types, key] : cfg.params.types.filter(v => v !== key)}})}/>)}</div>
        <div className="form-grid">
          {([
            ["강도", "intensity", .01, 1, .05], ["후보 사각형 면적 비율", "size_ratio", .001, 1, .01],
            ["경계 부드러움", "feather", 0, 1, .05], ["불규칙도", "roughness", 0, 1, .05],
            ["이미지당 패턴 수", "count", 1, 100, 1]
          ] as const).map(([label, key, min, max, step]) => <Field key={key} label={label}>
            <input type="number" min={min} max={max} step={step} value={cfg.params[key]} onChange={e => change({...cfg, params: {...cfg.params, [key]: Number(e.target.value)}})}/>
          </Field>)}
          <Field label="시드 (-1: 무작위)"><input type="number" min={-1} max={4294967295} value={cfg.params.seed ?? -1} onChange={e => change({...cfg, params: {...cfg.params, seed: Number(e.target.value) === -1 ? null : Number(e.target.value)}})}/></Field>
          <Field label="원본당 생성 수"><input type="number" min={1} max={100} value={cfg.per_image} onChange={e => change({...cfg, per_image: Number(e.target.value)})}/></Field>
        </div>
        <Check label="선택한 유형 혼합" value={cfg.params.mix_types} onChange={mix_types => change({...cfg, params: {...cfg.params, mix_types}})}/>
        <details><summary>ROI와 참조 질감</summary><div className="columns">
          <PathField label="ROI 마스크" value={cfg.roi} onChange={roi => change({...cfg, roi})} home={state.home} hint="원본과 동일한 크기 필요. 밝은 영역에만 생성합니다."/>
          <PathField label="참조 질감" value={cfg.texture} onChange={texture => change({...cfg, texture})} home={state.home}/>
        </div></details>
        <div className="toolbar">
          <button onClick={() => act(() => saveSettings(cfg))}>설정 저장</button>
          <button disabled={!cfg.folder || !cfg.output || !cfg.params.types.length} onClick={() => start(true)}>첫 이미지 샘플 생성</button>
          <button className="primary" disabled={!cfg.folder || !cfg.output || !cfg.params.types.length} onClick={() => start(false)}>대량 후보 생성</button>
          <span className="muted">{settingsMessage}</span>
        </div>
      </fieldset>
    </Panel>
    <Panel title="생성 결과 검수">
      {error && <p role="alert">{error}</p>}
      <Field label="합성 작업 이력"><select disabled={busy} value={jobId || ""} onChange={e => {setJobId(e.target.value || null); setMonitorId(e.target.value || null);}}>
        {!jobs.length && <option value="">생성 결과 없음</option>}
        {jobs.map(job => <option key={job.id} value={job.id}>{job.created_at.slice(0,19)} | {job.status}</option>)}
      </select></Field>
      {current && jobId && <>
        <Check label="원본 크기로 확대" value={zoom} onChange={setZoom}/>
        <div className={`defect-comparison ${zoom ? "native-size" : ""}`}>
          {([['original', '원본'], ['image', '합성 결과'], ['mask', '실제 변경 마스크']] as const).map(([kind, label]) =>
            <figure key={kind}><figcaption>{label}</figcaption><div><img alt={label} src={`/api/defects/${jobId}/image/${current.id}?kind=${kind}&size=${zoom ? 4096 : 1024}`}/></div></figure>)}
        </div>
        <p className="muted">{filename(current.source)} | 시드 {current.seed} | 실제 변경 {(current.changed_fraction * 100).toFixed(2)}% | 생성 {current.generation_sec?.toFixed(3)}초</p>
      </>}
      <div className="defect-candidates">{samples.map(sample => <div key={sample.id} className={sample.id === current?.id ? 'active' : ''}>
        <Check label={`${sample.id} 후보 선택`} value={selected.has(sample.id)} onChange={checked => toggle(sample.id, checked)}/>
        <button onClick={() => setCurrent(sample)}>{sample.id} {filename(sample.source)}</button>
      </div>)}</div>
      <div className="toolbar">
        <button disabled={!samples.length} onClick={() => setSelected(old => new Set([...old, ...samples.map(s => s.id)]))}>표시된 결과 선택</button>
        <button disabled={!selected.size} onClick={() => setSelected(new Set())}>선택 해제</button>
        <button className="primary" disabled={busy || !selected.size || !jobId} onClick={() => act(async () => {
          const job = await run('/jobs/defect-publish', { project_path: project.filepath, source_job: jobId, sample_ids: [...selected].sort(), output: cfg.output });
          setMonitorId(job.id);
        })}>선택한 {selected.size}장 그대로 저장</button>
        <span className="muted">후보 {total}장 / 선택 {selected.size}장</span>
      </div>
      <div className="toolbar">
        <button disabled={!page} onClick={() => setPage(page - 1)}>이전</button>
        <span>{page + 1} / {Math.max(1, Math.ceil(total / 48))}</span>
        <button disabled={(page + 1) * 48 >= total} onClick={() => setPage(page + 1)}>다음</button>
      </div>
      {output?.output_dir && <p>저장 폴더: {output.output_dir}</p>}
    </Panel>
    <JobMonitor id={monitorId || jobId}/>
  </>;
}
