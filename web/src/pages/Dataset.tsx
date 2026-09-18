import { useEffect, useRef, useState } from "react";
import { api, type Config, type Job, type Project, terminal } from "../api";
import { Empty, Field, JobMonitor, Panel, PathField, useJob } from "../components";
import { AnnotationEditor } from "../annotation-editor";
import { useDraft } from "../drafts";
import type { PageProps } from "../page-context";

export interface DatasetImage { path: string; name: string; split: string; classes: string[]; annotated: boolean; error: string; revision: string }
interface DatasetInfo { splits: { split: string; count: number; classes: Record<string, number> }[]; class_names: string[]; root: string }
export function Dataset({ project, state, act, refresh, busy }: PageProps & { project: Project }) {
  const [root, setRoot] = useDraft(project.filepath, "dataset-root", project.data.root as string);
  const [data, setData] = useState<DatasetInfo | null>(null);
  const [split, setSplit] = useDraft(project.filepath, "dataset-split", "train");
  const [filter, setFilter] = useDraft(project.filepath, "dataset-class", "");
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("all");
  const [offset, setOffset] = useState(0);
  const [images, setImages] = useState<{ total: number; images: DatasetImage[] }>({ total: 0, images: [] });
  const [selected, setSelected] = useState<string[]>([]);
  const [name, setName] = useState("");
  const [targetClass, setTargetClass] = useState(project.data.class_names[0] || "");
  const [sourceClass, setSourceClass] = useState(project.data.class_names[0] || "");
  const [targetSplit, setTargetSplit] = useState("val");
  const [preview, setPreview] = useState<Config | null>(null);
  const [dialog, setDialog] = useState<"import" | "rename" | "delete" | null>(null);
  const [folder, setFolder] = useState("");
  const [job, setJob] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const [loading, setLoading] = useState(false);
  const [opened, setOpened] = useState<DatasetImage | null>(null);
  const [annotationJob, setAnnotationJob] = useState<string | null>(null);
  const [annotationError, setAnnotationError] = useState("");
  const upload = useRef<HTMLInputElement>(null);
  const { view } = useJob(job);
  const done = terminal(view?.job);
  useEffect(() => {
    if (!annotationJob || view?.job.id !== annotationJob || !terminal(view.job)) return;
    if (view.job.status === "completed") setOpened(null);
    else setAnnotationError(view.job.error || "정답 저장이 완료되지 않았습니다. 다시 저장해 주세요.");
    setAnnotationJob(null);
  }, [annotationJob, view?.job.id, view?.job.status]);
  useEffect(() => { if (done) { setRevision(v => v + 1); void refresh(); } }, [done]);
  useEffect(() => {
    let active = true;
    api<DatasetInfo>("/dataset").then(d => active && setData(d)).catch(e => active && setError(e.message));
    return () => { active = false; };
  }, [project.modified, revision]);
  useEffect(() => { setOffset(0); setSelected([]); }, [split, filter, search, status]);
  useEffect(() => {
    let active = true;
    setLoading(true);
    const timer = setTimeout(() => {
      const query = new URLSearchParams({ split, class_name: filter, q: search, status, offset: String(offset) });
      api<typeof images>(`/dataset/images?${query}`).then(d => { if (active) { setImages(d); setError(""); } })
        .catch(e => active && setError(e.message)).finally(() => active && setLoading(false));
    }, 150);
    return () => { active = false; clearTimeout(timer); };
  }, [split, filter, search, status, offset, project.modified, revision]);
  const edit = async (action: string, extra: Config = {}) => {
    const j = await api<Job>("/dataset/edit", "POST", { action, paths: selected, split, target_split: split, ...extra });
    setJob(j.id); setSelected([]); setDialog(null); await refresh(); return j.id;
  };
  const toggle = (path: string) => setSelected(old => old.includes(path) ? old.filter(p => p !== path) : [...old, path]);
  const counts = data?.splits.find(s => s.split === split)?.classes || {};
  const names: string[] = project.data.class_names;
  return <>
    <div className="dataset-toolbar">
      <div className="split-tabs" role="group" aria-label="데이터 분할">
        {["train", "val", "test"].map(s => <button key={s} className={split === s ? "active" : ""} onClick={() => setSplit(s)}>{s.toUpperCase()} <span>{data?.splits.find(d => d.split === s)?.count ?? "—"}</span></button>)}
      </div>
      <button className="primary" disabled={busy} onClick={() => setDialog("import")}>＋ 이미지 추가</button>
    </div>
    <details className="dataset-location"><summary>데이터 위치: {root}</summary>
      <PathField label="데이터 루트 폴더" directory value={root} onChange={setRoot} home={state.home} />
      <button disabled={busy || !root} onClick={() => act(async () => { await api("/project", "PUT", { data_root: root }); await refresh(); })}>경로 적용</button>
    </details>
    <div className="data-workspace">
      <aside className="class-sidebar">
        <h2>클래스</h2>
        <button className={!filter ? "class-filter active" : "class-filter"} onClick={() => setFilter("")}>전체 이미지 <span>{data?.splits.find(s => s.split === split)?.count || 0}</span></button>
        {Array.from(new Set([...names, ...Object.keys(counts)])).map((n, i) => <button key={n} className={filter === n ? "class-filter active" : "class-filter"} onClick={() => setFilter(n)}><span className="class-marker" style={{ background: classColor(i) }} />{n}<span>{counts[n] || 0}</span></button>)}
        <div className="class-actions">
          <Field label="새 클래스 이름"><input value={name} onChange={e => setName(e.target.value)} placeholder="예: scratch" /></Field>
          <button disabled={busy || !name.trim()} onClick={() => act(async () => { await api("/classes", "POST", { name: name.trim() }); setName(""); await refresh(); })}>클래스 추가</button>
          {filter && names.includes(filter) && <><button disabled={busy} onClick={() => { setName(filter); setDialog("rename"); }}>이름 변경</button>
            <button className="danger" disabled={busy} onClick={() => act(async () => setPreview(await api("/classes/preview-delete", "POST", { name: filter })))}>클래스 삭제</button></>}
        </div>
      </aside>
      <section className="dataset-content">
        <div className="dataset-search">
          <input aria-label="이미지 검색" placeholder="파일 이름으로 검색" value={search} onChange={e => setSearch(e.target.value)} />
          <select aria-label="라벨 상태" value={status} onChange={e => setStatus(e.target.value)}><option value="all">모든 라벨 상태</option><option value="unlabeled">미라벨 이미지</option><option value="error">정답 오류</option></select>
          <button disabled={loading} onClick={() => act(async () => { await api("/dataset?rescan=true"); setRevision(v => v + 1); })}>새로고침</button>
        </div>
        <div className="selection-bar">
          <label className="check"><input type="checkbox" aria-label="현재 페이지 전체 선택" checked={images.images.length > 0 && images.images.every(i => selected.includes(i.path))} onChange={e => setSelected(e.target.checked ? images.images.map(i => i.path) : [])} />{selected.length ? `${selected.length}장 선택` : `${images.total}장`}</label>
          {selected.length > 0 && <div className="bulk-actions"><select aria-label="이동할 분할" value={targetSplit} onChange={e => setTargetSplit(e.target.value)}>{["train", "val", "test"].map(s => <option key={s}>{s}</option>)}</select><button disabled={busy || split === targetSplit} onClick={() => act(() => edit("move", { target_split: targetSplit }))}>분할 이동</button>
            {!["classify", "anomaly"].includes(project.task) && <select aria-label="변경할 원래 클래스" value={sourceClass} onChange={e => setSourceClass(e.target.value)}>{names.map(n => <option key={n}>{n}</option>)}</select>}
            <select aria-label="변경할 클래스" value={targetClass} onChange={e => setTargetClass(e.target.value)}>{names.map(n => <option key={n}>{n}</option>)}</select><button disabled={busy || !targetClass} onClick={() => act(() => edit("reclass", { class_name: targetClass, source_class: sourceClass }))}>클래스 변경</button>
            <button className="danger" disabled={busy} onClick={() => setDialog("delete")}>이미지 삭제</button></div>}
        </div>
        {error && <p className="error" role="alert">{error}</p>}
        {loading && <p className="muted" role="status">이미지 목록 불러오는 중</p>}
        {images.images.length ? <div className="dataset-grid-viewport"><div className="dataset-grid">{images.images.map(item => <article key={item.path} className={selected.includes(item.path) ? "image-card selected" : "image-card"}>
          <label className="image-select"><input type="checkbox" aria-label={`${item.name} 선택`} checked={selected.includes(item.path)} onChange={() => toggle(item.path)} /></label>
          <button className="image-open" aria-label={`${item.name} 열기`} onClick={() => { setAnnotationError(""); setOpened(item); }}><img loading="lazy" src={`/api/image?size=256&path=${encodeURIComponent(item.path)}&v=${item.revision}`} alt={item.name} /></button>
          <div className="image-card-meta"><strong title={item.name}>{item.name}</strong><div>{item.classes.length ? item.classes.map(n => <span key={n} className="image-class">{n}</span>) : <span className="muted">{item.annotated ? "배경 / 정답 없음" : "미라벨"}</span>}</div>{item.error && <span className="error" title={item.error}>정답 오류</span>}</div>
        </article>)}</div></div> : !loading && <Empty>조건에 맞는 이미지가 없습니다. 이미지를 추가하거나 필터를 변경하세요.</Empty>}
        <div className="pagination"><button disabled={!offset} onClick={() => { setOffset(Math.max(0, offset - 36)); setSelected([]); }}>이전</button><span>{images.total ? offset + 1 : 0}–{Math.min(offset + 36, images.total)} / {images.total}</span><button disabled={offset + 36 >= images.total} onClick={() => { setOffset(offset + 36); setSelected([]); }}>다음</button></div>
      </section>
    </div>
    <JobMonitor id={job} />
    {dialog && <DatasetDialog onClose={() => setDialog(null)} title={dialog === "import" ? "이미지 추가" : dialog === "rename" ? "클래스 이름 변경" : "이미지 삭제"}>
      {dialog === "import" ? <>
        <p>{split.toUpperCase()}에 이미지를 복사합니다. 원본 파일은 유지됩니다.</p>
        {["classify", "anomaly"].includes(project.task) && <Field label="추가할 클래스"><select value={targetClass} onChange={e => setTargetClass(e.target.value)}>{names.map(n => <option key={n}>{n}</option>)}</select></Field>}
        <input ref={upload} type="file" multiple accept="image/*,.txt" aria-label="추가할 이미지 파일" onChange={e => { const files = Array.from(e.target.files || []); act(async () => {
          const batch = crypto.randomUUID().replaceAll("-", ""); const paths: string[] = [];
          for (const file of files) { const res = await fetch(`/api/dataset/upload?name=${encodeURIComponent(file.name)}&batch=${batch}`, { method: "POST", headers: { "X-Studio-Request": "1" }, body: file }); const value = await res.json(); if (!res.ok) throw new Error(value.detail || "파일 추가 실패"); if (!file.name.toLowerCase().endsWith(".txt")) paths.push(value.path); }
          await edit("import", { paths, class_name: targetClass });
        }); }} />
        <p className="muted">탐지/폴리곤은 같은 이름의 .txt 정답도 함께 선택하세요. 픽셀 마스크가 있는 데이터는 images/labels/masks 구조의 폴더로 가져오세요.</p>
        <PathField label="또는 로컬 이미지 폴더" directory value={folder} onChange={setFolder} home={state.home} />
        <button className="primary" disabled={busy || !folder} onClick={() => act(() => edit("import", { paths: [], folder, class_name: targetClass }))}>폴더에서 가져오기</button>
      </> : dialog === "rename" ? <><Field label={`${filter}의 새 이름`}><input value={name} onChange={e => setName(e.target.value)} /></Field><button className="primary" disabled={busy || !name || name === filter} onClick={() => act(async () => { await edit("rename_class", { paths: [], source_class: filter, new_name: name }); setFilter(""); })}>이름 변경</button></> : <><p>선택한 이미지 {selected.length}장과 연결된 정답을 백업한 뒤 데이터셋에서 삭제합니다.</p><button className="danger" disabled={busy} onClick={() => act(() => edit("delete"))}>백업 후 삭제</button></>}
    </DatasetDialog>}
    {preview && <DatasetDialog title={`${preview.class_name} 클래스 삭제`} onClose={() => setPreview(null)}><p>이미지 {preview.image_count}장, 제거할 정답 {preview.removed_annotations}개, 재번호화할 정답 {preview.remapped_annotations}개, 제거할 마스크 픽셀 {preview.removed_pixels}개</p><p>백업 후 데이터를 변경합니다. 기존 모델의 클래스 목록은 유지됩니다.</p><button className="danger" disabled={busy} onClick={() => act(async () => { const j = await api<Job>("/classes/delete", "POST", { name: preview.class_name, preview_digest: preview.preview_digest }); setJob(j.id); setPreview(null); setFilter(""); await refresh(); })}>백업 후 클래스 삭제</button></DatasetDialog>}
    {opened && (["detect", "segment", "obb"].includes(project.task) ? <AnnotationEditor key={opened.path} item={opened} project={project} busy={busy || !!annotationJob} saveError={annotationError} onClose={() => { if (!annotationJob) setOpened(null); }} onSave={rows => act(async () => { setAnnotationError(""); setAnnotationJob(await edit("annotations", { paths: [opened.path], split: opened.split, annotations: rows })); })} /> : <DatasetImageViewer item={opened} onClose={() => setOpened(null)} />)}
  </>;
}
export const classColor = (index: number) => ["#1428a0", "#007e88", "#c65316", "#7542ad", "#b72d66", "#586522"][index % 6];

function DatasetImageViewer({ item, onClose }: { item: DatasetImage; onClose: () => void }) {
  const [zoom, setZoom] = useState(100);
  return <DatasetDialog title={`${item.name} 이미지 보기`} onClose={onClose}>
    <div className="dataset-image-toolbar">
      <Field label="확대"><input aria-label="데이터셋 이미지 확대" type="range" min="25" max="400" step="25" value={zoom} onChange={e => setZoom(Number(e.target.value))} /></Field>
      <span>{zoom}%</span>
      <button onClick={() => setZoom(100)}>화면 맞춤</button>
    </div>
    <div className="dataset-image-viewport"><img src={`/api/image?size=4096&path=${encodeURIComponent(item.path)}&v=${item.revision}`} alt={item.name} style={{ width: `${zoom}%`, maxWidth: "none" }} /></div>
    <p className="muted">격자에서는 썸네일만 표시합니다. 이미지를 크게 보려면 썸네일을 선택하세요.</p>
  </DatasetDialog>;
}

export function DatasetDialog({ title, children, onClose }: { title: string; children: React.ReactNode; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { ref.current?.showModal(); }, []);
  return <dialog ref={ref} className="dataset-dialog" onCancel={event => { event.preventDefault(); onClose(); }}><div className="panel-head"><h2>{title}</h2><button aria-label="닫기" onClick={onClose}>닫기</button></div>{children}</dialog>;
}
