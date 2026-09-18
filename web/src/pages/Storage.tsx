import { useEffect, useState } from "react";
import { api, type Job } from "../api";
import { Panel } from "../components";
import type { PageProps } from "../page-context";
interface Usage { bytes: number; upload_bytes: number; warnings: string[]; jobs: (Job & { bytes: number })[] }
const bytes = (n: number) => n > 1073741824 ? `${(n / 1073741824).toFixed(2)} GB` : `${(n / 1048576).toFixed(1)} MB`;
export function Storage({ state, act, refresh }: PageProps) {
  const [usage, setUsage] = useState<Usage | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [days, setDays] = useState(30);
  const reload = async () => setUsage(await api<Usage>("/storage"));
  useEffect(() => { void reload().catch(e => setError(e.message)); }, [state.active_job]);
  return <Panel title="작업 기록과 저장 공간" actions={<button onClick={() => act(reload)}>새로고침</button>}>
    {error && <p className="error">{error}</p>}
    <p>작업 결과 {bytes(usage?.bytes || 0)} / 업로드 임시 파일 {bytes(usage?.upload_bytes || 0)}</p>
    <p className="muted">추론 결과와 로그를 ZIP으로 내보내거나 선택해 삭제할 수 있습니다. 모델 가중치와 데이터셋 원본은 삭제하지 않습니다.</p>
    {usage?.warnings.map((w, i) => <p className="error" key={i}>{w}</p>)}
    <div className="toolbar"><label className="field"><span>보관 기간 (일)</span><input type="number" min="1" max="3650" value={days} onChange={e => setDays(Number(e.target.value))} /></label><button disabled={!Number.isFinite(days) || days < 1} onClick={() => setSelected((usage?.jobs || []).filter(j => j.id !== state.active_job && Date.parse(j.created_at) < Date.now() - days * 86400000).map(j => j.id))}>기간이 지난 기록 선택</button><button className="danger" disabled={!selected.length} onClick={() => { if (window.confirm(`선택한 작업 기록 ${selected.length}개와 미리보기를 삭제할까요?`)) act(async () => { await api("/storage/delete", "POST", { ids: selected }); setSelected([]); await reload(); await refresh(); }); }}>선택 기록 삭제</button><button disabled={!!state.active_job} onClick={() => act(async () => { await api("/storage/clear-uploads", "POST"); await reload(); })}>업로드 임시 파일 정리</button></div>
    {usage?.jobs.map(j => <div className="storage-row" key={j.id}><input type="checkbox" aria-label={`${j.id} 선택`} disabled={j.id === state.active_job} checked={selected.includes(j.id)} onChange={e => setSelected(old => e.target.checked ? [...old, j.id] : old.filter(id => id !== j.id))} /><div><strong>{j.kind} / {j.status}</strong><small>{j.created_at} / {j.project_path || "프로젝트 연결 없음"}</small></div><span>{bytes(j.bytes)}</span>{j.id !== state.active_job && <a className="button" href={`/api/history/${j.id}/archive`}>기록 내보내기</a>}</div>)}
  </Panel>;
}
