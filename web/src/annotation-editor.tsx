import { useEffect, useRef, useState } from "react";
import { obbFromThreePoints } from "./obb";
import { api, type Project } from "./api";
import { Field } from "./components";
import { classColor, DatasetDialog, type DatasetImage } from "./pages/Dataset";
import { TeachingEditor, type MaskPayload } from "./teaching-editor";
export interface Annotation { class_id: number; coordinates: number[] }
interface AnnotationView { width: number; height: number; has_mask: boolean; annotations: Annotation[] }
export interface AnnotationEditorProps { item: DatasetImage; project: Project; busy: boolean; saveError?: string; onSave: (rows: Annotation[] | MaskPayload, next?: number) => void; onClose: () => void; navigation?: { index: number; count: number } }
export function AnnotationEditor(props: AnnotationEditorProps) {
  return props.project.task === "obb" ? <LegacyAnnotationEditor {...props} /> : <TeachingEditor {...props} />;
}
function LegacyAnnotationEditor({ item, project, busy, saveError, onSave, onClose }: AnnotationEditorProps) {
  const [data, setData] = useState<AnnotationView | null>(null);
  const [rows, setRows] = useState<Annotation[]>([]);
  const [classId, setClassId] = useState(0);
  const [zoom, setZoom] = useState(100);
  const [points, setPoints] = useState<number[]>([]);
  const [error, setError] = useState("");
  const [dirty, setDirty] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const anchor = useRef<number[] | null>(null);
  const pointsRef = useRef<number[]>([]);
  const submittingRef = useRef(false);
  const updatePoints = (next: number[]) => { pointsRef.current = next; setPoints(next); };
  useEffect(() => { let active = true; api<AnnotationView>(`/dataset/annotations?path=${encodeURIComponent(item.path)}&split=${item.split}`).then(v => { if (active) { setData(v); setRows(v.annotations); } }).catch(e => active && setError(e.message)); return () => { active = false; }; }, [item.path]);
  useEffect(() => { if (!busy) { submittingRef.current = false; setSubmitting(false); } }, [busy]);
  useEffect(() => { if (saveError) { submittingRef.current = false; setSubmitting(false); } }, [saveError]);
  const editable = ["detect", "segment", "obb"].includes(project.task) && !data?.has_mask && !busy && project.data.class_names.length > 0;
  const coordinate = (event: React.PointerEvent<SVGSVGElement>) => { const rect = event.currentTarget.getBoundingClientRect(); return [Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))]; };
  const add = (coords: number[]) => { setRows(old => [...old, { class_id: classId, coordinates: coords }]); setDirty(true); updatePoints([]); };
  return <DatasetDialog title={item.name} onClose={() => { if ((!dirty && !points.length) || window.confirm("저장하지 않은 정답 변경을 버리고 닫을까요?")) onClose(); }}>
    {(error || saveError) && <p className="error" role="alert">{error || saveError}</p>}
    <div className="annotation-toolbar"><Field label="확대"><input type="range" min="50" max="400" step="25" value={zoom} onChange={e => setZoom(Number(e.target.value))} /></Field><span>{zoom}%</span>{editable && <Field label="정답 클래스"><select value={classId} onChange={e => setClassId(Number(e.target.value))}>{project.data.class_names.map((n: string, i: number) => <option key={n} value={i}>{n}</option>)}</select></Field>}</div>
    {editable && <p className="muted">{project.task === "detect" ? "이미지 위에서 드래그해 박스를 추가하세요." : project.task === "obb" ? "첫 두 번 클릭으로 한 변을 정하고, 세 번째 클릭으로 폭을 정하세요." : "윤곽을 따라 꼭짓점을 클릭한 뒤 폴리곤 완료를 누르세요."} 정답 목록에서 클래스 변경과 삭제가 가능합니다.</p>}
    <div className="annotation-canvas-scroll">{data && <svg className="annotation-canvas" viewBox={`0 0 ${data.width} ${data.height}`} style={{ width: `${zoom}%`, minWidth: `${zoom}%`, aspectRatio: `${data.width}/${data.height}` }} onPointerDown={e => { if (!editable || e.button !== 0) return; e.currentTarget.setPointerCapture(e.pointerId); if (project.task === "detect") anchor.current = coordinate(e); else {
        const next = [...pointsRef.current, ...coordinate(e)];
        if (project.task === "obb" && next.length === 6) {
          try { add(obbFromThreePoints(next, data.width, data.height)); setError(""); }
          catch (err) { setError(err instanceof Error ? err.message : String(err)); updatePoints([]); }
        } else updatePoints(next);
      } }} onPointerUp={e => { if (!anchor.current || !data) return; const [x1, y1] = anchor.current; const [x2, y2] = coordinate(e); anchor.current = null; const w = Math.abs(x2 - x1), h = Math.abs(y2 - y1); if (w * data.width >= 2 && h * data.height >= 2) add([(x1 + x2) / 2, (y1 + y2) / 2, w, h]); }}>
      <image href={`/api/image?path=${encodeURIComponent(item.path)}&size=4096&v=${item.revision}`} width={data.width} height={data.height} />
      {rows.map((row, i) => { const c = row.coordinates; const color = classColor(row.class_id); return <g key={i} stroke={color} strokeWidth={2} fill={color} fillOpacity={0.12} style={{ pointerEvents: "none" }}>{project.task === "detect" ? <rect x={(c[0] - c[2] / 2) * data.width} y={(c[1] - c[3] / 2) * data.height} width={c[2] * data.width} height={c[3] * data.height} vectorEffect="non-scaling-stroke" /> : <polygon points={c.reduce<string[]>((a, v, j) => { if (j % 2 === 0) a.push(`${v * data.width},${c[j + 1] * data.height}`); return a; }, []).join(" ")} vectorEffect="non-scaling-stroke" />}</g>; })}
      {points.length > 0 && <polyline points={points.reduce<string[]>((a, v, i) => { if (i % 2 === 0) a.push(`${v * data.width},${points[i + 1] * data.height}`); return a; }, []).join(" ")} fill="none" stroke="#1428a0" strokeWidth="3" vectorEffect="non-scaling-stroke" />}
    </svg>}</div>
    {data?.has_mask && <p>픽셀 마스크 포함: 데이터셋의 클래스 변경으로 선택 클래스 픽셀을 변경할 수 있습니다.</p>}
    {project.task === "segment" && editable && <div className="row"><button disabled={points.length < 6} onClick={() => add(points)}>폴리곤 완료</button><button onClick={() => updatePoints([])}>그리기 취소</button></div>}
    {project.task === "obb" && editable && <button disabled={!points.length} onClick={() => updatePoints([])}>그리기 취소</button>}
    {editable && <><div className="annotation-list">{rows.map((r, i) => <div key={i}><span>#{i + 1}</span><select aria-label={`정답 ${i + 1} 클래스`} value={r.class_id} onChange={e => { setRows(old => old.map((row, j) => j === i ? { ...row, class_id: Number(e.target.value) } : row)); setDirty(true); }}>{project.data.class_names.map((n: string, j: number) => <option key={n} value={j}>{n}</option>)}</select><button className="danger" onClick={() => { setRows(old => old.filter((_, j) => i !== j)); setDirty(true); }}>정답 삭제</button></div>)}</div><button className="primary" disabled={busy || submitting || !dirty || points.length > 0} onClick={() => { if (submittingRef.current) return; submittingRef.current = true; setSubmitting(true); onSave(rows); }}>정답 저장</button></>}
  </DatasetDialog>;
}
