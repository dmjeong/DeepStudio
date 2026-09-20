import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import { Field } from "./components";
import { classColor, DatasetDialog } from "./pages/Dataset";
import type { Annotation, AnnotationEditorProps } from "./annotation-editor";

type Shape = { kind: "polygon" | "stroke" | "box"; class_id: number; points: number[]; width?: number };
export interface MaskPayload { schema_version: number; width: number; height: number; base_png: string; shapes: Shape[]; persisted?: boolean }
type Mode = "polygon" | "rectangle" | "brush" | "erase" | "select" | "pan";
type Gesture = { type: string; start: number[]; original?: Shape; index: number; handle: number; points: number[] };
const copy = (shapes: Shape[]) => structuredClone(shapes);
const clamp = (value: number) => Math.max(0, Math.min(1, value));
const boxPoints = (p: number[]) => [p[0]-p[2]/2, p[1]-p[3]/2, p[0]+p[2]/2, p[1]-p[3]/2,
  p[0]+p[2]/2, p[1]+p[3]/2, p[0]-p[2]/2, p[1]+p[3]/2];
const polygonArea = (p: number[]) => Math.abs(p.reduce((sum, _, i) => i % 2 ? sum :
  sum + p[i]*p[(i+3)%p.length] - p[i+1]*p[(i+2)%p.length], 0)) / 2;

export function TeachingEditor({ item, project, busy, saveError, onSave, onClose, navigation }: AnnotationEditorProps) {
  const segment = project.task === "segment";
  const names: string[] = project.data.class_names;
  const [data, setData] = useState<MaskPayload | null>(null);
  const [shapes, setShapes] = useState<Shape[]>([]);
  const [mode, setMode] = useState<Mode>(segment ? "polygon" : "rectangle");
  const [classId, setClassId] = useState(segment && names.length > 1 ? 1 : 0);
  const [brush, setBrush] = useState(20);
  const [opacity, setOpacity] = useState(45);
  const [zoom, setZoom] = useState(100);
  const [fitWidth, setFitWidth] = useState(640);
  const [selected, setSelected] = useState(-1);
  const [points, setPoints] = useState<number[]>([]);
  const [preview, setPreview] = useState<Shape | null>(null);
  const [maskUrl, setMaskUrl] = useState("");
  const [base, setBase] = useState<HTMLCanvasElement | null>(null);
  const [dirty, setDirty] = useState(false);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const undo = useRef<Shape[][]>([]), redo = useRef<Shape[][]>([]);
  const gesture = useRef<Gesture | null>(null), space = useRef(false);
  const scroll = useRef<HTMLDivElement>(null);
  const submitLock = useRef(false);
  useEffect(() => {
    const element = scroll.current; if (!element || !data) return;
    const fit = () => setFitWidth(Math.max(1, Math.min(element.clientWidth-4, (element.clientHeight-4)*data.width/data.height)));
    const observer = new ResizeObserver(fit); observer.observe(element); fit();
    return () => observer.disconnect();
  }, [data]);
  useEffect(() => {
    let active = true;
    const query = `path=${encodeURIComponent(item.path)}&split=${item.split}`;
    (segment ? api<MaskPayload>(`/dataset/mask-annotations?${query}`) :
      api<{ width: number; height: number; annotations: Annotation[] }>(`/dataset/annotations?${query}`).then(v => ({
        schema_version: 1, width: v.width, height: v.height, base_png: "",
        shapes: v.annotations.map(row => ({ kind: "box" as const, class_id: row.class_id, points: row.coordinates })),
      }))).then(value => { if (active) { setData(value); setShapes(value.shapes); } })
      .catch(e => active && setError(e.message));
    return () => { active = false; };
  }, [item.path, item.split, segment]);
  useEffect(() => { if (!busy || saveError) { setSubmitting(false); submitLock.current = false; } }, [busy, saveError]);
  useEffect(() => {
    if (!segment || !data) return;
    let active = true;
    const image = new Image();
    image.onload = () => {
      if (!active) return;
      const canvas = document.createElement("canvas");
      const scale = Math.min(1, 2048 / Math.max(data.width, data.height));
      canvas.width = Math.max(1, Math.round(data.width*scale)); canvas.height = Math.max(1, Math.round(data.height*scale));
      const context = canvas.getContext("2d")!;
      context.imageSmoothingEnabled = false;
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < pixels.data.length; i += 4) {
        const id = pixels.data[i];
        const color = id === 255 ? "#ffffff" : classColor(id);
        pixels.data[i] = parseInt(color.slice(1, 3), 16); pixels.data[i+1] = parseInt(color.slice(3, 5), 16);
        pixels.data[i+2] = parseInt(color.slice(5, 7), 16); pixels.data[i+3] = id === 0 ? 0 : id === 255 ? 110 : 255;
      }
      context.putImageData(pixels, 0, 0); setBase(canvas);
    };
    image.onerror = () => active && setError("원본 마스크를 읽을 수 없습니다.");
    image.src = `data:image/png;base64,${data.base_png}`;
    return () => { active = false; };
  }, [data, segment]);
  useEffect(() => {
    if (!base || !data) return;
    const canvas = document.createElement("canvas"); canvas.width = base.width; canvas.height = base.height;
    const context = canvas.getContext("2d")!; context.drawImage(base, 0, 0);
    for (const shape of shapes) {
      context.globalCompositeOperation = shape.class_id === 0 ? "destination-out" : "source-over";
      context.strokeStyle = context.fillStyle = shape.class_id === 255 ? "#ffffff" : classColor(shape.class_id);
      const p = shape.points;
      context.beginPath(); context.moveTo(p[0]*(canvas.width-1), p[1]*(canvas.height-1));
      for (let i = 2; i < p.length; i += 2) context.lineTo(p[i]*(canvas.width-1), p[i+1]*(canvas.height-1));
      if (shape.kind === "polygon") { context.closePath(); context.fill(); }
      else {
        context.lineWidth = (shape.width || 1)*canvas.width/data.width; context.lineCap = context.lineJoin = "round";
        if (p.length === 2) { context.arc(p[0]*(canvas.width-1), p[1]*(canvas.height-1), context.lineWidth/2, 0, 2*Math.PI); context.fill(); }
        else context.stroke();
      }
    }
    setMaskUrl(canvas.toDataURL());
  }, [base, data, shapes]);
  const change = (next: Shape[]) => { undo.current = [...undo.current.slice(-99), copy(shapes)]; redo.current = []; setShapes(next); setDirty(true); };
  const cancel = () => { gesture.current = null; setPreview(null); setPoints([]); };
  const chooseMode = (value: Mode) => { cancel(); setMode(value); };
  const history = (back: boolean) => {
    const source = back ? undo : redo, target = back ? redo : undo;
    if (!source.current.length) return;
    cancel(); target.current.push(copy(shapes)); setShapes(source.current.pop()!); setSelected(-1); setDirty(true);
  };
  const add = (shape: Shape) => { change([...shapes, shape]); setSelected(shapes.length); setPoints([]); setPreview(null); };
  const finish = () => {
    if (!data || points.length < 6 || polygonArea(points)*data.width*data.height < .5) { setError("면적이 있는 다각형을 꼭짓점 3개 이상으로 그리세요."); return; }
    setError(""); add({ kind: "polygon", class_id: classId, points });
  };
  const save = (next = 0) => {
    if (!data || busy || submitLock.current || points.length || gesture.current || !names.length || (segment && !base)) return;
    setSubmitting(true); submitLock.current = true;
    onSave(segment ? { ...data, shapes, persisted: undefined } : shapes.map(row => ({ class_id: row.class_id, coordinates: row.points })), next);
  };
  const remove = () => { if (selected >= 0) { change(shapes.filter((_, i) => i !== selected)); setSelected(-1); } };
  const duplicate = () => {
    const value = shapes[selected]; if (value?.kind !== "box") return;
    const [x, y, w, h] = value.points;
    add({ ...value, points: [Math.min(1-w/2, x+.02), Math.min(1-h/2, y+.02), w, h] });
  };
  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement).closest("input,select,textarea") || busy || submitting) return;
      const key = event.key.toLowerCase(), ctrl = event.ctrlKey || event.metaKey;
      if (key === " ") { space.current = true; event.preventDefault(); return; }
      if (ctrl && key === "z") { event.preventDefault(); history(!event.shiftKey); }
      else if (ctrl && key === "y") { event.preventDefault(); history(false); }
      else if (ctrl && key === "s") { event.preventDefault(); save(); }
      else if (ctrl && key === "d" && !segment) { event.preventDefault(); duplicate(); }
      else if (key === "escape" && (points.length || gesture.current)) { event.preventDefault(); event.stopPropagation(); cancel(); }
      else if (key === "enter" && points.length) { event.preventDefault(); finish(); }
      else if (key === "delete") { event.preventDefault(); remove(); }
      else if (key === "backspace" && points.length) { event.preventDefault(); setPoints(points.slice(0, -2)); }
      else if (!ctrl && /^[1-9]$/.test(key)) setClassId(Math.min(names.length-1, Number(key)-1));
      else if (key === "0") setZoom(100);
      else if (key === "k" && navigation && navigation.index+1 < navigation.count) save(1);
      else if (key === "j" && navigation && navigation.index > 0) save(-1);
      else if (!ctrl && ({ p: segment, b: segment, e: segment, r: true, v: true, h: true } as Record<string, boolean>)[key])
        chooseMode(({ p: "polygon", b: "brush", e: "erase", r: "rectangle", v: "select", h: "pan" } as Record<string, Mode>)[key]);
    };
    const up = (event: KeyboardEvent) => { if (event.key === " ") space.current = false; };
    const blur = () => { space.current = false; gesture.current = null; setPreview(null); };
    window.addEventListener("keydown", down, true); window.addEventListener("keyup", up); window.addEventListener("blur", blur);
    return () => { window.removeEventListener("keydown", down, true); window.removeEventListener("keyup", up); window.removeEventListener("blur", blur); };
  });
  useEffect(() => {
    const element = scroll.current; if (!element) return;
    const wheel = (event: WheelEvent) => { event.preventDefault(); setZoom(value => Math.max(25, Math.min(800, value*(event.deltaY < 0 ? 1.2 : 1/1.2)))); };
    element.addEventListener("wheel", wheel, { passive: false });
    return () => element.removeEventListener("wheel", wheel);
  }, []);
  const coordinate = (event: React.PointerEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return [clamp((event.clientX-rect.left)/rect.width * (segment ? data!.width/Math.max(1, data!.width-1) : 1)),
      clamp((event.clientY-rect.top)/rect.height * (segment ? data!.height/Math.max(1, data!.height-1) : 1))];
  };
  const down = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!data || busy || submitting || !names.length) return;
    event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId);
    const p = coordinate(event);
    if (event.button === 1 || space.current || mode === "pan") {
      gesture.current = { type: "pan", start: [event.clientX, event.clientY, scroll.current!.scrollLeft, scroll.current!.scrollTop], index: -1, handle: -1, points: [] }; return;
    }
    if (event.button !== 0) return;
    const target = event.target as Element;
    const handle = target.getAttribute("data-handle"), index = target.closest("[data-shape]")?.getAttribute("data-shape");
    if (mode === "select") {
      const i = handle !== null ? selected : index === undefined ? -1 : Number(index);
      setSelected(i);
      if (i >= 0 && shapes[i].kind !== "stroke") gesture.current = { type: handle !== null ? "vertex" : "move", start: p, original: copy([shapes[i]])[0], index: i, handle: Number(handle), points: [] };
    } else if (mode === "polygon") {
      if (points.length >= 6 && Math.hypot((points[0]-p[0])*event.currentTarget.getBoundingClientRect().width, (points[1]-p[1])*event.currentTarget.getBoundingClientRect().height) < 10) finish();
      else setPoints([...points, ...p]);
    } else gesture.current = { type: mode, start: p, index: -1, handle: -1, points: p };
  };
  const move = (event: React.PointerEvent<SVGSVGElement>): Shape | null => {
    const value = gesture.current; if (!value || !data) return null;
    if (value.type === "pan") { scroll.current!.scrollLeft = value.start[2]+value.start[0]-event.clientX; scroll.current!.scrollTop = value.start[3]+value.start[1]-event.clientY; return null; }
    const p = coordinate(event), [x, y] = p, [sx, sy] = value.start;
    let shape: Shape;
    if (value.type === "brush" || value.type === "erase") {
      if (value.points.at(-2) !== x || value.points.at(-1) !== y) value.points.push(...p);
      shape = { kind: "stroke", class_id: value.type === "erase" ? 0 : classId, points: [...value.points], width: brush };
    } else if (value.type === "rectangle") {
      shape = { kind: segment ? "polygon" : "box", class_id: classId,
        points: segment ? [sx, sy, x, sy, x, y, sx, y] : [(sx+x)/2, (sy+y)/2, Math.abs(x-sx), Math.abs(y-sy)] };
    } else {
      shape = copy([value.original!])[0];
      const values = shape.kind === "box" ? boxPoints(shape.points) : shape.points;
      if (value.type === "vertex") {
        if (shape.kind === "box") {
          const opposite = (value.handle+2)%4, ox = values[2*opposite], oy = values[2*opposite+1];
          shape.points = [(ox+x)/2, (oy+y)/2, Math.abs(x-ox), Math.abs(y-oy)];
        } else shape.points.splice(value.handle*2, 2, x, y);
      } else {
        const xs = values.filter((_, i) => i%2 === 0), ys = values.filter((_, i) => i%2 === 1);
        const dx = Math.max(-Math.min(...xs), Math.min(1-Math.max(...xs), x-sx)), dy = Math.max(-Math.min(...ys), Math.min(1-Math.max(...ys), y-sy));
        shape.points = shape.kind === "box" ? [shape.points[0]+dx, shape.points[1]+dy, ...shape.points.slice(2)] : values.map((v, i) => v+(i%2 ? dy : dx));
      }
    }
    setPreview(shape); return shape;
  };
  const up = (event: React.PointerEvent<SVGSVGElement>) => {
    const value = gesture.current, shape = move(event); gesture.current = null; setPreview(null);
    if (!value || !shape || !data) return;
    if ((shape.kind === "box" && (shape.points[2]*data.width < 2 || shape.points[3]*data.height < 2)) ||
        (shape.kind === "polygon" && polygonArea(shape.points)*data.width*data.height < .5)) return;
    if (value.index >= 0) change(shapes.map((row, i) => i === value.index ? shape : row)); else add(shape);
  };
  const drawWidth = data ? data.width-(segment ? 1 : 0) : 1, drawHeight = data ? data.height-(segment ? 1 : 0) : 1;
  const svgPoints = (p: number[]) => p.reduce<string[]>((a, v, i) => { if (i%2 === 0) a.push(`${v*drawWidth},${p[i+1]*drawHeight}`); return a; }, []).join(" ");
  const renderShape = (shape: Shape, index: number, live = false) => {
    const p = shape.kind === "box" ? boxPoints(shape.points) : shape.points;
    const color = live || index === selected ? "#ffb000" : classColor(shape.class_id);
    return shape.kind === "stroke" ? <polyline key={index} data-shape={index} points={svgPoints(p)} fill="none" stroke={live ? color : "transparent"} strokeWidth={shape.width} strokeLinecap="round" strokeLinejoin="round" /> :
      <polygon key={index} data-shape={index} points={svgPoints(p)} fill={segment ? "transparent" : color} fillOpacity={segment ? 1 : .12} stroke={segment && index !== selected && !live ? "transparent" : color} strokeWidth={2} vectorEffect="non-scaling-stroke" />;
  };
  const disabled = busy || submitting || !data || !!points.length || !names.length || (segment && !base);
  return <DatasetDialog className="teaching-dialog" title={`데이터 티칭 — ${item.name}`} onClose={() => { if (!busy && !submitting && ((!dirty && !points.length) || window.confirm("저장하지 않은 변경을 버리고 닫을까요?"))) onClose(); }}>
    {(error || saveError) && <p className="error" role="alert">{error || saveError}</p>}
    <div className="annotation-toolbar"><Field label="정답 클래스"><select value={classId} disabled={busy || submitting} onChange={e => setClassId(Number(e.target.value))}>{names.map((name, i) => <option key={i} value={i}>{i}: {name}</option>)}</select></Field>
      {segment && <><Field label="브러시 px"><input type="number" min="1" max="1024" value={brush} onChange={e => setBrush(Math.max(1, Math.min(1024, Number(e.target.value) || 1)))} /></Field><Field label="마스크 농도"><input type="range" min="0" max="100" value={opacity} onChange={e => setOpacity(Number(e.target.value))} /></Field></>}
      <span>{zoom.toFixed(0)}%</span><button onClick={() => setZoom(100)}>화면 맞춤 0</button></div>
    <div className="row teaching-tools">{([...(segment ? [["polygon", "다각형 P"], ["brush", "브러시 B"], ["erase", "지우개 E"]] : []), ["rectangle", "사각형 R"], ["select", "선택/수정 V"], ["pan", "이동 H"]] as [Mode, string][]).map(([key, label]) => <button key={key} aria-pressed={mode === key} disabled={busy || submitting} className={mode === key ? "primary" : ""} onClick={() => chooseMode(key)}>{label}</button>)}
      <button disabled={busy || submitting || !undo.current.length} onClick={() => history(true)}>실행 취소 Ctrl+Z</button><button disabled={busy || submitting || !redo.current.length} onClick={() => history(false)}>다시 실행</button></div>
    <p className="muted">휠: 확대 · Space+드래그: 이동 · 1~9: 클래스 · 선택 후 꼭짓점 드래그: 수정 · Delete: 삭제 · Ctrl+S: 저장{segment ? " · Enter/첫 점: 다각형 완료 · 지우개/미표시 영역은 클래스 0(배경)" : " · Ctrl+D: 박스 복제"}</p>
    <div ref={scroll} className="annotation-canvas-scroll" onContextMenu={e => { e.preventDefault(); cancel(); }}>{data && <svg aria-label="정답 그리기 캔버스" className="annotation-canvas" viewBox={`0 0 ${data.width} ${data.height}`} style={{ width: `${fitWidth*zoom/100}px`, minWidth: `${fitWidth*zoom/100}px`, margin: "auto", aspectRatio: `${data.width}/${data.height}`, cursor: mode === "pan" ? "grab" : mode === "select" ? "default" : "crosshair" }} onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={cancel}>
      <image href={`/api/image?path=${encodeURIComponent(item.path)}&size=4096&v=${item.revision}`} width={data.width} height={data.height} style={{ pointerEvents: "none" }} />
      {segment && maskUrl && <image href={maskUrl} width={data.width} height={data.height} opacity={opacity/100} style={{ pointerEvents: "none", imageRendering: "pixelated" }} />}
      {shapes.map((shape, i) => renderShape(shape, i))}{preview && renderShape(preview, -1, true)}
      {points.length > 0 && <polyline points={svgPoints(points)} fill="none" stroke="#ffb000" strokeWidth="2" vectorEffect="non-scaling-stroke" />}
      {mode === "select" && shapes[selected] && shapes[selected].kind !== "stroke" && (() => {
        const shape = preview || shapes[selected], p = shape.kind === "box" ? boxPoints(shape.points) : shape.points;
        return p.map((v, i) => i%2 ? null : <circle key={i} data-handle={i/2} cx={v*drawWidth} cy={p[i+1]*drawHeight} r={6*data.width/fitWidth*100/zoom} fill="#ffb000" stroke="#ffffff" strokeWidth="1" vectorEffect="non-scaling-stroke" />);
      })()}
    </svg>}</div>
    {points.length > 0 && <div className="row"><button disabled={points.length < 6} onClick={finish}>다각형 완료 Enter</button><button onClick={cancel}>취소 Esc</button></div>}
    <div className="annotation-list">{shapes.map((shape, i) => <div key={i} className={i === selected ? "teaching-selected" : ""}>
      <button onClick={() => { setSelected(i); chooseMode("select"); }}>#{i+1} {shape.kind === "box" ? "박스" : shape.kind === "polygon" ? "다각형" : "브러시"}</button>
      <select aria-label={`정답 ${i+1} 클래스`} value={shape.class_id} disabled={busy || submitting} onChange={e => change(shapes.map((v, j) => j === i ? { ...v, class_id: Number(e.target.value) } : v))}>{names.map((name, n) => <option key={n} value={n}>{name}</option>)}{shape.class_id === 255 && <option value={255}>학습 제외 (255)</option>}</select>
      <button className="danger" disabled={busy || submitting} onClick={() => { change(shapes.filter((_, n) => i !== n)); setSelected(-1); }}>삭제</button></div>)}</div>
    <div className="row"><button className="primary" disabled={!!disabled} onClick={() => save()}>정답 저장 Ctrl+S</button>{navigation && navigation.index >= 0 && <><button disabled={!!disabled || navigation.index === 0} onClick={() => save(-1)}>저장하고 이전 J</button><span>현재 목록 {navigation.index+1}/{navigation.count}</span><button disabled={!!disabled || navigation.index+1 >= navigation.count} onClick={() => save(1)}>저장하고 다음 K</button></>}</div>
  </DatasetDialog>;
}
