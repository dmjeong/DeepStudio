import { useEffect, useState } from "react";
import { useDraft } from "./drafts";
import { api, filename, number, timing, type Config } from "./api";
import {
  Check,
  Empty,
  Field,
  JobMonitor,
  Panel,
  PathField,
  Stat,
  useJob,
} from "./components";
import type { PageProps } from "./pages";

export function Inference({ state, run, act, busy }: PageProps) {
  const projectKey = state.project?.filepath || "no-project";
  const task = state.project?.task || "classify";
  const [weights, setWeights] = useDraft(projectKey, "infer-weights", state.project?.runs.at(-1)?.checkpoint_path || "");
  const [folder, setFolder] = useDraft<string>(projectKey, "infer-folder", state.project?.data.test_dir || "");
  const [device, setDevice] = useDraft(projectKey, "infer-device", "cpu");
  const [computeCam, setComputeCam] = useDraft(projectKey, "infer-computeCam", true);
  const [threads, setThreads] = useDraft(projectKey, "infer-threads", 4);
  const [cropMode, setCropMode] = useDraft(projectKey, "infer-crop-mode", "model");
  const [cropJson, setCropJson] = useDraft(projectKey, "infer-crop-json", "");
  const [cropPreview, setCropPreview] = useState<Config | null>(null);
  useEffect(() => {
    let active = true;
    setCropPreview(null);
    if (cropMode !== "json" || !cropJson.trim()) return;
    const timer = setTimeout(() => {
      void api<Config>(`/inference/region?path=${encodeURIComponent(cropJson.trim())}`)
        .then(value => { if (active) setCropPreview({...value, path: cropJson}); })
        .catch(e => { if (active) setCropPreview({path: cropJson, error: (e as Error).message}); });
    }, 200);
    return () => { active = false; clearTimeout(timer); };
  }, [cropMode, cropJson, projectKey]);
  const cropReady = cropMode !== "json" || (cropPreview?.path === cropJson && !cropPreview?.error);
  const [id, setId] = useState<string | null>(
    () => state.jobs.find((j) => j.kind === "infer" && j.project_path === (state.project?.filepath || null))?.id || null,
  );
  const [page, setPage] = useState(0);
  const [results, setResults] = useState<Config[]>([]);
  const [total, setTotal] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const [review, setReview] = useState<Config>({});
  const [classFilter, setClassFilter] = useState("");
  const [decisionFilter, setDecisionFilter] = useState("");
  const [search, setSearch] = useState("");
  const [checked, setChecked] = useState<number[]>([]);
  const [selectedOnly, setSelectedOnly] = useState(false);
  const [sort, setSort] = useState("index");
  const [descending, setDescending] = useState(false);
  const [reviewThreshold, setReviewThreshold] = useState("");
  const threshold = reviewThreshold.trim() !== "" && Number.isFinite(Number(reviewThreshold)) ? Number(reviewThreshold) : null;
  const [resultQuery, setResultQuery] = useState("");
  useEffect(() => setPage(0), [classFilter, decisionFilter, search, selectedOnly, sort, descending, threshold]);
  useEffect(() => {
    setClassFilter(""); setDecisionFilter(""); setSearch(""); setChecked([]);
    setSelectedOnly(false); setReviewThreshold(""); setReview({}); setPage(0);
  }, [id]);
  useEffect(() => {
    const timer = setTimeout(() => {
      const params = new URLSearchParams({class_name: classFilter, decision: decisionFilter, search,
        selected: selectedOnly ? checked.join(",") : "", selected_only: String(selectedOnly), sort, descending: String(descending)});
      if (threshold !== null) params.set("threshold", String(threshold));
      setResultQuery(params.toString());
    }, 180);
    return () => clearTimeout(timer);
  }, [classFilter, decisionFilter, search, checked, selectedOnly, sort, descending, threshold]);
  const [lower, setLower] = useDraft(projectKey, "infer-lower", 0);
  const [upper, setUpper] = useDraft(projectKey, "infer-upper", 100);
  const [alpha, setAlpha] = useDraft(projectKey, "infer-alpha", 50);
  const [cam, setCam] = useDraft(projectKey, "infer-cam", true);
  const [query, setQuery] = useState("");
  const [zoom, setZoom] = useDraft(projectKey, "infer-zoom", 100);
  const [error, setError] = useState("");
  const [compare, setCompare] = useDraft(projectKey, "infer-compare", false);
  const [imageError, setImageError] = useState(false);
  const { view } = useJob(id);
  useEffect(() => {
    const timer = setTimeout(
      () =>
        setQuery(
          `cam=${cam}&lower=${lower / 100}&upper=${upper / 100}&alpha=${alpha / 100}`,
        ),
      180,
    );
    return () => clearTimeout(timer);
  }, [lower, upper, alpha, cam]);
  useEffect(() => {
    setResults([]);
    setTotal(0);
    setSelected(null);
  }, [id, page]);
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    if (!id) return;
    const poll = async () => {
      try {
        const value = await api<Config>(
          `/jobs/${id}/results?offset=${page * 60}&limit=60&${resultQuery}`,
        );
        if (!active) return;
        setResults(value.results);
        setTotal(value.filtered_total ?? value.total);
        setReview(value);
        setSelected(current => value.results.some((r: Config) => r.index === current) ? current : value.results[0]?.index ?? null);
        setError("");
      } catch (e) {
        if (active) setError((e as Error).message);
      }
      if (active && state.active_job === id) timer = setTimeout(poll, 1200);
    };
    void poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [id, page, state.active_job, resultQuery]);
  const result = results.find(r => r.index === selected);
  const savedThreshold = review.saved_thresholds?.length === 1 ? review.saved_thresholds[0] : null;
  const normalized = Boolean(review.score_normalized);
  const sliderValue = threshold ?? savedThreshold ?? (normalized ? .5 : review.score_range?.[0] ?? .5);
  const scoreMin = Math.min(review.score_range?.[0] ?? 0, sliderValue);
  const scoreMax = Math.max(review.score_range?.[1] ?? 1, sliderValue);
  const margin = Math.max((scoreMax - scoreMin) * .1, .01);
  const sliderMin = normalized ? 0 : scoreMin - margin;
  const sliderMax = normalized ? 1 : scoreMax + margin;
  const changeSort = (column: string) => {
    setDescending(sort === column ? !descending : false); setSort(column);
  };
  const imageUrl =
    result && id ? `/api/jobs/${id}/image/${result.index}?${query}&ready=${result.cache_ready ?? true}` : "";
  useEffect(() => setImageError(false), [imageUrl]);
  return (
    <>
      <section className="panel inference-input-compact" aria-label="추론 입력">
        <fieldset disabled={busy}>
          <div className="columns">
            <PathField
              label="모델 가중치"
              value={weights}
              onChange={setWeights}
              home={state.home}
            />
            <PathField
              label="이미지 폴더"
              directory
              value={folder}
              onChange={setFolder}
              home={state.home}
            />
          </div>
          <div className="toolbar">
            <Field label="연산 장치">
              <select
                value={device}
                onChange={(e) => setDevice(e.target.value)}
              >
                {["cpu", "auto", "cuda", "cuda:0", "cuda:1"].map((v) => (
                  <option key={v}>{v}</option>
                ))}
              </select>
            </Field>
            <Field label="CPU 스레드">
              <input
                type="number"
                min="1"
                max="64"
                value={threads}
                onChange={(e) => setThreads(Number(e.target.value))}
              />
            </Field>
            <Field label="입력 영역">
              <select className="crop-mode" value={cropMode}
                onChange={e => setCropMode(e.target.value)}>
                <option value="model">모델 설정 사용</option>
                <option value="json">JSON 설정 사용</option>
                <option value="full">원본 전체 사용</option>
              </select>
            </Field>
            <Check
              label="Grad-CAM 함께 계산"
              value={computeCam}
              onChange={setComputeCam}
            />
            <button
              className="primary push"
              disabled={!weights || !folder || !cropReady}
              onClick={() =>
                act(async () => {
                  const j = await run("/jobs/infer", {
                    weights,
                    folder,
                    device,
                    gradcam: computeCam,
                    threads,
                    crop_mode: cropMode,
                    crop_json: cropJson,
                  });
                  setId(j.id);
                  setPage(0);
                  setSelected(null);
                })
              }
            >
              추론 시작
            </button>
          </div>
          {cropMode === "json" && <div className="inference-crop-json">
            <PathField label="크롭 JSON" value={cropJson} onChange={setCropJson} home={state.home} />
            <span role="status" className={cropPreview?.error ? "error" : "muted"}>
              {cropPreview?.path === cropJson ? cropPreview.error || cropPreview.label :
                cropJson.trim() ? "JSON 설정 확인 중..." : "JSON 또는 프로젝트 파일을 선택해 주세요"}
            </span>
          </div>}
          {cropMode === "full" && <div className="inference-region-note muted">
            원본 전체를 모델 입력 크기에 맞춰 처리합니다.
          </div>}
        </fieldset>
      </section>
      <div className="row inference-job-row">
        <h2>저장된 추론 결과</h2>
        <select
          className="push job-select"
          aria-label="추론 작업 선택"
          value={id || ""}
          onChange={(e) => {
            setId(e.target.value);
            setPage(0);
          }}
        >
          <option value="">작업 선택</option>
          {state.jobs
            .filter((j) => j.kind === "infer" && j.project_path === (state.project?.filepath || null))
            .map((j) => (
              <option key={j.id} value={j.id}>
                {new Date(j.created_at).toLocaleString()} / {j.status}
              </option>
            ))}
        </select>
      </div>
      <div className="inference-layout">
        <div>
          <div className="viewer">
            <div className="viewer-toolbar">
              <span>
                {result ? filename(result.image_path) : "IMAGE VIEWER"}
              </span>
              <button
                className="push"
                disabled={!result}
                onClick={() => setZoom(Math.max(25, zoom - 25))}
                aria-label="축소"
              >
                −
              </button>
              <button disabled={!result} onClick={() => setZoom(100)}>
                {zoom}%
              </button>
              <button
                disabled={!result}
                onClick={() => setZoom(Math.min(400, zoom + 25))}
                aria-label="확대"
              >
                +
              </button>
            </div>
            <div className="image-stage">
              {result ? (
                <>
                  <div
                    className="result-label"
                    style={{ borderColor: result.color }}
                  >
                    <strong>{result.summary || result.status}</strong>
                    <span>{result.error || result.info}</span>
                  </div>
                  {!imageError ? (
                    <div className={compare ? "compare-images" : ""} style={{ width: zoom + "%" }}>
                    {compare && <img src={`/api/jobs/${id}/image/${result.index}?original=true&cam=false`} alt="원본 이미지" />}
                    <img
                      className="main-image"
                      style={{ width: "100%", maxWidth: "none" }}
                      src={imageUrl}
                      alt={result.summary || filename(result.image_path)}
                      onError={() => setImageError(true)}
                    />
                    </div>
                  ) : (
                    <p className="error">
                      캐시 이미지 표시 실패. 아래 오류와 작업 로그를 확인하세요.
                    </p>
                  )}
                </>
              ) : (
                <Empty>결과 목록에서 이미지를 선택하세요.</Empty>
              )}
            </div>
          </div>
          {(error || result?.error || result?.cache_error) && (
            <p role="alert" className="error">
              {error || result?.error || result?.cache_error}
            </p>
          )}
          {result && <TaskResultPanel task={result.task || task} result={result} />}
          <details className="panel inference-details" open={result?.task === "classify"}>
            <summary>선택 이미지 상세</summary>
            <div className="stats">
              <Stat
                label="추론"
                value={timing(result?.inference_sec)}
                note={result?.inference_status}
              />
              <Stat
                label="Grad-CAM"
                value={timing(result?.gradcam_sec)}
                note={result?.gradcam_status}
              />
              <Stat label="전체 처리" value={timing(result?.elapsed_sec)} />
            </div>
            {view?.job.output.inference_summary && (
              <p className="muted">
                {view.job.output.inference_summary}
                <br />
                {view.job.output.gradcam_summary}
              </p>
            )}
            {result?.details && (
              <details>
                <summary>상세 결과</summary>
                <pre className="log">
                  {JSON.stringify(result.details, null, 2)}
                </pre>
              </details>
            )}
          </details>
        <Panel title="시각화">
          <Check label="원본과 나란히 보기" value={compare} onChange={setCompare} />
          <Check label="히트맵 표시" value={cam} onChange={setCam} />
          {[
            [
              "표현 최소 강도",
              lower,
              (v: number) => setLower(Math.min(v, upper - 1)),
            ],
            [
              "최대 색상 시작 강도",
              upper,
              (v: number) => setUpper(Math.max(v, lower + 1)),
            ],
            ["히트맵 불투명도", alpha, setAlpha],
          ].map(([label, value, setter]) => (
            <Field key={String(label)} label={`${label}: ${value}%`}>
              <input
                type="range"
                min={label === "최대 색상 시작 강도" ? 1 : 0}
                max={label === "표현 최소 강도" ? 99 : 100}
                value={value as number}
                onChange={(e) =>
                  (setter as (v: number) => void)(Number(e.target.value))
                }
              />
            </Field>
          ))}
          <p className="muted">
            범위 조절은 저장된 활성화 지도에만 적용됩니다. 모델 추론을 다시
            실행하지 않습니다.
          </p>
          {result && !result.heatmap && (
            <p className="notice">
              이 이미지에 저장된 활성화 지도가 없습니다. Grad-CAM 계산 여부 또는
              실행 로그를 확인하세요.
            </p>
          )}
          <p className="muted">
            0–100%는 계산된 전체 강도를 표시합니다. 모델이 보지 않은 영역은 색을
            채우지 않습니다.
          </p>
          {result && (
            <a
              className="button"
              href={imageUrl}
              download={filename(result.image_path) + ".preview.png"}
            >
              현재 표시 이미지 저장
            </a>
          )}
        </Panel>
        </div>
        <Panel title={`배치 결과 ${total} / ${review.total ?? 0}장`}>
          {review.anomaly && <div className="threshold-control">
            <Field label="Anomaly 임계값">
              <input type="number" step="any" min={normalized ? 0 : undefined} max={normalized ? 1 : undefined} value={reviewThreshold}
                placeholder={savedThreshold === null ? "미보정" : String(savedThreshold)}
                onChange={e => setReviewThreshold(e.target.value === "" ? "" : normalized
                  ? String(Math.min(1, Math.max(0, Number(e.target.value)))) : e.target.value)} />
            </Field>
            <button onClick={() => setReviewThreshold("")}>저장된 임계값 복원</button>
            <input aria-label="Anomaly 임계값 슬라이더" type="range"
              min={sliderMin} max={sliderMax} step={(sliderMax - sliderMin) / 1000}
              value={sliderValue} onChange={e => setReviewThreshold(e.target.value)} />
            <p className="muted" title={(normalized ? "0에 가까울수록 정상 학습 데이터와 유사, 1에 가까울수록 멂. 불량 확률이 아닙니다. " : "") + "임계값 미만: OK, 이상: NG. 입력을 비우면 저장된 임계값을 사용합니다. 모델 파일과 원본 결과는 변경하지 않습니다."}>
              {normalized ? "0~1 | " : ""}스코어 ≥ 임계값: NG | 저장값: {review.saved_thresholds?.join(", ") || "미보정"}</p>
            <div className="review-counts">OK {review.counts?.OK ?? 0} / NG {review.counts?.NG ?? 0} / 오류 {review.counts?.ERROR ?? 0}</div>
          </div>}
          <div className="review-filters">
            <Field label="결과 클래스">
              <select value={classFilter} onChange={e => setClassFilter(e.target.value)}>
                <option value="">전체 클래스</option>
                {(review.classes || []).map((name: string) => <option key={name} value={name || "__unknown__"}>{name || "—"}</option>)}
              </select>
            </Field>
            <Field label="결과 판정">
              <select value={decisionFilter} onChange={e => setDecisionFilter(e.target.value)}>
                <option value="">전체 판정</option>
                {["OK", "NG", "미보정", "ERROR"].map(name => <option key={name}>{name}</option>)}
              </select>
            </Field>
            <Field label="결과 파일명 검색"><input value={search} onChange={e => setSearch(e.target.value)} placeholder="파일명 검색" /></Field>
          </div>
          <div className="review-selection">
            <Check label="선택한 행만 보기" value={selectedOnly} onChange={setSelectedOnly} />
            <button onClick={() => setChecked(current => [...new Set([...current, ...results.map(r => r.index)])])}>표시 행 선택</button>
            <button onClick={() => setChecked([])}>선택 해제</button>
            <span>선택 {checked.length}장</span>
          </div>
          <div className="table-scroll inference-results">
            <table aria-label="추론 결과 표">
              <thead><tr><th>선택</th>{[
                ["index", "인덱스"], ["source_class", "클래스"], ["filename", "파일명"],
                ["decision", "판정"], ["score", "스코어"], ["inference_sec", "추론 시간"],
              ].map(([key, label]) => <th key={key} aria-sort={sort === key ? descending ? "descending" : "ascending" : "none"}>
                <button onClick={() => changeSort(key)} aria-label={`${label} 정렬`}>{label}{sort === key ? descending ? " ↓" : " ↑" : ""}</button>
              </th>)}</tr></thead>
              <tbody>{results.map(r => <tr key={r.index} data-index={r.index} className={r.index === selected ? "selected" : ""}
                onClick={() => {setSelected(r.index); setZoom(100);}}>
                <td><input type="checkbox" aria-label={`${r.index + 1}번 결과 선택`} checked={checked.includes(r.index)}
                  onClick={e => e.stopPropagation()}
                  onChange={e => setChecked(current => e.target.checked ? [...current, r.index] : current.filter(i => i !== r.index))} /></td>
                <td>{r.index + 1}</td><td>{r.source_class || "—"}</td>
                <td title={r.image_path}><button onClick={() => {setSelected(r.index); setZoom(100);}}>{r.filename}</button></td>
                <td style={{color: r.color}} title={r.error || r.summary}>{r.decision}</td>
                <td>{typeof r.score === "number" ? number(r.score, 6) : "—"}</td>
                <td>{timing(r.inference_sec)}</td>
              </tr>)}</tbody>
            </table>
          </div>
          {!results.length && <Empty>조건에 맞는 결과가 없습니다.</Empty>}
          <p className="muted">클래스는 데이터셋의 원래 클래스입니다. 확인할 수 없는 경우 —로 표시합니다.</p>
          <div className="row">
            <button disabled={!page} onClick={() => setPage(page - 1)}>이전</button>
            <span>{page + 1} / {Math.max(1, Math.ceil(total / 60))}</span>
            <button disabled={(page + 1) * 60 >= total} onClick={() => setPage(page + 1)}>다음</button>
          </div>
        </Panel>
      </div>
      <JobMonitor id={id} />
    </>
  );
}

function TaskResultPanel({ task, result }: { task: string; result: Config }) {
  const details = result.details || {};
  if (task === "classify" && Array.isArray(details.probabilities)) {
    const names = Array.isArray(details.class_names) ? details.class_names : [];
    const probabilities = details.probabilities as number[];
    const classCount = Math.max(names.length, probabilities.length);
    return <section className="task-result-panel" aria-label="분류 결과">
      <strong>분류 클래스별 확률</strong>
      <div className="probabilities">
        {Array.from({length: classCount}, (_, i) => {
          const raw = Number(probabilities[i] ?? 0);
          const value = Number.isFinite(raw) ? Math.min(1, Math.max(0, raw)) : 0;
          return <div key={i}><span>{names[i] ?? `클래스 ${i}`}</span><meter min="0" max="1" value={value} /><strong>{number(value * 100, 1)}%</strong></div>;
        })}
      </div>
    </section>;
  }
  if (["detect", "obb"].includes(task) && Array.isArray(details.detections)) {
    return <section className="task-result-panel" aria-label="검출 결과">
      <strong>{task === "obb" ? "OBB 회전 박스" : "검출 박스"} {details.detections.length}개</strong>
      <div className="task-result-list">{details.detections.map((item: Config, i: number) => <span key={i}>{(item.class_name ?? item.class_id ?? "객체")} | {number(Number(item.confidence ?? item.score ?? 0) * 100, 1)}%{task === "obb" && ` | ${number(item.angle_deg, 1)}° | ${number(item.width_px, 1)}×${number(item.height_px, 1)} px`}</span>)}</div>
    </section>;
  }
  if (task === "detect" && details.num_detections != null) {
    return <section className="task-result-panel" aria-label="검출 결과">
      <strong>검출 박스 {Number(details.num_detections)}개</strong>
    </section>;
  }
  if (task === "segment" && details.pixel_counts) {
    const total = Object.values(details.pixel_counts as Record<string, number>).reduce((a, b) => a + Number(b), 0);
    return <section className="task-result-panel" aria-label="세그멘테이션 결과">
      <strong>세그멘테이션 픽셀 분포</strong>
      <div className="task-result-list">{Object.entries(details.pixel_counts as Record<string, number>).sort(([a], [b]) => Number(a) - Number(b)).map(([key, value]) => <span key={key}>클래스 {key} | {Number(value).toLocaleString()} px | {total ? number(Number(value) / total * 100, 1) : "0.0"}%</span>)}</div>
    </section>;
  }
  if (task === "segment" && details.num_detections != null) {
    return <section className="task-result-panel" aria-label="세그멘테이션 결과">
      <strong>세그멘테이션 결과</strong>
      <p>분할 객체 {Number(details.num_detections)}개</p>
    </section>;
  }
  if (task === "anomaly" && typeof result.score === "number") {
    const decision = result.threshold == null ? "미보정" : result.score >= result.threshold ? "NG" : "OK";
    return <section className="task-result-panel" aria-label="이상 탐지 결과">
      <strong>이상 탐지 점수</strong>
      <p>스코어 {number(result.score, 6)} | {result.threshold == null ? "임계값 미보정" : `임계값 ${number(result.threshold, 6)} | ${result.decision || decision}`}</p>
    </section>;
  }
  return null;
}
