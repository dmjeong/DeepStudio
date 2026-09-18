import { useJob } from "./job-store";
import { useEffect, useRef, useState, useId, isValidElement, cloneElement, type ReactElement, type ReactNode } from "react";
import {
  api,
  duration,
  number,
  type Job,
  terminal,
} from "./api";

export function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
}) {
  const id = useId();
  const control = isValidElement(children) && typeof children.type === "string" && ["input", "select", "textarea"].includes(children.type)
    ? cloneElement(children as ReactElement<Record<string, unknown>>, { "aria-labelledby": id }) : children;
  return (
    <label className="field">
      <span id={id}>{label}</span>
      {control}
      {hint && <small>{hint}</small>}
    </label>
  );
}
export function Panel({
  title,
  children,
  actions,
}: {
  title: string;
  children: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>{title}</h2>
        {actions}
      </div>
      {children}
    </section>
  );
}
export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="empty">
      <span className="empty-icon">□</span>
      <p>{children}</p>
    </div>
  );
}
export function Stat({
  label,
  value,
  note,
}: {
  label: string;
  value: ReactNode;
  note?: string;
}) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
      {note && <small>{note}</small>}
    </div>
  );
}
export function Check({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="check">
      <input
        type="checkbox"
        checked={value}
        onChange={(e) => onChange(e.target.checked)}
      />
      {label}
    </label>
  );
}
export function PathField({
  label,
  value,
  onChange,
  directory = false,
  home,
  hint,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  directory?: boolean;
  home: string;
  hint?: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Field label={label} hint={hint}>
        <span className="path-field">
          <input
            aria-label={label}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder="절대 경로를 입력하거나 찾아보기"
          />
          <button type="button" onClick={() => setOpen(true)}>
            찾아보기
          </button>
        </span>
      </Field>
      {open && (
        <FilePicker
          directory={directory}
          initial={directory && value ? value : home}
          onClose={() => setOpen(false)}
          onSelect={(v) => {
            onChange(v);
            setOpen(false);
          }}
        />
      )}
    </>
  );
}
function FilePicker({
  directory,
  initial,
  onClose,
  onSelect,
}: {
  directory: boolean;
  initial: string;
  onClose: () => void;
  onSelect: (v: string) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [path, setPath] = useState(initial);
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState("");
  const [draft, setDraft] = useState(initial);
  useEffect(() => {
    dialog.current?.showModal();
  }, []);
  useEffect(() => {
    let active = true;
    setError("");
    setData(null);
    api("/files?path=" + encodeURIComponent(path) + "&offset=" + offset)
      .then((d) => {
        if (active) {
          setData(d);
          setDraft(d.path);
        }
      })
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
    };
  }, [path, offset]);
  const navigate = (p: string) => {
    setPath(p);
    setOffset(0);
  };
  return (
    <dialog ref={dialog} onCancel={onClose} className="file-dialog">
      <div className="panel-head">
        <h2>{directory ? "폴더 선택" : "파일 선택"}</h2>
        <button onClick={onClose} aria-label="닫기">
          ✕
        </button>
      </div>
      <form
        className="row"
        onSubmit={(e) => {
          e.preventDefault();
          navigate(draft);
        }}
      >
        <input
          aria-label="폴더 경로"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
        />
        <button>이동</button>
      </form>
      <div className="row">
        <button onClick={() => navigate("")}>드라이브</button>
        <button disabled={!data?.path} onClick={() => navigate(data.parent)}>
          상위 폴더
        </button>
      </div>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      <div className="file-list">
        {data?.entries.map((entry: any) => (
          <button
            key={entry.path}
            disabled={!entry.directory && directory}
            onClick={() =>
              entry.directory ? navigate(entry.path) : onSelect(entry.path)
            }
          >
            <span>{entry.directory ? "▣" : "▤"}</span>
            {entry.name}
            {entry.directory && <span className="push">›</span>}
          </button>
        ))}
      </div>
      <div className="row">
        <button
          disabled={offset === 0}
          onClick={() => setOffset(Math.max(0, offset - 150))}
        >
          이전
        </button>
        <span>
          {offset + 1}–{Math.min(offset + 150, data?.total || 0)} /{" "}
          {data?.total || 0}
        </span>
        <button
          disabled={offset + 150 >= (data?.total || 0)}
          onClick={() => setOffset(offset + 150)}
        >
          다음
        </button>
        {directory && (
          <button
            className="primary push"
            disabled={!data?.path}
            onClick={() => onSelect(data.path)}
          >
            이 폴더 선택
          </button>
        )}
      </div>
    </dialog>
  );
}
export { useJob } from "./job-store";
export const statusLabel = (status: string) =>
  ({
    running: "실행 중",
    cancelling: "중단 처리 중",
    completed: "완료",
    failed: "실패",
    cancelled: "중단됨",
    interrupted: "서버 종료로 중단",
  })[status] || status;
export function JobMonitor({ id }: { id: string | null }) {
  const { view, events, error } = useJob(id);
  const [cancelError, setCancelError] = useState("");
  const progress = events
    .filter((e) => e.event === "progress_updated")
    .at(-1)?.args;
  const batch = events.filter(e => e.event === "batch_finished").at(-1)?.args;
  const logs = events
    .filter((e) => ["log_message", "training_error"].includes(e.event))
    .map((e) => String(e.args[0]));
  if (!id) return null;
  return (
    <Panel
      title="작업 로그"
      actions={
        view && (
          <span className={"badge " + view.job.status}>
            {statusLabel(view.job.status)}
          </span>
        )
      }
    >
      {progress && (
        <div className="progress-line">
          <progress value={progress[0]} max={progress[1] || 1} />
          <span>
            {progress[0]} / {progress[1]}
          </span>
        </div>
      )}
      {batch && !terminal(view?.job) && (!progress || batch[0] > progress[0]) && (
        <p role="status">Epoch {batch[0]} | Batch {batch[1]} / {batch[2]}</p>
      )}
      {(error || cancelError || view?.job.error) && (
        <p role="alert" className="error">
          {error || cancelError || view?.job.error}
        </p>
      )}
      {view?.job.project_error && (
        <p role="alert" className="error">{view.job.project_error}</p>
      )}
      {view && !terminal(view.job) && (
        <button
          className="danger"
          disabled={view.job.status === "cancelling"}
          onClick={() =>
            api(`/jobs/${id}/cancel`, "POST").catch((e) =>
              setCancelError(e.message),
            )
          }
        >
          안전하게 중단
        </button>
      )}
      <pre className="log" aria-label="작업 로그">
        {logs.slice(-500).join("\n") || "계산 프로세스 준비 중…"}
      </pre>
      {view?.console && (
        <details>
          <summary>엔진 실행 로그</summary>
          <pre className="log">{view.console}</pre>
        </details>
      )}
      {view && terminal(view.job) && (
        <p className="muted">
          전체 작업 시간 {duration(view.job.duration_sec)}
        </p>
      )}
      {view && terminal(view.job) && (
        <div className="row">
          {Object.entries({
            output_path: "ONNX 다운로드",
            config_path: "배포 설정 다운로드",
            archive_path: "결함 결과 ZIP 다운로드",
          })
            .filter(([key]) => view.job.output?.[key])
            .map(([key, label]) => (
              <a key={key} href={`/api/jobs/${id}/download/${key}`}>
                {label}
              </a>
            ))}
        </div>
      )}
      {view?.job.output && Object.keys(view.job.output).length > 0 && (
        <details>
          <summary>저장된 작업 결과</summary>
          <pre className="log">{JSON.stringify(view.job.output, null, 2)}</pre>
        </details>
      )}
    </Panel>
  );
}
export function LossChart({
  train,
  val,
  best,
  epochs,
}: {
  train: (number | null)[];
  val: (number | null)[];
  best?: number;
  epochs?: number[];
}) {
  const valid = [...train, ...val].filter(
    (v): v is number => v != null && Number.isFinite(v),
  );
  if (!valid.length)
    return <Empty>첫 에폭이 완료되면 손실 그래프가 표시됩니다.</Empty>;
  const maximum = Math.max(...valid, 0.001),
    minimum = Math.min(0, ...valid),
    count = Math.max(train.length, val.length);
  const numbers = epochs || Array.from({ length: count }, (_, i) => i + 1);
  const bestIndex = numbers.indexOf(best || 0);
  const x = (i: number) => 52 + (i / Math.max(count - 1, 1)) * 688;
  const y = (v: number) => 210 - ((v - minimum) / (maximum - minimum)) * 185;
  const segments = (values: (number | null)[]) => {
    let parts: string[] = [];
    let points: string[] = [];
    values.forEach((v, i) => {
      if (v == null || !Number.isFinite(v)) {
        if (points.length) parts.push(points.join(" "));
        points = [];
      } else points.push(`${x(i)},${y(v)}`);
    });
    if (points.length) parts.push(points.join(" "));
    return parts;
  };
  return (
    <>
      <div className="legend">
        <span className="train-key">Train loss</span>
        <span className="val-key">Val loss</span>
        <span>Best epoch {best || "—"}</span>
      </div>
      <svg
        className="chart"
        viewBox="0 0 780 248"
        role="img"
        aria-label="에폭별 Train loss와 Val loss 그래프"
      >
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line
              x1="52"
              x2="740"
              y1={y(minimum + (maximum - minimum) * t)}
              y2={y(minimum + (maximum - minimum) * t)}
              stroke="#2b3846"
            />
            <text x="3" y={y(minimum + (maximum - minimum) * t) + 5}>
              {number(minimum + (maximum - minimum) * t, 2)}
            </text>
          </g>
        ))}
        {bestIndex >= 0 ? (
          <line
            x1={x(bestIndex)}
            x2={x(bestIndex)}
            y1="20"
            y2="210"
            stroke="#b6c56d"
            strokeDasharray="4 4"
          />
        ) : null}
        {[train, val].map((values, k) => (
          <g key={k}>
            {segments(values).map((points, i) => (
              <polyline
                key={i}
                points={points}
                fill="none"
                stroke={k ? "#b6a0ed" : "#55cfc2"}
                strokeWidth="2.5"
              />
            ))}
            {values.map((v, i) =>
              v != null && Number.isFinite(v) ? (
                <circle
                  key={i}
                  cx={x(i)}
                  cy={y(v)}
                  r="2.5"
                  fill={k ? "#b6a0ed" : "#55cfc2"}
                >
                  <title>
                    Epoch {numbers[i]}: {number(v)}
                  </title>
                </circle>
              ) : null,
            )}
          </g>
        ))}
        <text x="52" y="238">
          {numbers[0]}
        </text>
        <text x="680" y="238">
          Epoch {numbers.at(-1)}
        </text>
      </svg>
    </>
  );
}
