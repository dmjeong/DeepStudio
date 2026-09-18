import { useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, type Job, type Project, type StudioState } from "./api";
import { Dataset, Projects, Training, type PageProps } from "./pages";
import { Storage } from "./pages/Storage";
import { Inference } from "./inference";
import { Defects, Export } from "./tools";
import { Empty, statusLabel } from "./components";
import "./style.css";

const tabs = [
  ["storage", "작업 기록", "M4 4h16v16H4z M8 8h8 M8 12h8 M8 16h5"],
  ["projects", "프로젝트", "M3 7h7l2 2h9v11H3z"],
  [
    "dataset",
    "데이터셋",
    "M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z",
  ],
  ["training", "학습", "M3 20V4 M3 20h18 M6 16l4-7 5 3 6-8"],
  [
    "inference",
    "추론",
    "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12 M9 12a3 3 0 106 0 3 3 0 10-6 0",
  ],
  ["export", "내보내기", "M12 3v12 M7 8l5-5 5 5 M4 15v6h16v-6"],
  ["defects", "Defect Gen", "M4 4h16v16H4z M8 8l8 8 M15 7l-3 7-4 3"],
] as const;
type Tab = (typeof tabs)[number][0];
function App() {
  const [state, setState] = useState<StudioState | null>(null);
  const [tab, setTab] = useState<Tab>(() => {
    const saved = localStorage.getItem("studio-tab");
    return tabs.some((v) => v[0] === saved) ? (saved as Tab) : "projects";
  });
  const [error, setError] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [pending, setPending] = useState(false);
  const refresh = useCallback(async () => {
    const data = await api<StudioState>("/state");
    setState(data);
    setConnectionError("");
  }, []);
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    let known: StudioState | null = null;
    const poll = async () => {
      try {
        const data = await api<StudioState>("/state?summary=true");
        if (data.project_summary) {
          data.project = known?.project_summary?.filepath === data.project_summary.filepath && known?.project_summary?.revision === data.project_summary.revision
            ? known?.project ?? null : await api<Project>("/project/view");
        }
        known = data;
        if (active) {
          setState(data);
          setConnectionError("");
        }
      } catch (e) {
        if (active) setConnectionError((e as Error).message);
      }
      if (active) timer = setTimeout(poll, 1800);
    };
    void poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, []);
  const navigate = (next: Tab) => {
    setTab(next);
    localStorage.setItem("studio-tab", next);
    setError("");
  };
  const act = (work: () => Promise<unknown>) => {
    if (pending) return;
    setPending(true);
    setError("");
    void work()
      .catch((e) => setError((e as Error).message))
      .finally(() => setPending(false));
  };
  const run = async (path: string, body?: unknown) => {
    const job = await api<Job>(path, "POST", body);
    await refresh();
    return job;
  };
  const props: PageProps | undefined = state
    ? { state, refresh, act, run, busy: pending || !!state.active_job }
    : undefined;
  const active = state?.jobs.find((j) => j.id === state.active_job);
  return (
    <div className="app">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            navigate("projects");
          }}
        >
          <svg viewBox="0 0 32 32" aria-hidden="true">
            <path
              d="M5 5h9v5H10v4H5z M18 5h9v9h-5v-4h-4z M5 18h5v4h4v5H5z M22 18h5v9h-9v-5h4z"
              fill="currentColor"
            />
            <circle cx="16" cy="16" r="4" fill="currentColor" />
          </svg>
          <span>
            Deep Vision
            <br />
            <strong>Studio</strong>
          </span>
        </a>
        <div className="workspace-label">WORKSPACE</div>
        <nav>
          {tabs.map(([key, label, path], i) => (
            <button
              key={key}
              aria-label={label}
              className={tab === key ? "active" : ""}
              aria-current={tab === key ? "page" : undefined}
              onClick={() => navigate(key)}
            >
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d={path} />
              </svg>
              <span>{label}</span>
              <small>{String(i + 1).padStart(2, "0")}</small>
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="local-status">
            <span className={connectionError ? "dot offline" : "dot"} />
            {connectionError ? "서버 연결 끊김" : "로컬 워크스페이스"}
          </div>
          <span>
            React Edition <strong>{state?.version ? `v${state.version}` : "연결 중"}</strong>
          </span>
          <small>이 PC에서 데이터와 모델을 처리합니다.</small>
        </div>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <span>
            WORKSPACE <span className="slash">/</span>{" "}
            {state?.project?.name || "프로젝트 선택"}
          </span>
          <span className="badge">LOCAL</span>
        </header>
        <main>
          <div className="page-heading">
            <div>
              <p className="eyebrow">DEEP VISION STUDIO</p>
              <h1>{tabs.find((v) => v[0] === tab)?.[1]}</h1>
            </div>
            {active && (
              <button
                className="job-indicator"
                onClick={() =>
                  navigate(
                    active.kind === "train"
                      ? "training"
                      : active.kind === "infer"
                        ? "inference"
                        : active.kind === "export"
                          ? "export"
                          : active.kind === "defects"
                            ? "defects"
                            : "dataset",
                  )
                }
              >
                <span className="dot" />
                {statusLabel(active.status)} <span>↗</span>
              </button>
            )}
          </div>
          {(error || connectionError || state?.error) && (
            <div role="alert" className="error banner">
              {error || connectionError || state?.error}
              <button
                onClick={() => {
                  setError("");
                  void refresh().catch((e) => setConnectionError(e.message));
                }}
              >
                다시 확인
              </button>
            </div>
          )}
          {!state || !props ? (
            <Empty>로컬 서버에 연결하는 중입니다.</Empty>
          ) : (
            <div key={state.project?.filepath || "none"}>
              {tab === "storage" && <Storage {...props} />}
              {tab === "projects" && <Projects {...props} />}{" "}
              {tab === "dataset" &&
                (state.project ? (
                  <Dataset {...props} project={state.project} />
                ) : (
                  <Empty>프로젝트 화면에서 프로젝트를 열어주세요.</Empty>
                ))}
              {tab === "training" &&
                (state.project ? (
                  <Training {...props} project={state.project} />
                ) : (
                  <Empty>프로젝트 화면에서 프로젝트를 열어주세요.</Empty>
                ))}
              {tab === "inference" && <Inference {...props} />}{" "}
              {tab === "export" && <Export {...props} />}{" "}
              {tab === "defects" && <Defects key={state.project?.filepath || "no-project"} {...props} />}
            </div>
          )}
        </main>
        <footer>
          Deep Vision Studio <span>로컬 파일은 설정한 경로에 저장됩니다.</span>
        </footer>
      </div>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
