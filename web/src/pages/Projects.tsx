import { useState } from "react";
import { api, filename, join } from "../api";
import { Empty, Field, Panel, PathField, Stat } from "../components";
import { tasks, type PageProps } from "../page-context";

const taskDetails: Record<string, [string, string]> = {
  classify: ["CLS", "이미지 전체를 보고 클래스를 판정합니다."],
  detect: ["DET", "객체의 위치와 크기를 사각 박스로 찾습니다."],
  obb: ["OBB", "기울어진 객체를 회전 박스로 찾습니다."],
  segment: ["SEG", "대상의 윤곽을 픽셀 단위로 구분합니다."],
  anomaly: ["AD", "정상 이미지와 다른 영역을 찾습니다."],
};

export function Projects({ state, refresh, act, busy }: PageProps) {
  const [name, setName] = useState("");
  const [task, setTask] = useState("classify");
  const [parent, setParent] = useState(state.home);
  const [classes, setClasses] = useState("");
  const [path, setPath] = useState("");
  const classNames = classes
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  return (
    <>
      <div className="columns">
        <Panel title="프로젝트 열기">
          <PathField
            label="프로젝트 파일"
            value={path}
            onChange={setPath}
            home={state.home}
            hint="기존 데스크톱 버전의 .dvproj 파일도 열 수 있습니다."
          />
          <button
            className="primary"
            disabled={busy || !path}
            onClick={() =>
              act(async () => {
                await api("/projects/open", "POST", { path });
                await refresh();
              })
            }
          >
            프로젝트 열기
          </button>
          <h3>최근 프로젝트</h3>
          <div className="recent-list">
            {state.recent.length ? (
              state.recent.map((p) => (
                <button
                  key={p}
                  disabled={busy}
                  onClick={() =>
                    act(async () => {
                      await api("/projects/open", "POST", { path: p });
                      await refresh();
                    })
                  }
                >
                  <strong>{filename(p)}</strong>
                  <small>{p}</small>
                  <span>↗</span>
                </button>
              ))
            ) : (
              <Empty>최근 프로젝트가 없습니다.</Empty>
            )}
          </div>
        </Panel>
        <Panel title="새 프로젝트">
          <div className="form-grid">
            <Field label="프로젝트 이름">
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="예: Surface inspection"
              />
            </Field>

          </div>
          <fieldset className="project-task-choices" disabled={busy}>
            <legend>작업 유형</legend>
            <div className="project-task-grid">
              {Object.entries(tasks).map(([value, label]) => (
                <label key={value} className={`project-task-card ${task === value ? "selected" : ""}`}>
                  <input type="radio" name="project-task" value={value} checked={task === value} onChange={() => setTask(value)} />
                  <span className="project-task-code">{taskDetails[value][0]}</span>
                  <strong>{label}</strong>
                  <span className="project-task-description">{taskDetails[value][1]}</span>
                </label>
              ))}
            </div>
          </fieldset>
          <PathField
            label="상위 저장 폴더"
            directory
            value={parent}
            onChange={setParent}
            home={state.home}
          />
          <Field
            label="클래스 이름"
            hint="쉼표로 구분합니다. 데이터셋 화면에서 추가하거나 삭제할 수 있습니다."
          >
            <input
              value={classes}
              onChange={(e) => setClasses(e.target.value)}
              disabled={task === "anomaly"}
              placeholder={
                task === "anomaly"
                  ? "정상 데이터 폴더는 자동 생성됩니다"
                  : "예: good, scratch, dent"
              }
            />
          </Field>
          <p className="muted">
            저장 위치: {join(parent, name || "프로젝트 이름")}
          </p>
          <button
            className="primary"
            disabled={
              busy ||
              !name.trim() ||
              !parent ||
              (task !== "anomaly" && classNames.length === 0)
            }
            onClick={() =>
              act(async () => {
                await api("/projects", "POST", {
                  name: name.trim(),
                  task,
                  parent,
                  class_names: task === "anomaly" ? [] : classNames,
                });
                await refresh();
              })
            }
          >
            프로젝트 만들기
          </button>
        </Panel>
      </div>
      {state.project && (
        <Panel title="현재 프로젝트">
          <div className="stats">
            <Stat label="프로젝트" value={state.project.name} />
            <Stat label="작업" value={tasks[state.project.task]} />
            <Stat
              label="클래스"
              value={state.project.data.class_names.length}
            />
            <Stat label="학습 기록" value={state.project.runs.length} />
          </div>
          <p className="muted path">{state.project.filepath}</p>
        </Panel>
      )}
    </>
  );
}
