import { useState } from "react";
import { join } from "./api";
import { Check, Field, JobMonitor, Panel, PathField } from "./components";
import type { PageProps } from "./pages";

export function Export({ state, run, act, busy }: PageProps) {
  const [weights, setWeights] = useState(
    state.project?.runs.at(-1)?.checkpoint_path || "",
  );
  const [output, setOutput] = useState(
    state.project ? join(state.project.project_dir, "exports/model.onnx") : "",
  );
  const [opset, setOpset] = useState(17);
  const [dynamic, setDynamic] = useState(false);
  const [id, setId] = useState<string | null>(
    state.jobs.find((j) => j.kind === "export")?.id || null,
  );
  return (
    <>
      <Panel title="ONNX 내보내기">
        <fieldset disabled={busy}>
          <div className="columns">
            <PathField
              label="학습 체크포인트"
              value={weights}
              onChange={setWeights}
              home={state.home}
            />
            <Field label="출력 파일의 절대 경로">
              <input
                value={output}
                onChange={(e) => setOutput(e.target.value)}
                placeholder="예: C:/projects/inspection/exports/model.onnx"
              />
            </Field>
          </div>
          <div className="toolbar">
            <Field label="ONNX opset">
              <select
                value={opset}
                onChange={(e) => setOpset(Number(e.target.value))}
              >
                {[17, 18, 19, 20].map((v) => (
                  <option key={v}>{v}</option>
                ))}
              </select>
            </Field>
            <Check label="동적 배치" value={dynamic} onChange={setDynamic} />
            <button
              className="primary push"
              disabled={!weights || !output}
              onClick={() =>
                act(async () =>
                  setId(
                    (
                      await run("/jobs/export", {
                        weights,
                        output,
                        opset,
                        dynamic_batch: dynamic,
                      })
                    ).id,
                  ),
                )
              }
            >
              변환 및 검증
            </button>
          </div>
        </fieldset>
        <p className="muted">
          체크포인트의 전처리와 클래스 정보를 함께 내보내고 ONNX Runtime 결과를
          검증합니다. PatchCore의 메모리 뱅크는 이 ONNX 경로를 지원하지
          않습니다.
        </p>
        <p className="muted">
          변환 중 중단 요청은 즉시 엔진을 종료하지 않습니다. 현재 변환을 마칠
          때까지 기다려주세요.
        </p>
      </Panel>
      <JobMonitor id={id} />
    </>
  );
}
export { Defects } from "./defects";
