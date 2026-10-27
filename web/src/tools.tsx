import { useEffect, useState } from "react";
import { api, join } from "./api";
import { useDraft } from "./drafts";
import { Check, Field, JobMonitor, Panel, PathField } from "./components";
import type { PageProps } from "./pages";

export function Export({ state, run, act, busy }: PageProps) {
  const projectKey = state.project?.filepath || "no-project";
  const [weights, setWeights] = useState(
    state.project?.runs.at(-1)?.checkpoint_path || "",
  );
  const [output, setOutput] = useDraft(projectKey, "export-output",
    state.project ? join(state.project.project_dir, "exports/model.onnx") : "",
  );
  // Keep an explicit encryption choice when this page unmounts on navigation.
  // Drafts are scoped to the project and contain only the key's path.
  const [encrypted, setEncrypted] = useDraft(projectKey, "export-encrypted", false);
  const [keyPath, setKeyPath] = useDraft(projectKey, "export-key-path", "");
  const [keyNotice, setKeyNotice] = useState("");
  const [opset, setOpset] = useState(17);
  const [dynamic, setDynamic] = useState(false);
  const efficientnet = state.project?.task === "classify" && String(state.project.model.model_id || "").startsWith("efficientnet_");
  const [datasetValidation, setDatasetValidation] = useState(efficientnet);
  const [validationDir, setValidationDir] = useState("");
  const [precisionFallback, setPrecisionFallback] = useState(false);
  useEffect(() => {
    setDatasetValidation(efficientnet);
    setValidationDir("");
    setPrecisionFallback(false);
    setKeyNotice("");
    setWeights(state.project?.runs.at(-1)?.checkpoint_path || "");
  }, [state.project?.filepath, efficientnet]);
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
          <Check label="모델과 설정을 암호화해서 내보내기 (.dvsenc)" value={encrypted}
            onChange={value => { setEncrypted(value); setOutput(output.replace(/\.(onnx|dvsenc)$/i, value ? ".dvsenc" : ".onnx")); }} />
          {encrypted && <>
            <PathField label="암호키 파일 (.key): 기존 키를 선택하거나 새로 저장할 경로 입력" value={keyPath}
              onChange={value => { setKeyPath(value); setKeyNotice(""); }} home={state.home} />
            <button disabled={!keyPath} onClick={() => act(async () => {
              await api("/model-keys", "POST", { path: keyPath });
              setKeyNotice("새 암호키를 만들었습니다. 이 키를 별도로 보관하세요.");
            })}>입력한 경로에 새 키 생성</button>
            {keyNotice && <p role="status">{keyNotice}</p>}
            <p className="muted">기존 키는 덮어쓰지 않습니다. 키를 별도로 보관하세요. 복호화는 초기화 때 한 번만 수행합니다.</p>
          </>}
          {efficientnet && <>
            <Check label="실제 이미지 추가 검증: PyTorch·ONNX 판정 일치·확률 오차 0.1%p 이하"
              value={datasetValidation} onChange={setDatasetValidation} />
            {datasetValidation ? <>
              <PathField label="비교 이미지 폴더 (비우면 프로젝트 val → test → train에서 자동 선택)"
                value={validationDir} onChange={setValidationDir} home={state.home} directory />
              <p className="muted">평면 폴더와 클래스별 하위 폴더를 모두 지원합니다. 이미지 0장 클래스는 제외하며, 정답률이 아닌 PyTorch·ONNX 출력 일치를 검사합니다.</p>
            </> : <Check label="고정밀 호환 재시도 허용 (추론이 느려질 수 있음)"
              value={precisionFallback} onChange={setPrecisionFallback} />}
          </>}
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
              disabled={!weights || !output || (encrypted && !keyPath)}
              onClick={() =>
                act(async () =>
                  setId(
                    (
                      await run("/jobs/export", {
                        weights,
                        output,
                        opset,
                        encryption_key_path: encrypted ? keyPath : "",
                        dynamic_batch: dynamic,
                        dataset_validation: efficientnet && datasetValidation,
                        validation_dir: validationDir,
                        allow_precision_fallback: efficientnet && !datasetValidation && precisionFallback,
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
          검증합니다. 암호화는 모든 태스크와 SAM2의 다중 그래프를 지원합니다.
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
