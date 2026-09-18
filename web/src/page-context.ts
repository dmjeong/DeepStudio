import type { Job, StudioState } from "./api";

export interface PageProps {
  state: StudioState;
  refresh: () => Promise<void>;
  run: (path: string, body?: unknown) => Promise<Job>;
  act: (work: () => Promise<unknown>) => void;
  busy: boolean;
}
export const tasks: Record<string, string> = {
  classify: "이미지 분류",
  detect: "객체 탐지",
  segment: "세그멘테이션",
  anomaly: "이상 탐지",
};
