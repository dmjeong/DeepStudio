import { useEffect, useState, type Dispatch, type SetStateAction } from "react";

export function readDraft<T>(storage: Pick<Storage, "getItem">, key: string, fallback: T): T {
  try { const raw = storage.getItem(key); return raw === null ? fallback : JSON.parse(raw) as T; } catch { return fallback; }
}
export function useDraft<T>(project: string, field: string, initial: T): [T, Dispatch<SetStateAction<T>>] {
  const key = `studio-draft:${project}:${field}`;
  const [value, setValue] = useState<T>(() => readDraft(sessionStorage, key, initial));
  useEffect(() => { try { sessionStorage.setItem(key, JSON.stringify(value)); } catch { /* 현재 화면의 입력은 유지 */ } }, [key, value]);
  return [value, setValue];
}
