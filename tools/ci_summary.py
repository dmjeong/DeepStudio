"""CI 결과 파일을 짧은 상태와 사람이 읽을 수 있는 요약으로 변환한다."""

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def summarize(kind, path):
    path = Path(path)
    if not path.exists():
        return "결과 파일 없음 (설치 또는 실행 단계 확인)", ""
    if kind == "lint":
        entries = json.loads(path.read_text(encoding="utf-8"))
        details = [
            f"{Path(item['filename']).name}:{item['location']['row']} {item['code']}"
            for item in entries
        ]
        return f"{len(entries)} errors; " + ", ".join(details[:8]), "\n".join(details)
    root = ET.parse(path).getroot()
    suites = list(root.iter("testsuite"))
    totals = {key: sum(int(s.get(key, 0)) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    failed = [case.get("name", "unknown") for case in root.iter("testcase")
              if case.find("failure") is not None or case.find("error") is not None]
    passed = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    summary = (f"{passed} passed, {totals['failures']} failed, "
               f"{totals['errors']} errors, {totals['skipped']} skipped")
    if failed:
        summary += "; " + ", ".join(failed[:5])
    return summary, "\n".join(failed)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    summary, detail = summarize(sys.argv[1], sys.argv[2])
    print(summary)
    print(detail)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:
            out.write("summary=" + summary.replace("\n", " ") + "\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as out:
            out.write(summary + "\n\n```text\n" + detail + "\n```\n")
