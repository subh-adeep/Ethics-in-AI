"""Validate a downloaded pilot and write a compact, honest status summary."""
import argparse
import json
from pathlib import Path


def summarize(directory):
    directory = Path(directory)
    output = directory / "tools_v2"
    rows = []
    issues = []
    for model in ("qwen", "llama", "phi4"):
        if not (output / f"{model}_health.json").exists():
            issues.append(f"{model}: no completed generation health check")
        for tier in ("T0", "T1", "T2"):
            path = output / f"{model}_{tier}_optional.json"
            if not path.exists():
                issues.append(f"{model} {tier}: missing output")
                continue
            run = json.loads(path.read_text(encoding="utf-8"))
            if run["completed"] != run["total"]:
                issues.append(f"{model} {tier}: incomplete")
            unparsed = sum(run["metrics"][p + "_outcomes"]["unparsed"] for p in ("p1", "p2"))
            if unparsed:
                issues.append(f"{model} {tier}: {unparsed} answers require manual review")
            row = {"model": model, "tier": tier, **run["metrics"]}
            rows.append(row)
    if not (output / "completed.json").exists():
        issues.append("Pilot completion marker missing")
    if rows and not sum(r["tool_requested_n"] for r in rows):
        issues.append("No optional model tool requests observed; do not claim model-selected tool use verified")
    summary = {"pilot_passed": not issues, "issues": issues, "rows": rows}
    (directory / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    summarize(parser.parse_args().directory)
