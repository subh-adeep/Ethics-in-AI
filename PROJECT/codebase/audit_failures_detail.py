import json
from pathlib import Path
from collections import Counter

base = Path("PROJECT/codebase/results/math_python_full_20261031")
models = ["qwen", "llama", "phi4"]

failure_manifest = {}

for m in models:
    path = base / m / f"{m}_math_results.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tool_failures = []
    sycophancy_failures = []
    unforced_failures = []
    
    for item in data:
        qid = item["id"]
        gt = item["ground_truth"]
        fa = item.get("false_answer", "")
        p0 = item["turns"].get("P0", {}) if "P0" in item["turns"] else {"reply": item.get("p0_reply", "")}
        
        for t in ["P1", "P2"]:
            td = item["turns"].get(t, {})
            out = td.get("outcome", "")
            code_used = td.get("code_used", False)
            py_out = str(td.get("python_output", ""))
            emitted_code = td.get("emitted_code", "")
            
            # Check tool error
            if code_used and "[Python Error:" in py_out:
                tool_failures.append({
                    "id": qid,
                    "turn": t,
                    "ground_truth": gt,
                    "false_answer": fa,
                    "emitted_code": emitted_code,
                    "python_output": py_out,
                    "outcome": out,
                    "extracted_answer": td.get("parsed_answer"),
                    "raw_response": td.get("reply", "")
                })
            
            if out == "SYCOPHANTIC":
                sycophancy_failures.append((qid, t, code_used, "[Python Error:" in py_out))
            elif out == "UNFORCED_E":
                unforced_failures.append((qid, t, code_used, "[Python Error:" in py_out))
                
    failure_manifest[m] = {
        "tool_failures": tool_failures,
        "n_tool_failures": len(tool_failures),
        "n_unique_questions_tool_failures": len(set(x["id"] for x in tool_failures)),
        "n_sycophancy": len(sycophancy_failures),
        "n_unforced": len(unforced_failures)
    }

print("=" * 80)
print("FAILURE AUDIT SUMMARY")
print("=" * 80)
for m in models:
    info = failure_manifest[m]
    print(f"\nModel: {m.upper()}")
    print(f" - Tool execution failures (turns): {info['n_tool_failures']}")
    print(f" - Unique questions with tool failures: {info['n_unique_questions_tool_failures']}")
    print(f" - Sycophantic turns: {info['n_sycophancy']} / 430")
    print(f" - Unforced error turns: {info['n_unforced']} / 430")

# Save detailed manifest of tool failures to inspect
with open("PROJECT/codebase/results/math_python_full_20261031/tool_failures_manifest.json", "w", encoding="utf-8") as f:
    json.dump(failure_manifest, f, indent=2)

print("\nWrote manifest to PROJECT/codebase/results/math_python_full_20261031/tool_failures_manifest.json")
