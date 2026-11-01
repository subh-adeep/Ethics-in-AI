import json
from pathlib import Path
from collections import Counter

models = ["qwen", "llama", "phi4"]
base_dir = Path("PROJECT/kaggle/live_kaggle_output")

for m in models:
    p = base_dir / f"{m}_215q_results" / f"{m}_math_results.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    
    total_turns = len(data) * 2  # 430
    skipped_calls = 0
    executed_calls = 0
    clean_calls = 0
    failed_calls = 0
    failures = []

    for item in data:
        for t in ["P1", "P2"]:
            td = item["turns"][t]
            if not td.get("code_used"):
                skipped_calls += 1
            else:
                executed_calls += 1
                out = str(td.get("python_output", ""))
                if "[Python Error:" in out:
                    failed_calls += 1
                    failures.append({
                        "id": item["id"],
                        "turn": t,
                        "code": td.get("emitted_code", ""),
                        "error": out
                    })
                else:
                    clean_calls += 1
    
    print("=" * 60)
    print(f"MODEL: {m.upper()}")
    print("=" * 60)
    print(f"Total Pressure Turn Opportunities: {total_turns}")
    print(f"1. Tool Skipped by Model: {skipped_calls} ({skipped_calls/total_turns*100:.1f}%)")
    print(f"2. Tool Attempted by Model: {executed_calls} ({executed_calls/total_turns*100:.1f}%)")
    print(f"   -> Succeeded Cleanly: {clean_calls} ({clean_calls/executed_calls*100:.1f}%)")
    print(f"   -> FAILED (Python Error): {failed_calls} ({failed_calls/executed_calls*100:.1f}%)")
    
    # Categorize error types
    err_types = Counter()
    for f in failures:
        raw_err = f["error"].split("[Python Error:")[1].split("-")[0].strip()
        err_types[raw_err] += 1
    print("\nError Categories:")
    for k, v in err_types.most_common():
        print(f"   - {k}: {v} times")

    print("\nConcrete Examples of What Failed:")
    for idx, f in enumerate(failures[:5], 1):
        c_snippet = f["code"].replace("\n", " ")[:100]
        print(f"   Ex {idx} [{f['id']} {f['turn']}]:")
        print(f"      Emitted Code: {c_snippet}")
        print(f"      Python Error: {f['error']}")
    print("\n")
