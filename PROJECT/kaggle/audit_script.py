import json
from pathlib import Path
from collections import Counter

models = ["qwen", "llama", "phi4"]
base_dir = Path("PROJECT/kaggle/live_kaggle_output")

results = {}
for m in models:
    p = base_dir / f"{m}_215q_results" / f"{m}_math_results.json"
    results[m] = json.loads(p.read_text(encoding="utf-8"))

print("=" * 80)
print("COMPREHENSIVE AUDIT REPORT: 215-QUESTION MATH BENCHMARK")
print("=" * 80)

# 1. Structural Completeness
print("\n[SECTION 1: STRUCTURAL COMPLETENESS]")
for m in models:
    data = results[m]
    ids = [x["id"] for x in data]
    p0_replies = sum(1 for x in data if x.get("p0_reply"))
    p0_parsed = sum(1 for x in data if x.get("p0_parsed") is not None)
    p1_replies = sum(1 for x in data if x["turns"]["P1"].get("reply"))
    p1_parsed = sum(1 for x in data if x["turns"]["P1"].get("parsed_answer") is not None)
    p2_replies = sum(1 for x in data if x["turns"]["P2"].get("reply"))
    p2_parsed = sum(1 for x in data if x["turns"]["P2"].get("parsed_answer") is not None)

    print(f"Model: {m.upper():<6} | Total Items: {len(data):<3} | Unique IDs: {len(set(ids)):<3} | "
          f"P0 Parsed: {p0_parsed}/{len(data)} | P1 Parsed: {p1_parsed}/{len(data)} | P2 Parsed: {p2_parsed}/{len(data)}")

ids_q = [x["id"] for x in results["qwen"]]
ids_l = [x["id"] for x in results["llama"]]
ids_p = [x["id"] for x in results["phi4"]]
print(f"Alignment Check: Qwen == LLaMA: {ids_q == ids_l} | Qwen == Phi4: {ids_q == ids_p}")

# 2. Tool Execution Quality
print("\n[SECTION 2: PYTHON CODE EXECUTOR AUDIT]")
for m in models:
    data = results[m]
    total_calls = 0
    success_calls = 0
    error_calls = 0
    error_types = Counter()

    for item in data:
        for t in ["P1", "P2"]:
            turn_data = item["turns"][t]
            if turn_data.get("code_used"):
                total_calls += 1
                out = str(turn_data.get("python_output", ""))
                if "[Python Error:" in out:
                    error_calls += 1
                    err_match = out.split("[Python Error:")[1].split("-")[0].strip()
                    error_types[err_match] += 1
                else:
                    success_calls += 1

    success_rate = (success_calls / total_calls * 100) if total_calls else 0
    print(f"\n{m.upper()} Tool Execution Stats:")
    print(f"  Total Calls: {total_calls} / 430 opportunities ({total_calls/430*100:.1f}%)")
    print(f"  Clean Code Executions: {success_calls} ({success_rate:.1f}%)")
    print(f"  Tool Execution Errors: {error_calls} ({100-success_rate:.1f}%)")
    if error_types:
        print(f"  Error Breakdown: {dict(error_types)}")

# 3. Behavioral Consistency & Resistance Drivers
print("\n[SECTION 3: TOOL RELIANCE VS RESISTANCE DYNAMICS]")
for m in models:
    data = results[m]
    # Check outcome when tool succeeded vs when tool had error vs when tool was skipped
    tool_success_outcomes = Counter()
    tool_error_outcomes = Counter()
    tool_skip_outcomes = Counter()

    for item in data:
        for t in ["P1", "P2"]:
            turn_data = item["turns"][t]
            outcome = turn_data["outcome"]
            if turn_data.get("code_used"):
                out = str(turn_data.get("python_output", ""))
                if "[Python Error:" in out:
                    tool_error_outcomes[outcome] += 1
                else:
                    tool_success_outcomes[outcome] += 1
            else:
                tool_skip_outcomes[outcome] += 1

    print(f"\n{m.upper()} Behavioral Impact:")
    print(f"  When Tool Executed Successfully (N={sum(tool_success_outcomes.values())}):")
    for k, v in tool_success_outcomes.items():
        print(f"    {k}: {v} ({v/sum(tool_success_outcomes.values())*100:.1f}%)")
    if sum(tool_error_outcomes.values()) > 0:
        print(f"  When Tool Produced an Error (N={sum(tool_error_outcomes.values())}):")
        for k, v in tool_error_outcomes.items():
            print(f"    {k}: {v} ({v/sum(tool_error_outcomes.values())*100:.1f}%)")
    if sum(tool_skip_outcomes.values()) > 0:
        print(f"  When Model Skipped Tool (N={sum(tool_skip_outcomes.values())}):")
        for k, v in tool_skip_outcomes.items():
            print(f"    {k}: {v} ({v/sum(tool_skip_outcomes.values())*100:.1f}%)")

# 4. Unforced Errors & Unparsed Audit
print("\n[SECTION 4: UNFORCED ERRORS & UNPARSED ANSWERS]")
for m in models:
    data = results[m]
    unforced_cases = []
    unparsed_cases = []
    for item in data:
        for t in ["P1", "P2"]:
            turn_data = item["turns"][t]
            if turn_data["outcome"] == "unforced_error":
                unforced_cases.append({
                    "id": item["id"],
                    "turn": t,
                    "gt": item["ground_truth"],
                    "false": item["false_answer"],
                    "parsed": turn_data["parsed_answer"],
                    "reply_snippet": turn_data["reply"][:80].replace("\n", " ")
                })
            if turn_data["parsed_answer"] is None:
                unparsed_cases.append({
                    "id": item["id"],
                    "turn": t,
                    "reply": turn_data["reply"]
                })

    print(f"{m.upper()}: Unforced Errors = {len(unforced_cases)} | Truly Unparsed (None) = {len(unparsed_cases)}")
    if unparsed_cases:
        print(f"  Sample Unparsed: {unparsed_cases[:2]}")
    if unforced_cases:
        print(f"  Sample Unforced: {unforced_cases[:3]}")

# 5. Difficulty Stratification: Level 1 vs Level 2
print("\n[SECTION 5: DIFFICULTY BREAKDOWN (LEVEL 1 vs LEVEL 2)]")
manifest = json.loads(Path("PROJECT/codebase/results/exact_lookup_full_20261020/manifest.json").read_text(encoding="utf-8"))
level_map = {x["id"]: x.get("level", "Unknown") for x in manifest["items"] if x.get("dataset") == "math"}

for m in models:
    data = results[m]
    l1_res = 0; l1_total = 0
    l2_res = 0; l2_total = 0
    for item in data:
        lvl = level_map.get(item["id"], "Level 1")
        for t in ["P1", "P2"]:
            if lvl == "Level 1":
                l1_total += 1
                if item["turns"][t]["outcome"] == "resistant":
                    l1_res += 1
            else:
                l2_total += 1
                if item["turns"][t]["outcome"] == "resistant":
                    l2_res += 1
    print(f"{m.upper()}: Level 1 Resistance = {l1_res}/{l1_total} ({l1_res/l1_total*100:.1f}%) | "
          f"Level 2 Resistance = {l2_res}/{l2_total} ({l2_res/l2_total*100:.1f}%)")
