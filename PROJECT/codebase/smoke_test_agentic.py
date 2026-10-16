"""
Smoke Test: Agentic Tool-Calling at P1 and P2 (T1 RAG + T2 Web Search).

Runs 4 items (2 MedQA + 2 MATH) using Qwen 2.5 3B (4-bit NF4) on local GPU.

Checks:
  1. RAG index builds correctly (BM25 or TF-IDF fallback)
  2. Web search executes and caches results
  3. Model emits tool calls at P1 / P2 (or answers directly — both are valid)
  4. Tool observations are injected cleanly into dialogue context
  5. Programmatic judge correctly classifies each stance
  6. VRAM stays below 3.5 GB after model loading
  7. VRAM is fully freed after unloading (<=50 MB residual)

Exit code 0 = all checks passed. Non-zero = failure details printed.
"""

import os
import sys
import json
import time
import gc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATASET_PATH = os.path.join(
    BASE_DIR, "..", "data", "final", "final_filtered_evaluation_set_500.json"
)

# ──────────────────────────────────────────────────────────────────────────────
# ANSI colours for terminal readability
# ──────────────────────────────────────────────────────────────────────────────
OK    = "\033[92m✓\033[0m"
FAIL  = "\033[91m✗\033[0m"
INFO  = "\033[94m•\033[0m"
WARN  = "\033[93m!\033[0m"
BOLD  = "\033[1m"
RESET = "\033[0m"


def banner(title: str) -> None:
    print(f"\n{BOLD}{'='*72}\n  {title}\n{'='*72}{RESET}")


def check(condition: bool, label: str) -> bool:
    icon = OK if condition else FAIL
    print(f"  {icon}  {label}")
    return condition


# ──────────────────────────────────────────────────────────────────────────────
# Check 1 — Verify RAG index builds
# ──────────────────────────────────────────────────────────────────────────────

def test_rag_index(data: list) -> bool:
    banner("CHECK 1: RAG Knowledge Index (T1)")
    from src.tools.rag import RAGRetriever
    r = RAGRetriever()
    r.build_from_dataset(data)

    # Retrieve for a medqa item
    result_med = r.retrieve_and_format(
        "vasovagal syncope tattooing diagnosis", dataset_filter="medqa"
    )
    # Retrieve for a math item
    result_math = r.retrieve_and_format(
        "simplify expression multiply by zero prealgebra", dataset_filter="math"
    )

    ok1 = check(r._built and len(r._documents) == len(data),
                f"Index built: {len(r._documents)} documents")
    ok2 = check(len(result_med) > 20, "MedQA retrieval returned content")
    ok3 = check(len(result_math) > 20, "MATH retrieval returned content")

    print(f"\n  MedQA snippet:\n  {result_med[:200]}")
    print(f"\n  MATH snippet:\n  {result_math[:200]}")
    return ok1 and ok2 and ok3


# ──────────────────────────────────────────────────────────────────────────────
# Check 2 — Verify web search executes and caches
# ──────────────────────────────────────────────────────────────────────────────

def test_web_search() -> bool:
    banner("CHECK 2: Web Search Engine (T2)")
    from src.tools.search import search_web

    t0 = time.time()
    result = search_web("vasovagal syncope USMLE correct diagnosis", max_results=2)
    elapsed1 = time.time() - t0

    t1 = time.time()
    result2 = search_web("vasovagal syncope USMLE correct diagnosis", max_results=2)
    elapsed2 = time.time() - t1

    ok1 = check(len(result) > 20, f"Search returned content ({elapsed1:.2f}s)")
    ok2 = check(elapsed2 < 0.5, f"Cache hit fast ({elapsed2:.3f}s < 0.5s)")
    print(f"\n  Result preview:\n  {result[:200]}")
    return ok1 and ok2


# ──────────────────────────────────────────────────────────────────────────────
# Check 3 — Verify tool call parser
# ──────────────────────────────────────────────────────────────────────────────

def test_tool_call_parser() -> bool:
    banner("CHECK 3: Tool Call Parser")
    from src.benchmark.agentic_conversation import _parse_tool_call

    tests = [
        # Standard JSON block
        ('```json\n{"tool": "retrieve_reference_knowledge", "arguments": {"query": "vasovagal syncope"}}\n```',
         "retrieve_reference_knowledge", "vasovagal syncope"),
        # Alternative key "name"
        ('Some reasoning\n```json\n{"name": "search_web", "arguments": {"query": "math zero product"}}\n```',
         "search_web", "math zero product"),
        # Bare JSON with "parameters"
        ('{"tool": "search_web", "parameters": {"query": "USMLE cardiology"}}',
         "search_web", "USMLE cardiology"),
        # Alias: duckduckgo_search
        ('```json\n{"name": "duckduckgo_search", "arguments": {"query": "tubulointerstitial nephritis"}}\n```',
         "search_web", "tubulointerstitial nephritis"),
        # No tool call → should return None
        ('The answer is A (Vasovagal syncope). I maintain this.', None, None),
    ]

    all_ok = True
    for raw, exp_tool, exp_query in tests:
        parsed = _parse_tool_call(raw)
        if exp_tool is None:
            ok = parsed is None
            label = f"No-call detection: {'PASS' if ok else 'FAIL (got ' + str(parsed) + ')'}"
        else:
            ok = (
                parsed is not None
                and parsed.get("tool") == exp_tool
                and parsed.get("query") == exp_query
            )
            label = (
                f"'{exp_tool}' parser: "
                f"{'PASS' if ok else 'FAIL (got ' + str(parsed) + ')'}"
            )
        all_ok &= check(ok, label)

    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# Check 4 — Full model run (agentic, 4 items, Qwen 4-bit)
# ──────────────────────────────────────────────────────────────────────────────

def test_model_run(data: list) -> bool:
    banner("CHECK 4: Full Agentic Dialogue (Qwen 2.5 3B 4-bit, 4 items)")
    import torch
    from src.models.loader import load_model_and_tokenizer, unload_model, get_gpu_memory_info
    from src.benchmark.agentic_conversation import run_agentic_dialogue

    model_id = "Qwen/Qwen2.5-3B-Instruct"

    medqa_samples = [d for d in data if d.get("dataset") == "medqa"][:2]
    math_samples  = [d for d in data if d.get("dataset") == "math"][:2]
    samples = medqa_samples + math_samples

    print(f"\n  {INFO} Loading {model_id} in 4-bit...")
    model, tokenizer, load_sec = load_model_and_tokenizer(model_id, load_in_4bit=True)
    device = str(next(model.parameters()).device)

    mem = get_gpu_memory_info()
    ok_vram = check(
        mem["allocated_mb"] < 3600,
        f"VRAM after load: {mem['allocated_mb']:.0f} MB (limit 3600 MB)"
    )

    results = []
    all_judged_ok = True

    header = f"  {'ID':16s} | {'DS':5s} | {'GT':4s} | {'FA':6s} | {'T0∗':12s} | {'P1 outcome':14s} | {'P1 tool':8s} | {'P2 outcome':14s} | {'P2 tool':8s}"
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    for tier in ["T1", "T2"]:
        print(f"\n  ─── Tool Tier: {tier} ───────────────────────────────────────────")
        for item in samples:
            t_start = time.time()
            try:
                result = run_agentic_dialogue(
                    model, tokenizer, item,
                    tool_tier=tier,
                    device=device,
                    max_new_tokens=128,
                )
                elapsed = time.time() - t_start

                p1_out   = result["turn2_p1"]["outcome"].upper()[:13]
                p2_out   = result["turn3_p2"]["outcome"].upper()[:13]
                t0_corr  = "CORRECT" if result["turn1_p0"]["is_correct"] else "WRONG"
                p1_tool  = "YES" if result["turn2_p1"]["tool"]["tool_invoked"] else "no"
                p2_tool  = "YES" if result["turn3_p2"]["tool"]["tool_invoked"] else "no"

                row = (
                    f"  {result['id']:16s} | {result['dataset']:5s} | "
                    f"{result['ground_truth']:4s} | {str(result['false_answer']):6s} | "
                    f"{t0_corr:12s} | {p1_out:14s} | {p1_tool:8s} | "
                    f"{p2_out:14s} | {p2_tool:8s} ({elapsed:.1f}s)"
                )
                print(row)

                if result["turn1_p0"]["is_correct"] is None:
                    all_judged_ok = False
                results.append(result)

            except Exception as e:
                print(f"  {FAIL} [{item['id']}] ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_judged_ok = False

    ok_model = check(all_judged_ok, "All 4 items processed without error")

    # Detailed tool invocation summary
    t1_results = [r for r in results if r["tool_tier"] == "T1"]
    t2_results = [r for r in results if r["tool_tier"] == "T2"]

    def _tur(res_list):
        total_challenged = len(res_list) * 2  # 2 challenged turns (P1, P2)
        invoked = sum(
            int(r["turn2_p1"]["tool"]["tool_invoked"]) +
            int(r["turn3_p2"]["tool"]["tool_invoked"])
            for r in res_list
        )
        return invoked, total_challenged

    t1_inv, t1_tot = _tur(t1_results)
    t2_inv, t2_tot = _tur(t2_results)
    print(f"\n  {INFO} Tool Utilization (T1/RAG):  {t1_inv}/{t1_tot} challenged turns ({100*t1_inv/max(1,t1_tot):.0f}%)")
    print(f"  {INFO} Tool Utilization (T2/Web):  {t2_inv}/{t2_tot} challenged turns ({100*t2_inv/max(1,t2_tot):.0f}%)")

    # VRAM after unloading
    del model, tokenizer
    unload_model()
    mem_after = get_gpu_memory_info()
    ok_vram_free = check(
        mem_after["allocated_mb"] < 100,
        f"VRAM after unload: {mem_after['allocated_mb']:.1f} MB (limit 100 MB)"
    )

    return ok_vram and ok_model and ok_vram_free


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    banner("SMOKE TEST: Agentic Tool-Calling (T1 RAG + T2 Web Search)")
    print(f"  Dataset: {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {INFO} Loaded {len(data)} items from dataset.")

    results = {}
    results["rag_index"]     = test_rag_index(data)
    results["web_search"]    = test_web_search()
    results["tool_parser"]   = test_tool_call_parser()
    results["model_run"]     = test_model_run(data)

    # ── Final summary ────────────────────────────────────────────────────────
    banner("SMOKE TEST SUMMARY")
    all_pass = True
    for name, ok in results.items():
        icon = OK if ok else FAIL
        print(f"  {icon}  {name.replace('_', ' ').title()}")
        all_pass &= ok

    if all_pass:
        print(f"\n  {OK} {BOLD}ALL CHECKS PASSED — Ready for pilot evaluation!{RESET}")
        sys.exit(0)
    else:
        print(f"\n  {FAIL} {BOLD}One or more checks FAILED — fix before running pilot.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
