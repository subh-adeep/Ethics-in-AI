"""
Week 4 Pilot Evaluation — Agentic Tool-Calling (T1 RAG + T2 Web Search).

Evaluates the first 25 MedQA + 25 MATH questions from the intersection dataset
(items where ALL 3 models answered correctly at baseline T0).

Runs each model sequentially on both T1 (RAG) and T2 (Web Search) tiers.
Fully compatible with local GTX 1650 (4GB VRAM), 4-bit NF4 quantization.

Output:
  results/week4_logs/<model_id>_t1_pilot.json
  results/week4_logs/<model_id>_t2_pilot.json
  results/week4_pilot_report.json   ← comparative T0 vs T1 vs T2 summary

Checkpointing every 5 items with auto-resume on restart.
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import json
import time
import gc
from datetime import datetime
from typing import Dict, Any, List, Optional

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(
    BASE_DIR, "..", "data", "final", "final_filtered_evaluation_set_500.json"
)
RESULTS_DIR  = os.path.join(BASE_DIR, "results", "week4_logs")
REPORT_PATH  = os.path.join(BASE_DIR, "results", "week4_pilot_report.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Pilot configuration
# ──────────────────────────────────────────────────────────────────────────────

PILOT_PER_DOMAIN = 25   # 25 MedQA + 25 MATH = 50 items total
CHECKPOINT_EVERY = 5    # Save partial results every N items
COOLDOWN_EVERY   = 60   # GPU cooldown break every N items
COOLDOWN_SECS    = 120  # Cooldown duration in seconds

MODELS = [
    {
        "id": "qwen",
        "name": "Qwen 2.5 3B",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
    },
    {
        "id": "llama",
        "name": "LLaMA 3.2 3B",
        "model_id": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    },
    {
        "id": "phi4",
        "name": "Phi-4 Mini 3.8B",
        "model_id": "unsloth/Phi-4-mini-instruct-bnb-4bit",
    },
]

TOOL_TIERS = ["T1", "T2"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_header(title: str) -> None:
    print(f"\n{'='*72}\n {title}\n{'='*72}", flush=True)


def load_pilot_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Loads the first PILOT_PER_DOMAIN items per domain from the dataset."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_items = json.load(f)

    medqa_items = [d for d in all_items if d.get("dataset") == "medqa"][:PILOT_PER_DOMAIN]
    math_items  = [d for d in all_items if d.get("dataset") == "math"][:PILOT_PER_DOMAIN]
    pilot = medqa_items + math_items

    print(
        f"[Data] Pilot cohort: {len(medqa_items)} MedQA + {len(math_items)} MATH "
        f"= {len(pilot)} total items.",
        flush=True,
    )
    return pilot


def compute_metrics(dialogues: List[Dict[str, Any]], tier: str) -> Dict[str, Any]:
    """Computes per-tier metrics from evaluated dialogues."""
    n = len(dialogues)
    if n == 0:
        return {}

    medqa = [d for d in dialogues if d["dataset"] == "medqa"]
    math  = [d for d in dialogues if d["dataset"] == "math"]

    def _count(records, key, val):
        return sum(1 for r in records if r.get(key) == val)

    def _domain_stats(records):
        total = len(records)
        if total == 0:
            return {}
        p1_resistant = sum(1 for r in records if r["turn2_p1"]["outcome"] == "resistant")
        p1_syco      = sum(1 for r in records if r["turn2_p1"]["outcome"] == "sycophantic")
        p2_resistant = sum(1 for r in records if r["turn3_p2"]["outcome"] == "resistant")
        p2_syco      = sum(1 for r in records if r["turn3_p2"]["outcome"] == "sycophantic")
        cumulative_flip = sum(1 for r in records if r["trajectory"]["cumulative_flip"])
        rem_resistant   = sum(1 for r in records if r["trajectory"]["remained_resistant"])

        # Tool Utilization Rate (TUR)
        challenged_turns = total * 2
        tools_invoked = sum(
            int(r["turn2_p1"]["tool"]["tool_invoked"]) +
            int(r["turn3_p2"]["tool"]["tool_invoked"])
            for r in records
        )
        # Evidence Override Rate (EOR)
        tool_then_syco = sum(
            int(r["turn2_p1"]["evidence_override"]) +
            int(r["turn3_p2"]["evidence_override"])
            for r in records
        )
        grounded_resistant = sum(
            int(r["turn2_p1"]["tool"]["tool_invoked"] and r["turn2_p1"]["outcome"] == "resistant") +
            int(r["turn3_p2"]["tool"]["tool_invoked"] and r["turn3_p2"]["outcome"] == "resistant")
            for r in records
        )

        def pct(num, denom):
            return round(100 * num / max(1, denom), 1)

        return {
            "n": total,
            "p1_resistant_pct":    pct(p1_resistant, total),
            "p1_sycophantic_pct":  pct(p1_syco, total),
            "p2_resistant_pct":    pct(p2_resistant, total),
            "p2_sycophantic_pct":  pct(p2_syco, total),
            "cumulative_flip_pct": pct(cumulative_flip, total),
            "remained_resistant_pct": pct(rem_resistant, total),
            # Agentic-specific
            "tool_utilization_rate_pct": pct(tools_invoked, challenged_turns),
            "evidence_override_rate_pct": pct(tool_then_syco, max(1, tools_invoked)),
            "tool_assisted_resistance_pct": pct(grounded_resistant, max(1, tools_invoked)),
        }

    return {
        "tier": tier,
        "overall": _domain_stats(dialogues),
        "medqa":   _domain_stats(medqa),
        "math":    _domain_stats(math),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Single model × tier evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_model_tier(
    model_cfg: Dict[str, str],
    pilot_data: List[Dict[str, Any]],
    tool_tier: str,
) -> List[Dict[str, Any]]:
    """
    Runs the agentic evaluation for one model on one tool tier.

    Returns the list of evaluated dialogue records.
    """
    from src.models.loader import (
        load_model_and_tokenizer,
        unload_model,
        get_gpu_memory_info,
    )
    from src.benchmark.agentic_conversation import run_agentic_dialogue

    model_id   = model_cfg["model_id"]
    model_name = model_cfg["name"]
    short_id   = model_cfg["id"]
    out_file   = os.path.join(RESULTS_DIR, f"{short_id}_{tool_tier.lower()}_pilot.json")

    print_header(f"{model_name} | {tool_tier} | {PILOT_PER_DOMAIN*2} pilot items")

    # ── Auto-resume from checkpoint ─────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    completed_ids = set()
    if os.path.exists(out_file):
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            results = existing.get("dialogues", [])
            completed_ids = {r["id"] for r in results}
            print(
                f"[Resume] Found {len(results)} completed items — skipping.",
                flush=True,
            )
        except (json.JSONDecodeError, KeyError):
            results = []
            completed_ids = set()

    remaining = [it for it in pilot_data if it["id"] not in completed_ids]
    if not remaining:
        print(f"[Done] All {len(results)} items already evaluated.", flush=True)
        return results

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\n[Model] Loading {model_id} (4-bit NF4)...", flush=True)
    model, tokenizer, load_sec = load_model_and_tokenizer(model_id, load_in_4bit=True)
    device = str(next(model.parameters()).device)
    mem = get_gpu_memory_info()
    print(
        f"[VRAM] Loaded in {load_sec:.1f}s — "
        f"Allocated: {mem['allocated_mb']:.0f} MB",
        flush=True,
    )

    # ── Evaluation loop ──────────────────────────────────────────────────────
    gpu_start_time = time.time()
    total = len(pilot_data)
    evaluated = len(results)

    for idx, item in enumerate(remaining, start=1):
        global_idx = evaluated + idx
        item_start = time.time()

        try:
            dialogue = run_agentic_dialogue(
                model, tokenizer, item,
                tool_tier=tool_tier,
                device=device,
                max_new_tokens=128,
            )
        except Exception as e:
            print(f"\n[Error] {item['id']}: {e}", flush=True)
            import traceback; traceback.print_exc()
            continue

        elapsed = time.time() - item_start
        results.append(dialogue)

        # Live per-item print
        p1_out  = dialogue["turn2_p1"]["outcome"].upper()[:10]
        p2_out  = dialogue["turn3_p2"]["outcome"].upper()[:10]
        p1_tool = "TOOL" if dialogue["turn2_p1"]["tool"]["tool_invoked"] else "  -  "
        p2_tool = "TOOL" if dialogue["turn3_p2"]["tool"]["tool_invoked"] else "  -  "
        p1_eor  = "EOR!" if dialogue["turn2_p1"]["evidence_override"] else "    "
        p2_eor  = "EOR!" if dialogue["turn3_p2"]["evidence_override"] else "    "

        print(
            f"[{global_idx:3d}/{total}] {item['id']:18s} ({item['dataset']:5s}) | "
            f"P1: {p1_tool} {p1_out:10s} {p1_eor} | "
            f"P2: {p2_tool} {p2_out:10s} {p2_eor} ({elapsed:.1f}s)",
            flush=True,
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (global_idx % CHECKPOINT_EVERY == 0) or (idx == len(remaining)):
            snapshot = {
                "model_id":   model_id,
                "model_name": model_name,
                "tool_tier":  tool_tier,
                "timestamp":  datetime.utcnow().isoformat(),
                "evaluated":  len(results),
                "dialogues":  results,
            }
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
            print(f"  [Checkpoint] {len(results)} items saved → {out_file}", flush=True)

        # ── GPU Cooldown ──────────────────────────────────────────────────────
        if (global_idx % COOLDOWN_EVERY == 0) and (idx != len(remaining)):
            elapsed_gpu = time.time() - gpu_start_time
            print(
                f"\n[Cooldown] {COOLDOWN_SECS}s GPU break after "
                f"{elapsed_gpu/60:.1f} min continuous run...",
                flush=True,
            )
            torch.cuda.empty_cache()
            time.sleep(COOLDOWN_SECS)
            gpu_start_time = time.time()

    # ── Unload model ─────────────────────────────────────────────────────────
    del model, tokenizer
    unload_model()
    mem_after = get_gpu_memory_info()
    print(
        f"\n[VRAM] Unloaded — Allocated: {mem_after['allocated_mb']:.0f} MB",
        flush=True,
    )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Week 4 Pilot Evaluation — Agentic Tool-Calling")
    parser.add_argument("--model", type=str, default="all", choices=["all", "qwen", "llama", "phi4"],
                        help="Model to run (default: all)")
    parser.add_argument("--tier", type=str, default="all", choices=["all", "T1", "T2"],
                        help="Tool tier to run (default: all)")
    parser.add_argument("--limit", type=int, default=PILOT_PER_DOMAIN,
                        help=f"Number of items per domain (default: {PILOT_PER_DOMAIN})")
    args = parser.parse_args()

    limit = args.limit
    models_to_run = MODELS if args.model == "all" else [m for m in MODELS if m["id"] == args.model]
    tiers_to_run = TOOL_TIERS if args.tier == "all" else [args.tier]

    print_header("ETHICS IN AI — WEEK 4 PILOT (T1 RAG + T2 WEB SEARCH)")
    print(f"Pilot size: {limit} MedQA + {limit} MATH = {limit*2} items per model × tier")
    print(f"Models: {[m['name'] for m in models_to_run]}")
    print(f"Tiers:  {tiers_to_run}")

    # Pre-warm RAG index so it only builds once per process
    print("\n[RAG] Pre-building reference index...", flush=True)
    from src.tools.rag import get_retriever
    retriever = get_retriever(DATASET_PATH)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        all_items = json.load(f)
    medqa_items = [d for d in all_items if d.get("dataset") == "medqa"][:limit]
    math_items  = [d for d in all_items if d.get("dataset") == "math"][:limit]
    pilot_data = medqa_items + math_items

    all_metrics: Dict[str, Any] = {}

    for model_cfg in models_to_run:
        for tier in tiers_to_run:
            key = f"{model_cfg['id']}_{tier}"
            dialogues = run_model_tier(model_cfg, pilot_data, tier)
            metrics = compute_metrics(dialogues, tier)
            all_metrics[key] = {
                "model_id":   model_cfg["model_id"],
                "model_name": model_cfg["name"],
                "tool_tier":  tier,
                "metrics":    metrics,
            }

            # ── Per-tier console summary ─────────────────────────────────────
            print_header(f"SUMMARY: {model_cfg['name']} | {tier}")
            ov = metrics.get("overall", {})
            print(f"  Pilot Items:            {ov.get('n', '?')}")
            print(f"  P1 Resistant:           {ov.get('p1_resistant_pct', '?')}%")
            print(f"  P1 Sycophantic:         {ov.get('p1_sycophantic_pct', '?')}%")
            print(f"  P2 Resistant:           {ov.get('p2_resistant_pct', '?')}%")
            print(f"  P2 Sycophantic:         {ov.get('p2_sycophantic_pct', '?')}%")
            print(f"  Cumulative Flip Rate:   {ov.get('cumulative_flip_pct', '?')}%")
            print(f"  Remained Resistant:     {ov.get('remained_resistant_pct', '?')}%")
            print(f"  Tool Utilization Rate:  {ov.get('tool_utilization_rate_pct', '?')}%")
            print(f"  Evidence Override Rate: {ov.get('evidence_override_rate_pct', '?')}%")
            print(f"  Tool-Assisted Resist.:  {ov.get('tool_assisted_resistance_pct', '?')}%")

    # ── Save comparative report ───────────────────────────────────────────────
    report = {
        "experiment": "Week 4 Pilot — Agentic Tool-Calling (T1 RAG + T2 Web Search)",
        "timestamp":  datetime.utcnow().isoformat(),
        "pilot_size": {"medqa": PILOT_PER_DOMAIN, "math": PILOT_PER_DOMAIN},
        "models":     [m["name"] for m in MODELS],
        "tiers":      TOOL_TIERS,
        "results":    all_metrics,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_header("PILOT EVALUATION COMPLETE")
    print(f"Full comparative report saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
