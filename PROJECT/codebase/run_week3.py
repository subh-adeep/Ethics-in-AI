"""
Week 3 Execution Pipeline: Multi-Turn Continuous-Context T0 Baseline Benchmark.

Evaluates 3 models sequentially on the 497-item intersection dataset (247 MedQA + 250 MATH):
1. Qwen 2.5 3B Instruct
2. LLaMA 3.2 3B Instruct
3. Phi-4 Mini 3.8B Instruct

Safety & Resource Safeguards:
- 100% Offline execution on GTX 1650 (4GB VRAM) with greedy decoding (temperature=0.0).
- Checkpointing after every 5 items with instant auto-resume if interrupted.
- GPU Health: Mandatory 10-minute cooldown break every 1 hour of continuous GPU execution.
- Complete VRAM unloading & garbage collection between models.
- Publication-quality metrics compilation into week3_t0_benchmark_report.json.
"""

import os
import sys
import json
import time
import gc
from datetime import datetime
from typing import Dict, Any, List
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.loader import (
    load_model_and_tokenizer,
    unload_model,
    get_gpu_memory_info,
)
from src.benchmark.conversation import run_multiturn_dialogue

DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "final", "final_filtered_evaluation_set_500.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "week3_logs")
REPORT_PATH = os.path.join(BASE_DIR, "results", "week3_t0_benchmark_report.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = [
    {
        "id": "qwen",
        "name": "Qwen 2.5 3B",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "output_file": os.path.join(RESULTS_DIR, "qwen_t0_results.json"),
    },
    {
        "id": "llama",
        "name": "LLaMA 3.2 3B",
        "model_id": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "output_file": os.path.join(RESULTS_DIR, "llama_t0_results.json"),
    },
    {
        "id": "phi4",
        "name": "Phi-4 Mini 3.8B",
        "model_id": "unsloth/Phi-4-mini-instruct-bnb-4bit",
        "output_file": os.path.join(RESULTS_DIR, "phi4_t0_results.json"),
    },
]


def gpu_cooldown_break(duration_seconds: int = 600, reason: str = "1 hour of continuous execution reached"):
    """
    Pauses execution to allow GTX 1650 silicon and fans to return to idle temperatures (~40°C).
    """
    print(f"\n{'='*75}", flush=True)
    print(f"[GPU Health Cooldown] Taking a {duration_seconds // 60}-minute thermal break ({reason})", flush=True)
    print(f"[GPU Health Cooldown] Flushing cache to allow GPU silicon to cool down...", flush=True)
    print(f"{'='*75}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    for remaining in range(duration_seconds, 0, -60):
        mins = remaining // 60
        print(f"  [GPU Cooldown] {mins} minute(s) remaining...", flush=True)
        time.sleep(min(60, remaining))

    print(f"[GPU Cooldown] Break complete! Resuming evaluation with fresh, cooled GPU.\n", flush=True)


def compute_model_metrics(dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes comprehensive sycophancy, resistance, and unforced error metrics."""
    total = len(dialogues)
    if total == 0:
        return {}

    medqa_items = [d for d in dialogues if d["dataset"] == "medqa"]
    math_items = [d for d in dialogues if d["dataset"] == "math"]

    def _calc_subset(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        count = len(items)
        if count == 0:
            return {}

        t1_correct = sum(1 for d in items if d["turn1_p0"]["is_correct"])
        p1_resistant = sum(1 for d in items if d["turn2_p1"]["outcome"] == "resistant")
        p1_flipped = sum(1 for d in items if d["turn2_p1"]["outcome"] == "sycophantic")
        p1_unforced = sum(1 for d in items if d["turn2_p1"]["outcome"] == "unforced_error")

        p2_resistant = sum(1 for d in items if d["turn3_p2"]["outcome"] == "resistant")
        p2_flipped = sum(1 for d in items if d["turn3_p2"]["outcome"] == "sycophantic")
        p2_unforced = sum(1 for d in items if d["turn3_p2"]["outcome"] == "unforced_error")

        cumulative_flipped = sum(1 for d in items if d["trajectory"]["flipped_at_p1"] or d["trajectory"]["flipped_at_p2"])
        consistently_resistant = sum(1 for d in items if d["trajectory"]["remained_resistant"])

        return {
            "total_questions": count,
            "turn1_baseline_accuracy": round((t1_correct / count) * 100, 2),
            "p1_suggestive_pressure": {
                "resistance_count": p1_resistant,
                "resistance_rate_pct": round((p1_resistant / count) * 100, 2),
                "flip_count": p1_flipped,
                "flip_rate_pct": round((p1_flipped / count) * 100, 2),
                "unforced_error_count": p1_unforced,
                "unforced_error_rate_pct": round((p1_unforced / count) * 100, 2),
            },
            "p2_authority_pressure": {
                "resistance_count": p2_resistant,
                "resistance_rate_pct": round((p2_resistant / count) * 100, 2),
                "flip_count": p2_flipped,
                "flip_rate_pct": round((p2_flipped / count) * 100, 2),
                "unforced_error_count": p2_unforced,
                "unforced_error_rate_pct": round((p2_unforced / count) * 100, 2),
            },
            "cumulative_trajectory": {
                "cumulative_flip_count": cumulative_flipped,
                "cumulative_flip_rate_pct": round((cumulative_flipped / count) * 100, 2),
                "consistently_resistant_count": consistently_resistant,
                "consistently_resistant_rate_pct": round((consistently_resistant / count) * 100, 2),
            }
        }

    return {
        "overall": _calc_subset(dialogues),
        "by_dataset": {
            "medqa": _calc_subset(medqa_items),
            "math": _calc_subset(math_items),
        }
    }


def evaluate_single_model(model_info: Dict[str, Any], dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Runs continuous multi-turn evaluation for a single model with checkpointing and cooldowns."""
    model_name = model_info["name"]
    model_id = model_info["model_id"]
    output_file = model_info["output_file"]

    print("\n" + "=" * 80)
    print(f" STARTING EVALUATION: {model_name.upper()} ({model_id})")
    print(f" Target questions: {len(dataset)} | Output file: {output_file}")
    print("=" * 80)

    # 1. Load existing checkpoints
    results_map: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                saved_results = json.load(f)
                if isinstance(saved_results, dict) and "dialogues" in saved_results:
                    for d in saved_results["dialogues"]:
                        results_map[d["id"]] = d
                elif isinstance(saved_results, list):
                    for d in saved_results:
                        results_map[d["id"]] = d
            print(f"[Checkpoint] Resuming: Found {len(results_map)}/{len(dataset)} already evaluated questions.")
        except Exception as e:
            print(f"[Checkpoint] Warning loading checkpoint: {e}")

    # Check if already 100% completed
    if len(results_map) >= len(dataset):
        print(f"[Checkpoint] Model {model_name} is already completely evaluated! Skipping to next model.")
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # 2. Load model into VRAM
    model, tokenizer, load_sec = load_model_and_tokenizer(model_id, load_in_4bit=True)
    device = next(model.parameters()).device
    mem = get_gpu_memory_info()
    print(f"[VRAM] {model_name} loaded in {load_sec:.1f}s | VRAM: {mem['allocated_mb']} MB (Peak: {mem['max_allocated_mb']} MB)")

    last_cooldown_time = time.time()
    model_start_time = time.time()

    try:
        for idx, item in enumerate(dataset, start=1):
            item_id = item["id"]

            if item_id in results_map:
                continue

            # Check if 1 hour of continuous GPU execution elapsed
            if time.time() - last_cooldown_time >= 3600:
                gpu_cooldown_break(duration_seconds=600, reason="1 hour of continuous GPU execution")
                last_cooldown_time = time.time()

            t_start = time.time()
            dialogue_res = run_multiturn_dialogue(model, tokenizer, item, device=device, max_new_tokens=96)
            elapsed = time.time() - t_start

            results_map[item_id] = dialogue_res

            # Print progress log
            p1_outcome = dialogue_res["turn2_p1"]["outcome"].upper()
            p2_outcome = dialogue_res["turn3_p2"]["outcome"].upper()
            print(f"[{idx:3d}/{len(dataset)}] ID: {item_id:14s} ({item['dataset']:5s}) | P1: {p1_outcome:14s} | P2: {p2_outcome:14s} ({elapsed:.1f}s)", flush=True)

            # Checkpoint every 5 items
            if idx % 5 == 0 or idx == len(dataset):
                temp_data = {
                    "model_id": model_id,
                    "model_name": model_name,
                    "last_updated": datetime.now().isoformat(),
                    "total_evaluated": len(results_map),
                    "dialogues": list(results_map.values()),
                }
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(temp_data, f, indent=2)

    finally:
        unload_model(model, tokenizer)
        mem_after = get_gpu_memory_info()
        print(f"\n[Cleanup] Unloaded {model_name} from GPU | VRAM: {mem_after['allocated_mb']} MB")

    # 3. Final metrics compilation for this model
    all_dialogues = list(results_map.values())
    metrics = compute_model_metrics(all_dialogues)

    final_model_output = {
        "model_id": model_id,
        "model_name": model_name,
        "completed_at": datetime.now().isoformat(),
        "total_duration_sec": round(time.time() - model_start_time, 2),
        "total_evaluated": len(all_dialogues),
        "metrics": metrics,
        "dialogues": all_dialogues,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_model_output, f, indent=2)

    print(f"\n[Finished] {model_name} evaluated completely. Results saved to {output_file}")
    return final_model_output


def run_week3_benchmark():
    """Main benchmark runner for all 3 models."""
    print("=" * 80)
    print(" ETHICS IN AI - WEEK 3 MULTI-TURN CONTINUOUS EVALUATION (T0 BASELINE)")
    print("=" * 80)

    # 1. Load full evaluation dataset (497 items)
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)

    medqa_count = sum(1 for d in full_dataset if d.get("dataset") == "medqa")
    math_count = sum(1 for d in full_dataset if d.get("dataset") == "math")
    print(f"[Dataset] Loaded evaluation set: {len(full_dataset)} total questions (MedQA: {medqa_count}, MATH: {math_count})")

    benchmark_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(full_dataset),
        "dataset_composition": {"medqa": medqa_count, "math": math_count},
        "models_evaluated": {},
    }

    total_start = time.time()

    # 2. Sequential evaluation across the 3 models
    for model_info in MODELS:
        model_result = evaluate_single_model(model_info, full_dataset)
        benchmark_summary["models_evaluated"][model_info["id"]] = {
            "model_name": model_info["name"],
            "model_id": model_info["model_id"],
            "total_duration_sec": model_result.get("total_duration_sec", 0),
            "metrics": model_result.get("metrics", {}),
        }
        # Take a 2-minute GPU cool-down pause between switching models
        time.sleep(120)

    # 3. Save publication benchmark report
    benchmark_summary["grand_total_duration_sec"] = round(time.time() - total_start, 2)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print("\n" + "=" * 80)
    print(" ALL 3 MODELS EVALUATED SUCCESSFULLY (WEEK 3 T0 BASELINE COMPLETE)")
    print("=" * 80)
    print(f"Grand Summary Report: {REPORT_PATH}")
    print("\nModel Comparison Summary (Cumulative Flip Rate by Turn 3):")
    for mid, mdata in benchmark_summary["models_evaluated"].items():
        overall = mdata.get("metrics", {}).get("overall", {})
        cum = overall.get("cumulative_trajectory", {})
        print(f"  - {mdata['model_name']:18s}: Cumulative Flip Rate = {cum.get('cumulative_flip_rate_pct', 'N/A')}% | Resistant = {cum.get('consistently_resistant_rate_pct', 'N/A')}%")


if __name__ == "__main__":
    run_week3_benchmark()
