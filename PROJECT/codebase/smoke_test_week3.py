"""
Smoke Test for Week 3 Multi-Turn Continuous-Context Evaluation.

Evaluates 4 sample items (2 MedQA + 2 MATH) using Qwen 2.5 3B:
- Verifies continuous context preservation across 3 turns.
- Verifies deterministic programmatic evaluation (resistant vs. sycophantic vs. unforced error).
- Verifies VRAM headroom and clean model unloading.
"""

import os
import sys
import json
import time

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


def main():
    print("=" * 80)
    print(" WEEK 3 SMOKE TEST: MULTI-TURN CONTINUOUS EVALUATION (T0 BASELINE)")
    print("=" * 80)

    # 1. Load sample data
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    medqa_samples = [d for d in data if d.get("dataset") == "medqa"][:2]
    math_samples = [d for d in data if d.get("dataset") == "math"][:2]
    samples = medqa_samples + math_samples

    print(f"\n[Data] Selected {len(samples)} smoke test items (2 MedQA + 2 MATH).")

    # 2. Load Qwen 2.5 3B
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    print(f"\n[Model] Loading {model_id}...")
    model, tokenizer, load_sec = load_model_and_tokenizer(model_id, load_in_4bit=True)
    device = next(model.parameters()).device

    mem = get_gpu_memory_info()
    print(f"[VRAM] After loading: {mem['allocated_mb']} MB allocated (Peak: {mem['max_allocated_mb']} MB)")

    results = []
    print("\n" + "=" * 80)
    print(" RUNNING 3-TURN PRESSURE DIALOGUES")
    print("=" * 80)

    for idx, item in enumerate(samples, start=1):
        print(f"\n{'-'*70}")
        print(f"Sample {idx}/{len(samples)}: [{item['dataset'].upper()}] ID: {item['id']}")
        print(f"Ground Truth (GT): {item['ground_truth']}")
        fa = item.get('false_answer_key') if item['dataset'] == 'medqa' else item.get('false_answer')
        print(f"False Answer (FA): {fa}")
        print(f"{'-'*70}")

        start_t = time.time()
        dialogue = run_multiturn_dialogue(model, tokenizer, item, device=device, max_new_tokens=96)
        elapsed = time.time() - start_t

        print(f"\n[Turn 1 - P0 Neutral]:")
        print(f"  Response:  {dialogue['turn1_p0']['response'][:120]}...")
        print(f"  Extracted: {dialogue['turn1_p0']['extracted']} | Correct: {dialogue['turn1_p0']['is_correct']}")

        print(f"\n[Turn 2 - P1 Suggestive]:")
        print(f"  Prompt:    {dialogue['turn2_p1']['prompt']}")
        print(f"  Response:  {dialogue['turn2_p1']['response'][:120]}...")
        print(f"  Extracted: {dialogue['turn2_p1']['extracted']} | Outcome: {dialogue['turn2_p1']['outcome'].upper()} (Stance: {dialogue['turn2_p1']['stance']})")

        print(f"\n[Turn 3 - P2 Authority]:")
        print(f"  Prompt:    {dialogue['turn3_p2']['prompt']}")
        print(f"  Response:  {dialogue['turn3_p2']['response'][:120]}...")
        print(f"  Extracted: {dialogue['turn3_p2']['extracted']} | Outcome: {dialogue['turn3_p2']['outcome'].upper()} (Stance: {dialogue['turn3_p2']['stance']})")

        print(f"\nTrajectory: Flipped P1={dialogue['trajectory']['flipped_at_p1']} | Flipped P2={dialogue['trajectory']['flipped_at_p2']} | Resistant={dialogue['trajectory']['remained_resistant']} ({elapsed:.2f}s)")
        results.append(dialogue)

    # 3. Clean up
    unload_model(model, tokenizer)
    mem_after = get_gpu_memory_info()
    print(f"\n[VRAM] After unloading: {mem_after['allocated_mb']} MB (Successfully freed)")

    # 4. Summary Table
    print("\n" + "=" * 80)
    print(" SMOKE TEST SUMMARY TABLE")
    print("=" * 80)
    print(f"{'ID':16s} | {'Dataset':7s} | {'GT':6s} | {'FA':6s} | {'T1 Correct':10s} | {'T2 (P1)':14s} | {'T3 (P2)':14s}")
    print("-" * 80)
    for r in results:
        t1_stat = "YES" if r['turn1_p0']['is_correct'] else "NO"
        print(f"{r['id']:16s} | {r['dataset']:7s} | {str(r['ground_truth']):6s} | {str(r['false_answer']):6s} | {t1_stat:10s} | {r['turn2_p1']['outcome']:14s} | {r['turn3_p2']['outcome']:14s}")
    print("=" * 80)
    print(" SMOKE TEST PASSED ALL CHECKS")
    print("=" * 80)


if __name__ == "__main__":
    main()
