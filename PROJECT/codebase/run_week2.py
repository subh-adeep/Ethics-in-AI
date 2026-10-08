"""
Week 2 Execution Script:
1. Verifies candidate pools (MedQA 1,400 & MATH Level 1-2 ~1,944).
2. Sequentially evaluates Qwen 2.5 3B, LLaMA 3.2 3B, and Phi-4 Mini 3.8B on GTX 1650.
3. Completely frees VRAM between models.
4. Computes 3-model simultaneous intersection and downsamples to exactly 250 per dataset.
5. Generates deterministic false answers and P1/P2 pressure prompts.
6. Outputs full publication logs and final evaluation dataset.
"""

import os
import sys
import json
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data.download import prepare_medqa_candidates, prepare_math_candidates_level1_2
from src.benchmark.filter import run_week2_filter

CANDIDATE_DIR = os.path.join(BASE_DIR, "..", "data", "candidate")
MEDQA_CAND = os.path.join(CANDIDATE_DIR, "candidate_medqa_1400.json")
MATH_CAND = os.path.join(CANDIDATE_DIR, "candidate_math_level1_2.json")


def main():
    print("=" * 80)
    print(" ETHICS IN AI - WEEK 2 BASELINE FILTERING & DATASET PIPELINE")
    print("=" * 80)

    # Step 1: Ensure candidate files exist
    if not os.path.exists(MEDQA_CAND) or not os.path.exists(MATH_CAND):
        print("\n[Step 1] Preparing Candidate Pools (1,400 MedQA, All Level 1-2 Numerical MATH)...")
        if not os.path.exists(MEDQA_CAND):
            prepare_medqa_candidates(target_count=1400)
        if not os.path.exists(MATH_CAND):
            prepare_math_candidates_level1_2()
    else:
        print("\n[Step 1] Candidate Pools already prepared:")
        print(f"  - MedQA: {MEDQA_CAND}")
        print(f"  - MATH:  {MATH_CAND}")

    # Step 2: Execute sequential filtering
    print("\n[Step 2] Executing Baseline Filtering (P0 under T0) across all 3 models...")
    summary = run_week2_filter(target_count=250, random_seed=42)

    # Step 3: Print summary
    print("\n" + "=" * 80)
    print(" WEEK 2 FILTERING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nFinal Dataset Location: {summary['final_dataset_path']}")
    print(f"Retained Questions: MedQA={summary['final_dataset_retained']['medqa_retained']}, MATH={summary['final_dataset_retained']['math_retained']} (Total: {summary['final_dataset_retained']['total']})")
    print("\nIndividual Accuracies:")
    for model_name, acc in summary["individual_accuracies"].items():
        print(f"  - {model_name:18s}: MedQA={acc['medqa_correct']}/{summary['candidate_counts']['medqa']} | MATH={acc['math_correct']}/{summary['candidate_counts']['math_level1_2']}")
    print(f"\nSimultaneous 3-Model Intersection:")
    print(f"  - MedQA: {summary['simultaneous_intersection']['medqa_count']}")
    print(f"  - MATH:  {summary['simultaneous_intersection']['math_count']}")


if __name__ == "__main__":
    main()
