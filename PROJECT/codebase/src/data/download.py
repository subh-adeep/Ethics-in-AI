"""
Dataset Downloader and Candidate Pool Generator.
1. MedQA: Preserves the 300 existing evaluated candidates and expands to 1,400 questions.
2. MATH: Scans all configs across train & test splits and extracts ALL Level 1 & Level 2 strictly numerical problems (~1,944).
"""

import os
import json
import re
import random
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CANDIDATE_DIR = os.path.join(DATA_DIR, "candidate")
FINAL_DIR = os.path.join(DATA_DIR, "final")
LOGS_DIR = os.path.join(PROJECT_ROOT, "codebase", "results", "week2_logs")
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "model")

os.environ["HF_HOME"] = MODEL_CACHE_DIR

for d in [DATA_DIR, RAW_DIR, CANDIDATE_DIR, FINAL_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

from datasets import load_dataset
from src.data.prompts import is_strictly_numerical


def extract_last_boxed(solution: str) -> Optional[str]:
    """Extracts the exact content inside the final \\boxed{...} with balanced brace matching."""
    idx = solution.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    content = []
    for ch in solution[start:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(content).strip()
        content.append(ch)
    return None


def prepare_medqa_candidates(
    target_count: int = 1400,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Expands MedQA candidate pool to target_count (1,400),
    strictly preserving the first 300 candidates already evaluated.
    """
    print(f"\n[Data] Preparing MedQA dataset (target: {target_count} candidates)...", flush=True)
    existing_300_path = os.path.join(CANDIDATE_DIR, "candidate_medqa_300.json")
    
    candidates = []
    seen_questions = set()
    
    # 1. Load and preserve existing 300 questions
    if os.path.exists(existing_300_path):
        with open(existing_300_path, "r", encoding="utf-8") as f:
            existing_items = json.load(f)
            for it in existing_items:
                candidates.append(it)
                seen_questions.add(it["question"].strip())
        print(f"[Data] Loaded {len(candidates)} existing MedQA candidates from: {existing_300_path}", flush=True)

    # 2. Fetch new candidates from test and train splits
    for split_name in ["test", "train"]:
        if len(candidates) >= target_count:
            break
        print(f"[Data] Pulling additional MedQA questions from split '{split_name}'...", flush=True)
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split_name, cache_dir=RAW_DIR)
        
        pool = []
        for item in ds:
            q_text = item.get("question", "").strip()
            options = item.get("options", {})
            ans_idx = item.get("answer_idx", "").strip().upper()

            if not q_text or len(options) != 4 or ans_idx not in ("A", "B", "C", "D"):
                continue
            if ans_idx not in options or q_text in seen_questions:
                continue

            pool.append({
                "dataset": "medqa",
                "question": q_text,
                "options": options,
                "ground_truth": ans_idx,
                "ground_truth_text": options[ans_idx],
            })

        # Shuffle the available split items with fixed seed
        rng = random.Random(seed)
        rng.shuffle(pool)

        for item in pool:
            if len(candidates) >= target_count:
                break
            seen_questions.add(item["question"])
            item["id"] = f"medqa_{len(candidates) + 1:04d}"
            candidates.append(item)

    out_path = os.path.join(CANDIDATE_DIR, f"candidate_medqa_{len(candidates)}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    print(f"[Data] Successfully saved {len(candidates)} MedQA candidate questions to: {out_path}", flush=True)
    return candidates


def prepare_math_candidates_level1_2(
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Extracts ALL strictly numerical Level 1 and Level 2 problems from Hendrycks MATH
    across train and test splits (~1,944 questions).
    """
    print(f"\n[Data] Extracting ALL Level 1 & Level 2 strictly numerical MATH questions...", flush=True)
    configs = ["prealgebra", "algebra", "number_theory", "counting_and_probability", "geometry"]
    
    candidates = []
    seen_problems = set()

    for cfg in configs:
        for split_name in ["train", "test"]:
            try:
                ds = load_dataset("EleutherAI/hendrycks_math", cfg, split=split_name, cache_dir=RAW_DIR)
                for item in ds:
                    problem = item.get("problem", "").strip()
                    solution = item.get("solution", "").strip()
                    level_str = item.get("level", "").strip()

                    # Check for Level 1 or Level 2
                    lvl_match = re.search(r"\d+", level_str)
                    if not lvl_match or int(lvl_match.group(0)) not in (1, 2):
                        continue

                    if not problem or problem in seen_problems:
                        continue

                    boxed_ans = extract_last_boxed(solution)
                    if not boxed_ans:
                        continue

                    clean_ans = boxed_ans.replace("$", "").replace(",", "").strip()
                    if is_strictly_numerical(clean_ans):
                        seen_problems.add(problem)
                        candidates.append({
                            "id": f"math_l12_{len(candidates) + 1:04d}",
                            "dataset": "math",
                            "category": cfg,
                            "level": f"Level {lvl_match.group(0)}",
                            "question": problem,
                            "ground_truth": clean_ans,
                            "solution": solution,
                        })
            except Exception as e:
                print(f"[Data] Warning: Error loading MATH config '{cfg}' split '{split_name}': {e}", flush=True)

    print(f"[Data] Extracted {len(candidates)} total Level 1 & Level 2 numerical MATH questions.", flush=True)
    
    # Shuffle with seed for reproducible order
    rng = random.Random(seed)
    rng.shuffle(candidates)
    # Re-index nicely
    for idx, c in enumerate(candidates):
        c["id"] = f"math_l12_{idx + 1:04d}"

    out_path = os.path.join(CANDIDATE_DIR, "candidate_math_level1_2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    print(f"[Data] Saved {len(candidates)} MATH Level 1 & 2 candidate questions to: {out_path}", flush=True)
    return candidates


if __name__ == "__main__":
    medqa = prepare_medqa_candidates(target_count=1400)
    math_items = prepare_math_candidates_level1_2()
    print(f"\n[Done] Final Candidate Pools: MedQA ({len(medqa)}), MATH Level 1-2 ({len(math_items)})", flush=True)
