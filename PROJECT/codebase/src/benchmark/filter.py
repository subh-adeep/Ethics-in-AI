"""
Sequential Baseline Filtering Pipeline for Week 2 (Scaled 250-Overlap).
Evaluates Qwen, LLaMA, and Phi-4 one at a time.
1. MedQA: Evaluates 1,400 questions incrementally (preserving 300 already evaluated).
2. MATH: Evaluates all Level 1 & Level 2 strictly numerical questions (~1,944).
Saves checkpoints every 50 questions and completely unloads VRAM between models.
Computes 3-model simultaneous intersection and downsamples to exactly 250 per dataset.
Generates deterministic false answers & P1/P2 pressure prompts.
"""

import os
import gc
import json
import re
import random
import time
from typing import List, Dict, Any, Optional, Tuple
import torch

from src.models.loader import load_model_and_tokenizer, unload_model
from src.data.prompts import (
    format_p0_prompt,
    format_p1_prompt,
    format_p2_prompt,
    derive_math_false_answer,
    select_medqa_false_answer,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CANDIDATE_DIR = os.path.join(PROJECT_ROOT, "data", "candidate")
FINAL_DIR = os.path.join(PROJECT_ROOT, "data", "final")
LOGS_DIR = os.path.join(PROJECT_ROOT, "codebase", "results", "week2_logs")

os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

MODELS = [
    {"id": "qwen", "name": "Qwen 2.5 3B", "model_id": "Qwen/Qwen2.5-3B-Instruct"},
    {"id": "llama", "name": "LLaMA 3.2 3B", "model_id": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"},
    {"id": "phi4", "name": "Phi-4 Mini 3.8B", "model_id": "unsloth/Phi-4-mini-instruct-bnb-4bit"},
]
def cooldown_break(duration_seconds: int = 600, reason: str = ""):
    """Takes a timed cooldown break to let GPU temperature drop back to idle (~40C) and flush buffers."""
    print(f"\n{'='*70}\n[GPU Cooldown] Taking a {duration_seconds // 60}-minute break ({reason})\n[GPU Cooldown] Allowing GPU fans and silicon to return to idle temperatures (~40C)...\n{'='*70}", flush=True)
    for remaining in range(duration_seconds, 0, -60):
        print(f"  [GPU Cooldown] {remaining // 60} minute(s) remaining...", flush=True)
        time.sleep(min(60, remaining))
    print("[GPU Cooldown] Break complete! Resuming evaluation with fresh, cooled GPU.\n", flush=True)



def extract_medqa_answer(response_text: str) -> Optional[str]:
    """Extracts predicted option letter (A, B, C, D) from model output."""
    clean = response_text.strip()
    match = re.search(r"^(?:(?:the\s+)?(?:correct\s+)?answer\s+is\s*[:\*\#]*\s*|\*+|\#+)?\(?([A-D])\)?(?:\s|[:\.\)\*\#]|$)", clean, re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    match = re.search(r"\b(?:option|answer|choice)\s*[:\*\#]*\s*\(?([A-D])\)?\b", clean, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"\b([A-D])\b", clean[:100])
    if match:
        return match.group(1).upper()

    return None


def extract_math_answer(response_text: str) -> Optional[str]:
    """Extracts predicted numerical answer from model output."""
    clean = response_text.strip()

    # Priority 1: Final answer: [number]
    match = re.search(r"(?:final\s+answer|the\s+answer\s+is)\s*[:\*\#]*\s*\$?(-?\d+(?:\.\d+)?)\$?", clean, re.IGNORECASE)
    if match:
        return match.group(1)

    # Priority 2: \\boxed{...}
    idx = clean.rfind(r"\boxed{")
    if idx != -1:
        start = idx + len(r"\boxed{")
        depth = 1
        chars = []
        for ch in clean[start:]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            chars.append(ch)
        val = "".join(chars).replace("$", "").replace(",", "").strip()
        num_match = re.search(r"^-?\d+(?:\.\d+)?$", val)
        if num_match:
            return num_match.group(0)

    # Priority 3: First line pure number
    first_line = clean.split("\n")[0]
    num_first = re.search(r"^-?\d+(?:\.\d+)?$", first_line.strip())
    if num_first:
        return num_first.group(0)

    # Priority 4: Look for last number in last line
    last_line = clean.split("\n")[-1]
    numbers = re.findall(r"-?\d+(?:\.\d+)?", last_line)
    if numbers:
        return numbers[-1]

    return None


def evaluate_single_model(
    model_cfg: Dict[str, str],
    medqa_items: List[Dict[str, Any]],
    math_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluates one model sequentially.
    - MedQA: Incremental (skips items already in log).
    - MATH: Evaluates all Level 1 & 2 items with checkpoints every 50 questions.
    - Completely unloads model from VRAM upon completion.
    """
    model_key = model_cfg["id"]
    model_id = model_cfg["model_id"]
    log_file = os.path.join(LOGS_DIR, f"{model_key}_eval.json")

    results = {
        "model_id": model_id,
        "model_name": model_cfg["name"],
        "medqa_eval": {},
        "math_eval": {},
        "medqa_correct_count": 0,
        "math_correct_count": 0,
        "total_eval_time_s": 0.0,
    }

    # Load existing log to resume / reuse completed MedQA evaluations
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                results["medqa_eval"] = prev_data.get("medqa_eval", {})
                results["math_eval"] = prev_data.get("math_eval", {})
                print(f"[Filter] Loaded existing log for {model_cfg['name']}: {len(results['medqa_eval'])} MedQA, {len(results['math_eval'])} MATH already recorded.", flush=True)
        except Exception as e:
            print(f"[Filter] Warning reading existing log: {e}", flush=True)

    # Filter items that still need evaluation
    needed_medqa = [it for it in medqa_items if it["id"] not in results["medqa_eval"]]
    needed_math = [it for it in math_items if it["id"] not in results["math_eval"]]

    print(f"\n{'='*70}\n[Filter] {model_cfg['name']}: To Evaluate -> {len(needed_medqa)} MedQA items, {len(needed_math)} MATH Level 1-2 items\n{'='*70}", flush=True)

    if not needed_medqa and not needed_math:
        print(f"[Filter] All items already evaluated for {model_cfg['name']}! Skipping GPU load.", flush=True)
        results["medqa_correct_count"] = sum(1 for v in results["medqa_eval"].values() if v.get("is_correct"))
        results["math_correct_count"] = sum(1 for v in results["math_eval"].values() if v.get("is_correct"))
        return results

    # Load Model on GPU
    model, tokenizer, load_dur = load_model_and_tokenizer(model_id, device="cuda", load_in_4bit=True)
    start_eval_t = time.time()
    last_cooldown_t = time.time()

    # --- 1. Evaluate MedQA ---
    if needed_medqa:
        print(f"\n[Filter] Running {len(needed_medqa)} MedQA questions on {model_cfg['name']}...", flush=True)
        for idx, item in enumerate(needed_medqa):
            prompt = format_p0_prompt("medqa", item)
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                formatted_input = prompt

            inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=60,
                    temperature=0.01,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

            pred = extract_medqa_answer(response)
            gt = item["ground_truth"].strip().upper()
            is_correct = (pred == gt)

            results["medqa_eval"][item["id"]] = {
                "predicted": pred,
                "ground_truth": gt,
                "is_correct": is_correct,
                "response_snippet": response[:150],
            }

            # Periodic log update and save checkpoint
            if (idx + 1) % 20 == 0 or (idx + 1) == len(needed_medqa):
                cur_correct = sum(1 for v in results["medqa_eval"].values() if v.get("is_correct"))
                total_done = len(results["medqa_eval"])
                print(f"  [MedQA] {total_done}/{len(medqa_items)} done | Correct: {cur_correct} ({cur_correct/total_done*100:.1f}%)", flush=True)

            if (idx + 1) % 50 == 0:
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

            # Hourly 10-minute cooldown check
            if time.time() - last_cooldown_t >= 3600:
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                cooldown_break(600, f"Hourly GPU Protection: 1 hour elapsed ({model_cfg['name']})")
                last_cooldown_t = time.time()

        # Save MedQA completion
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # 10-minute cooldown break between MedQA and MATH datasets
        if needed_math:
            cooldown_break(600, f"{model_cfg['name']}: MedQA completed -> Cooling down before MATH Level 1-2")

    # --- 2. Evaluate MATH Level 1 & 2 ---
    if needed_math:
        print(f"\n[Filter] Running {len(needed_math)} MATH Level 1 & 2 questions on {model_cfg['name']}...", flush=True)
        for idx, item in enumerate(needed_math):
            prompt = format_p0_prompt("math", item)
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                formatted_input = prompt

            inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.01,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

            pred = extract_math_answer(response)
            gt = item["ground_truth"].strip()
            is_correct = False
            if pred is not None:
                try:
                    is_correct = (float(pred) == float(gt))
                except ValueError:
                    is_correct = (pred == gt)

            results["math_eval"][item["id"]] = {
                "predicted": pred,
                "ground_truth": gt,
                "is_correct": is_correct,
                "response_snippet": response[:150],
            }

            if (idx + 1) % 20 == 0 or (idx + 1) == len(needed_math):
                cur_correct = sum(1 for v in results["math_eval"].values() if v.get("is_correct"))
                total_done = len(results["math_eval"])
                print(f"  [MATH L1-2] {total_done}/{len(math_items)} done | Correct: {cur_correct} ({cur_correct/total_done*100:.1f}%)", flush=True)

            if (idx + 1) % 50 == 0:
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

            # Hourly 10-minute cooldown check
            if time.time() - last_cooldown_t >= 3600:
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                cooldown_break(600, f"Hourly GPU Protection: 1 hour elapsed ({model_cfg['name']})")
                last_cooldown_t = time.time()

    results["medqa_correct_count"] = sum(1 for v in results["medqa_eval"].values() if v.get("is_correct"))
    results["math_correct_count"] = sum(1 for v in results["math_eval"].values() if v.get("is_correct"))
    results["total_eval_time_s"] = round(time.time() - start_eval_t, 2)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Filter] Saved complete evaluation log for {model_cfg['name']} to: {log_file}", flush=True)

    # Cleanly offload VRAM
    print(f"[Filter] Unloading {model_cfg['name']} and clearing VRAM...", flush=True)
    unload_model(model, tokenizer)
    cooldown_break(600, f"Model switch: {model_cfg['name']} completed -> Cooling down before next model")
    return results


def run_week2_filter(
    target_count: int = 250,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Orchestrates the Week 2 baseline filtering across all 3 models."""
    medqa_cand_file = os.path.join(CANDIDATE_DIR, "candidate_medqa_1400.json")
    math_cand_file = os.path.join(CANDIDATE_DIR, "candidate_math_level1_2.json")

    if not os.path.exists(medqa_cand_file) or not os.path.exists(math_cand_file):
        raise FileNotFoundError("Candidate files missing. Run src.data.download first.")

    with open(medqa_cand_file, "r", encoding="utf-8") as f:
        medqa_items = json.load(f)
    with open(math_cand_file, "r", encoding="utf-8") as f:
        math_items = json.load(f)

    print(f"\n[Week 2 Filter] Loaded {len(medqa_items)} MedQA and {len(math_items)} MATH Level 1-2 candidates.", flush=True)

    # 1. Sequentially evaluate each model
    model_evaluations = {}
    for m in MODELS:
        eval_res = evaluate_single_model(m, medqa_items, math_items)
        model_evaluations[m["id"]] = eval_res

    # 2. Compute Intersection for MedQA
    print(f"\n{'='*70}\n[Week 2 Intersection] Calculating Questions Correct Across ALL 3 Models\n{'='*70}", flush=True)
    medqa_intersection = []
    for item in medqa_items:
        q_id = item["id"]
        all_correct = all(
            model_evaluations[m["id"]]["medqa_eval"].get(q_id, {}).get("is_correct", False)
            for m in MODELS
        )
        if all_correct:
            medqa_intersection.append(item)

    # 3. Compute Intersection for MATH
    math_intersection = []
    for item in math_items:
        q_id = item["id"]
        all_correct = all(
            model_evaluations[m["id"]]["math_eval"].get(q_id, {}).get("is_correct", False)
            for m in MODELS
        )
        if all_correct:
            math_intersection.append(item)

    print(f"\n[Statistics for Research Paper]:", flush=True)
    for m in MODELS:
        m_id = m["id"]
        m_res = model_evaluations[m_id]
        print(f"  - {m['name']}: MedQA {m_res['medqa_correct_count']}/{len(medqa_items)} ({m_res['medqa_correct_count']/len(medqa_items)*100:.1f}%) | MATH {m_res['math_correct_count']}/{len(math_items)} ({m_res['math_correct_count']/len(math_items)*100:.1f}%)", flush=True)

    print(f"\n  [*] Simultaneous 3-Model Intersection (MedQA): {len(medqa_intersection)} questions", flush=True)
    print(f"  [*] Simultaneous 3-Model Intersection (MATH):  {len(math_intersection)} questions", flush=True)

    # 4. Downsample to target_count (250) if available
    rng = random.Random(random_seed)
    
    if len(medqa_intersection) >= target_count:
        final_medqa = rng.sample(medqa_intersection, target_count)
    else:
        final_medqa = list(medqa_intersection)
        
    if len(math_intersection) >= target_count:
        final_math = rng.sample(math_intersection, target_count)
    else:
        final_math = list(math_intersection)

    # 5. Generate False Answers & Pressure Prompts (P1 & P2)
    print(f"\n[Prompts] Generating deterministic false answers and P1/P2 prompts...", flush=True)
    final_eval_dataset = []

    for item in final_medqa:
        false_key, false_text = select_medqa_false_answer(item["ground_truth"], item["options"], seed=random_seed)
        enriched = dict(item)
        enriched["false_answer_key"] = false_key
        enriched["false_answer_text"] = false_text
        enriched["p0_prompt"] = format_p0_prompt("medqa", enriched)
        enriched["p1_prompt"] = format_p1_prompt("medqa", enriched)
        enriched["p2_prompt"] = format_p2_prompt("medqa", enriched)
        final_eval_dataset.append(enriched)

    for item in final_math:
        false_ans = derive_math_false_answer(item["ground_truth"])
        enriched = dict(item)
        enriched["false_answer"] = false_ans
        enriched["p0_prompt"] = format_p0_prompt("math", enriched)
        enriched["p1_prompt"] = format_p1_prompt("math", enriched)
        enriched["p2_prompt"] = format_p2_prompt("math", enriched)
        final_eval_dataset.append(enriched)

    # 6. Save Final Dataset & Summary Report
    final_dataset_path = os.path.join(FINAL_DIR, "final_filtered_evaluation_set_500.json")
    with open(final_dataset_path, "w", encoding="utf-8") as f:
        json.dump(final_eval_dataset, f, indent=2, ensure_ascii=False)

    summary = {
        "candidate_counts": {"medqa": len(medqa_items), "math_level1_2": len(math_items)},
        "individual_accuracies": {
            m["name"]: {
                "medqa_correct": model_evaluations[m["id"]]["medqa_correct_count"],
                "math_correct": model_evaluations[m["id"]]["math_correct_count"],
            }
            for m in MODELS
        },
        "simultaneous_intersection": {
            "medqa_count": len(medqa_intersection),
            "math_count": len(math_intersection),
        },
        "final_dataset_retained": {
            "medqa_retained": len(final_medqa),
            "math_retained": len(final_math),
            "total": len(final_eval_dataset),
        },
        "final_dataset_path": final_dataset_path,
    }

    summary_file = os.path.join(LOGS_DIR, "week2_filtering_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Complete] Final filtered dataset saved to: {final_dataset_path}", flush=True)
    print(f"[Complete] Full statistical summary saved to: {summary_file}", flush=True)
    return summary


if __name__ == "__main__":
    run_week2_filter(target_count=250)
