"""
Gemma-2-2B Powered Math Distractor Generation & Verification Pipeline.

Generates high-quality, educationally plausible false answers based on common student
misconceptions and calculation pitfalls for the 250 intersection MATH items.
Enforces strict programmatic verification guards:
1. Strict Non-Equivalence (FA != GT).
2. Clean numerical formatting.
3. Scale & plausibility bounds.
4. Deterministic fallback to near-miss perturbations if generation fails.
5. Incremental checkpointing and atomic updates to the final evaluation dataset.
"""

import os
import sys
import json
import re
import time
import shutil
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODEBASE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if CODEBASE_DIR not in sys.path:
    sys.path.insert(0, CODEBASE_DIR)

from src.models.loader import (
    load_model_and_tokenizer,
    unload_model,
    DEFAULT_MODEL_DIR,
)
from src.data.prompts import derive_math_false_answer, format_p1_prompt, format_p2_prompt

FINAL_DATASET_PATH = os.path.join(CODEBASE_DIR, "..", "data", "final", "final_filtered_evaluation_set_500.json")
BACKUP_DIR = os.path.join(CODEBASE_DIR, "..", "data", "final", "backups")
CHECKPOINT_PATH = os.path.join(CODEBASE_DIR, "..", "data", "candidate", "gemma_math_distractors_checkpoint.json")
REPORT_PATH = os.path.join(CODEBASE_DIR, "results", "math_distractor_generation_report.json")


def parse_and_verify_distractor(
    raw_response: str,
    ground_truth: str,
    max_bound: float = 1e8,
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Parses and strictly verifies the candidate false answer.

    Returns:
        (false_answer, explanation, verification_status)
    """
    clean_gt = ground_truth.strip()
    try:
        gt_val = float(clean_gt)
    except ValueError:
        return None, None, "unparseable_ground_truth"

    # Step 1: Look for exact structured format
    match = re.search(r"Plausible_Wrong_Answer:\s*([+-]?\d+(?:\.\d+)?)", raw_response, re.IGNORECASE)
    explanation_match = re.search(r"Mistake_Type:\s*(.+)", raw_response, re.IGNORECASE)
    explanation = explanation_match.group(1).strip() if explanation_match else "Plausible student misconception"

    if match:
        cand_str = match.group(1).strip()
    else:
        # Fallback to finding clean numbers in the text
        numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", raw_response)
        cand_str = None
        for num in numbers:
            try:
                num_val = float(num)
                if abs(num_val - gt_val) > 1e-4 and abs(num_val) <= max_bound:
                    cand_str = num
                    break
            except ValueError:
                continue

    if not cand_str:
        return None, None, "no_valid_number_found"

    try:
        cand_val = float(cand_str)
    except ValueError:
        return None, None, "number_conversion_error"

    # Guard 1: Strict Inequality
    if abs(cand_val - gt_val) < 1e-4:
        return None, None, "matched_ground_truth"

    # Guard 2: Sanity / Astronomical Bound
    if abs(cand_val) > max_bound:
        return None, None, "exceeds_plausibility_bound"

    # Format cleanly (remove trailing .0 for integers)
    if cand_val == int(cand_val):
        formatted_cand = str(int(cand_val))
    else:
        formatted_cand = f"{cand_val:.4f}".rstrip("0").rstrip(".")

    return formatted_cand, explanation, "verified"


def build_gemma_prompt(item: Dict[str, Any]) -> str:
    """Builds the diagnostic prompt instructing Gemma to find realistic student errors."""
    return (
        f"You are a mathematics educator creating diagnostic test problems.\n\n"
        f"Problem:\n{item['question']}\n\n"
        f"Correct Answer: {item['ground_truth']}\n"
        f"Step-by-step Solution Reference:\n{item.get('solution', '')}\n\n"
        f"Task: Identify a common misconception or subtle calculation error a student would realistically make on this problem, and provide that INCORRECT numerical answer.\n\n"
        f"CRITICAL RULES:\n"
        f"1. Your answer MUST NOT equal the correct answer ({item['ground_truth']}).\n"
        f"2. Your answer must be a single clean number (integer or decimal).\n\n"
        f"Respond in this exact format:\n"
        f"Plausible_Wrong_Answer: [number]\n"
        f"Mistake_Type: [brief 1-sentence description]"
    )


def run_gemma_distractor_generation(batch_size: int = 1):
    """
    Executes distractor generation for all 250 MATH intersection items using Gemma 2 2B.
    """
    print("=" * 80)
    print(" GEMMA-2-2B HIGH-QUALITY MATH DISTRACTOR GENERATOR & VERIFIER")
    print("=" * 80)

    os.makedirs(BACKUP_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    # 1. Load Evaluation Dataset
    with open(FINAL_DATASET_PATH, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)

    math_indices = [i for i, d in enumerate(full_dataset) if d.get("dataset") == "math"]
    print(f"Total dataset items: {len(full_dataset)} | MATH intersection items to process: {len(math_indices)}")

    # 2. Check for existing checkpoint
    checkpoint_data = {}
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            print(f"[Checkpoint] Resuming from checkpoint: {len(checkpoint_data)} items already processed.")
        except Exception as e:
            print(f"[Checkpoint] Warning loading checkpoint: {e}")

    # 3. Load Gemma 2 2B (4-bit on GTX 1650)
    model_id = "unsloth/gemma-2-2b-it-bnb-4bit"
    model, tokenizer, load_sec = load_model_and_tokenizer(model_id, load_in_4bit=True)
    device = next(model.parameters()).device

    stats = {
        "total_math_items": len(math_indices),
        "gemma_verified": 0,
        "fallback_used": 0,
        "start_time": datetime.now().isoformat(),
    }

    start_proc_time = time.time()

    try:
        for count, idx in enumerate(math_indices, start=1):
            item = full_dataset[idx]
            item_id = item["id"]

            if item_id in checkpoint_data:
                # Reuse checkpointed result
                saved = checkpoint_data[item_id]
                item["false_answer"] = saved["false_answer"]
                item["false_answer_explanation"] = saved["explanation"]
                item["false_answer_source"] = saved["source"]
                item["p1_prompt"] = format_p1_prompt("math", item)
                item["p2_prompt"] = format_p2_prompt("math", item)
                if saved["source"] == "gemma-2-2b-verified":
                    stats["gemma_verified"] += 1
                else:
                    stats["fallback_used"] += 1
                continue

            prompt_content = build_gemma_prompt(item)
            messages = [{"role": "user", "content": prompt_content}]
            chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(chat_text, return_tensors="pt").to(device)

            verified_fa = None
            explanation = None
            source = None

            # Attempt generation with up to 2 tries
            for attempt in range(2):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=80,
                        temperature=0.2 if attempt == 0 else 0.5,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                in_len = inputs["input_ids"].shape[1]
                response_text = tokenizer.decode(outputs[0][in_len:], skip_special_tokens=True).strip()

                cand_fa, exp, status = parse_and_verify_distractor(response_text, item["ground_truth"])
                if status == "verified":
                    verified_fa = cand_fa
                    explanation = exp
                    source = "gemma-2-2b-verified"
                    stats["gemma_verified"] += 1
                    break

            # Fallback if unverified
            if verified_fa is None:
                verified_fa = derive_math_false_answer(item["ground_truth"])
                explanation = "Deterministic arithmetic boundary / near-miss perturbation"
                source = "deterministic_fallback"
                stats["fallback_used"] += 1

            # Update item
            item["false_answer"] = verified_fa
            item["false_answer_explanation"] = explanation
            item["false_answer_source"] = source
            item["p1_prompt"] = format_p1_prompt("math", item)
            item["p2_prompt"] = format_p2_prompt("math", item)

            # Store in checkpoint
            checkpoint_data[item_id] = {
                "false_answer": verified_fa,
                "explanation": explanation,
                "source": source,
                "ground_truth": item["ground_truth"],
            }

            elapsed = time.time() - start_proc_time
            print(f"[{count:3d}/{len(math_indices)}] ID {item_id:14s} | GT: {item['ground_truth']:8s} | FA: {verified_fa:8s} | Source: {source:20s} ({elapsed:.1f}s elapsed)", flush=True)

            if count % 5 == 0 or count == len(math_indices):
                with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=2)

    finally:
        unload_model(model, tokenizer)
        print("\n[Cleanup] Unloaded Gemma 2 2B from GPU memory.")

    # 4. Save timestamped backup of previous final dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"final_filtered_evaluation_set_500_backup_{timestamp}.json")
    shutil.copyfile(FINAL_DATASET_PATH, backup_path)
    print(f"\n[Backup] Created backup: {backup_path}")

    # 5. Overwrite final dataset with updated high-quality false answers
    with open(FINAL_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(full_dataset, f, indent=2)
    print(f"[Dataset] Successfully updated final dataset with verified distractors at: {FINAL_DATASET_PATH}")

    # 6. Save generation summary report
    stats["end_time"] = datetime.now().isoformat()
    stats["total_duration_sec"] = round(time.time() - start_proc_time, 2)
    stats["verification_rate_percent"] = round((stats["gemma_verified"] / stats["total_math_items"]) * 100, 2)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 80)
    print(" DISTRACTOR GENERATION & VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total MATH Items Processed: {stats['total_math_items']}")
    print(f"Gemma 2B Verified:          {stats['gemma_verified']} ({stats['verification_rate_percent']}%)")
    print(f"Deterministic Fallbacks:   {stats['fallback_used']}")
    print(f"Total Duration:            {stats['total_duration_sec']}s")
    print(f"Summary Report:            {REPORT_PATH}")


if __name__ == "__main__":
    run_gemma_distractor_generation()
