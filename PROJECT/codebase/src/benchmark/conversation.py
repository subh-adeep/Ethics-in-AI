"""
Multi-Turn Continuous-Context Conversation Runner for T0 Baseline Evaluation.

Executes the continuous 3-turn pressure protocol:
Turn 1: P0 (Neutral Baseline) -> R1
Turn 2: P1 (Suggestive Pressure) -> R2 (conditioned on [U1, R1, U2])
Turn 3: P2 (Authority Pressure)  -> R3 (conditioned on [U1, R1, U2, R2, U3])

Preserves conversation history natively via tokenizer.apply_chat_template.
Evaluates every turn with src.benchmark.judge.evaluate_response.
"""

from typing import Dict, Any, List
import torch
from src.benchmark.judge import evaluate_response


def generate_turn_response(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    device: str = "cuda",
    max_new_tokens: int = 96,
) -> str:
    """Generates the next turn response using greedy decoding for deterministic evaluation."""
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for exact reproducibility
            pad_token_id=tokenizer.pad_token_id,
        )

    in_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][in_len:], skip_special_tokens=True).strip()
    return response


def run_multiturn_dialogue(
    model,
    tokenizer,
    item: Dict[str, Any],
    device: str = "cuda",
    max_new_tokens: int = 96,
) -> Dict[str, Any]:
    """
    Executes a complete 3-turn continuous-context evaluation for a single benchmark item.

    Args:
        model: Hugging Face CausalLM
        tokenizer: Hugging Face PreTrainedTokenizer
        item: Benchmark item from final_filtered_evaluation_set_500.json
        device: 'cuda' or 'cpu'
        max_new_tokens: Maximum generation length per turn

    Returns:
        Structured dialogue record with all turn prompts, responses, and evaluated outcomes.
    """
    dataset_type = item["dataset"]
    gt = item["ground_truth"]
    fa = item.get("false_answer_key") if dataset_type == "medqa" else item.get("false_answer")
    
    p0 = item["p0_prompt"]
    p1 = item["p1_prompt"]
    p2 = item["p2_prompt"]

    # --- Turn 1: P0 Baseline ---
    messages = [{"role": "user", "content": p0}]
    r1 = generate_turn_response(model, tokenizer, messages, device=device, max_new_tokens=max_new_tokens)
    eval_t1 = evaluate_response(dataset_type, r1, ground_truth=gt, false_answer=fa, is_challenged_turn=False)

    # --- Turn 2: P1 Suggestive Pressure ---
    messages.append({"role": "assistant", "content": r1})
    messages.append({"role": "user", "content": p1})
    r2 = generate_turn_response(model, tokenizer, messages, device=device, max_new_tokens=max_new_tokens)
    eval_t2 = evaluate_response(dataset_type, r2, ground_truth=gt, false_answer=fa, is_challenged_turn=True)

    # --- Turn 3: P2 Authority Pressure ---
    messages.append({"role": "assistant", "content": r2})
    messages.append({"role": "user", "content": p2})
    r3 = generate_turn_response(model, tokenizer, messages, device=device, max_new_tokens=max_new_tokens)
    eval_t3 = evaluate_response(dataset_type, r3, ground_truth=gt, false_answer=fa, is_challenged_turn=True)

    # Summarize flip trajectory
    # Did the model flip at P1? Did it flip at P2?
    flipped_at_p1 = (eval_t2["outcome"] == "sycophantic")
    flipped_at_p2 = (eval_t3["outcome"] == "sycophantic")
    remained_resistant = (eval_t2["outcome"] == "resistant" and eval_t3["outcome"] == "resistant")

    return {
        "id": item["id"],
        "dataset": dataset_type,
        "ground_truth": gt,
        "false_answer": fa,
        "turn1_p0": {
            "prompt": p0,
            "response": r1,
            "extracted": eval_t1["extracted_answer"],
            "is_correct": eval_t1["is_correct"],
        },
        "turn2_p1": {
            "prompt": p1,
            "response": r2,
            "extracted": eval_t2["extracted_answer"],
            "outcome": eval_t2["outcome"],  # 'resistant' | 'sycophantic' | 'unforced_error'
            "stance": eval_t2["stance"],
        },
        "turn3_p2": {
            "prompt": p2,
            "response": r3,
            "extracted": eval_t3["extracted_answer"],
            "outcome": eval_t3["outcome"],  # 'resistant' | 'sycophantic' | 'unforced_error'
            "stance": eval_t3["stance"],
        },
        "trajectory": {
            "flipped_at_p1": flipped_at_p1,
            "flipped_at_p2": flipped_at_p2,
            "remained_resistant": remained_resistant,
        }
    }
