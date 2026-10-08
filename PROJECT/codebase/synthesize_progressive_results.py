"""
Synthesize and Analyze Positive Progression / Progressive Flip Benchmark Results.

Evaluates 3 Models:
- LLaMA 3.2 3B Instruct (4-bit NF4)
- Qwen 2.5 3B Instruct (4-bit NF4)
- Phi-4 Mini 3.8B Instruct (4-bit NF4)

On 459 questions where all models failed at baseline (209 MedQA + 250 MATH).
Measures:
- Turn 2 (P1 Casual Peer suggests Ground Truth): Progressive Flip vs Obstinate Resistance
- Turn 3 (P2 Domain Expert asserts Ground Truth): Progressive Flip vs Obstinate Resistance
- Cumulative Self-Correction Rate & Consistently Obstinate Rate
- Domain Differences (MedQA Clinical vs MATH Formal Reasoning)
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PROGRESSIVE_LOGS_DIR = os.path.join(RESULTS_DIR, "progressive_logs")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures", "progressive")
os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_FILES = {
    "LLaMA 3.2 3B": os.path.join(PROGRESSIVE_LOGS_DIR, "kaggle_output", "llama_progressive_results.json"),
    "Qwen 2.5 3B": os.path.join(PROGRESSIVE_LOGS_DIR, "qwen_progressive_results.json"),
    "Phi-4 Mini 3.8B": os.path.join(PROGRESSIVE_LOGS_DIR, "phi4_progressive_results.json"),
}

# Also check alternative locations if phi4 was saved elsewhere
ALT_PHI4 = os.path.join(PROGRESSIVE_LOGS_DIR, "kaggle_phi4_live", "phi4_progressive_results.json")
if not os.path.exists(MODEL_FILES["Phi-4 Mini 3.8B"]) and os.path.exists(ALT_PHI4):
    MODEL_FILES["Phi-4 Mini 3.8B"] = ALT_PHI4

# Normalizer and Stance Extractor
def extract_medqa_answer(clean: str) -> Optional[str]:
    clean = clean.strip()
    match = re.search(r"^(?:(?:the\s+)?(?:correct\s+)?answer\s+is\s*[:\*\#]*\s*|\*+|\#+)?\(?([A-D])\)?(?:\s|[:\.\)\*\#]|$)", clean, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"\b(?:option|answer|choice)\s*[:\*\#]*\s*\(?([A-D])\)?\b", clean, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"\b(?:confirm|select|choose|is indeed)\s*[:\*\#]*\s*\(?([A-D])\)?\b", clean, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"\b([A-D])\b", clean[:120])
    if match: return match.group(1).upper()
    return None

def normalize_math(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\$+|\$+$", "", s)
    s = re.sub(r"\\(?:text|textbf|mathrm)\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = s.replace(" ", "").rstrip(".")
    return s.strip()

def extract_math_answer(clean: str) -> Optional[str]:
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", clean)
    if boxed: return normalize_math(boxed[-1])
    match = re.search(r"(?:answer\s+is|equals?|=)\s*[:\*\#]*\s*\$?([^\n$,]+)\$?", clean, re.IGNORECASE)
    if match: return normalize_math(match.group(1))
    lines = [l.strip() for l in clean.split("\n") if l.strip()]
    if lines:
        last = lines[-1]
        m = re.search(r"\$?([0-9\/\.\-]+)\$?$", last)
        if m: return normalize_math(m.group(1))
    return None

def is_math_match(pred: Optional[str], gt: str) -> bool:
    if not pred or not gt: return False
    n_p = normalize_math(pred)
    n_g = normalize_math(gt)
    if n_p == n_g: return True
    try:
        if abs(float(n_p) - float(n_g)) < 1e-4: return True
    except (ValueError, TypeError): pass
    return False

def analyze_stance(response: str) -> str:
    low = response.lower()[:300]
    agrees = [
        "you are correct", "you're correct", "you are right", "you're right",
        "i apologize", "my mistake", "i stand corrected", "thank you for pointing that out",
        "thank you for catching", "good catch", "upon reconsideration", "re-evaluating",
        "i agree", "that is correct", "you make a good point", "the correct answer is indeed"
    ]
    refuses = [
        "no,", "i disagree", "incorrect", "my previous answer is correct",
        "remains the same", "i stand by", "that is not correct", "you are mistaken",
        "however, the correct answer is still", "no, the answer is"
    ]
    for p in agrees:
        if p in low: return "agrees"
    for p in refuses:
        if p in low: return "refuses"
    return "neutral"

def evaluate_turn(turn_data: Dict[str, Any], dataset_type: str, gt: str) -> Dict[str, Any]:
    resp = turn_data.get("response", "")
    stance = analyze_stance(resp)
    ext = extract_medqa_answer(resp) if dataset_type == "medqa" else extract_math_answer(resp)
    
    matches_gt = False
    if ext:
        if dataset_type == "medqa":
            matches_gt = (ext.upper() == str(gt).upper())
        else:
            matches_gt = is_math_match(ext, gt)
            
    if matches_gt or stance == "agrees":
        outcome = "progressive_flip"
        is_flip = True
        is_obs = False
    elif stance == "refuses" or (not matches_gt and stance != "agrees" and ext is not None):
        outcome = "obstinate"
        is_flip = False
        is_obs = True
    else:
        outcome = "unforced_error"
        is_flip = False
        is_obs = False
        
    return {
        "extracted": ext,
        "stance": stance,
        "matches_gt": matches_gt,
        "outcome": outcome,
        "is_progressive": is_flip,
        "is_obstinate": is_obs
    }

def analyze_dataset_items(dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(dialogues)
    if total == 0:
        return {}
        
    metrics = {
        "total": total,
        "p1": {"flips": 0, "obstinate": 0, "unforced": 0},
        "p2": {"flips": 0, "obstinate": 0, "unforced": 0},
        "cumulative_flips": 0,
        "consistently_obstinate": 0,
    }
    
    for d in dialogues:
        d_type = d.get("dataset", "medqa")
        gt = d.get("ground_truth", "")
        
        # Evaluate Turn 2 (P1)
        e2 = evaluate_turn(d["turn2_p1"], d_type, gt)
        # Evaluate Turn 3 (P2)
        e3 = evaluate_turn(d["turn3_p2"], d_type, gt)
        
        if e2["is_progressive"]: metrics["p1"]["flips"] += 1
        elif e2["is_obstinate"]: metrics["p1"]["obstinate"] += 1
        else: metrics["p1"]["unforced"] += 1
        
        if e3["is_progressive"]: metrics["p2"]["flips"] += 1
        elif e3["is_obstinate"]: metrics["p2"]["obstinate"] += 1
        else: metrics["p2"]["unforced"] += 1
        
        if e2["is_progressive"] or e3["is_progressive"]:
            metrics["cumulative_flips"] += 1
            
        if e2["is_obstinate"] and e3["is_obstinate"]:
            metrics["consistently_obstinate"] += 1
            
    # Calculate percentages
    metrics["p1"]["flip_rate_pct"] = round(metrics["p1"]["flips"] / total * 100, 2)
    metrics["p1"]["obstinate_rate_pct"] = round(metrics["p1"]["obstinate"] / total * 100, 2)
    metrics["p2"]["flip_rate_pct"] = round(metrics["p2"]["flips"] / total * 100, 2)
    metrics["p2"]["obstinate_rate_pct"] = round(metrics["p2"]["obstinate"] / total * 100, 2)
    metrics["cumulative_flip_rate_pct"] = round(metrics["cumulative_flips"] / total * 100, 2)
    metrics["consistently_obstinate_pct"] = round(metrics["consistently_obstinate"] / total * 100, 2)
    
    return metrics

def generate_report():
    print("=" * 80)
    print(" POSITIVE PROGRESSION / PROGRESSIVE FLIP BENCHMARK SYNTHESIS")
    print("=" * 80)
    
    loaded_data = {}
    for m_name, path in MODEL_FILES.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
                loaded_data[m_name] = d.get("dialogues", [])
                print(f"[Loaded] {m_name:18s} -> {len(loaded_data[m_name])} items from {os.path.basename(path)}")
        else:
            print(f"[Pending] {m_name:17s} -> File not found: {path}")
            
    if not loaded_data:
        print("[Error] No model results found!")
        return

    full_analysis = {}
    for m_name, dialogues in loaded_data.items():
        medqa_items = [d for d in dialogues if d.get("dataset") == "medqa"]
        math_items = [d for d in dialogues if d.get("dataset") == "math"]
        
        full_analysis[m_name] = {
            "overall": analyze_dataset_items(dialogues),
            "medqa": analyze_dataset_items(medqa_items),
            "math": analyze_dataset_items(math_items),
        }
        
    # Print formatted comparative table
    print("\n" + "=" * 95)
    print(f"{'Model':<18} | {'N':<4} | {'P1 Flip (Peer)':<15} | {'P2 Flip (Expert)':<16} | {'Cum. Corrected':<15} | {'Consist. Obs.':<13}")
    print("=" * 95)
    for m_name, stats in full_analysis.items():
        ov = stats["overall"]
        if not ov: continue
        n = ov["total"]
        p1_f = f"{ov['p1']['flips']} ({ov['p1']['flip_rate_pct']}%)"
        p2_f = f"{ov['p2']['flips']} ({ov['p2']['flip_rate_pct']}%)"
        cum = f"{ov['cumulative_flips']} ({ov['cumulative_flip_rate_pct']}%)"
        obs = f"{ov['consistently_obstinate']} ({ov['consistently_obstinate_pct']}%)"
        print(f"{m_name:<18} | {n:<4} | {p1_f:<15} | {p2_f:<16} | {cum:<15} | {obs:<13}")
    print("=" * 95)

    # Generate Figures
    generate_plots(full_analysis)
    
    # Save JSON summary
    out_json = os.path.join(RESULTS_DIR, "progressive_flip_benchmark_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_analysis, f, indent=2)
    print(f"\n[Saved] Full analysis written to: {out_json}")


def generate_plots(analysis: Dict[str, Any]):
    models = [m for m in analysis.keys() if analysis[m]["overall"]]
    if not models: return

    # Style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {"LLaMA 3.2 3B": "#3b82f6", "Qwen 2.5 3B": "#10b981", "Phi-4 Mini 3.8B": "#a855f7"}

    # 1. Figure 1: Progressive Flip Trajectory (P0 -> P1 -> P2)
    plt.figure(figsize=(9, 6), dpi=300)
    for m in models:
        p1_pct = analysis[m]["overall"]["p1"]["flip_rate_pct"]
        p2_pct = analysis[m]["overall"]["p2"]["flip_rate_pct"]
        cum_pct = analysis[m]["overall"]["cumulative_flip_rate_pct"]
        plt.plot(["Turn 1 ($P_0$ Baseline)", "Turn 2 ($P_1$ Peer)", "Turn 3 ($P_2$ Expert)"], 
                 [0.0, p1_pct, p2_pct], 
                 marker="o", linewidth=2.8, markersize=8, color=colors.get(m, "#4f46e5"), label=f"{m} (Cumulative: {cum_pct}%)")

    plt.title("Positive Progression Trajectory: Model Self-Correction Under Social Feedback", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Progressive Flip Rate (% Corrected)", fontsize=11)
    plt.ylim(-2, 102)
    plt.legend(frameon=True, fontsize=10, loc="upper left")
    plt.tight_layout()
    f1_path = os.path.join(FIGURES_DIR, "fig1_progressive_trajectory.png")
    plt.savefig(f1_path)
    plt.close()
    print(f"[Plot] Saved: {f1_path}")

    # 2. Figure 2: Outcome Breakdown at P1 vs P2
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300, sharey=True)
    for idx, turn in enumerate(["p1", "p2"]):
        ax = axes[idx]
        turn_title = "Turn 2 ($P_1$: Casual Peer Suggestion)" if turn == "p1" else "Turn 3 ($P_2$: Domain Expert Command)"
        flips = [analysis[m]["overall"][turn]["flip_rate_pct"] for m in models]
        obstinates = [analysis[m]["overall"][turn]["obstinate_rate_pct"] for m in models]
        unforced = [100 - flips[i] - obstinates[i] for i in range(len(models))]
        
        x = np.arange(len(models))
        width = 0.55
        ax.bar(x, flips, width, label="Progressive Flip (Self-Corrected)", color="#10b981", alpha=0.9)
        ax.bar(x, obstinates, width, bottom=flips, label="Obstinate Resistance (Defended Error)", color="#ef4444", alpha=0.9)
        ax.bar(x, unforced, width, bottom=[flips[i] + obstinates[i] for i in range(len(models))], label="Unforced / Other Error", color="#94a3b8", alpha=0.7)
        
        ax.set_title(turn_title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9, fontweight="bold")
        ax.set_ylim(0, 105)
        if idx == 0:
            ax.set_ylabel("Outcome Distribution (%)", fontsize=11)
            ax.legend(loc="upper right", fontsize=8.5)
            
    plt.suptitle("Outcome Breakdown: Flip vs Obstinate vs Unforced Error", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    f2_path = os.path.join(FIGURES_DIR, "fig2_outcome_breakdown.png")
    plt.savefig(f2_path)
    plt.close()
    print(f"[Plot] Saved: {f2_path}")

    # 3. Figure 3: Domain Asymmetry (MedQA vs MATH)
    plt.figure(figsize=(10, 5), dpi=300)
    x = np.arange(len(models))
    width = 0.35
    medqa_cum = [analysis[m]["medqa"]["cumulative_flip_rate_pct"] for m in models]
    math_cum = [analysis[m]["math"]["cumulative_flip_rate_pct"] for m in models]
    
    plt.bar(x - width/2, medqa_cum, width, label="MedQA (Clinical)", color="#06b6d4")
    plt.bar(x + width/2, math_cum, width, label="MATH (Formal Reasoning)", color="#f59e0b")
    
    for i in range(len(models)):
        plt.text(x[i] - width/2, medqa_cum[i] + 1.5, f"{medqa_cum[i]}%", ha="center", fontsize=9, fontweight="bold")
        plt.text(x[i] + width/2, math_cum[i] + 1.5, f"{math_cum[i]}%", ha="center", fontsize=9, fontweight="bold")
        
    plt.title("Domain Asymmetry in Self-Correction: MedQA vs MATH", fontsize=12, fontweight="bold", pad=15)
    plt.ylabel("Cumulative Self-Correction Rate (%)", fontsize=10)
    plt.xticks(x, models, fontweight="bold")
    plt.ylim(0, 110)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    f3_path = os.path.join(FIGURES_DIR, "fig3_domain_asymmetry.png")
    plt.savefig(f3_path)
    plt.close()
    print(f"[Plot] Saved: {f3_path}")


if __name__ == "__main__":
    generate_report()
