import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Set aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16

REPORT_PATH = r"D:\collage\7th sem\Ethics in ai\PROJECT\codebase\results\week3_t0_benchmark_report.json"
OUT_DIR = r"D:\collage\7th sem\Ethics in ai\PROJECT\codebase\results\figures"
os.makedirs(OUT_DIR, exist_ok=True)

with open(REPORT_PATH, "r", encoding="utf-8") as f:
    report = json.load(f)

models_data = report["models_evaluated"]
qwen = models_data["qwen"]["metrics"]
llama = models_data["llama"]["metrics"]
phi4 = models_data["phi4"]["metrics"]

models = ["Qwen 2.5 3B", "LLaMA 3.2 3B", "Phi-4 Mini 3.8B"]
model_colors = ["#2b5c8f", "#d95f02", "#7570b3"]

# ==============================================================================
# FIGURE 1: Trajectory of Sycophancy Across Multi-Turn Dialogue
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

turns = ["Turn 1 (P0)\nBaseline", "Turn 2 (P1)\nSuggestive Bias", "Turn 3 (P2)\nAuthoritative Bias"]
qwen_traj = [0.0, qwen["overall"]["p1_suggestive_pressure"]["flip_rate_pct"], qwen["overall"]["p2_authority_pressure"]["flip_rate_pct"]]
llama_traj = [0.0, llama["overall"]["p1_suggestive_pressure"]["flip_rate_pct"], llama["overall"]["p2_authority_pressure"]["flip_rate_pct"]]
phi4_traj = [0.0, phi4["overall"]["p1_suggestive_pressure"]["flip_rate_pct"], phi4["overall"]["p2_authority_pressure"]["flip_rate_pct"]]

ax.plot(turns, qwen_traj, marker="o", linewidth=3, markersize=9, color="#2b5c8f", label="Qwen 2.5 3B")
ax.plot(turns, llama_traj, marker="s", linewidth=3, markersize=9, color="#d95f02", label="LLaMA 3.2 3B")
ax.plot(turns, phi4_traj, marker="^", linewidth=3, markersize=9, color="#7570b3", label="Phi-4 Mini 3.8B")

for x_idx, (y_q, y_l, y_p) in enumerate(zip(qwen_traj, llama_traj, phi4_traj)):
    if x_idx > 0:
        ax.annotate(f"{y_q:.1f}%", (x_idx, y_q), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color="#2b5c8f")
        ax.annotate(f"{y_l:.1f}%", (x_idx, y_l), textcoords="offset points", xytext=(0, -18), ha='center', fontweight='bold', color="#d95f02")
        ax.annotate(f"{y_p:.1f}%", (x_idx, y_p), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color="#7570b3")

ax.set_ylim(-5, 100)
ax.set_ylabel("Sycophantic Flip Rate (%)", fontweight="bold")
ax.set_title("Multi-Turn Sycophancy Escalation (P0 -> P1 -> P2)", pad=15, fontweight="bold")
ax.legend(frameon=True, facecolor="white", edgecolor="#ccc", loc="upper left")
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
fig1_path = os.path.join(OUT_DIR, "figure1_sycophancy_trajectory.png")
plt.savefig(fig1_path)
plt.close()
print(f"Saved: {fig1_path}")


# ==============================================================================
# FIGURE 2: 3-Way Outcome Breakdown (Resistant vs Sycophantic vs Unforced Error)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, dpi=300)

outcomes = ["Resistant", "Sycophantic", "Unforced Error"]
outcome_colors = ["#2ca02c", "#d62728", "#7f7f7f"]

# Data P1
q_p1 = [qwen["overall"]["p1_suggestive_pressure"]["resistance_rate_pct"], qwen["overall"]["p1_suggestive_pressure"]["flip_rate_pct"], qwen["overall"]["p1_suggestive_pressure"]["unforced_error_rate_pct"]]
l_p1 = [llama["overall"]["p1_suggestive_pressure"]["resistance_rate_pct"], llama["overall"]["p1_suggestive_pressure"]["flip_rate_pct"], llama["overall"]["p1_suggestive_pressure"]["unforced_error_rate_pct"]]
p_p1 = [phi4["overall"]["p1_suggestive_pressure"]["resistance_rate_pct"], phi4["overall"]["p1_suggestive_pressure"]["flip_rate_pct"], phi4["overall"]["p1_suggestive_pressure"]["unforced_error_rate_pct"]]

# Data P2
q_p2 = [qwen["overall"]["p2_authority_pressure"]["resistance_rate_pct"], qwen["overall"]["p2_authority_pressure"]["flip_rate_pct"], qwen["overall"]["p2_authority_pressure"]["unforced_error_rate_pct"]]
l_p2 = [llama["overall"]["p2_authority_pressure"]["resistance_rate_pct"], llama["overall"]["p2_authority_pressure"]["flip_rate_pct"], llama["overall"]["p2_authority_pressure"]["unforced_error_rate_pct"]]
p_p2 = [phi4["overall"]["p2_authority_pressure"]["resistance_rate_pct"], phi4["overall"]["p2_authority_pressure"]["flip_rate_pct"], phi4["overall"]["p2_authority_pressure"]["unforced_error_rate_pct"]]

x = np.arange(len(models))
width = 0.25

# Panel 1: Turn 2 (P1 - Suggestive)
b1 = ax1.bar(x - width, [q_p1[0], l_p1[0], p_p1[0]], width, label="Resistant", color=outcome_colors[0], alpha=0.9)
b2 = ax1.bar(x, [q_p1[1], l_p1[1], p_p1[1]], width, label="Sycophantic", color=outcome_colors[1], alpha=0.9)
b3 = ax1.bar(x + width, [q_p1[2], l_p1[2], p_p1[2]], width, label="Unforced Error", color=outcome_colors[2], alpha=0.9)

ax1.set_xticks(x)
ax1.set_xticklabels(models, fontweight="bold")
ax1.set_ylabel("Outcome Proportion (%)", fontweight="bold")
ax1.set_title("Turn 2 (P1): Suggestive Pressure Response", pad=12, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# Panel 2: Turn 3 (P2 - Authoritative)
b4 = ax2.bar(x - width, [q_p2[0], l_p2[0], p_p2[0]], width, label="Resistant", color=outcome_colors[0], alpha=0.9)
b5 = ax2.bar(x, [q_p2[1], l_p2[1], p_p2[1]], width, label="Sycophantic", color=outcome_colors[1], alpha=0.9)
b6 = ax2.bar(x + width, [q_p2[2], l_p2[2], p_p2[2]], width, label="Unforced Error", color=outcome_colors[2], alpha=0.9)

ax2.set_xticks(x)
ax2.set_xticklabels(models, fontweight="bold")
ax2.set_title("Turn 3 (P2): Authoritative Pressure Response", pad=12, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(frameon=True, facecolor="white", edgecolor="#ccc", loc="upper right")

for bars in [b4, b5, b6]:
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
fig2_path = os.path.join(OUT_DIR, "figure2_outcome_breakdown_p1_p2.png")
plt.savefig(fig2_path)
plt.close()
print(f"Saved: {fig2_path}")


# ==============================================================================
# FIGURE 3: Domain Asymmetry (MedQA vs MATH)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, dpi=300)

# Baseline Accuracy
medqa_acc = [qwen["by_dataset"]["medqa"]["turn1_baseline_accuracy"], llama["by_dataset"]["medqa"]["turn1_baseline_accuracy"], phi4["by_dataset"]["medqa"]["turn1_baseline_accuracy"]]
math_acc = [qwen["by_dataset"]["math"]["turn1_baseline_accuracy"], llama["by_dataset"]["math"]["turn1_baseline_accuracy"], phi4["by_dataset"]["math"]["turn1_baseline_accuracy"]]

x = np.arange(len(models))
width = 0.35

rects1 = ax1.bar(x - width/2, medqa_acc, width, label="MedQA (USMLE)", color="#1b9e77", alpha=0.9)
rects2 = ax1.bar(x + width/2, math_acc, width, label="MATH (Level 1-2)", color="#d95f02", alpha=0.9)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontweight="bold")
ax1.set_ylabel("Accuracy (%)", fontweight="bold")
ax1.set_title("P0 Baseline Accuracy by Domain", pad=12, fontweight="bold")
ax1.set_ylim(0, 115)
ax1.legend(frameon=True, facecolor="white", edgecolor="#ccc")
ax1.grid(True, linestyle="--", alpha=0.5)

for rects in [rects1, rects2]:
    for rect in rects:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight="bold")

# Cumulative Sycophantic Flip Rate by Domain
medqa_flip = [qwen["by_dataset"]["medqa"]["cumulative_trajectory"]["cumulative_flip_rate_pct"],
              llama["by_dataset"]["medqa"]["cumulative_trajectory"]["cumulative_flip_rate_pct"],
              phi4["by_dataset"]["medqa"]["cumulative_trajectory"]["cumulative_flip_rate_pct"]]

math_flip = [qwen["by_dataset"]["math"]["cumulative_trajectory"]["cumulative_flip_rate_pct"],
             llama["by_dataset"]["math"]["cumulative_trajectory"]["cumulative_flip_rate_pct"],
             phi4["by_dataset"]["math"]["cumulative_trajectory"]["cumulative_flip_rate_pct"]]

rects3 = ax2.bar(x - width/2, medqa_flip, width, label="MedQA (Clinical)", color="#1b9e77", alpha=0.9)
rects4 = ax2.bar(x + width/2, math_flip, width, label="MATH (Formal Reasoning)", color="#d95f02", alpha=0.9)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontweight="bold")
ax2.set_ylabel("Cumulative Flip Rate (%)", fontweight="bold")
ax2.set_title("Cumulative Sycophancy Susceptibility by Domain", pad=12, fontweight="bold")
ax2.set_ylim(0, 115)
ax2.legend(frameon=True, facecolor="white", edgecolor="#ccc")
ax2.grid(True, linestyle="--", alpha=0.5)

for rects in [rects3, rects4]:
    for rect in rects:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight="bold")

plt.tight_layout()
fig3_path = os.path.join(OUT_DIR, "figure3_domain_asymmetry.png")
plt.savefig(fig3_path)
plt.close()
print(f"Saved: {fig3_path}")


# ==============================================================================
# FIGURE 4: Overall Resilience Profile (Cumulative Flip vs Consistently Resistant)
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

cum_flip = [qwen["overall"]["cumulative_trajectory"]["cumulative_flip_rate_pct"],
            llama["overall"]["cumulative_trajectory"]["cumulative_flip_rate_pct"],
            phi4["overall"]["cumulative_trajectory"]["cumulative_flip_rate_pct"]]

consistent_res = [qwen["overall"]["cumulative_trajectory"]["consistently_resistant_rate_pct"],
                  llama["overall"]["cumulative_trajectory"]["consistently_resistant_rate_pct"],
                  phi4["overall"]["cumulative_trajectory"]["consistently_resistant_rate_pct"]]

width = 0.35
x = np.arange(len(models))

b1 = ax.bar(x - width/2, cum_flip, width, label="Cumulative Sycophantic Flip Rate", color="#e41a1c", alpha=0.85)
b2 = ax.bar(x + width/2, consistent_res, width, label="Consistently Resistant (All Turns)", color="#377eb8", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(models, fontweight="bold")
ax.set_ylabel("Percentage (%)", fontweight="bold")
ax.set_ylim(0, 105)
ax.set_title("Overall Robustness Profile: Capitulation vs Resistance", pad=15, fontweight="bold")
ax.legend(frameon=True, facecolor="white", edgecolor="#ccc", loc="upper right")
ax.grid(True, linestyle="--", alpha=0.5)

for bar in b1:
    h = bar.get_height()
    ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold', color="#990000")

for bar in b2:
    h = bar.get_height()
    ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold', color="#084594")

plt.tight_layout()
fig4_path = os.path.join(OUT_DIR, "figure4_model_resilience_comparison.png")
plt.savefig(fig4_path)
plt.close()
print(f"Saved: {fig4_path}")

print("\nAll 4 benchmark figures generated successfully in:", OUT_DIR)
