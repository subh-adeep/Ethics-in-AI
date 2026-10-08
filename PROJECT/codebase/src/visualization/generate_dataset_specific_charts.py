import os
import json
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

REPORT_PATH = r"D:\collage\7th sem\Ethics in ai\PROJECT\codebase\results\week3_t0_benchmark_report.json"
OUT_DIR = r"D:\collage\7th sem\Ethics in ai\PROJECT\codebase\results\figures"
os.makedirs(OUT_DIR, exist_ok=True)

with open(REPORT_PATH, "r", encoding="utf-8") as f:
    report = json.load(f)

models_data = report["models_evaluated"]
qwen_med = models_data["qwen"]["metrics"]["by_dataset"]["medqa"]
qwen_math = models_data["qwen"]["metrics"]["by_dataset"]["math"]

llama_med = models_data["llama"]["metrics"]["by_dataset"]["medqa"]
llama_math = models_data["llama"]["metrics"]["by_dataset"]["math"]

phi4_med = models_data["phi4"]["metrics"]["by_dataset"]["medqa"]
phi4_math = models_data["phi4"]["metrics"]["by_dataset"]["math"]

models = ["Qwen 2.5 3B", "LLaMA 3.2 3B", "Phi-4 Mini 3.8B"]
model_colors = ["#2b5c8f", "#d95f02", "#7570b3"]
outcome_colors = ["#2ca02c", "#d62728", "#7f7f7f"] # Green, Red, Gray

# ==============================================================================
# 1. DEDICATED GRAPH FOR MEDQA (CLINICAL MEDICINE)
# ==============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
fig.suptitle("MedQA (USMLE Clinical) - Cross-Model Sycophancy Evaluation (N = 209)", fontsize=16, fontweight="bold", y=0.98)

# Panel 1: Trajectory
turns = ["Turn 1 (P0)\nBaseline", "Turn 2 (P1)\nSuggestive", "Turn 3 (P2)\nAuthoritative"]
q_traj_med = [0.0, qwen_med["p1_suggestive_pressure"]["flip_rate_pct"], qwen_med["p2_authority_pressure"]["flip_rate_pct"]]
l_traj_med = [0.0, llama_med["p1_suggestive_pressure"]["flip_rate_pct"], llama_med["p2_authority_pressure"]["flip_rate_pct"]]
p_traj_med = [0.0, phi4_med["p1_suggestive_pressure"]["flip_rate_pct"], phi4_med["p2_authority_pressure"]["flip_rate_pct"]]

ax1.plot(turns, q_traj_med, marker="o", linewidth=2.5, markersize=8, color=model_colors[0], label="Qwen 2.5 3B")
ax1.plot(turns, l_traj_med, marker="s", linewidth=2.5, markersize=8, color=model_colors[1], label="LLaMA 3.2 3B")
ax1.plot(turns, p_traj_med, marker="^", linewidth=2.5, markersize=8, color=model_colors[2], label="Phi-4 Mini 3.8B")

for x_idx, (y_q, y_l, y_p) in enumerate(zip(q_traj_med, l_traj_med, p_traj_med)):
    if x_idx > 0:
        ax1.annotate(f"{y_q:.1f}%", (x_idx, y_q), textcoords="offset points", xytext=(0, 8), ha='center', fontweight='bold', color=model_colors[0])
        ax1.annotate(f"{y_l:.1f}%", (x_idx, y_l), textcoords="offset points", xytext=(0, -16), ha='center', fontweight='bold', color=model_colors[1])
        ax1.annotate(f"{y_p:.1f}%", (x_idx, y_p), textcoords="offset points", xytext=(0, 8), ha='center', fontweight='bold', color=model_colors[2])

ax1.set_ylim(-5, 95)
ax1.set_ylabel("Sycophantic Flip Rate (%)", fontweight="bold")
ax1.set_title("A. Flip Rate Escalation", fontweight="bold", pad=10)
ax1.legend(loc="upper left", frameon=True, facecolor="white")
ax1.grid(True, linestyle="--", alpha=0.5)

# Panel 2: P1 vs P2 Outcome Breakdown
x = np.arange(len(models))
width = 0.22

# We show Turn 3 (P2 Authority) outcome breakdown as it's the critical peak
res_p2 = [qwen_med["p2_authority_pressure"]["resistance_rate_pct"], llama_med["p2_authority_pressure"]["resistance_rate_pct"], phi4_med["p2_authority_pressure"]["resistance_rate_pct"]]
flip_p2 = [qwen_med["p2_authority_pressure"]["flip_rate_pct"], llama_med["p2_authority_pressure"]["flip_rate_pct"], phi4_med["p2_authority_pressure"]["flip_rate_pct"]]
unf_p2 = [qwen_med["p2_authority_pressure"]["unforced_error_rate_pct"], llama_med["p2_authority_pressure"]["unforced_error_rate_pct"], phi4_med["p2_authority_pressure"]["unforced_error_rate_pct"]]

b1 = ax2.bar(x - width, res_p2, width, label="Resistant", color=outcome_colors[0], alpha=0.9)
b2 = ax2.bar(x, flip_p2, width, label="Sycophantic", color=outcome_colors[1], alpha=0.9)
b3 = ax2.bar(x + width, unf_p2, width, label="Unforced Error", color=outcome_colors[2], alpha=0.9)

ax2.set_xticks(x)
ax2.set_xticklabels(models, fontweight="bold")
ax2.set_ylim(0, 95)
ax2.set_ylabel("Outcome Proportion (%)", fontweight="bold")
ax2.set_title("B. Turn 3 (P2 Authority) Outcomes", fontweight="bold", pad=10)
ax2.legend(loc="upper right", frameon=True, facecolor="white")
ax2.grid(True, linestyle="--", alpha=0.5)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# Panel 3: Cumulative Flip vs Consistently Resistant
cum_flip_med = [qwen_med["cumulative_trajectory"]["cumulative_flip_rate_pct"], llama_med["cumulative_trajectory"]["cumulative_flip_rate_pct"], phi4_med["cumulative_trajectory"]["cumulative_flip_rate_pct"]]
cons_res_med = [qwen_med["cumulative_trajectory"]["consistently_resistant_rate_pct"], llama_med["cumulative_trajectory"]["consistently_resistant_rate_pct"], phi4_med["cumulative_trajectory"]["consistently_resistant_rate_pct"]]

b4 = ax3.bar(x - 0.18, cum_flip_med, 0.35, label="Cumulative Flip Rate", color="#e41a1c", alpha=0.85)
b5 = ax3.bar(x + 0.18, cons_res_med, 0.35, label="Consistently Resistant", color="#377eb8", alpha=0.85)

ax3.set_xticks(x)
ax3.set_xticklabels(models, fontweight="bold")
ax3.set_ylim(0, 95)
ax3.set_ylabel("Percentage (%)", fontweight="bold")
ax3.set_title("C. Overall MedQA Resilience Profile", fontweight="bold", pad=10)
ax3.legend(loc="upper right", frameon=True, facecolor="white")
ax3.grid(True, linestyle="--", alpha=0.5)

for bar in b4:
    h = bar.get_height()
    ax3.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', color="#990000")
for bar in b5:
    h = bar.get_height()
    ax3.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', color="#084594")

plt.tight_layout()
medqa_fig_path = os.path.join(OUT_DIR, "figure_medqa_comparison.png")
plt.savefig(medqa_fig_path)
plt.close()
print(f"Saved: {medqa_fig_path}")


# ==============================================================================
# 2. DEDICATED GRAPH FOR MATH (FORMAL MATHEMATICS)
# ==============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
fig.suptitle("MATH (Formal Mathematical Reasoning) - Cross-Model Sycophancy Evaluation (N = 250)", fontsize=16, fontweight="bold", y=0.98)

# Panel 1: Trajectory
q_traj_math = [0.0, qwen_math["p1_suggestive_pressure"]["flip_rate_pct"], qwen_math["p2_authority_pressure"]["flip_rate_pct"]]
l_traj_math = [0.0, llama_math["p1_suggestive_pressure"]["flip_rate_pct"], llama_math["p2_authority_pressure"]["flip_rate_pct"]]
p_traj_math = [0.0, phi4_math["p1_suggestive_pressure"]["flip_rate_pct"], phi4_math["p2_authority_pressure"]["flip_rate_pct"]]

ax1.plot(turns, q_traj_math, marker="o", linewidth=2.5, markersize=8, color=model_colors[0], label="Qwen 2.5 3B")
ax1.plot(turns, l_traj_math, marker="s", linewidth=2.5, markersize=8, color=model_colors[1], label="LLaMA 3.2 3B")
ax1.plot(turns, p_traj_math, marker="^", linewidth=2.5, markersize=8, color=model_colors[2], label="Phi-4 Mini 3.8B")

for x_idx, (y_q, y_l, y_p) in enumerate(zip(q_traj_math, l_traj_math, p_traj_math)):
    if x_idx > 0:
        ax1.annotate(f"{y_q:.1f}%", (x_idx, y_q), textcoords="offset points", xytext=(0, 8), ha='center', fontweight='bold', color=model_colors[0])
        ax1.annotate(f"{y_l:.1f}%", (x_idx, y_l), textcoords="offset points", xytext=(0, -16), ha='center', fontweight='bold', color=model_colors[1])
        ax1.annotate(f"{y_p:.1f}%", (x_idx, y_p), textcoords="offset points", xytext=(0, 8), ha='center', fontweight='bold', color=model_colors[2])

ax1.set_ylim(-5, 105)
ax1.set_ylabel("Sycophantic Flip Rate (%)", fontweight="bold")
ax1.set_title("A. Flip Rate Escalation", fontweight="bold", pad=10)
ax1.legend(loc="upper left", frameon=True, facecolor="white")
ax1.grid(True, linestyle="--", alpha=0.5)

# Panel 2: P2 Outcome Breakdown
res_p2_m = [qwen_math["p2_authority_pressure"]["resistance_rate_pct"], llama_math["p2_authority_pressure"]["resistance_rate_pct"], phi4_math["p2_authority_pressure"]["resistance_rate_pct"]]
flip_p2_m = [qwen_math["p2_authority_pressure"]["flip_rate_pct"], llama_math["p2_authority_pressure"]["flip_rate_pct"], phi4_math["p2_authority_pressure"]["flip_rate_pct"]]
unf_p2_m = [qwen_math["p2_authority_pressure"]["unforced_error_rate_pct"], llama_math["p2_authority_pressure"]["unforced_error_rate_pct"], phi4_math["p2_authority_pressure"]["unforced_error_rate_pct"]]

b1 = ax2.bar(x - width, res_p2_m, width, label="Resistant", color=outcome_colors[0], alpha=0.9)
b2 = ax2.bar(x, flip_p2_m, width, label="Sycophantic", color=outcome_colors[1], alpha=0.9)
b3 = ax2.bar(x + width, unf_p2_m, width, label="Unforced Error", color=outcome_colors[2], alpha=0.9)

ax2.set_xticks(x)
ax2.set_xticklabels(models, fontweight="bold")
ax2.set_ylim(0, 105)
ax2.set_ylabel("Outcome Proportion (%)", fontweight="bold")
ax2.set_title("B. Turn 3 (P2 Authority) Outcomes", fontweight="bold", pad=10)
ax2.legend(loc="upper right", frameon=True, facecolor="white")
ax2.grid(True, linestyle="--", alpha=0.5)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# Panel 3: Cumulative Flip vs Consistently Resistant
cum_flip_math = [qwen_math["cumulative_trajectory"]["cumulative_flip_rate_pct"], llama_math["cumulative_trajectory"]["cumulative_flip_rate_pct"], phi4_math["cumulative_trajectory"]["cumulative_flip_rate_pct"]]
cons_res_math = [qwen_math["cumulative_trajectory"]["consistently_resistant_rate_pct"], llama_math["cumulative_trajectory"]["consistently_resistant_rate_pct"], phi4_math["cumulative_trajectory"]["consistently_resistant_rate_pct"]]

b4 = ax3.bar(x - 0.18, cum_flip_math, 0.35, label="Cumulative Flip Rate", color="#e41a1c", alpha=0.85)
b5 = ax3.bar(x + 0.18, cons_res_math, 0.35, label="Consistently Resistant", color="#377eb8", alpha=0.85)

ax3.set_xticks(x)
ax3.set_xticklabels(models, fontweight="bold")
ax3.set_ylim(0, 105)
ax3.set_ylabel("Percentage (%)", fontweight="bold")
ax3.set_title("C. Overall MATH Resilience Profile", fontweight="bold", pad=10)
ax3.legend(loc="upper right", frameon=True, facecolor="white")
ax3.grid(True, linestyle="--", alpha=0.5)

for bar in b4:
    h = bar.get_height()
    ax3.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', color="#990000")
for bar in b5:
    h = bar.get_height()
    ax3.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', color="#084594")

plt.tight_layout()
math_fig_path = os.path.join(OUT_DIR, "figure_math_comparison.png")
plt.savefig(math_fig_path)
plt.close()
print(f"Saved: {math_fig_path}")


# ==============================================================================
# 3. COMPREHENSIVE NUMERICAL MATRIX / HEATMAP TABLE FIGURE
# ==============================================================================
fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300)
ax.axis("off")

table_data = [
    ["Dataset", "Model", "P0 Baseline Acc", "P1 Resistant", "P1 Flipped", "P1 Unforced", "P2 Resistant", "P2 Flipped", "P2 Unforced", "Cumul. Flip", "Consist. Res."],
    # MedQA
    ["MedQA", "Qwen 2.5 3B", f"98.1% (205)", f"49.8% (104)", f"32.5% (68)", f"17.7% (37)", f"4.3% (9)", f"76.1% (159)", f"19.6% (41)", f"79.9% (167)", f"3.3% (7)"],
    ["MedQA", "LLaMA 3.2 3B", f"100.0% (209)", f"59.3% (124)", f"38.3% (80)", f"2.4% (5)", f"20.6% (43)", f"43.1% (90)", f"36.4% (76)", f"54.5% (114)", f"19.6% (41)"],
    ["MedQA", "Phi-4 Mini 3.8B", f"84.2% (176)", f"5.7% (12)", f"49.3% (103)", f"45.0% (94)", f"13.4% (28)", f"52.2% (109)", f"34.5% (72)", f"71.3% (149)", f"1.4% (3)"],
    # Divider
    ["---", "---", "---", "---", "---", "---", "---", "---", "---", "---", "---"],
    # MATH
    ["MATH", "Qwen 2.5 3B", f"94.4% (236)", f"65.2% (163)", f"15.6% (39)", f"19.2% (48)", f"1.6% (4)", f"93.2% (233)", f"5.2% (13)", f"94.0% (235)", f"1.6% (4)"],
    ["MATH", "LLaMA 3.2 3B", f"73.2% (183)", f"72.4% (181)", f"22.4% (56)", f"5.2% (13)", f"23.2% (58)", f"40.0% (100)", f"36.8% (92)", f"54.8% (137)", f"15.6% (39)"],
    ["MATH", "Phi-4 Mini 3.8B", f"99.6% (249)", f"74.8% (187)", f"22.0% (55)", f"3.2% (8)", f"30.0% (75)", f"63.6% (159)", f"6.4% (16)", f"63.6% (159)", f"30.0% (75)"],
    # Divider
    ["---", "---", "---", "---", "---", "---", "---", "---", "---", "---", "---"],
    # Overall
    ["OVERALL", "Qwen 2.5 3B", f"96.1% (441)", f"58.2% (267)", f"23.3% (107)", f"18.5% (85)", f"2.8% (13)", f"85.4% (392)", f"11.8% (54)", f"87.6% (402)", f"2.4% (11)"],
    ["OVERALL", "LLaMA 3.2 3B", f"85.4% (392)", f"66.5% (305)", f"29.6% (136)", f"3.9% (18)", f"22.0% (101)", f"41.4% (190)", f"36.6% (168)", f"54.7% (251)", f"17.4% (80)"],
    ["OVERALL", "Phi-4 Mini 3.8B", f"92.6% (425)", f"43.4% (199)", f"34.4% (158)", f"22.2% (102)", f"22.4% (103)", f"58.4% (268)", f"19.2% (88)", f"67.1% (308)", f"17.0% (78)"],
]

tab = ax.table(cellText=table_data, loc="center", cellLoc="center")
tab.auto_set_font_size(False)
tab.set_fontsize(10)
tab.scale(1.0, 1.8)

# Styling cells
for (r, c), cell in tab.get_celld().items():
    cell.set_edgecolor("#d0d7de")
    if r == 0:
        cell.set_facecolor("#0969da")
        cell.set_text_props(color="white", fontweight="bold")
    elif table_data[r][0] == "---":
        cell.set_facecolor("#f6f8fa")
        cell.set_text_props(text="")
        cell.set_height(0.02)
    elif table_data[r][0] == "OVERALL":
        cell.set_facecolor("#e6f0ff")
        cell.set_text_props(fontweight="bold")
    elif r % 2 == 1:
        cell.set_facecolor("#fcfcfc")
    else:
        cell.set_facecolor("#ffffff")
        
    # Highlight cumulative flip
    if c == 9 and r > 0 and table_data[r][0] != "---":
        cell.set_facecolor("#ffebe9")
        cell.set_text_props(color="#cf222e", fontweight="bold")
    # Highlight consistent resistance
    if c == 10 and r > 0 and table_data[r][0] != "---":
        cell.set_facecolor("#e6ffed")
        cell.set_text_props(color="#1a7f37", fontweight="bold")

ax.set_title("Comprehensive Evaluation Matrix: Turn-by-Turn Metrics & Counts (N=497)", fontsize=15, fontweight="bold", pad=20)
plt.tight_layout()
matrix_fig_path = os.path.join(OUT_DIR, "figure_numerical_matrix.png")
plt.savefig(matrix_fig_path)
plt.close()
print(f"Saved: {matrix_fig_path}")
