"""build_figures.py

Builds every figure and data table of the final report directly from the saved
result files (no number is typed by hand).

Main source (final labels = parser v4 + strict manual review of unparsed replies):
  codebase/results/final_analysis_20261121/manual_unparsed_review_strict/
      metrics_table_strict.csv    per dataset x model x condition x turn
      significance_strict.csv     42 paired tests, tool condition vs no tool
Other sources:
  final_analysis_20261121/rescore/python_reliability_v4.csv   (Python tool, v4 labels)
  final_analysis_20261121/rescore/self_correction_v4.csv      (repair loop, v4 labels)
  qwen_tool_choice_probe_20261110/kaggle_output/probe/*_probe.jsonl

Outputs:
  figures/fig_resistance_heatmap.pdf, fig_delta_forest.pdf, fig_access_trust.pdf,
  fig_python.pdf, fig_qwen_probe.pdf
  tables/tab_metrics_full.tex, tab_significance.tex, tab_qwen_probe.tex, numbers.tex

Run (from the 'Final report' folder):  python scripts/build_figures.py
Requires: matplotlib.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parents[1]                 # .../PROJECT/Final report
RES = HERE.parent / "codebase" / "results"
FINAL = RES / "final_analysis_20261121"
STRICT = FINAL / "manual_unparsed_review_strict"
FIG = HERE / "figures"
TAB = HERE / "tables"

# ---- style: report font (Palatino), palette validated for colour-blind safety ----
for name in ["pala.ttf", "palab.ttf", "palai.ttf", "palabi.ttf"]:
    path = Path("C:/Windows/Fonts") / name
    if path.exists():
        font_manager.fontManager.addfont(str(path))
plt.rcParams.update({
    # a list of families = per-glyph fallback (matplotlib >= 3.6): Palatino for text,
    # DejaVu for the up/down triangles that Palatino lacks
    "font.family": ["Palatino Linotype", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "text.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b",
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
TEXT_W = 5.5                                                # template \textwidth (in)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODELS = ["Qwen", "LLaMA", "Phi-4"]
MODEL_COLOR = {"Qwen": "#2a78d6", "LLaMA": "#eb6834", "Phi-4": "#1baf7a"}   # slots 1-3
MODEL_MARK = {"Qwen": "o", "LLaMA": "s", "Phi-4": "^"}
MODEL_LABEL = {"Qwen": "Qwen2.5-3B", "LLaMA": "Llama-3.2-3B", "Phi-4": "Phi-4-mini"}
BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
U_MARK = 10.0   # heatmap cells with at least this % unparsed replies show the U share
# outcome colours (R, S, O, U): categorical slots blue, red, yellow, violet;
# validated with the dataviz validator (adjacent CVD dE >= 15, normal-vision dE >= 20)
OUTCOMES = [("resistant", "R  correct answer", "#2a78d6"),
            ("flip", "S  user's wrong answer", "#e34948"),
            ("other_wrong", "O  other wrong answer", "#eda100"),
            ("unparsed", "U  no clear final answer", "#6250d6")]

CONDITIONS = {
    "MedQA": ["T0", "T1", "T2 web", "Exact", "T3 textbook"],
    "MATH": ["T0", "T1", "Exact", "Python"],
}
COND_LABEL = {"T0": "None", "T1": "BM25\ntop-2", "T2 web": "Web\nsearch", "Exact": "Exact\nlookup",
              "T3 textbook": "Textbook\npassages", "Python": "Python\nexec."}
COND_SHORT = {"T1": "BM25 top-2", "T2 web": "Web search", "Exact": "Exact lookup",
              "T3 textbook": "Textbook", "Python": "Python"}


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def num(value):
    return float(value) if value not in ("", None) else None


METRICS = {(r["dataset"], r["model"], r["condition"], r["turn"]): r
           for r in read_csv(STRICT / "metrics_table_strict.csv")}
SIG = {(r["dataset"], r["model"], r["condition"], r["turn"]): r
       for r in read_csv(STRICT / "significance_strict.csv")}


def style(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Figure: heatmap
def fig_heatmap():
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    rows = [(m, t) for m in MODELS for t in ("P1", "P2")]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.4),
                             gridspec_kw={"width_ratios": [5, 4], "wspace": 0.08})
    for ax, dataset in zip(axes, ("MedQA", "MATH")):
        conds = CONDITIONS[dataset]
        for i, (model, turn) in enumerate(rows):
            for j, cond in enumerate(conds):
                r = METRICS[(dataset, model, cond, turn)]
                value = float(r["resistant_pct"])
                ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, color=cmap(value / 100), lw=0))
                text_color = "white" if value >= 55 else INK
                label = f"{value:.0f}"
                sig = SIG.get((dataset, model, cond, turn))
                if sig and float(sig["holm_p"]) < 0.05:
                    label += " \u25B2" if float(sig["diff_R_pp"]) > 0 else " \u25BC"
                unparsed = float(r["unparsed_pct"])
                if unparsed >= U_MARK:
                    # cells where many replies have no clear final answer: show the U share
                    ax.text(j + 0.48, i + 0.33, label, ha="center", va="center",
                            fontsize=7.6, color=text_color)
                    ax.text(j + 0.48, i + 0.70, f"U {unparsed:.0f}%", ha="center", va="center",
                            fontsize=5.6, color=text_color)
                else:
                    ax.text(j + 0.48, i + 0.46, label, ha="center", va="center",
                            fontsize=7.6, color=text_color)
        ax.set_xlim(0, len(conds))
        ax.set_ylim(len(rows), 0)
        ax.set_xticks([j + 0.48 for j in range(len(conds))])
        ax.set_xticklabels([COND_LABEL[c] for c in conds], fontsize=7)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(f"{dataset} (n = {'146' if dataset == 'MedQA' else '208–215'} per cell)",
                     pad=22, fontsize=8.5)
        if dataset == "MedQA":
            ax.set_yticks([i + 0.46 for i in range(len(rows))])
            ax.set_yticklabels([f"{MODEL_LABEL[m]}  {t}" for m, t in rows], fontsize=7.5)
        else:
            ax.set_yticks([])
        for k in (2, 4):
            ax.axhline(k - 0.04, color="white", lw=2.0)
    fig.savefig(FIG / "fig_resistance_heatmap.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure: R/S/O/U composition
def fig_outcomes():
    """100% stacked horizontal bars: every challenged turn split into R, S, O and U."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 6.4), gridspec_kw={"wspace": 0.42})
    for ax, dataset in zip(axes, ("MedQA", "MATH")):
        conds = CONDITIONS[dataset]
        y, ticks, labels, groups = 0.0, [], [], []
        for model in MODELS:
            for turn in ("P1", "P2"):
                groups.append((y, f"{MODEL_LABEL[model]}  {turn}"))
                y += 0.85                                    # room for the group header
                for cond in conds:
                    r = METRICS[(dataset, model, cond, turn)]
                    left = 0.0
                    for key, _, color in OUTCOMES:
                        share = float(r[f"{key}_pct"])
                        if share > 0:
                            # 2 px surface gap between segments (edgecolor = surface)
                            ax.barh(y, share, left=left, height=0.78, color=color,
                                    edgecolor="#fcfcfb", linewidth=0.7)
                            big = share >= 14 and key in ("resistant", "flip", "unparsed")
                            if big:
                                ax.text(left + share / 2, y, f"{share:.0f}", ha="center", va="center",
                                        fontsize=5.8, color="white")
                        left += share
                    ticks.append(y)
                    labels.append(COND_LABEL[cond].replace("\n", " "))
                    y += 1.0
                y += 0.35
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=6.3)
        for gy, text in groups:
            ax.text(0, gy + 0.28, text, ha="left", va="center", fontsize=6.8,
                    fontweight="bold", color=INK)
        ax.set_ylim(y - 0.2, -0.4)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=6.5)
        ax.tick_params(axis="y", length=0, pad=2)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"{dataset} (n = {'146' if dataset == 'MedQA' else '208–215'})", fontsize=8.5, pad=4)
    handles = [Patch(facecolor=c, edgecolor="none", label=lab) for _, lab, c in OUTCOMES]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.025), handlelength=1.2, columnspacing=1.2)
    fig.savefig(FIG / "fig_outcomes.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure: forest plot
def fig_forest():
    combos = [("MedQA", c) for c in CONDITIONS["MedQA"][1:]] + [("MATH", c) for c in CONDITIONS["MATH"][1:]]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.7), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    offsets = {"Qwen": -0.22, "LLaMA": 0.0, "Phi-4": 0.22}
    for ax, turn in zip(axes, ("P1", "P2")):
        style(ax, "x")
        ax.axvline(0, color=INK2, lw=0.8)
        for k, (dataset, cond) in enumerate(combos):
            for model in MODELS:
                r = SIG[(dataset, model, cond, turn)]
                d, lo, hi = float(r["diff_R_pp"]), float(r["ci_low_pp"]), float(r["ci_high_pp"])
                y = k + offsets[model]
                significant = float(r["holm_p"]) < 0.05
                ax.plot([lo, hi], [y, y], color=MODEL_COLOR[model], lw=1.1, solid_capstyle="round")
                ax.plot(d, y, MODEL_MARK[model], ms=4.6, mec=MODEL_COLOR[model], mew=1.1,
                        mfc=MODEL_COLOR[model] if significant else "white", zorder=3)
            if k == 3:
                ax.axhline(k + 0.5, color=INK2, lw=0.6, ls=(0, (2, 2)))
        ax.set_xlim(-70, 70)
        ax.set_xticks([-60, -40, -20, 0, 20, 40, 60])
        ax.set_xlabel(f"{turn}: change in resistant % vs no tool (pp)")
        ax.set_ylim(len(combos) - 0.5, -0.5)
    axes[0].set_yticks(range(len(combos)))
    axes[0].set_yticklabels([f"{d}: {COND_SHORT[c]}" for d, c in combos])
    axes[0].tick_params(axis="y", length=0)
    handles = [Line2D([], [], color=MODEL_COLOR[m], marker=MODEL_MARK[m], lw=1.1, ms=4.6,
                      mfc=MODEL_COLOR[m], label=MODEL_LABEL[m]) for m in MODELS]
    handles.append(Line2D([], [], color=INK2, marker="o", lw=0, ms=4.6, mfc="white",
                          label="hollow = not significant after Holm"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.55, 1.04), handlelength=1.6, columnspacing=1.2)
    fig.savefig(FIG / "fig_delta_forest.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure: access / trust
def pooled(dataset, model, cond):
    p1, p2 = METRICS[(dataset, model, cond, "P1")], METRICS[(dataset, model, cond, "P2")]
    opportunities = int(p1["n"]) + int(p2["n"])
    used = int(p1["tool_use_n"] or 0) + int(p2["tool_use_n"] or 0)
    evidence = int(p1["evidence_correct_n"] or 0) + int(p2["evidence_correct_n"] or 0)
    override = int(p1["evidence_override_n"] or 0) + int(p2["evidence_override_n"] or 0)
    return used, opportunities, override, evidence


def fig_access_trust():
    combos = [("MedQA", c) for c in CONDITIONS["MedQA"][1:]] + [("MATH", c) for c in CONDITIONS["MATH"][1:]]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.75), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    bar_h = 0.24
    offsets = {"Qwen": -bar_h, "LLaMA": 0.0, "Phi-4": bar_h}
    for k, (dataset, cond) in enumerate(combos):
        for model in MODELS:
            used, opp, override, evidence = pooled(dataset, model, cond)
            y = k + offsets[model]
            use_pct = 100 * used / opp
            axes[0].barh(y, use_pct, height=bar_h * 0.92, color=MODEL_COLOR[model], lw=0)
            axes[0].text(use_pct + 1.5, y, f"{use_pct:.0f}", va="center", fontsize=6.3, color=INK2)
            if evidence >= 10:
                ovr = 100 * override / evidence
                axes[1].barh(y, ovr, height=bar_h * 0.92, color=MODEL_COLOR[model], lw=0)
                axes[1].text(ovr + 1.5, y, f"{ovr:.0f}", va="center", fontsize=6.3, color=INK2)
            # fewer than 10 evidence turns (Qwen rarely called the tool): no bar; see caption
    for ax, label in zip(axes, ("(a) Chose to call the tool (% of P1+P2 turns)",
                                "(b) Agreed with the user although the tool\nresult supported the correct answer (%)")):
        style(ax, "x")
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel(label)
        ax.axhline(3.5, color=INK2, lw=0.6, ls=(0, (2, 2)))
    axes[0].set_yticks(range(len(combos)))
    axes[0].set_yticklabels([f"{d}: {COND_SHORT[c]}" for d, c in combos])
    axes[0].set_ylim(len(combos) - 0.5, -0.5)
    axes[0].tick_params(axis="y", length=0)
    handles = [Patch(color=MODEL_COLOR[m], label=MODEL_LABEL[m]) for m in MODELS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.03))
    fig.savefig(FIG / "fig_access_trust.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure: Python tool
def fig_python():
    reliability = {r["model"]: r for r in read_csv(FINAL / "rescore" / "python_reliability_v4.csv")}
    repair = {r["model"]: r for r in read_csv(FINAL / "rescore" / "self_correction_v4.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 1.85), gridspec_kw={"wspace": 0.3})
    width, ok_color, bad_color = 0.36, "#2a78d6", "#eb6834"
    x = range(len(MODELS))
    # (a) resistant when code ran cleanly vs crashed
    ax = axes[0]
    style(ax, "y")
    for i, model in enumerate(MODELS):
        r = reliability[model]
        clean, crash = float(r["clean_R_pct"]), float(r["crashed_R_pct"])
        n_clean, n_crash = int(r["ran_clean"]), int(r["crashed"])
        ax.bar(i - width / 2, clean, width * 0.94, color=ok_color, lw=0)
        ax.bar(i + width / 2, crash, width * 0.94, color=bad_color, lw=0)
        ax.text(i - width / 2, clean + 2, f"{clean:.0f}\n(n={n_clean})", ha="center", va="bottom", fontsize=6.2, color=INK2)
        ax.text(i + width / 2, crash + 2, f"{crash:.0f}\n(n={n_crash})", ha="center", va="bottom", fontsize=6.2, color=INK2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{MODEL_LABEL[m]}\n " for m in MODELS])
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Resistant (%)")
    ax.set_title("(a) Code ran cleanly vs. crashed", loc="left")
    ax.legend(handles=[Patch(color=ok_color, label="code ran"), Patch(color=bad_color, label="code crashed")],
              loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0.0, 1.02))
    # (b) self-correction on the crashed turns
    ax = axes[1]
    style(ax, "y")
    for i, model in enumerate(MODELS):
        r = repair[model]
        turns = int(r["turns"])
        before = 100 * int(r["R_before"]) / turns
        after = float(r["R_after_pct"])
        ax.bar(i - width / 2, before, width * 0.94, color=bad_color, lw=0)
        ax.bar(i + width / 2, after, width * 0.94, color=ok_color, lw=0)
        ax.text(i - width / 2, before + 2, f"{before:.0f}", ha="center", va="bottom", fontsize=6.2, color=INK2)
        ax.text(i + width / 2, after + 2, f"{after:.0f}", ha="center", va="bottom", fontsize=6.2, color=INK2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{MODEL_LABEL[m]}\n(n={repair[m]['turns']})" for m in MODELS])
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("(b) Crashed turns: before vs. after repair", loc="left")
    ax.legend(handles=[Patch(color=bad_color, label="before"), Patch(color=ok_color, label="after error feedback")],
              loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0.0, 1.02))
    fig.savefig(FIG / "fig_python.pdf")
    plt.close(fig)
    return reliability, repair


# ---------------------------------------------------------------- Qwen probe
PROBE_ORDER = [("original", "Original menu (1 = use tool, 2 = answer)"),
               ("swapped_order (1=answer, 2=tool)", "Order swapped (1 = answer, 2 = use tool)"),
               ("name=search_web", "Tool named \\texttt{search\\_web}"),
               ("name=retrieve_answer_sheet", "Tool named \\texttt{retrieve\\_answer\\_sheet}"),
               ("name=lookup_tool (neutral)", "Tool named \\texttt{lookup\\_tool} (neutral)"),
               ("with description", "One-line description of what the tool returns"),
               ("no assistant prefill", "No ``I will choose my next action.'' prefill")]


def probe_table():
    picks = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0]))
    for model in ("qwen", "phi4"):
        path = RES / "qwen_tool_choice_probe_20261110" / "kaggle_output" / "probe" / f"{model}_probe.jsonl"
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            cell = picks[row["variant"]][model]
            # Same rule as the experiment runner (runner.py, choose()): option "1" wins ties
            # (probabilities[0] >= probabilities[1]). p_use is P(tool) normalised over the two
            # digits, and "1" is the tool except in the swapped variant.
            uses_tool = row["p_use"] > 0.5 or (row["p_use"] == 0.5 and not row.get("swap", False))
            cell[0] += int(uses_tool)
            cell[1] += 1
            cell[2] += row["p_use"]
    variants = list(picks)
    return picks, variants


def write_probe_table(picks, variants):
    known = dict(PROBE_ORDER)
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"& \multicolumn{2}{c}{Qwen2.5-3B} & \multicolumn{2}{c}{Phi-4-mini}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Router variant & mean $P(\text{use})$ & picks tool & mean $P(\text{use})$ & picks tool\\", r"\midrule"]
    for variant in [v for v, _ in PROBE_ORDER if v in picks] + [v for v in variants if v not in known]:
        q, p = picks[variant]["qwen"], picks[variant]["phi4"]
        label = known.get(variant, variant.replace("_", r"\_"))
        lines.append(f"{label} & {q[2]/q[1]:.2f} & {q[0]}/{q[1]} & {p[2]/p[1]:.2f} & {p[0]}/{p[1]}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_qwen_probe.tex").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- appendix tables
def write_metrics_table():
    lines = [r"\begin{tabular}{llll rrrr rr}", r"\toprule",
             r"Data & Model & Condition & Turn & R\,\% & S\,\% & O\,\% & U\,\% & Tool\,\% & Override\,\%\\", r"\midrule"]
    previous = None
    for key, r in METRICS.items():
        dataset, model, cond, turn = key
        if previous and (previous[0], previous[1]) != (dataset, model):
            lines.append(r"\midrule")
        tool = r["tool_use_pct"] or "--"
        ovr = r["evidence_override_pct"] or "--"
        cond_tex = cond.replace("T2 web", "T2 web").replace("T3 textbook", "T3 textbook")
        lines.append(f"{dataset} & {MODEL_LABEL[model]} & {cond_tex} & {turn} & {r['resistant_pct']} & "
                     f"{r['flip_pct']} & {r['other_wrong_pct']} & {r['unparsed_pct']} & {tool} & {ovr}\\\\")
        previous = key
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_metrics_full.tex").write_text("\n".join(lines), encoding="utf-8")


def fmt_p(p):
    p = float(p)
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def write_significance_table():
    lines = [r"\begin{tabular}{llll rrr rr l}", r"\toprule",
             r"Data & Model & Tool & Turn & T0 R\% & Tool R\% & $\Delta$ (95\% CI) & +/$-$ & Holm $p$ & Verdict\\",
             r"\midrule"]
    for key, r in SIG.items():
        dataset, model, cond, turn = key
        delta = f"{float(r['diff_R_pp']):+.1f} [{float(r['ci_low_pp']):+.1f}, {float(r['ci_high_pp']):+.1f}]"
        lines.append(f"{dataset} & {MODEL_LABEL[model]} & {COND_SHORT.get(cond, cond)} & {turn} & "
                     f"{r['T0_R_pct']} & {r['tool_R_pct']} & {delta} & {r['improved']}/{r['worsened']} & "
                     f"{fmt_p(r['holm_p'])} & {r['verdict']}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_significance.tex").write_text("\n".join(lines), encoding="utf-8")


def write_week3_tables():
    """Appendix: the earlier 459-question no-tool baseline and the positive-progression run
    (earlier scorer; read from week3_t0_benchmark_report.json and progressive_flip_benchmark_report.json)."""
    week3 = json.load(open(RES / "week3_t0_benchmark_report.json", encoding="utf-8"))["models_evaluated"]
    order = [("qwen", "Qwen"), ("llama", "LLaMA"), ("phi4", "Phi-4")]
    lines = [r"\begin{tabular}{ll r rr rr r}", r"\toprule",
             r"& & P0 & \multicolumn{2}{c}{P1} & \multicolumn{2}{c}{P2} & Flipped at\\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Data & Model & correct\,\% & R\,\% & S\,\% & R\,\% & S\,\% & P1 or P2\,\%\\", r"\midrule"]
    for dataset, label in (("medqa", "MedQA"), ("math", "MATH")):
        for key, model in order:
            m = week3[key]["metrics"]["by_dataset"][dataset]
            p1, p2 = m["p1_suggestive_pressure"], m["p2_authority_pressure"]
            lines.append(f"{label} ({m['total_questions']}) & {MODEL_LABEL[model]} & {m['turn1_baseline_accuracy']:.1f} & "
                         f"{p1['resistance_rate_pct']:.1f} & {p1['flip_rate_pct']:.1f} & "
                         f"{p2['resistance_rate_pct']:.1f} & {p2['flip_rate_pct']:.1f} & "
                         f"{m['cumulative_trajectory']['cumulative_flip_rate_pct']:.1f}\\\\")
        if dataset == "medqa":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_week3.tex").write_text("\n".join(lines), encoding="utf-8")

    progression = json.load(open(RES / "progressive_flip_benchmark_report.json", encoding="utf-8"))
    names = {"Qwen": "Qwen 2.5 3B", "LLaMA": "LLaMA 3.2 3B", "Phi-4": "Phi-4 Mini 3.8B"}
    lines = [r"\begin{tabular}{ll rr rr rr}", r"\toprule",
             r"& & \multicolumn{2}{c}{P1} & \multicolumn{2}{c}{P2} & Corrected at & Never\\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Data & Model & corrected\,\% & kept wrong\,\% & corrected\,\% & kept wrong\,\% & P1 or P2\,\% & corrected\,\%\\",
             r"\midrule"]
    for dataset, label in (("medqa", "MedQA"), ("math", "MATH")):
        for model in MODELS:
            m = progression[names[model]][dataset]
            lines.append(f"{label} ({m['total']}) & {MODEL_LABEL[model]} & {m['p1']['flip_rate_pct']:.1f} & "
                         f"{m['p1']['obstinate_rate_pct']:.1f} & {m['p2']['flip_rate_pct']:.1f} & "
                         f"{m['p2']['obstinate_rate_pct']:.1f} & {m['cumulative_flip_rate_pct']:.1f} & "
                         f"{m['consistently_obstinate_pct']:.1f}\\\\")
        if dataset == "medqa":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_progression.tex").write_text("\n".join(lines), encoding="utf-8")


def write_numbers(reliability, repair):
    """LaTeX macros for numbers quoted in the text, so text and data cannot drift apart."""
    m = lambda d, mo, c, t, f="resistant_pct": METRICS[(d, mo, c, t)][f]
    s = lambda d, mo, c, t, f="diff_R_pp": SIG[(d, mo, c, t)][f]
    sig_count = sum(float(r["holm_p"]) < 0.05 for r in SIG.values())
    better = sum(float(r["holm_p"]) < 0.05 and float(r["diff_R_pp"]) > 0 for r in SIG.values())
    worse = sum(float(r["holm_p"]) < 0.05 and float(r["diff_R_pp"]) < 0 for r in SIG.values())
    total = repair["Total"]
    macros = {
        "nTests": len(SIG), "nSig": sig_count, "nSigBetter": better, "nSigWorse": worse,
        "medqaQwenPtwoTzero": m("MedQA", "Qwen", "T0", "P2"),
        "medqaLlamaPoneTzero": m("MedQA", "LLaMA", "T0", "P1"),
        "medqaLlamaPoneExact": m("MedQA", "LLaMA", "Exact", "P1"),
        "medqaPhiPtwoTzero": m("MedQA", "Phi-4", "T0", "P2"),
        "medqaPhiPtwoTone": m("MedQA", "Phi-4", "T1", "P2"),
        "medqaPhiPtwoExact": m("MedQA", "Phi-4", "Exact", "P2"),
        "medqaPhiPtwoWeb": m("MedQA", "Phi-4", "T2 web", "P2"),
        "medqaPhiPtwoText": m("MedQA", "Phi-4", "T3 textbook", "P2"),
        "medqaQwenPoneWeb": m("MedQA", "Qwen", "T2 web", "P1"),
        "medqaQwenPoneTzero": m("MedQA", "Qwen", "T0", "P1"),
        "mathQwenPtwoTzero": m("MATH", "Qwen", "T0", "P2"),
        "mathQwenPtwoPy": m("MATH", "Qwen", "Python", "P2"),
        "mathLlamaPtwoTzero": m("MATH", "LLaMA", "T0", "P2"),
        "mathLlamaPtwoPy": m("MATH", "LLaMA", "Python", "P2"),
        "mathPhiPtwoTzero": m("MATH", "Phi-4", "T0", "P2"),
        "mathPhiPtwoPy": m("MATH", "Phi-4", "Python", "P2"),
        "mathPhiPtwoExact": m("MATH", "Phi-4", "Exact", "P2"),
        "mathPhiPtwoTone": m("MATH", "Phi-4", "T1", "P2"),
        "mathLlamaPoneTzero": m("MATH", "LLaMA", "T0", "P1"),
        "mathLlamaPonePy": m("MATH", "LLaMA", "Python", "P1"),
        "mathLlamaPoneTone": m("MATH", "LLaMA", "T1", "P1"),
        "mathLlamaPoneExact": m("MATH", "LLaMA", "Exact", "P1"),
        "mathQwenPoneTzero": m("MATH", "Qwen", "T0", "P1"),
        "mathQwenPonePy": m("MATH", "Qwen", "Python", "P1"),
        "deltaQwenPyPtwo": s("MATH", "Qwen", "Python", "P2"),
        "deltaLlamaPyPtwo": s("MATH", "LLaMA", "Python", "P2"),
        "deltaPhiPyPtwo": s("MATH", "Phi-4", "Python", "P2"),
        "deltaLlamaPyPone": s("MATH", "LLaMA", "Python", "P1"),
        "deltaPhiTextPtwo": s("MedQA", "Phi-4", "T3 textbook", "P2"),
        "repairTurns": total["turns"], "repairFixed": total["repaired"],
        "repairRecovered": total["recovered"], "repairNotR": total["not_R_before"],
        "repairRecoveredPct": total["recovered_pct_of_not_R"],
        "pyCleanQwen": reliability["Qwen"]["clean_R_pct"], "pyCrashQwen": reliability["Qwen"]["crashed_R_pct"],
        "pyCleanLlama": reliability["LLaMA"]["clean_R_pct"], "pyCrashLlama": reliability["LLaMA"]["crashed_R_pct"],
        "pyCleanPhi": reliability["Phi-4"]["clean_R_pct"], "pyCrashPhi": reliability["Phi-4"]["crashed_R_pct"],
    }
    text = "% Generated by scripts/build_figures.py from the saved result files. Do not edit by hand.\n"
    text += "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items()) + "\n"
    (TAB / "numbers.tex").write_text(text, encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    fig_heatmap()
    fig_outcomes()
    fig_forest()
    fig_access_trust()
    reliability, repair = fig_python()
    picks, variants = probe_table()
    write_probe_table(picks, variants)
    write_metrics_table()
    write_significance_table()
    write_week3_tables()
    write_numbers(reliability, repair)
    print("probe variants:", {v: {m: picks[v][m][:2] for m in picks[v]} for v in variants})
    print("figures ->", FIG)
    print("tables  ->", TAB)


if __name__ == "__main__":
    main()
