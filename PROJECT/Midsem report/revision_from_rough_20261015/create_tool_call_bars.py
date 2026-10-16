from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = {
    "medqa": ("MedQA", 146, [0, 146, 143], [3, 146, 142]),
    "math": ("MATH", 215, [0, 215, 214], [8, 215, 214]),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
for key, (name, n, p1, p2) in DATA.items():
    fig, ax = plt.subplots(figsize=(6.0, 2.05))
    fig.subplots_adjust(left=.105, right=.985, bottom=.19, top=.79)
    x = np.arange(3)
    width = .29
    for offset, counts, label, color, hatch in [
        (-width/2, p1, "P1: ordinary user", "#3568A8", ""),
        ( width/2, p2, "P2: claimed expertise", "#BB6A25", "//")]:
        bars = ax.bar(x+offset, counts, width, label=label,
                      color=color, hatch=hatch, edgecolor="white", linewidth=.4, zorder=3)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x()+bar.get_width()/2, count+n*.035,
                    f"{count}\n({100*count/n:.1f}%)", ha="center", va="bottom",
                    fontsize=8, linespacing=1.05)
    ax.set_xticks(x, ["Qwen", "LLaMA", "Phi-4"])
    ax.set_ylabel("Tool calls (count)")
    ax.set_ylim(0, n*1.34)
    ax.set_yticks([0,50,100,146] if n==146 else [0,75,150,215])
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linewidth=.55)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(axis="both", length=3)
    fig.suptitle(f"{name}: BM25 tool calls with the tool available (T1)", y=.985,
                 fontsize=10, fontweight="semibold")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.55,.89), ncol=2, frameon=False,
               handlelength=1.5, columnspacing=1.7, fontsize=8.5)
    for extension in ["pdf", "png"]:
        fig.savefig(ROOT/"figures"/f"{key}_tool_calls_bars.{extension}", dpi=220,
                    facecolor="white", metadata={"Creator":"Matplotlib"} if extension=="pdf" else None)
    plt.close(fig)
