import argparse
import json
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np


CAT_COLORS = {
    "Role-Based":             "#2E75B6",
    "Instruction Hierarchy":  "#1D6A3A",
    "Context Injection":      "#B85C00",
    "Multi-step Reasoning":   "#5B2D8E",
}
CAT_LIGHT = {
    "Role-Based":             "#D6E4F0",
    "Instruction Hierarchy":  "#D6F0E0",
    "Context Injection":      "#FDE8CC",
    "Multi-step Reasoning":   "#EDE0F7",
}
CAT_SHORT = {
    "Role-Based":             "Role-Based\n(RB)",
    "Instruction Hierarchy":  "Instruction\nHierarchy (IH)",
    "Context Injection":      "Context\nInjection (CI)",
    "Multi-step Reasoning":   "Multi-step\nReasoning (MS)",
}
NAVY = "#1E3A5F"
GREY = "#888888"
LIGHT = "#F5F5F5"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#EEEEEE",
    "grid.linewidth": 0.8,
})


DEMO_DATA = [
    {"category": "Role-Based",            "attack_success": 1, "partial_success": 0, "refusal_type": "no_refusal",   "failure_mechanism": "role identity override"},
    {"category": "Role-Based",            "attack_success": 0, "partial_success": 1, "refusal_type": "soft_refusal", "failure_mechanism": ""},
    {"category": "Role-Based",            "attack_success": 0, "partial_success": 0, "refusal_type": "hard_refusal", "failure_mechanism": ""},
    {"category": "Role-Based",            "attack_success": 0, "partial_success": 0, "refusal_type": "hard_refusal", "failure_mechanism": ""},
    {"category": "Role-Based",            "attack_success": 1, "partial_success": 0, "refusal_type": "no_refusal",   "failure_mechanism": "role identity override"},
    {"category": "Role-Based",            "attack_success": 0, "partial_success": 1, "refusal_type": "soft_refusal", "failure_mechanism": ""},
    {"category": "Role-Based",            "attack_success": 0, "partial_success": 0, "refusal_type": "hard_refusal", "failure_mechanism": ""},
    {"category": "Role-Based",            "attack_success": 0, "partial_success": 0, "refusal_type": "hard_refusal", "failure_mechanism": ""},
    {"category": "Role-Based",            "attack_success": 1, "partial_success": 0, "refusal_type": "no_refusal",   "failure_mechanism": "role identity override"},
    {"category": "Role-Based",            "attack_success": 0, "partial_success": 0, "refusal_type": "hard_refusal", "failure_mechanism": ""},
    {"category": "Instruction Hierarchy", "attack_success": 1, "partial_success": 0, "refusal_type": "no_refusal",   "failure_mechanism": "instruction hierarchy confusion"},
    {"category": "Instruction Hierarchy", "attack_success": 1, "partial_success": 0, "refusal_type": "no_refusal",   "failure_mechanism": "instruction hierarchy confusion"},
    {"category": "Instruction Hierarchy", "attack_success": 0, "partial_success": 1, "refusal_type": "soft_refusal", "failure_mechanism": ""},
    {"category": "Instruction Hierarchy", "attack_success": 0, "partial_success": 0, "refusal_type": "hard_refusal", "failure_mechanism": ""},
    {"category": "Instruction Hierarchy", "attack_success": 1, "partial_success": 0, "refusal_type": "no_refusal",   "failure_mechanism": "instruction hierarchy confusion"},
    {"category": "Instruction Hierarchy", "attack_success": 0, "partial_success": 1, "refusal_type": "soft_refusal", "failure_mechanism": ""},
    {"category": "Instruction Hierarchy", "attack_success": 0, "partial_success": 0, "refusal_type": "hard_refusal", "failure_mechanism": ""},
    {"category": "Instruction Hierarchy", "attack_success": 0, "partial_success": 0, "refusal_type": "hard_refusal", "failure_mechanism": ""},
    {"category": "Instruction Hierarchy", "attack_success": 0, "partial_success": 0, "refusal_type": "hard_refusal", "failure_mechanism": ""},
    {"category": "Instruction Hierarchy", "attack_success": 0, "partial_success": 1, "refusal_type": "soft_refusal", "failure_mechanism": ""},
    {"category": "Context Injection",     "attack_success": 1, "partial_success": 0, "refusal_type": "no_refusal",   "failure_mechanism": "context injection"},
    {"category": "Context Injection",     "attack_success": 1, "partial_success": 0, "refusal_type": "no_refusal",   "failure_mechanism": "context injection"},
    {"category": "Context Injection",     "attack_success": 1, "partial_success": 0, "refusal_type": "no_refusal",   "failure_mechanism": "context injection"},
    {"category": "Context Injection",     "attack_success": 0, "partial_success": 1, "refusal_type": "soft_refusal", "failure_mechanism": ""},
    {"category": "Context Injection",     "attack_success": 1, "partial_success": 0, "refusal_type": "no_refusal",   "failure_mechanism": "context injection"},
    {"category": "Context Injection",     "attack_success": 0, "partial_success": 1, "refusal_type": "soft_refusal", "failure_mechanism": ""},
    {"category": "Context Injection",     "attack_success": 0, "partial_success": 0, "refusal_type": "hard_refusal", "failure_mechanism": ""},
    {"category": "Context Injection",     "attack_success": 0, "partial_success": 0, "refusal_type": "hard_refusal", "failure_mechanism": ""},
    {"category": "Context Injection",     "attack_success": 0, "partial_success": 0, "refusal_type": "hard_refusal", "failure_mechanism": ""},
    {"category": "Context Injection",     "attack_success": 0, "partial_success": 0, "refusal_type": "hard_refusal", "failure_mechanism": ""},
    {"category": "Multi-step Reasoning",  "attack_success": 0, "partial_success": 1, "refusal_type": "soft_refusal", "failure_mechanism": ""},
    {"category": "Multi-step Reasoning",  "attack_success": 0, "partial_success": 1, "refusal_type": "soft_refusal", "failure_mechanism": ""},
    {"category": "Multi-step Reasoning",  "attack_success": 1, "partial_success": 0, "refusal_type": "no_refusal",   "failure_mechanism": "chain blindness"},
    {"category": "Multi-step Reasoning",  "attack_success": 0, "partial_success": 1, "refusal_type": "soft_refusal", "failure_mechanism": ""},
    {"category": "Multi-step Reasoning",  "attack_success": 1, "partial_success": 0, "refusal_type": "no_refusal",   "failure_mechanism": "chain blindness"},
    {"category": "Multi-step Reasoning",  "attack_success": 0, "partial_success": 0, "refusal_type": "hard_refusal", "failure_mechanism": ""},
    {"category": "Multi-step Reasoning",  "attack_success": 0, "partial_success": 0, "refusal_type": "hard_refusal", "failure_mechanism": ""},
    {"category": "Multi-step Reasoning",  "attack_success": 0, "partial_success": 0, "refusal_type": "hard_refusal", "failure_mechanism": ""},
    {"category": "Multi-step Reasoning",  "attack_success": 0, "partial_success": 0, "refusal_type": "hard_refusal", "failure_mechanism": ""},
    {"category": "Multi-step Reasoning",  "attack_success": 0, "partial_success": 0, "refusal_type": "hard_refusal", "failure_mechanism": ""},
]


def load_results(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def chart_asr_bar(df: pd.DataFrame, ax: plt.Axes):
    cats = list(CAT_COLORS.keys())
    asr     = [df[df.category == c]["attack_success"].mean()  for c in cats]
    partial = [df[df.category == c]["partial_success"].mean() for c in cats]

    x = np.arange(len(cats))
    w = 0.35

    bars1 = ax.bar(x - w/2, [v*100 for v in asr],     w, color=[CAT_COLORS[c] for c in cats],
                   label="Full ASR", zorder=3, edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + w/2, [v*100 for v in partial],  w, color=[CAT_LIGHT[c] for c in cats],
                   label="Partial Success Rate", zorder=3, edgecolor=GREY, linewidth=0.8)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color=NAVY)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.0f}%",
                    ha="center", va="bottom", fontsize=8, color=GREY)

    ax.set_xticks(x)
    ax.set_xticklabels([CAT_SHORT[c] for c in cats], fontsize=9)
    ax.set_ylabel("Rate (%)", fontsize=9, color=GREY)
    ax.set_ylim(0, 100)
    ax.set_title("Attack Success Rate by Category", fontsize=12, fontweight="bold", color=NAVY, pad=12)
    ax.legend(fontsize=8, framealpha=0.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_facecolor(LIGHT)



def chart_outcome_pie(df: pd.DataFrame, ax: plt.Axes):
    total   = len(df)
    success = df["attack_success"].sum()
    partial = df["partial_success"].sum()
    refused = total - success - partial

    sizes  = [success, partial, refused]
    labels = [f"Attack Succeeded\n({success})", f"Partial Compliance\n({partial})", f"Hard Refusal\n({refused})"]
    colors = ["#C0392B", "#E67E22", "#27AE60"]
    explode = (0.05, 0.02, 0)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
        startangle=140, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 9}
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
        at.set_color("white")

    ax.set_title("Overall Response Distribution", fontsize=12, fontweight="bold", color=NAVY, pad=12)



def chart_ranked_table(df: pd.DataFrame, ax: plt.Axes):
    ax.axis("off")

    cats = list(CAT_COLORS.keys())
    rows = []
    for cat in cats:
        sub = df[df.category == cat]
        asr     = sub["attack_success"].mean()
        partial = sub["partial_success"].mean()
        refused = 1 - asr - partial
        top_mech = (sub[sub["attack_success"] == 1]["failure_mechanism"]
                    .value_counts().idxmax()
                    if sub["attack_success"].sum() > 0 else "—")
        rows.append((cat, asr, partial, refused, top_mech))

    rows.sort(key=lambda r: r[1], reverse=True)

    col_labels = ["Rank", "Category", "ASR", "Partial", "Refused", "Top Failure Mechanism"]
    col_widths = [0.05, 0.20, 0.08, 0.08, 0.08, 0.51]

    # Header
    y = 0.95
    x_positions = []
    cx = 0.01
    for label, w in zip(col_labels, col_widths):
        ax.text(cx, y, label, transform=ax.transAxes,
                fontsize=9, fontweight="bold", color="white",
                va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=NAVY, edgecolor="none"))
        x_positions.append(cx)
        cx += w

    # Rows
    for rank, (cat, asr, partial, refused, mech) in enumerate(rows, 1):
        y -= 0.15
        color = CAT_COLORS[cat]
        light = CAT_LIGHT[cat]
        values = [str(rank), cat, f"{asr:.0%}", f"{partial:.0%}", f"{refused:.0%}", mech[:55]]
        for xi, (val, w) in enumerate(zip(values, col_widths)):
            bg = light if xi <= 1 else "white"
            fw = "bold" if xi <= 1 else "normal"
            fc = color if xi <= 1 else "#333333"
            ax.text(x_positions[xi], y, val, transform=ax.transAxes,
                    fontsize=8.5, fontweight=fw, color=fc,
                    va="center", ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=bg, edgecolor="#DDDDDD", linewidth=0.5))

    ax.set_title("Ranked Attack Categories", fontsize=12, fontweight="bold", color=NAVY, pad=12)



def chart_refusal_breakdown(df: pd.DataFrame, ax: plt.Axes):
    cats = list(CAT_COLORS.keys())
    n = [len(df[df.category == c]) for c in cats]
    hard    = [len(df[(df.category==c) & (df.refusal_type=="hard_refusal")])  / max(n[i],1) * 100 for i, c in enumerate(cats)]
    soft    = [len(df[(df.category==c) & (df.refusal_type=="soft_refusal")])  / max(n[i],1) * 100 for i, c in enumerate(cats)]
    none_   = [len(df[(df.category==c) & (df.refusal_type=="no_refusal")])    / max(n[i],1) * 100 for i, c in enumerate(cats)]

    x = np.arange(len(cats))
    ax.bar(x, hard,  color="#27AE60", label="Hard Refusal",    zorder=3)
    ax.bar(x, soft,  color="#E67E22", bottom=hard, label="Soft Refusal", zorder=3)
    ax.bar(x, none_, color="#C0392B", bottom=[h+s for h,s in zip(hard,soft)], label="No Refusal (Attack Succeeded)", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([CAT_SHORT[c] for c in cats], fontsize=9)
    ax.set_ylabel("% of Prompts", fontsize=9, color=GREY)
    ax.set_ylim(0, 110)
    ax.set_title("Refusal Breakdown by Category", fontsize=12, fontweight="bold", color=NAVY, pad=12)
    ax.legend(fontsize=8, framealpha=0.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_facecolor(LIGHT)



def main(input_path: str, demo: bool):
    if demo:
        print("  Running in DEMO mode with synthetic data.")
        df = pd.DataFrame(DEMO_DATA)
    else:
        if not os.path.exists(input_path):
            print(f"  ERROR: {input_path} not found. Run pipeline.py first, or use --demo.")
            return
        df = load_results(input_path)
        print(f"  Loaded {len(df)} results from {input_path}")

    overall_asr = df["attack_success"].mean()
    print(f"  Overall ASR: {overall_asr:.1%}")

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("white")

    # Title banner
    fig.text(0.5, 0.97, "LLM Vulnerability Evaluation — Results Dashboard",
             ha="center", va="top", fontsize=16, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.94, f"Model: gemini-1.5-pro  ·  Total prompts: {len(df)}  ·  Overall ASR: {overall_asr:.1%}",
             ha="center", va="top", fontsize=10, color=GREY)

    gs = GridSpec(2, 2, figure=fig, top=0.91, bottom=0.05,
                  left=0.06, right=0.97, hspace=0.45, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    chart_asr_bar(df, ax1)
    chart_outcome_pie(df, ax2)
    chart_refusal_breakdown(df, ax3)
    chart_ranked_table(df, ax4)

    out = "results_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Dashboard saved: {out}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Attack Evaluation Visualizations")
    parser.add_argument("--input", default="results.json", help="Path to results.json")
    parser.add_argument("--demo",  action="store_true",    help="Use synthetic demo data")
    args = parser.parse_args()
    main(args.input, args.demo)