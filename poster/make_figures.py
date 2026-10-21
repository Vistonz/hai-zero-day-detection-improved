"""Figures for the plant-version-shift poster, built from the recorded CSVs."""
from __future__ import annotations
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "fig"; OUT.mkdir(exist_ok=True)

# Validated categorical palette, slots 1-3 (all-pairs safe).
C = {"pca": "#2a78d6", "svdd": "#eb6834", "vae": "#1baf7a"}
LAB = {"pca": "PCA", "svdd": "Deep SVDD", "vae": "VAE"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.labelcolor": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})

rows = list(csv.DictReader(open(ROOT / "data/metrics_summary.csv")))
def g(r, k):
    try: return float(r[k + "_mean"])
    except (KeyError, ValueError): return float("nan")

# ---------------------------------------------------------------- fig A: PR-AUC drop
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for i, m in enumerate(["pca", "svdd", "vae"]):
    a = next(r for r in rows if r["model"] == m and r["domain"] == "21.03" and r["arm"] == "frozen")
    b = next(r for r in rows if r["model"] == m and r["domain"] == "23.05" and r["arm"] == "frozen")
    y = len(C) - 1 - i
    ax.plot([g(a, "pr_auc"), g(b, "pr_auc")], [y, y], color=C[m], lw=3, alpha=0.55, zorder=2)
    ax.plot(g(a, "pr_auc"), y, "o", ms=13, color=C[m], zorder=3)
    ax.plot(g(b, "pr_auc"), y, "o", ms=13, color="white", mec=C[m], mew=3, zorder=3)
    ax.annotate(f"{g(a,'pr_auc'):.2f}", (g(a, "pr_auc"), y + 0.26), ha="center",
                fontsize=13, color=INK, fontweight="bold")
    ax.annotate(f"{g(b,'pr_auc'):.2f}", (g(b, "pr_auc"), y + 0.26), ha="center",
                fontsize=13, color=C[m], fontweight="bold")
ax.set_yticks(range(3)); ax.set_yticklabels([LAB[m] for m in ["vae", "svdd", "pca"]], fontsize=15)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 0.60); ax.set_ylim(-0.6, 2.55)
ax.set_xlabel("PR-AUC (threshold-free)", fontsize=15)
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
fig.text(0.5, -0.06, "filled = 21.03 in-domain   hollow = 23.05 cross-version",
         ha="center", fontsize=13, color=MUTED)
fig.tight_layout(); fig.savefig(OUT / "fig_prauc.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- fig B: the collapse
fig, ax = plt.subplots(figsize=(6.8, 4.3))
ax.axvspan(1000, 30000, color="#e34948", alpha=0.10, lw=0)
ax.axvline(10, color=MUTED, ls="--", lw=1.8)
ax.annotate("target 10/day", (9.3, 0.06), rotation=90,
            ha="right", va="bottom", fontsize=11, color=MUTED)
ax.annotate("alarm collapse", (5500, 1.27), ha="center", fontsize=13,
            color="#b3312f", fontweight="bold")

def P(m, dom):
    r = next(x for x in rows if x["model"] == m and x["domain"] == dom and x["arm"] == "frozen")
    return g(r, "false_alarms_per_day"), g(r, "attack_detection_rate")

# 21.03 points
for m in ("pca", "svdd", "vae"):
    fa, rec = P(m, "21.03")
    ax.plot(fa, rec, "o", ms=14, color=C[m], zorder=3)
# 23.05: VAE honest, PCA and SVDD identical -> drawn side by side and labelled together
fa, rec = P("vae", "23.05")
ax.plot(fa, rec, "o", ms=14, mfc="white", mec=C["vae"], mew=3, zorder=3)
ax.annotate("VAE 0.154", (fa, rec), xytext=(15, 0), textcoords="offset points",
            ha="left", va="center", fontsize=12, color=C["vae"], fontweight="bold")

fa, rec = P("pca", "23.05")
ax.plot(fa * 0.72, rec, "o", ms=14, mfc="white", mec=C["pca"], mew=3, zorder=3)
ax.plot(fa * 1.38, rec, "o", ms=14, mfc="white", mec=C["svdd"], mew=3, zorder=3)
ax.annotate("PCA and Deep SVDD\nrecall 1.000 at 8297 alarms/day", (fa, rec),
            xytext=(0, -46), textcoords="offset points", ha="center", fontsize=11.5,
            color="#b3312f")

ax.set_xscale("log"); ax.set_xlim(3, 3e4); ax.set_ylim(0, 1.32)
ax.set_xlabel("observed false alarms per day (log)", fontsize=15)
ax.set_ylabel("event detection rate", fontsize=15)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
handles = [plt.Line2D([], [], marker="o", ls="", ms=12, color=C[m], label=LAB[m]) for m in C]
handles += [plt.Line2D([], [], marker="o", ls="", ms=12, color=MUTED, label="21.03 (filled)"),
            plt.Line2D([], [], marker="o", ls="", ms=12, mfc="white", mec=MUTED, mew=2,
                       label="23.05 (hollow)")]
ax.legend(handles=handles, frameon=False, fontsize=11.5, ncol=5,
          loc="upper center", bbox_to_anchor=(0.5, 1.24))
fig.tight_layout(); fig.savefig(OUT / "fig_collapse.png", dpi=220, bbox_inches="tight"); plt.close(fig)

for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}: {f.stat().st_size/1024:.0f} KB")
