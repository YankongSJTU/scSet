# -*- coding: utf-8 -*-
"""Backbone (scGPT) orthogonal validation figure — journal panels + editable PPTX.
(a) 双基座并排条形：STATE vs scGPT 四配置 mean
(b) scGPT 配对逐队列差 vs fm_sup（strip）
(c) 身份/语义镜像反转：STATE vs scGPT fm_mean
数据：results_backbone_pulled/（本地直算），与主文 NPG 配色一致。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results_backbone_pulled"
PANELS = ROOT / "figures_en_20261106" / "panels"
PANELS.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED = "#ffffff", "#1a1a1a", "#52514e", "#8a8880"
DEEP, GRID = "#22355c", "#e1e0d9"
C_ENC, C_RED, C_GOLD, C_TEAL, C_VIOL = "#355191", "#e64b35", "#d9a020", "#00a087", "#4dbbd5"

rcParams.update({
    "font.sans-serif": ["Arial", "DejaVu Sans"], "font.family": "sans-serif",
    "font.size": 10, "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
})

df = pd.read_csv(RES / "cohort_rows.csv")
df = df[df.cohort.notna()].drop_duplicates(subset=["method", "cohort", "round", "seed"])
wide = df.groupby(["method", "cohort"])["cohort_balanced_accuracy"].mean().unstack(0)

# STATE 对照值（主文记录）
state_vals = {"Pooled mean": 0.488, "Sup pooled (control)": 0.536,
              "scSet-CE": 0.596, "scSet-Contrastive": 0.555}
scg_vals = {"Pooled mean": float(wide["scgpt_fm_mean"].mean()),
            "Sup pooled (control)": float(wide["scgpt_fm_sup_mean_mlp"].mean()),
            "scSet-CE": float(wide["scgpt_scset"].mean()),
            "scSet-Contrastive": float(wide["scgpt_scset_contrast"].mean())}

# ---------------- (a) 双基座并排 ----------------
fig, ax = plt.subplots(figsize=(5.6, 3.7), dpi=200)
names = list(state_vals)
x = np.arange(len(names))
w = 0.36
cols = [C_GOLD, C_TEAL, C_ENC, C_RED]
b1 = ax.bar(x - w / 2, [state_vals[n] for n in names], w, color=cols,
            alpha=0.45, edgecolor=SURFACE, linewidth=0.8, label="STATE (2,058-d)")
b2 = ax.bar(x + w / 2, [scg_vals[n] for n in names], w, color=cols,
            edgecolor=SURFACE, linewidth=0.8, label="scGPT (512-d)")
for xi, n in zip(x, names):
    ax.text(xi - w / 2, state_vals[n] + 0.012, f"{state_vals[n]:.3f}",
            ha="center", fontsize=8.5, color=INK2)
    ax.text(xi + w / 2, scg_vals[n] + 0.012, f"{scg_vals[n]:.3f}",
            ha="center", fontsize=8.5, color=INK, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["fm_mean", "fm_sup_mean_mlp", "scSet-CE", "scSet-Contrastive"],
                   fontsize=9.5, rotation=14, ha="right")
ax.set_ylim(0, 0.72)
ax.set_ylabel("Cohort-level balanced accuracy", fontsize=10.5)
ax.set_title("Orthogonal backbone check · 42 cohorts, same folds & probes",
             loc="left", fontsize=12, pad=12)
ax.legend(loc="upper right", fontsize=9)
ax.yaxis.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0, -1, "", fontsize=1)
fig.tight_layout()
fig.savefig(PANELS / "fig12a_backbone.png", bbox_inches="tight")
plt.close(fig)

# ---------------- (b) scGPT 配对差 strip ----------------
fig, ax = plt.subplots(figsize=(4.2, 3.7), dpi=200)
base = wide["scgpt_fm_sup_mean_mlp"]
for i, (m, c) in enumerate([("scgpt_fm_mean", C_GOLD),
                            ("scgpt_scset", C_ENC),
                            ("scgpt_scset_contrast", C_RED)]):
    j = pd.concat([wide[m], base], axis=1).dropna()
    d = j.iloc[:, 0] - j.iloc[:, 1]
    try:
        p = wilcoxon(j.iloc[:, 0], j.iloc[:, 1]).pvalue
    except ValueError:
        p = float("nan")
    yi = 2 - i
    ax.scatter(d, np.full(len(d), yi) + np.random.default_rng(i).normal(0, 0.05, len(d)),
               s=16, color=c, alpha=0.55, edgecolors=SURFACE, linewidths=0.5, zorder=3)
    ax.hlines(yi, d.min(), d.max(), color=c, lw=1.0, alpha=0.4, zorder=2)
    ax.plot([d.mean(), d.mean()], [yi - 0.22, yi + 0.22], color=c, lw=2.6, zorder=4)
    ax.text(0.30, yi, f"Δ {d.mean():+.3f}  p={p:.3f}", fontsize=9.5,
            va="center", color=INK,
            bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=GRID, lw=0.6))
ax.axvline(0, color=INK2, lw=0.9)
ax.set_yticks([2, 1, 0])
ax.set_yticklabels(["fm_mean", "scSet", "scSet-Contrastive"], fontsize=10)
ax.set_xlabel("Δ balanced accuracy vs fm_sup_mean_mlp (paired, 42 cohorts)",
              fontsize=10)
ax.set_xlim(-0.42, 0.55)
ax.set_title("Paired deltas on scGPT", loc="left", fontsize=12, pad=12)
ax.xaxis.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0, -1, "", fontsize=1)
fig.tight_layout()
fig.savefig(PANELS / "fig12b_backbone.png", bbox_inches="tight")
plt.close(fig)

# ---------------- (c) 身份-语义镜像反转 ----------------
fig, ax = plt.subplots(figsize=(4.6, 3.7), dpi=200)
metrics = ["Self-retrieval P@1\n(identity)", "Disease-neighbour P@1\n(semantics)"]
state_m = [0.999, 0.379]
scg_m = [0.0, 0.824]
x = np.arange(2)
w = 0.36
ax.bar(x - w / 2, state_m, w, color=C_GOLD, alpha=0.85, label="STATE fm_mean")
ax.bar(x + w / 2, scg_m, w, color=C_VIOL, label="scGPT fm_mean")
for xi, (sv, cv) in zip(x, zip(state_m, scg_m)):
    ax.text(xi - w / 2, sv + 0.02, f"{sv:.3f}", ha="center", fontsize=9.5, color=INK2)
    ax.text(xi + w / 2, cv + 0.02, f"{cv:.3f}", ha="center", fontsize=9.5,
            color=INK, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel("P@1 (64-cell query → 256-cell library)", fontsize=10)
ax.set_title("The antinomy inverts across backbones", loc="left", fontsize=12, pad=12)
ax.legend(loc="upper center", fontsize=9)
ax.yaxis.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0, -1, "", fontsize=1)
fig.tight_layout()
fig.savefig(PANELS / "fig12c_backbone.png", bbox_inches="tight")
plt.close(fig)
print("panels saved: fig12a/b/c_backbone.png")
