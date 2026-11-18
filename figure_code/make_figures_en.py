# -*- coding: utf-8 -*-
"""
中文稿结果图集 v2：汇报级版式。
- PPTX 层：深色标题带 + stat tile 数字卡片行 + 面板区 + takeaway 结论条 + 页码
- 面板层：条形叠加逐队列分布点、置信带、显著性括号线、面板字母徽章
数据来源与 v1 相同（本地直算 + STATUS.md 记录值硬编码）。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures_en_20261106"
PANELS = OUT / "panels"
PANELS.mkdir(parents=True, exist_ok=True)

# ---------------- palette (NPG-derived, validator-passed 2026-11-10) ----------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
DEEP = "#22355c"        # 标题带 / 强调深蓝
WASH = "#e9eef5"        # takeaway 浅底
C_CCE = "#355191"       # 深蓝：CCE 改进系
C_FULL = "#e64b35"      # 红：原始模型 / 有害变体
C_SUP = "#00a087"       # 青：sup_mean_mlp
C_STATEm = "#d9a020"    # 金：state_mean pooled对照
C_EXT = "#8e6fbf"       # 紫：外部深度方法（scVI / scPoli）
C_GRAY = "#b3b8c0"      # 去强调base
C_VIOL = "#4dbbd5"      # 浅蓝：concat / 几何语义轴

rcParams.update({
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": BASE,
    "axes.linewidth": 0.9,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "legend.fontsize": 11,
    "axes.titlesize": 14,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
})

PANELS_MADE = []


def new_ax(w=6.4, h=3.1):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PANELS / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    PANELS_MADE.append(f"{name}.png")


def badge(ax, letter, x=-0.14, y=1.04):
    """面板字母徽章（圆角方块 + 白字）。"""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=13, fontweight="bold",
            color="white", ha="center", va="center", zorder=12,
            bbox=dict(boxstyle="round,pad=0.32", fc=DEEP, ec="none"))


def barh(ax, labels, values, colors, letter=None, title="",
         xlabel="Cohort-level balanced accuracy", fmt="{:.3f}", vline=None, vline_label=None,
         dots=None, dot_color=None):
    """水平条形（值标条尾）+ 可选逐队列分布点叠加（dots: 与 labels 对齐的 Series 列表）。"""
    n = len(labels)
    y = np.arange(n)[::-1]
    ax.barh(y, values, height=0.60, color=colors, zorder=2)
    if dots is not None:
        rng = np.random.default_rng(11)
        for yi, col in zip(y, dots):
            if col is None:
                continue
            xs = np.asarray(col, dtype=float)
            spread = max(values) * 0.011
            ax.scatter(xs + rng.uniform(-spread, spread, len(xs)),
                       np.full(len(xs), yi), s=15, zorder=3, linewidths=0,
                       color=dot_color or INK, alpha=0.30)
            m = np.mean(xs)
            ax.plot([m, m], [yi - 0.30, yi + 0.30], color=INK, lw=1.8, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11.5, color=INK)
    ax.set_xlabel(xlabel, fontsize=11.5)
    if title:
        ax.set_title(title, loc="left", pad=14 if letter else 6)
    lo = min(0.0, min(values))
    span = max(values) - lo
    ax.set_xlim(lo, max(values) + span * 0.24 + 0.02)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    for yi, v in zip(y, values):
        off = span * 0.016 if v >= 0 else -span * 0.016
        ax.text(v + off, yi, fmt.format(v), va="center",
                ha="left" if v >= 0 else "right", fontsize=11, color=INK,
                fontweight="bold")
    if vline is not None:
        ax.axvline(vline, color=MUTED, lw=1.1)
        if vline_label:
            ax.text(vline, n - 0.30, " " + vline_label, fontsize=10.5, color=MUTED,
                    ha="left", va="center")
    if letter:
        badge(ax, letter)


def strip(ax, deltas, letter=None, title="", label="Δ (per-cohort paired)",
          note=None):
    """逐队列配对 Δ：分布点 + 均值线 + 四分位带。"""
    x = np.random.default_rng(7).uniform(-0.17, 0.17, len(deltas))
    q1, q3 = np.percentile(deltas, [25, 75])
    ax.axvspan(q1, q3, color=C_CCE, alpha=0.10, zorder=1)
    ax.axvline(0, color=BASE, lw=1.0, zorder=2)
    ax.scatter(x, deltas, s=22, color=C_CCE, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    m = float(np.mean(deltas))
    ax.axhline(m, color=C_FULL, lw=2.0, zorder=4)
    ax.text(0.55, m, f"mean {m:+.3f}", fontsize=11.5, color=C_FULL, va="bottom",
            fontweight="bold")
    ax.set_xticks([])
    ax.set_ylabel(label, fontsize=11.5)
    ax.set_xlim(-0.55, 0.85)
    ax.set_title(title, loc="left", pad=14 if letter else 6)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    if note:
        ax.text(0.03, 0.045, note, transform=ax.transAxes, fontsize=11, color=INK2)
    if letter:
        badge(ax, letter)


def sig_bracket(ax, x0, x1, y, text):
    """显著性括线（竖直比较用）：x0/x1 为两个条的中心。"""
    h = y * 0.035
    ax.plot([x0, x0, x1, x1], [y - h, y, y, y0 := y, ], color=INK2, lw=1.0)
    # 简化：直接画横线 + 文本
    ax.plot([x0, x1], [y, y], color=INK2, lw=1.0)
    ax.text((x0 + x1) / 2, y + h * 0.25, text, ha="center", fontsize=10.5, color=INK2)


# ---------------- data loading ----------------
full_wide = pd.read_csv(ROOT / "results_full_pulled/cohort_level_balanced_accuracy_full.csv")
SUMMARY = json.load(open(ROOT / "results_full_pulled/summary_full.json"))
full_means = {m: d["mean_balanced_accuracy"] for m, d in SUMMARY["cohort_level_summary"].items()}
fig5b = pd.read_csv(ROOT / "figures/source_data/fig5b_external_tasks.csv")
fig3c = pd.read_csv(ROOT / "figures/source_data/fig3c_retrieval_precision_at_k.csv")
countsweep = pd.read_csv(ROOT / "results_countsweep_pulled/summary.csv")
countsweep_coh = pd.read_csv(ROOT / "results_countsweep_pulled/cohort_rows.csv")
dimsat = pd.read_csv(ROOT / "results_dimsat_pulled/dimsat_summary.csv")
scvi_rows = pd.read_csv(ROOT / "results_scvi_pulled/cohort_rows.csv")
geom = pd.read_csv(ROOT / "figures/source_data/fig4_geometry_summary.csv")


def runs_means(run_dir, prefix):
    d = ROOT / run_dir
    rows = []
    for rd in sorted(d.glob(f"{prefix}_r*_s*_c*")):
        f = rd / "cohort_rows.csv"
        if f.exists():
            t = pd.read_csv(f)
            if len(t):
                rows.append(t["cohort_balanced_accuracy"].mean())
    return float(np.mean(rows)) if rows else np.nan


def runs_cohort_wide(run_dir, prefix):
    d = ROOT / run_dir
    rows = []
    for rd in sorted(d.glob(f"{prefix}_r*_s*_c*")):
        f = rd / "cohort_rows.csv"
        if f.exists():
            t = pd.read_csv(f)
            if len(t):
                rows.append(t)
    return pd.concat(rows).groupby("cohort")["cohort_balanced_accuracy"].mean()


# ---------------- 图1 主基准 ----------------
order1 = ["cce_ce", "cce_long", "cce_full", "sup_mean_mlp", "pseudobulk",
          "pca256", "state_mean", "state_moments", "composition"]
colmap1 = {"cce_ce": C_CCE, "cce_long": C_CCE, "cce_full": C_FULL,
           "sup_mean_mlp": C_SUP, "state_mean": C_STATEm}
LAB = {"cce_ce": "scSet-CE", "cce_long": "scSet-CE×2",
       "cce_full": "scSet-Contrastive", "cce_end2end": "scSet (e2e)",
       "notrans": "scSet"}
cols1 = [colmap1.get(m, C_GRAY) for m in order1]

# 花瓣图：9 方法一瓣（半径=Cohort-level balanced accuracy）
fig = plt.figure(figsize=(5.2, 4.6), dpi=200)
axp = fig.add_subplot(111, polar=True)
fig.subplots_adjust(top=0.84, bottom=0.06, left=0.10, right=0.92)
vals1 = [full_means[m] for m in order1]
n_pet = len(order1)
ang1 = np.linspace(0, 2 * np.pi, n_pet, endpoint=False) + np.pi / n_pet
axp.bar(ang1, vals1, width=2 * np.pi / n_pet * 0.62, bottom=0.0, color=cols1,
        edgecolor=SURFACE, linewidth=1.5, zorder=3)
# 选择性数值标注：仅四个关键角色（其余靠径向刻度）
KEY = {"cce_ce", "sup_mean_mlp", "state_mean", "cce_full"}
for a, v, m in zip(ang1, vals1, order1):
    if m in KEY:
        axp.text(a, v + 0.048, f"{v:.3f}", ha="center", va="center",
                 fontsize=11, color=INK, fontweight="bold")
# 方法名两圈交替，避免相邻碰撞
for i, (a, m) in enumerate(zip(ang1, order1)):
    deg = float(np.degrees(a)) % 360
    if 90 < deg < 270:
        rot, ha = deg + 180, "right"
    else:
        rot, ha = deg, "left"
    axp.text(a, 0.82 if i % 2 == 0 else 0.95, LAB.get(m, m), rotation=rot,
             rotation_mode="anchor", ha=ha, va="center", fontsize=10.5, color=INK2)
axp.set_ylim(0, 1.04)
axp.set_yticks([0.2, 0.4, 0.6])
axp.set_yticklabels(["0.2", "0.4", "0.6"], fontsize=9, color=MUTED)
axp.set_xticks([])
axp.grid(True, axis="y", color=GRID, lw=0.7)
axp.spines["polar"].set_color(GRID)
axp.set_facecolor(SURFACE)
fig.text(0.02, 0.97, "Pre-registered main endpoint · petal plot", fontsize=14, fontweight="bold", color=INK)
fig.text(0.02, 0.925, "42 cohorts × 6 rounds × 3 seeds (n=1,134 runs); scSet-CE vs sup +0.060 (Holm p=0.0138)",
         fontsize=9.5, color=INK2)
save(fig, "fig1a_petal")

fig, ax = new_ax(3.9, 3.55)
d = full_wide["cce_ce"] - full_wide["sup_mean_mlp"]
strip(ax, d.values, letter="b", title="Paired difference · scSet-CE − sup_mean_mlp",
      label="Δ balanced accuracy")
save(fig, "fig1b_paired")

# c/d 双联面板：检索 + 协议反转
fig, (axc, axd) = plt.subplots(1, 2, figsize=(7.6, 3.5), dpi=200)
for a in (axc, axd):
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.set_facecolor(SURFACE)
ret = fig3c.groupby("method")[["P1", "P5"]].mean().loc[["cce_ce", "cce_long", "sup_mean_mlp", "state_mean"]]
x = np.arange(len(ret))
w = 0.35
cols_r = [C_CCE, C_CCE, C_SUP, C_STATEm]
axc.bar(x - w / 2, ret["P1"], w, color=cols_r, label="P@1")
axc.bar(x + w / 2, ret["P5"], w, color=cols_r, alpha=0.42, label="P@5")
for xi, v in zip(x - w / 2, ret["P1"]):
    axc.text(xi, v + 0.016, f"{v:.3f}", ha="center", fontsize=10, color=INK, fontweight="bold")
for xi, v in zip(x + w / 2, ret["P5"]):
    axc.text(xi, v + 0.016, f"{v:.3f}", ha="center", fontsize=9, color=INK2)
axc.set_xticks(x)
axc.set_xticklabels([LAB.get(m, m) for m in ret.index], fontsize=10,
                    rotation=18, ha="right")
axc.set_ylim(0, 0.92)
axc.set_ylabel("Sample retrieval precision", fontsize=11)
axc.set_title("Disease-neighbour retrieval", loc="left", fontsize=12.5, pad=12)
axc.legend(loc="lower right", ncols=2, fontsize=9.5)
axc.yaxis.grid(True)
axc.set_axisbelow(True)
badge(axc, "", x=-0.16, y=1.06)

proto = ["In-atlas diagnostic\n (train-set classification)", "Cohort-held-out transfer\n (pre-registered)"]
cce_v = [0.998, full_means["cce_ce"]]
st_v = [0.966, full_means["state_mean"]]
x2 = np.arange(2)
w2 = 0.33
axd.bar(x2 - w2 / 2, cce_v, w2, color=C_CCE, label="scSet-CE")
axd.bar(x2 + w2 / 2, st_v, w2, color=C_STATEm, label="STATE mean")
for xi, v in zip(x2 - w2 / 2, cce_v):
    axd.text(xi, v + 0.018, f"{v:.3f}", ha="center", fontsize=10.5, color=INK, fontweight="bold")
for xi, v in zip(x2 + w2 / 2, st_v):
    axd.text(xi, v + 0.018, f"{v:.3f}", ha="center", fontsize=10.5, color=INK2)
axd.set_xticks(x2)
axd.set_xticklabels(proto, fontsize=10.5)
axd.set_ylim(0, 1.2)
axd.set_ylabel("Accuracy / balanced accuracy", fontsize=11)
axd.set_title("The protocol decides the ranking", loc="left", fontsize=12.5, pad=12)
axd.annotate("Leaked evaluation inflates both (>0.96)", xy=(0.5, 0.82), xytext=(0.52, 1.06),
             fontsize=10, color=INK2, ha="center",
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
axd.legend(loc="lower left", fontsize=9.5)
axd.yaxis.grid(True)
axd.set_axisbelow(True)
badge(axd, "", x=-0.16, y=1.06)
fig.tight_layout()
fig.savefig(PANELS / "fig1cd_retrieval_gap.png", bbox_inches="tight")
plt.close(fig)
PANELS_MADE.append("fig1cd_retrieval_gap.png")

# ---------------- 图2 comparator ----------------
scvi_mean = scvi_rows["cohort_balanced_accuracy"].mean()
cmp_order = ["cce_ce", "sup_mean_mlp", "pseudobulk", "pca256", "state_mean",
             "scPoli*", "scvi_mean", "composition"]
cmp_vals = [full_means["cce_ce"], full_means["sup_mean_mlp"], full_means["pseudobulk"],
            full_means["pca256"], full_means["state_mean"], 0.458, scvi_mean,
            full_means["composition"]]
cmp_cols = [C_CCE, C_SUP, C_GRAY, C_GRAY, C_STATEm, C_EXT, C_EXT, C_GRAY]
cmp_dots = [full_wide[m].values if m in full_wide.columns else None for m in cmp_order]
sc = scvi_rows.groupby("cohort")["cohort_balanced_accuracy"].mean()
cmp_dots[6] = sc.reindex(full_wide["cohort"]).values
fig, ax = new_ax(7.2, 3.55)
barh(ax, [LAB.get(m, m) for m in cmp_order], cmp_vals, cmp_cols, letter="a",
     title="Comparator matrix · 42-cohort held-out transfer", dots=cmp_dots)
ax.text(0.05, 6.6, "* scPoli: 66-sample subset + exposure caveat; scvi_mean: scArches, zero exposure per fold",
        fontsize=10.5, color=MUTED, va="center")
save(fig, "fig2a_comparator")

fig, ax = new_ax(3.9, 3.55)
bars = ["scPoli in-atlas", "scPoli held-out", "state_mean (subset)", "scSet-CE (subset)"]
vals = [0.988, 0.458, 0.667, 0.847]
cols = [C_EXT, C_EXT, C_STATEm, C_CCE]
y = np.arange(4)[::-1]
ax.barh(y, vals, height=0.60, color=cols, zorder=2)
ax.set_yticks(y)
ax.set_yticklabels(bars, fontsize=11)
for yi, v in zip(y, vals):
    ax.text(v + 0.015, yi, f"{v:.3f}", va="center", fontsize=11, color=INK, fontweight="bold")
ax.set_xlim(0, 1.22)
ax.set_xlabel("Accuracy / balanced accuracy", fontsize=11.5)
ax.set_title("scPoli: 0.988 → 0.458", loc="left", pad=14)
badge(ax, "")
ax.text(0.47, 2.42, "66 samples / 3 cancers / 6 folds\natlas saw held-out cohorts (in its favour)",
        fontsize=10.5, color=MUTED)
ax.xaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "fig2b_scpoli")

fig, ax = new_ax(3.9, 3.55)
merged = pd.concat([sc.rename("scvi"), full_wide.set_index("cohort")["state_mean"]], axis=1).dropna()
strip(ax, (merged["scvi"] - merged["state_mean"]).values, letter="c",
      title="Paired difference · scvi_mean − state_mean", label="Δ balanced accuracy",
      note="Wilcoxon p=0.002 (recorded)")
save(fig, "fig2c_scvi")

# ---------------- 图3 架构 ----------------
ab = {}
ab_coh = {}
for v in ["ab_ce_notrans", "ab_ce_nogat", "ab_ce_meanpool", "ab_ce_nolineage", "ab_ce_minimal"]:
    ab[v] = runs_means("runs_ablation_pulled/runs_ablation", v)
    ab_coh[v] = runs_cohort_wide("runs_ablation_pulled/runs_ablation", v)
arch_labels = ["scSet (GAT only)", "Transformer mixer (no GAT)", "mean pooling only",
               "no lineage covariates", "scSet-CE (full architecture)", "minimal stack (no mixing)"]
arch_vals = [ab["ab_ce_notrans"], ab["ab_ce_nogat"], ab["ab_ce_meanpool"],
             ab["ab_ce_nolineage"], full_means["cce_ce"], ab["ab_ce_minimal"]]
arch_cols = [C_CCE, C_CCE, C_GRAY, C_GRAY, C_FULL, C_GRAY]
fw = full_wide.set_index("cohort")
arch_dots = [ab_coh["ab_ce_notrans"].values, ab_coh["ab_ce_nogat"].values,
             ab_coh["ab_ce_meanpool"].values, ab_coh["ab_ce_nolineage"].values,
             fw["cce_ce"].values, ab_coh["ab_ce_minimal"].values]
fig, ax = new_ax(7.2, 3.55)
barh(ax, arch_labels, arch_vals, arch_cols, letter="a",
     title="Architecture grid · d256, 18 runs/variant, 42 cohorts", dots=arch_dots)
ax.text(0.35, 4.8, "scSet vs scSet-CE +0.027 (p=0.0012)\nminimal trails any mixing mechanism by 0.06+",
        fontsize=11.5, color=INK2, va="center")
save(fig, "fig3a_arch")

fig, ax = new_ax(3.8, 3.55)
nt = runs_cohort_wide("runs_ablation_pulled/runs_ablation", "ab_ce_notrans")
mm = pd.concat([nt.rename("nt"), fw["cce_ce"]], axis=1).dropna()
strip(ax, (mm["nt"] - mm["cce_ce"]).values, letter="b",
      title="Paired difference · scSet − scSet-CE", label="Δ balanced accuracy")
save(fig, "fig3b_paired")

fig, ax = new_ax(3.8, 3.55)
in_labels = ["scSet × d1024", "scSet (d256 ref)", "global state token", "lineage identity token"]
in_vals = [runs_means("runs_ablation_pulled/runs_ablation", "ab_ce_notrans_w"),
           ab["ab_ce_notrans"],
           runs_means("runs_ablation_pulled/runs_ablation", "ab_ce_ctx"),
           runs_means("runs_ablation_pulled/runs_ablation", "ab_ce_lintok")]
barh(ax, in_labels, in_vals, [C_CCE, C_CCE, C_FULL, C_FULL], letter="c",
     title="Input-side enrichment (3 variants)")
ax.text(0.46, 1.1, "token enrichment is harmful · d1024 +0.005 ns",
        fontsize=11, color=INK2, va="center")
save(fig, "fig3c_input")

# ---------------- 图4 损失 ----------------
fig, ax = new_ax(4.3, 3.55)
loss_labels = ["CE + cohort-adversarial", "pure CE (scSet)", "CE + 0.5·contrastive"]
loss_vals = [runs_means("runs_ablation_pulled/runs_ablation", "ab_ce_dann"),
             ab["ab_ce_notrans"],
             runs_means("runs_ablation_pulled/runs_ablation", "ab_ce_hyb")]
barh(ax, loss_labels, loss_vals, [C_CCE, C_CCE, C_FULL], letter="a",
     title="Loss grid (scSet base)")
ax.text(0.36, 1.2, "hybrid vs pure CE −0.049 (p=0.008)\npure CE vs sup +0.088 (p=0.0009)",
        fontsize=11, color=INK2, va="center")
save(fig, "fig4a_loss")

fig, ax = new_ax(4.3, 3.55)
bp_labels = ["twin head d256", "twin head d1024", "variance-spreading ×d1024",
             "hierarchical cohort supervision",
             "variance-spreading (λ=2)", "variance-spreading (λ=0.5)"]
bp_vals = [runs_means("runs_ablation_pulled/runs_ablation", "ab_twin"),
           runs_means("runs_ablation_pulled/runs_ablation", "ab_twin_w"),
           runs_means("runs_ablation_pulled/runs_ablation", "ab_wcv_w"),
           runs_means("runs_ablation_pulled/runs_ablation", "ab_hier"),
           runs_means("runs_ablation_pulled/runs_ablation", "ab_ce_wcvar20"),
           runs_means("runs_ablation_pulled/runs_ablation", "ab_ce_wcvar05")]
barh(ax, bp_labels, bp_vals, [C_FULL] * 6, letter="b",
     title="Four orthogonal within-class mechanisms (plan B)",
     vline=0.60, vline_label="threshold 0.60")
ax.text(0.32, 2.2, "variance-spreading vs scSet-CE −0.080 (p=0.0036)",
        fontsize=11, color=INK2, va="center")
save(fig, "fig4b_planb")

fig, ax = new_ax(4.3, 3.55)
b48_labels = ["scSet@2048", "label smoothing", "crop consistency", "masked pre-training"]
b48_vals = [runs_means("runs_cells2048_pulled/runs_cells2048", "ab_ce_notrans"),
            runs_means("runs_cells2048_pulled/runs_cells2048", "ab_ce_ls"),
            runs_means("runs_cells2048_pulled/runs_cells2048", "ab_ce_msc"),
            runs_means("runs_cells2048_pulled/runs_cells2048", "ab_ce_pre")]
barh(ax, b48_labels, b48_vals, [C_CCE, C_GRAY, C_GRAY, C_FULL], letter="c",
     title="@2048 batch · loss and self-supervision")
ax.text(0.45, 1.1, "ls exactly neutral (35 ties) · pre −0.036 (p=0.039)",
        fontsize=11, color=INK2, va="center")
save(fig, "fig4c_2048")

# ---------------- 图5 细胞数定标 ----------------
n256 = ab["ab_ce_notrans"]
n1024 = runs_means("runs_cells1024_pulled/runs_cells1024", "ab_ce_notrans")
n2048 = runs_means("runs_cells2048_pulled/runs_cells2048", "ab_ce_notrans")
nall = runs_means("runs_cells_all_pulled/runs_cells_all", "ab_ce_notrans")
# 逐 run 均值序列 → sd band
def run_series(run_dir, prefix):
    d = ROOT / run_dir
    return [float((pd.read_csv(rd / "cohort_rows.csv")["cohort_balanced_accuracy"].mean()))
            for rd in sorted(d.glob(f"{prefix}_r*_s*_c*"))
            if (rd / "cohort_rows.csv").exists()]

ser256 = run_series("runs_ablation_pulled/runs_ablation", "ab_ce_notrans")
ser1024 = run_series("runs_cells1024_pulled/runs_cells1024", "ab_ce_notrans")
ser2048 = run_series("runs_cells2048_pulled/runs_cells2048", "ab_ce_notrans")
base1024 = pd.read_csv(ROOT / "results_cells1024_pulled/baseline_c1024.csv").groupby("method")["cohort_balanced_accuracy"].mean()
base2048 = pd.read_csv(ROOT / "results_cells1024_pulled/baseline_c2048.csv").groupby("method")["cohort_balanced_accuracy"].mean()

fig, ax = new_ax(5.1, 3.55)
xs = np.array([256, 1024, 2048])
ys = np.array([n256, n1024, n2048])
sd = np.array([np.std(ser256), np.std(ser1024), np.std(ser2048)])
ax.fill_between(xs, ys - sd, ys + sd, color=C_CCE, alpha=0.13, lw=0)
ax.plot(xs, ys, "-o", color=C_CCE, lw=2.4, markersize=9, markeredgecolor=SURFACE,
        markeredgewidth=1.8, label="scSet (set encoder)", zorder=3)
ax.plot(xs, [full_means["state_mean"], base1024["state_mean"], base2048["state_mean"]],
        "-s", color=C_STATEm, lw=2.2, markersize=8, markeredgecolor=SURFACE,
        markeredgewidth=1.6, label="state_mean (pooled)", zorder=3)
ax.plot(xs, [full_means["sup_mean_mlp"], base1024["sup_mean_mlp"], base2048["sup_mean_mlp"]],
        "-^", color=C_SUP, lw=2.2, markersize=8, markeredgecolor=SURFACE,
        markeredgewidth=1.6, label="sup_mean_mlp", zorder=3)
for x, y in zip(xs, ys):
    ax.text(x, y - 0.0135, f"{y:.4f}", ha="center", fontsize=11, color=INK,
            fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", fc=SURFACE, ec="none", alpha=0.85))
ax.axhline(nall, color=C_CCE, lw=1.1, ls=(0, (4, 3)))
ax.text(240, nall + 0.0135, f"Uncapped (mean 5,190 cells/sample): {nall:.4f} (p=0.51, flat)",
        fontsize=10.5, color=INK2, ha="left", zorder=6)
ax.set_xscale("log")
ax.set_xticks(xs)
ax.set_xticklabels(["256", "1024", "2048"])
ax.minorticks_off()
ax.set_ylim(0.455, 0.685)
ax.set_xlabel("Cells per sample (log axis)", fontsize=11.5)
ax.set_ylabel("Cohort-level balanced accuracy", fontsize=11.5)
ax.set_title("Information intake · encoders saturate at 10³, pooling at 10²", loc="left", pad=14)
badge(ax, "")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncols=3, fontsize=10)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "fig5a_cellsat")

fig, ax = new_ax(3.7, 3.55)
lab = ["@1024 − @256", "@2048 − @1024", "uncapped − @2048"]
val = [n1024 - n256, n2048 - n1024, nall - n2048]
ps = ["p=0.075", "p=0.58 (saturated)", "p=0.51"]
y = np.arange(3)[::-1]
ax.barh(y, val, height=0.55, color=C_CCE, zorder=2)
ax.axvline(0, color=BASE, lw=0.9)
for yi, v, p in zip(y, val, ps):
    ax.text(v + 0.0008, yi, f"{v:+.4f}  {p}", va="center", fontsize=11,
            color=INK, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(lab, fontsize=11)
ax.set_xlim(-0.005, 0.030)
ax.set_xlabel("Δ balanced accuracy (paired)", fontsize=11.5)
ax.set_title("Marginal gains decay to zero", loc="left", pad=14)
badge(ax, "")
ax.xaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "fig5b_delta")

fig, ax = new_ax(3.7, 3.55)
bl = ["state_mean", "sup_mean_mlp", "pseudobulk", "pca256"]
dv = [base2048[m] - full_means[m] for m in bl]
y = np.arange(4)[::-1]
ax.barh(y, dv, height=0.55, color=C_GRAY, zorder=2)
ax.axvline(0, color=BASE, lw=0.9)
for yi, v in zip(y, dv):
    ax.text(v + 0.0006 if v >= 0 else v - 0.0006, yi, f"{v:+.4f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=11, color=INK, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(bl, fontsize=11)
ax.set_xlabel("Δ balanced accuracy (@2048 − @256, all ns)", fontsize=11.5)
ax.set_title("Baselines are count-insensitive", loc="left", pad=14)
badge(ax, "")
ax.xaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "fig5c_baseline")

# ---------------- 图6 数目扫描（加跨队列 sd band） ----------------
coh_sd = countsweep_coh.groupby(["method", "cells"])["cohort_balanced_accuracy"].std().reset_index()

fig, ax = new_ax(5.1, 3.55)
series = {"cce_ce": (C_CCE, 2.4), "sup_mean_mlp": (C_SUP, 2.4),
          "state_mean": (C_STATEm, 2.4), "pca256": (C_GRAY, 1.5),
          "composition": (C_GRAY, 1.3), "cce_full": (C_FULL, 1.8)}
for meth, (c, lw) in series.items():
    t = countsweep[countsweep["method"] == meth].sort_values("n")
    s = coh_sd[coh_sd["method"] == meth].set_index("cells")["cohort_balanced_accuracy"].reindex(t["n"].values)
    sv = np.nan_to_num(np.asarray(s, dtype=float))
    if c != C_GRAY:
        ax.fill_between(np.asarray(t["n"]), np.asarray(t["cohort_bal_acc"]) - sv,
                        np.asarray(t["cohort_bal_acc"]) + sv, color=c, alpha=0.12, lw=0)
    ax.plot(t["n"], t["cohort_bal_acc"], "-o" if c != C_GRAY else "-", color=c, lw=lw,
            markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=LAB.get(meth, meth))
ax.set_xscale("log", base=2)
ax.set_xticks([8, 16, 32, 64, 128, 256])
ax.set_xticklabels(["8", "16", "32", "64", "128", "256"])
ax.minorticks_off()
ax.set_xlabel("Inference-time cells per sample n (log axis)", fontsize=11.5)
ax.set_ylabel("Cohort-level balanced accuracy", fontsize=11.5)
ax.set_title("Discriminative endpoint · the encoder gap grows with n", loc="left", pad=14)
badge(ax, "")
t8 = countsweep[countsweep["method"] == "cce_ce"].set_index("n")["cohort_bal_acc"]
ts = countsweep[countsweep["method"] == "sup_mean_mlp"].set_index("n")["cohort_bal_acc"]
ax.annotate(f"n=128: Δ={t8[128]-ts[128]:+.3f}", xy=(128, t8[128]), xytext=(46, 0.505),
            fontsize=11, color=INK2, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncols=3, fontsize=10)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "fig6a_acc")

fig, ax = new_ax(3.9, 3.55)
for meth, (c, lw) in series.items():
    t = countsweep[countsweep["method"] == meth].sort_values("n")
    ax.plot(t["n"], t["P1"], "-o" if c != C_GRAY else "-", color=c, lw=lw,
            markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=LAB.get(meth, meth))
ax.set_xscale("log", base=2)
ax.set_xticks([8, 16, 32, 64, 128, 256])
ax.set_xticklabels(["8", "16", "32", "64", "128", "256"])
ax.minorticks_off()
ax.set_xlabel("Inference-time cells per sample n", fontsize=11.5)
ax.set_ylabel("Cross-count self-retrieval P@1", fontsize=11.5)
ax.set_title("Pooled retrieves naturally · supervised crowds", loc="left", pad=14)
badge(ax, "")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncols=3, fontsize=10)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "fig6b_p1")

fig, ax = new_ax(3.9, 3.55)
for meth, (c, lw) in series.items():
    t = countsweep[countsweep["method"] == meth].sort_values("n")
    ax.plot(t["n"], t["cos_self"], "-o" if c != C_GRAY else "-", color=c, lw=lw,
            markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=LAB.get(meth, meth))
ax.set_xscale("log", base=2)
ax.set_xticks([8, 16, 32, 64, 128, 256])
ax.set_xticklabels(["8", "16", "32", "64", "128", "256"])
ax.minorticks_off()
ax.set_xlabel("Inference-time cells per sample n", fontsize=11.5)
ax.set_ylabel("cos(emb-n, emb-256)", fontsize=11.5)
ax.set_ylim(0.85, 1.02)
ax.set_title("Embedding stability: no family difference", loc="left", pad=14)
badge(ax, "")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncols=3, fontsize=10)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "fig6c_cosself")

# ---------------- 图7 宽度/维度/推断 ----------------
fig, ax = new_ax(4.6, 3.55)
for meth, c in [("cce_ce", C_CCE), ("cce_full", C_FULL), ("state_mean", C_STATEm)]:
    t = dimsat[dimsat["method"].str.startswith(meth + "_pca")].copy()
    if not len(t):
        continue
    t["k"] = t["method"].str.extract(r"_pca(\d+)$").astype(int)
    t = t.sort_values("k")
    ax.plot(t["k"], t["bal"], "-o", color=c, lw=2.2, markersize=5.5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label=LAB.get(meth, meth))
ax.set_xscale("log", base=2)
ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256])
ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128", "256"])
ax.minorticks_off()
ax.set_xlabel("PCA components kept k", fontsize=11.5)
ax.set_ylabel("Cohort-level balanced accuracy", fontsize=11.5)
ax.set_title("Dimensionality saturation · signal lives in 8–16 dims", loc="left", pad=14)
badge(ax, "")
t = dimsat[dimsat["method"] == "cce_ce_pca8"]
if len(t):
    ax.text(8, t["bal"].iloc[0] + 0.005, f"PCA-8 already {t['bal'].iloc[0]:.3f}",
            fontsize=11, color=INK, fontweight="bold")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncols=3, fontsize=10)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "fig7a_dimsat")

fig, ax = new_ax(3.8, 3.55)
d1024 = runs_means("runs_d1024_pulled/runs_d1024", "cce_ce")
barh(ax, ["scSet-CE d1024", "scSet-CE d256 (ref)"], [d1024, full_means["cce_ce"]],
     [C_CCE, C_CCE], letter="b", title="Width dose (256 cells)")
ax.text(0.52, 0.4, "Δ +0.023 p=0.10 (ns)\nvs sup +0.083 p=0.0024",
        fontsize=11, color=INK2, va="center")
save(fig, "fig7b_width")

bag_files = sorted((ROOT / "results_bagging_pulled").glob("cohort_rows_r*.csv"))
bag = pd.concat([pd.read_csv(f) for f in bag_files])
bag_m = bag.groupby("method")["cohort_balanced_accuracy"].mean()
ks = [1, 10, 25, 50, 100]
zm = [bag_m[f"bag{k}_zmean"] for k in ks]
lg = [bag_m[f"bag{k}_logit"] for k in ks]
fig, ax = new_ax(3.8, 3.55)
ax.plot(ks, zm, "-o", color=C_CCE, lw=2.4, markersize=7.5, markeredgecolor=SURFACE,
        markeredgewidth=1.7, label="embedding mean (zmean)")
ax.plot(ks, lg, "-s", color=C_GRAY, lw=2.0, markersize=6, markeredgecolor=SURFACE,
        markeredgewidth=1.4, label="logit vote")
ax.set_xscale("log")
ax.set_xticks(ks)
ax.set_xticklabels([str(k) for k in ks])
ax.set_xlabel("Bootstrap bagging count K", fontsize=11.5)
ax.set_ylabel("Cohort-level balanced accuracy", fontsize=11.5)
ax.set_ylim(0.593, 0.658)
ax.text(1.15, zm[0] + 0.001, f"K=1: {zm[0]:.4f}", fontsize=11, color=INK, fontweight="bold")
ax.text(20, 0.6045, "zmean flat within ±0.0005 (ns)\nbagging never beats using all cells once",
        fontsize=11, color=INK2)
ax.set_title("Test-side bagging: zero effect", loc="left", pad=14)
badge(ax, "")
ax.legend(loc="center right")
ax.yaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "fig7c_bagging")

# ---------------- 图8 外部患者级 ----------------
# 雷达图：3 方法 × 5 端点（每轴独立归一至该轴最优 = 1.0）
fig = plt.figure(figsize=(4.9, 4.2), dpi=200)
axr = fig.add_subplot(111, polar=True)
ax_names = ["ARTEMIS pCR/RD\n(AUROC)", "GSE306201 primary/metastasis\n(AUROC)",
            "ICB end-to-end\n(AUROC)", "Main-benchmark transfer\n(balanced acc.)", "Disease retrieval\n(P@1)"]
radar_series = [
    ("scSet (encoder)", [0.472, 0.589, 0.829, full_means["cce_ce"], ret.loc["cce_ce", "P1"]], C_CCE, "-"),
    ("sup_mean_mlp", [0.511, 0.561, 0.784, full_means["sup_mean_mlp"], ret.loc["sup_mean_mlp", "P1"]], C_SUP, "-"),
    ("state_mean pooled", [0.612, 0.906, 0.802, full_means["state_mean"], ret.loc["state_mean", "P1"]], C_STATEm, "--"),
]
mat = np.array([v for _, v, _, _ in radar_series])
amax = mat.max(axis=0)
ang5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
angc = np.concatenate([ang5, ang5[:1]])
for name, vals, c, ls in radar_series:
    nv = np.array(vals) / amax
    nvc = np.concatenate([nv, nv[:1]])
    axr.plot(angc, nvc, color=c, lw=2.2, ls=ls, label=name, zorder=3,
             marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.0)
    axr.fill(angc, nvc, color=c, alpha=0.08, zorder=2)
axr.set_xticks(ang5)
axr.set_xticklabels(ax_names, fontsize=9.5, color=INK2)
axr.tick_params(axis="x", pad=20)
axr.set_ylim(0, 1.12)
axr.set_yticks([0.25, 0.5, 0.75, 1.0])
axr.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=9, color=MUTED)
axr.grid(color=GRID, lw=0.8)
axr.spines["polar"].set_color(GRID)
axr.set_facecolor(SURFACE)
axr.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=1, fontsize=10)
fig.text(0.03, 0.965, "Five-endpoint radar · each axis normalised to its best method (= 1.0)", fontsize=14,
         fontweight="bold", color=INK)
fig.text(0.03, 0.925, "Encoder wins the three within-benchmark axes; pooling is perfect on both external axes — no all-rounder",
         fontsize=9.5, color=INK2)
save(fig, "fig8a_radar")

fig, ax = new_ax(3.9, 3.55)
jl = ["STATE mean · gene", "STATE mean · guide", "scSet · gene", "scSet · guide"]
jv = [0.721, 0.690, 0.553, 0.540]
jc = [C_STATEm, C_STATEm, C_CCE, C_CCE]
y2 = np.arange(4)[::-1]
ax.barh(y2, jv, height=0.60, color=jc, zorder=2)
ax.set_yticks(y2)
ax.set_yticklabels(jl, fontsize=11)
for yi, v in zip(y2, jv):
    ax.text(v + 0.010, yi, f"{v:.3f}", va="center", fontsize=11, color=INK, fontweight="bold")
ax.set_xlim(0, 0.90)
ax.set_xlabel("AUROC (cell-level retrieval, prior protocol)", fontsize=11.5)
ax.set_title("Jiang CRISPR replicate retrieval", loc="left", pad=14)
badge(ax, "")
ax.xaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "fig8c_jiang")

fig, ax = new_ax(3.9, 3.55)
dl = ["ARTEMIS\n (scSet−pooled)", "GSE306201\n (scSet−pooled)", "Jiang gene\n (scSet−state)"]
dv2 = [0.472 - 0.612, 0.589 - 0.906, 0.553 - 0.721]
y3 = np.arange(3)[::-1]
ax.barh(y3, dv2, height=0.55, color=C_FULL, zorder=2)
ax.axvline(0, color=BASE, lw=0.9)
for yi, v in zip(y3, dv2):
    ax.text(v - 0.010, yi, f"{v:+.3f}", va="center", ha="right", fontsize=11,
            color=INK, fontweight="bold")
ax.set_yticks(y3)
ax.set_yticklabels(dl, fontsize=11)
ax.set_xlim(-0.50, 0.07)
ax.set_xlabel("Δ (encoder − best pooled control)", fontsize=11.5)
ax.set_title("External 0/3 · the boundary closes on the objective", loc="left", pad=14)
badge(ax, "")
ax.xaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "fig8d_deficit")

# ---------------- 图9 ICB + 样本量定律 ----------------
fig, ax = new_ax(4.6, 3.55)
il = ["scSet (e2e)", "state_mean", "sup_mean_mlp"]
iv = [0.829, 0.802, 0.784]
ie = [0.132, 0.146, 0.156]
x = np.arange(3)
ax.bar(x, iv, 0.52, yerr=ie, capsize=4, color=[C_CCE, C_STATEm, C_SUP],
       error_kw=dict(ecolor=INK2, lw=1.2), zorder=2)
for xi, v in zip(x, iv):
    ax.text(xi, v + 0.168, f"{v:.3f}", ha="center", fontsize=12, color=INK, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(il, fontsize=11)
ax.set_ylim(0, 1.08)
ax.set_ylabel("AUROC (5-fold × 3 seeds)", fontsize=11.5)
ax.set_title("GSE120575 ICB response · end-to-end", loc="left", pad=14)
badge(ax, "")
ax.text(0.02, 0.05, "Paired Δ=+0.027, p=0.14 (parity, positive)", transform=ax.transAxes,
        fontsize=11, color=INK2)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "fig9a_icb")

fig, ax = new_ax(4.6, 3.55)
ns = [38, 342, 455]
deltas = [0.027, 0.060, 0.104]
ps = ["p=0.14 (parity)", "Holm p=0.014", "p<0.005"]
ax.fill_between([30, 550], -0.005, 0.135, color=WASH, lw=0, zorder=0)
ax.plot(ns, deltas, "-o", color=C_CCE, lw=2.6, markersize=9, markeredgecolor=SURFACE,
        markeredgewidth=1.8, zorder=3)
has = ["left", "center", "right"]
for (x0, y0, p), ha in zip(zip(ns, deltas, ps), has):
    ax.annotate(f"{y0:+.3f}\n{p}", xy=(x0, y0), xytext=(x0, y0 + 0.009),
                fontsize=11, color=INK, ha=ha, fontweight="bold")
ax.set_xscale("log")
ax.set_xticks(ns)
ax.set_xticklabels(["~38 (ICB)", "342 (benchmark)", "455 + 10³ cells"])
ax.minorticks_off()
ax.set_xlabel("Labelled training samples (log axis)", fontsize=11.5)
ax.set_ylabel("Δ (encoder − pooled)", fontsize=11.5)
ax.set_ylim(0, 0.145)
ax.set_title("Sample-size law · the advantage cashes in with labels", loc="left", pad=14)
badge(ax, "")
ax.yaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "fig9b_law")

fig, ax = new_ax(3.8, 3.55)
fl = ["ICB end-to-end\n (retrained)", "GSE306201\n (frozen)", "ARTEMIS\n (frozen)"]
fv = [0.829, 0.589, 0.472]
y4 = np.arange(3)[::-1]
ax.barh(y4, fv, height=0.60, color=[C_CCE, C_FULL, C_FULL], zorder=2)
ax.set_yticks(y4)
ax.set_yticklabels(fl, fontsize=11)
for yi, v in zip(y4, fv):
    ax.text(v + 0.015, yi, f"{v:.3f}", va="center", fontsize=11, color=INK, fontweight="bold")
ax.set_xlim(0, 1.04)
ax.set_xlabel("scSet AUROC", fontsize=11.5)
ax.set_title("Frozen fails · retraining reaches parity", loc="left", pad=14)
badge(ax, "")
ax.xaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "fig9c_frozen")

# ---------------- 图10 二律背反 ----------------
fig, ax = new_ax(5.1, 3.55)
cfgs = ["base", "T=0.05", "+uniformity", "large-pool anchor", "batch64"]
cos64 = [0.978, 0.978, 0.944, 0.979, 0.963]
icb1 = [0.27, 0.27, 0.27, 0.27, 0.27]
impute = [0.956, 0.951, 0.951, 0.954, 0.95]
x = np.arange(len(cfgs))
w = 0.26
ax.bar(x - w, cos64, w, color=C_CCE, label="cos@64 invariance", zorder=2)
ax.bar(x, icb1, w, color=C_SUP, label="ICB top-1 (chance 0.03)", zorder=2)
ax.bar(x + w, impute, w, color=C_STATEm, label="imputation decoding", zorder=2)
for xi, v in zip(x - w, cos64):
    ax.text(xi, v + 0.013, f"{v:.3f}".lstrip("0"), ha="center", fontsize=8.5, color=INK2)
for xi, v in zip(x + w, impute):
    ax.text(xi, v + 0.013, f"{v:.3f}".lstrip("0"), ha="center", fontsize=8.5, color=INK2)
ax.set_xticks(x)
ax.set_xticklabels(cfgs, fontsize=11)
ax.set_ylim(0, 1.26)
ax.set_ylabel("Metric value", fontsize=11.5)
ax.set_title("Five consistency-SSL configs · invariance and information pass", loc="left", pad=14)
badge(ax, "")
ax.text(0.02, 0.045, "Cross-count retrieval P@1 = 0 (all five) — metric collapse",
        transform=ax.transAxes, fontsize=11.5, color=C_FULL, fontweight="bold")
ax.legend(loc="upper center", ncols=3, fontsize=10)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "fig10a_ssl")

fig, ax = new_ax(3.9, 3.55)
rl = ["state_mean\n (pooled mean)", "scSet-CE @ n=8\n (supervised)", "consistency SSL\n (five configs)"]
rv = [0.999, 0.279, 0.0]
y5 = np.arange(3)[::-1]
ax.barh(y5, rv, height=0.60, color=[C_STATEm, C_CCE, C_FULL], zorder=2)
ax.set_yticks(y5)
ax.set_yticklabels(rl, fontsize=11)
for yi, v in zip(y5, rv):
    ax.text(v + 0.016, yi, f"{v:.3f}", va="center", fontsize=11, color=INK, fontweight="bold")
ax.set_xlim(0, 1.20)
ax.set_xlabel("Cross-count self-retrieval P@1 (64-cell query → 256-cell library)", fontsize=11.5)
ax.set_title("Identity solved constructively by the mean", loc="left", pad=14)
badge(ax, "")
ax.xaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "fig10b_identity")

fig, ax = new_ax(3.9, 3.55)
gm = geom.set_index("method").loc[["cce_ce", "cce_full", "pseudobulk", "state_mean"]]
x = np.arange(len(gm))
w = 0.32
ax.bar(x - w / 2, gm["within_class_cos"], w, color=C_CCE, label="within-class cos (identity)", zorder=2)
ax.bar(x + w / 2, gm["between_class_cos"], w, color=C_VIOL, label="between-class cos (semantics)", zorder=2)
for xi, v in zip(x - w / 2, gm["within_class_cos"]):
    ax.text(xi, v + 0.016, f"{v:.2f}", ha="center", fontsize=10.5, color=INK, fontweight="bold")
for xi, v in zip(x + w / 2, gm["between_class_cos"]):
    ax.text(xi, max(v, 0) + 0.016, f"{v:.2f}", ha="center", fontsize=10.5, color=INK2)
ax.axhline(0, color=BASE, lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels([LAB.get(m, m) for m in gm.index], fontsize=11)
ax.set_ylabel("Mean cosine similarity", fontsize=11.5)
ax.set_title("Supervised: semantics w/o identity · mean: identity w/o semantics", loc="left", pad=14)
badge(ax, "")
ax.legend(loc="upper right", fontsize=10)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "fig10c_geom")

# ================= PPTX assembly (English) =================
from PIL import Image

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

INK_RGB = RGBColor(0x0B, 0x0B, 0x0B)
INK2_RGB = RGBColor(0x52, 0x51, 0x4E)
MUT_RGB = RGBColor(0x89, 0x87, 0x81)
DEEP_RGB = RGBColor(0x0D, 0x36, 0x6B)
WASH_RGB = RGBColor(0xEA, 0xF2, 0xFD)
GRID_RGB = RGBColor(0xE1, 0xE0, 0xD9)
WHITE_RGB = RGBColor(0xFF, 0xFF, 0xFF)
SKY_RGB = RGBColor(0x9E, 0xC5, 0xF4)
PALE_RGB = RGBColor(0xC7, 0xDA, 0xF6)
FONT = "Arial"


def _set(run, text, size, color, bold=False):
    run.text = text
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold


def add_slide(kicker, title, subtitle, panels, tiles, takeaway, footer, page):
    s = prs.slides.add_slide(BLANK)
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.06))
    band.fill.solid(); band.fill.fore_color.rgb = DEEP_RGB
    band.line.fill.background(); band.shadow.inherit = False
    tb = s.shapes.add_textbox(Inches(0.45), Inches(0.07), Inches(12.4), Inches(0.30))
    _set(tb.text_frame.paragraphs[0].add_run(), kicker, 10.5, SKY_RGB, True)
    tb2 = s.shapes.add_textbox(Inches(0.45), Inches(0.30), Inches(12.4), Inches(0.42))
    _set(tb2.text_frame.paragraphs[0].add_run(), title, 21, WHITE_RGB, True)
    tb3 = s.shapes.add_textbox(Inches(0.45), Inches(0.72), Inches(12.4), Inches(0.30))
    _set(tb3.text_frame.paragraphs[0].add_run(), subtitle, 11, PALE_RGB)
    n_t = len(tiles)
    gap = 0.16
    cw = (12.45 - gap * (n_t - 1)) / n_t
    for i, (big, label) in enumerate(tiles):
        x = 0.45 + i * (cw + gap)
        card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.20),
                                  Inches(cw), Inches(0.92))
        card.adjustments[0] = 0.14
        card.fill.solid(); card.fill.fore_color.rgb = WHITE_RGB
        card.line.color.rgb = GRID_RGB; card.line.width = Pt(0.75)
        card.shadow.inherit = False
        tf = card.text_frame; tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = Inches(0.12); tf.margin_right = Inches(0.06)
        tf.margin_top = Inches(0.04); tf.margin_bottom = Inches(0.04)
        p1 = tf.paragraphs[0]; p1.alignment = PP_ALIGN.LEFT
        _set(p1.add_run(), big, 14 if len(big) > 7 else 19, DEEP_RGB, True)
        p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.LEFT
        _set(p2.add_run(), label, 9.5, INK2_RGB)
    top = Inches(2.28); avail_w = 12.45; avail_h = 4.42
    scale = avail_w / sum(w for _, w in panels)
    x = Inches(0.45)
    for name, w_in in panels:
        img = Image.open(PANELS / name)
        w = Inches(w_in * scale)
        h = Inches(w_in * scale * img.height / img.width)
        if h.inches > avail_h:
            k = avail_h / h.inches
            h = Inches(h.inches * k); w = Inches(w.inches * k)
        s.shapes.add_picture(str(PANELS / name), x,
                             top + Inches((avail_h - h.inches) / 2), width=w, height=h)
        x = Inches(x.inches + w.inches + 0.10)
    tk = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(6.80),
                            Inches(12.45), Inches(0.46))
    tk.adjustments[0] = 0.5
    tk.fill.solid(); tk.fill.fore_color.rgb = WASH_RGB
    tk.line.fill.background(); tk.shadow.inherit = False
    tf = tk.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.14)
    p = tf.paragraphs[0]
    _set(p.add_run(), "Takeaway  ", 11, DEEP_RGB, True)
    _set(p.add_run(), takeaway, 11, DEEP_RGB)
    fb = s.shapes.add_textbox(Inches(0.45), Inches(7.28), Inches(11.4), Inches(0.22))
    _set(fb.text_frame.paragraphs[0].add_run(), footer, 8, MUT_RGB)
    pg = s.shapes.add_textbox(Inches(12.55), Inches(7.24), Inches(0.6), Inches(0.26))
    p = pg.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    _set(p.add_run(), f"{page} / 11", 9, MUT_RGB)


add_slide(
    "Fig 1 · Architecture",
    "scSet: the terminal set encoder",
    "Permutation-invariant cell-set encoder over STATE embeddings; every hyper-parameter from the released implementation",
    [("architecture_main_en.png", 13.2)],
    [("~3.5M", "parameters at d=256"),
     ("85 s", "per training run (one A100)"),
     ("1×", "forward pass at deployment"),
     ("k = 15", "kNN graph, built in STATE space")],
    "Every component is earned by ablation: no Transformer, no contrastive term, no bagging — one GAT plus cross-entropy is the terminal recipe.",
    "Terminal recipe of the systematic architecture dissection (Figs 4–5); hyper-parameters in Methods",
    1,
)
add_slide(
    "Fig 2 · Main benchmark",
    "Pre-registered reversal: fixing the objective beats pooling",
    "scSet-CE 0.596 > sup_mean_mlp 0.536 (+0.060, Holm p=0.0138); scSet-Contrastive 0.555 does not clear the pooling control",
    [("fig1a_petal.png", 4.7), ("fig1b_paired.png", 3.7), ("fig1cd_retrieval_gap.png", 7.6)],
    [("+0.060", "scSet-CE vs sup paired difference (Holm p=0.0138)"),
     ("0.596", "scSet-CE cohort-level balanced accuracy (42 cohorts)"),
     ("1,134", "runs (6 rounds × 3 seeds × all methods)"),
     ("0.998→0.596", "same model: in-atlas diagnostic → cohort-held-out")],
    "The contrastive term, not the architecture, caused the prototype's underperformance; cross-entropy plus longer training reverses the verdict.",
    "Data: results_full_pulled/ (42 cohorts × 6 rounds × 3 seeds); deck generated by make_figures_en.py",
    2,
)
add_slide(
    "Fig 3 · Comparator matrix",
    "scSet-CE leads every internal and external method by 0.06–0.32",
    "scvi_mean (scArches, zero exposure per fold) 0.347 at the bottom; scPoli falls from 0.988 in-atlas to 0.458 cohort-held-out",
    [("fig2a_comparator.png", 7.0), ("fig2b_scpoli.png", 3.6), ("fig2c_scvi.png", 3.6)],
    [("0.596", "scSet-CE — top of the full matrix"),
     ("−0.25 / −0.14", "scvi_mean vs scSet-CE / vs state_mean"),
     ("0.988→0.458", "scPoli: in-atlas diagnostic → held-out"),
     ("0.847 vs 0.458", "same subset: scSet-CE vs scPoli (+0.39)")],
    "The evaluation protocol alone can invert method rankings — deep-generative and reference-adapted sample latents lose to a simple mean.",
    "Data: results_scvi_pulled/, scpoli_pulled/, fig5b_external_tasks.csv; subset numbers per STATUS.md 2026-11-05",
    3,
)
add_slide(
    "Fig 4 · Mechanism I: architecture",
    "One mixing mechanism is optimal; two over-smooth",
    "scSet (GAT only) 0.6237 is the strongest configuration; minimal 0.556 shows the whole gain is cell-level context mixing",
    [("fig3a_arch.png", 7.0), ("fig3b_paired.png", 3.4), ("fig3c_input.png", 3.6)],
    [("0.6237", "scSet (exploratory best)"),
     ("+0.027", "scSet vs scSet-CE (p=0.0012)"),
     ("−0.068", "minimal vs scSet (value of context mixing)"),
     ("−0.041", "global state token (p=0.016, harmful)")],
    "One cell-level mixing mechanism suffices: GAT or Transformer alone reaches 0.62+; both together 0.596; neither 0.556 — an inverted U.",
    "Data: runs_ablation_pulled/ (72 + 18 runs); input-enrichment verdicts 2026-10-31",
    4,
)
add_slide(
    "Fig 5 · Mechanism II: loss",
    "The contrastive term is a negative contribution everywhere",
    "Pure CE is optimal; CE + 0.5·contrastive −0.049 (p=0.008); four orthogonal within-class mechanisms all miss the 0.60 threshold",
    [("fig4a_loss.png", 4.0), ("fig4b_planb.png", 4.0), ("fig4c_2048.png", 4.0)],
    [("−0.049", "CE + contrastive vs pure CE (p=0.008)"),
     ("0 / 4", "within-class mechanisms passing (plan B)"),
     ("−0.080", "variance-spreading vs scSet-CE (p=0.0036)"),
     ("−0.036", "masked pre-training @2048 (p=0.039)")],
    "The component to delete from 'cancer-supervised contrastive' is the contrastive term; within-class preservation needs a new signal, not a new loss.",
    "Data: runs_ablation_pulled/, runs_cells2048_pulled/; p-values per STATUS.md",
    5,
)
add_slide(
    "Fig 6 · Mechanism III: cell count",
    "Two saturation scales: 10³ for encoders, 10² for pooling",
    "0.6237 @256 → 0.6404 @1024 → 0.6455 @2048 → 0.6440 uncapped (p=0.51); baselines never move",
    [("fig5a_cellsat.png", 5.0), ("fig5b_delta.png", 3.4), ("fig5c_baseline.png", 3.4)],
    [("10³", "encoder information-intake saturation (cells/sample)"),
     ("10²", "pooling saturation — one order of magnitude apart"),
     ("0.6455", "@2048 nominal record (vs @1024 p=0.58)"),
     ("1,024", "practical operating point (2048 statistically equal)")],
    "The essential difference between encoders and pooling is the scale at which compositional structure is read.",
    "Data: runs_cells1024/2048/all_pulled, results_cells1024_pulled/baseline_c*.csv",
    6,
)
add_slide(
    "Fig 7 · Count sweep",
    "The advantage grows with inference-time cells",
    "scSet-CE 0.492@8 → 0.593@256; sup ties at n=8 and trails by 0.060 at n=128 — opposite of count-insensitivity",
    [("fig6a_acc.png", 4.8), ("fig6b_p1.png", 3.6), ("fig6c_cosself.png", 3.6)],
    [("+0.060", "scSet-CE vs sup at n=128"),
     ("0.492", "scSet-CE at n=8 (ties sup)"),
     ("0.593", "scSet-CE at n=256"),
     ("0.95+", "cos-self at n=8 for all methods (equal stability)")],
    "Low retrieval for supervised encoders is same-class crowding, not instability — discriminative endpoints are the right yardstick.",
    "Data: results_countsweep_pulled/ (count-sweep v4)",
    7,
)
add_slide(
    "Fig 8 · Mechanism IV: width · dims · inference",
    "Everything saturates or is neutral: the ceiling is set by information",
    "Discriminative signal lives in 8–16 dims (PCA-8 reaches 0.600); d1024 +0.023 ns; bagging ±0.0005 — eight axes now closed",
    [("fig7a_dimsat.png", 4.5), ("fig7b_width.png", 3.5), ("fig7c_bagging.png", 3.6)],
    [("8–16", "dimensions carrying the discriminative signal"),
     ("+0.023", "d1024 vs d256 (p=0.10, ns)"),
     ("±0.0005", "bagging K=1→100 total change (ns)"),
     ("0.64", "terminal recipe (scSet + CE + d256 + ~1,024 cells)")],
    "Eight closed axes (architecture/loss/width/cells/SSL/dimensions/inference/within-class): the ceiling is STATE information content and the task.",
    "Data: results_dimsat_pulled/, runs_d1024_pulled/, results_bagging_pulled/",
    8,
)
add_slide(
    "Fig 9 · External patient-level endpoints",
    "0 / 3: the boundary closes on the objective, not the read-out",
    "Three deployment modes (alone / supervised projection / concat fusion) all trail pooling; the radar shows no all-rounder across five endpoints",
    [("fig8a_radar.png", 4.7), ("fig8c_jiang.png", 3.8), ("fig8d_deficit.png", 3.8)],
    [("−0.14", "ARTEMIS: scSet vs pooled (0.472 / 0.612)"),
     ("−0.32", "GSE306201: scSet vs pooled (0.589 / 0.906)"),
     ("+0.08~0.11", "concat information gain (cancer-discriminative)"),
     ("0 / 3", "external endpoints won (deployment modes exhausted)")],
    "The within-benchmark complement is cancer-discriminative, not response-relevant — closing the boundary needs within-class-preserving objectives.",
    "Data: STATUS.md 2026-11-01 v5/v8 records (3 seeds, server details); Jiang is a prior cell-level protocol",
    9,
)
add_slide(
    "Fig 10 · ICB end-to-end",
    "Sample-size law: frozen fails, retraining reaches parity",
    "GSE120575 (48 samples / 33 patients): scSet (e2e) 0.829 vs state_mean 0.802 (+0.027, p=0.14) — against frozen scores of 0.47–0.59",
    [("fig9a_icb.png", 4.4), ("fig9b_law.png", 4.6), ("fig9c_frozen.png", 3.4)],
    [("0.829", "scSet (e2e) AUROC (±0.132)"),
     ("38→342→455", "labels: parity → +0.06 → +0.10"),
     ("0.83 vs 0.80", "under scarce labels the encoder is at least harmless"),
     ("−0.14", "leakage-split variant (fold sensitivity, ED)")],
    "Deployment guide: ≥300 cross-cohort labels — use the encoder; ~40 — pooling or parity; frozen + within-class endpoint — retrain.",
    "Data: icb_benchmark_20261103 (server), values per STATUS.md 2026-11-03",
    10,
)
add_slide(
    "Fig 11 · The identity–semantics antinomy",
    "Every objective family sacrifices one half; having both is the open problem",
    "Mean pooling self-retrieval 0.999 (identity by construction); supervised encoders 0.89 within / −0.09 between; five SSL configs retrieve at 0",
    [("fig10a_ssl.png", 4.9), ("fig10b_identity.png", 3.6), ("fig10c_geom.png", 3.6)],
    [("0.999", "state_mean self-retrieval P@1 (for free)"),
     ("0.379", "state_mean disease neighbour (identity, no semantics)"),
     ("P@1 = 0", "five SSL configs + MMD-MDS (all fail)"),
     ("0.27", "SSL information carried (8× chance, constant)")],
    "The pooled mean is the control any new method must first beat at its own trivial task — a new supervision signal is the next paper.",
    "Data: results_id/ (server, per STATUS.md 2026-11-04), figures/source_data/fig4_geometry_summary.csv",
    11,
)

out_pptx = OUT / "Figure_deck_EN_v2.pptx"
prs.save(out_pptx)
print("panels:", len(PANELS_MADE))
print("saved:", out_pptx)
