# -*- coding: utf-8 -*-
"""
图 1 网络结构 v2 —— 极简高级风（CN/EN 双语）。
设计原则：一条水平主流程；同一个细胞点云三次出现（集合 → 图 → 池化）作为视觉母题；
细线、留白、柔和阴影；无标题带、无底部注记条（图注在正文图例中）。
超参照旧取自发布实现 cce_full_20261019（sc_cpsformer.py + train_variants.py notrans）。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Ellipse, Polygon

ROOT = Path(__file__).resolve().parent.parent

# NPG 派生调色（与图集 v3 一致）
SURFACE = "#ffffff"
INK = "#1a1a1a"
INK2 = "#52514e"
MUTED = "#8a8880"
DEEP = "#22355c"
C_CCE = "#355191"
C_FULL = "#e64b35"
C_STATEm = "#d9a020"
C_VIOL = "#4dbbd5"
C_AQUA = "#00a087"
EDGE_BLUE = "#c9d8ec"
FILL_BLUE = "#f3f7fc"
FILL_WARM = "#faf8f2"
EDGE_WARM = "#e5ddc8"
LIN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]


def render(lang: str, out_png: Path):
    is_cn = lang == "cn"
    fonts = ["Microsoft YaHei", "Segoe UI", "DejaVu Sans"] if is_cn else ["Arial", "DejaVu Sans"]
    rcParams.update({
        "font.sans-serif": fonts,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
    })

    T = {
        # 输入
        "sample": "样本 = 细胞集合" if is_cn else "Sample = set of cells",
        "ncells": "n ≈ 10³ 个细胞 · 按谱系着色" if is_cn else "n ≈ 10³ cells · coloured by lineage",
        "onecell": "单个细胞向量" if is_cn else "Single-cell vector",
        "state": "STATE 2,058",
        "lineage": "谱系 12" if is_cn else "lineage 12",
        # 投影
        "linear": "Linear",
        # GAT
        "gat": "图注意力 GAT × 2" if is_cn else "Graph attention GAT × 2",
        "gatsub": "kNN 细胞图 · k = 15（STATE 空间）· 4 头拼接" if is_cn else "kNN cell graph · k = 15 (STATE space) · 4 concatenated heads",
        "gatcap": "每个细胞读取其图邻域的上下文" if is_cn else "each cell reads its graph neighbourhood",
        # 池化
        "pool": "排列不变双池化" if is_cn else "Permutation-invariant dual pooling",
        "poolsub": "集合 → 固定长度向量" if is_cn else "set → fixed-length vector",
        "mean": "均值 → 256" if is_cn else "mean → 256",
        "query": "查询 ×4 → 4×256" if is_cn else "queries ×4 → 4×256",
        "concat": "concat",
        # MLP
        "mlp": "MLP 头" if is_cn else "MLP head",
        "mlpsub": "1280→512→256 · GELU · L2 归一化" if is_cn else "1280→512→256 · GELU · L2 norm",
        # z
        "zlab": "样本向量" if is_cn else "Sample vector",
        # 出口
        "trainonly": "仅训练" if is_cn else "training only",
        "trainhead": "Linear → 10 癌种" if is_cn else "Linear → 10 cancer types",
        "celoss": "交叉熵损失" if is_cn else "cross-entropy loss",
        "deploy": "部署 · 单次前向" if is_cn else "Deployment · single forward pass",
        "deploysub": "检索 · 迁移 · 监督探针" if is_cn else "retrieval · transfer · probing",
        "prov": ("scSet · 系统性架构消融的终版配方 · 消融见 图4 / 图5"
                 if is_cn else
                 "scSet · the terminal recipe of the systematic architecture dissection · ablations in Figs 4–5"),
        "d2070": "n × 2070",
        "d256": "n × 256",
        "d1280": "1280",
        "d256z": "256",
    }

    fig, ax = plt.subplots(figsize=(11.6, 5.15), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 45)
    ax.axis("off")

    # ---------- 基础绘制函数 ----------
    def txt(x, y, s, size=10, color=INK, weight="normal", ha="center", va="center",
            z=6, style="normal", family=None):
        ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
                zorder=z, style=style, family=family)

    def dim(x, y, s):
        txt(x, y, s, 8.4, MUTED, style="italic", family="DejaVu Sans")

    def arrow(x0, y0, x1, y1, color=DEEP, lw=1.5, ls="-", ms=11, z=4, rad=0.0,
              alpha=1.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=ms, color=color, lw=lw,
                                     linestyle=ls, zorder=z, alpha=alpha,
                                     connectionstyle=f"arc3,rad={rad}"))

    def softbox(x, y, w, h, face, edge, lw=1.2, ls="-", rad=0.9, shadow=True, z=2):
        if shadow:
            ax.add_patch(FancyBboxPatch((x + 0.32, y - 0.32), w, h,
                                        boxstyle=f"round,pad=0.35,rounding_size={rad}",
                                        fc="#22355c", ec="none", alpha=0.07, zorder=z - 1))
        b = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0.35,rounding_size={rad}",
                           fc=face, ec=edge, lw=lw, linestyle=ls, zorder=z)
        ax.add_patch(b)
        return b

    # ---------- 细胞点云（唯一母题，三次使用） ----------
    rng = np.random.default_rng(7)
    centers = np.array([[-2.6, 2.1], [1.9, 3.1], [3.1, -1.6], [-1.4, -2.9], [-0.2, 0.4], [2.4, 0.9]])
    cluster_lab = rng.integers(0, 6, len(centers))
    pts_raw, cols_raw = [], []
    for ci, (cx, cy) in enumerate(centers):
        n_i = rng.integers(5, 9)
        for _ in range(n_i):
            px = cx + rng.normal(0, 0.95)
            py = cy + rng.normal(0, 0.75)
            # 压入椭圆
            if (px / 4.6) ** 2 + (py / 3.6) ** 2 > 1.0:
                continue
            pts_raw.append([px, py])
            cols_raw.append(cluster_lab[ci])
    pts_raw = np.array(pts_raw)
    cols_raw = np.array(cols_raw)

    def draw_cloud(cx, cy, sx=1.0, sy=1.0, alpha=1.0, r=0.34, z=4, lw_edge=0.7):
        for (px, py), li in zip(pts_raw, cols_raw):
            ax.add_patch(Circle((cx + px * sx, cy + py * sy), r,
                                fc=LIN_COLORS[li], ec=SURFACE, lw=lw_edge,
                                alpha=alpha, zorder=z))

    # =========================================================
    # 1) 输入：细胞集合
    # =========================================================
    ax.add_patch(Ellipse((10.3, 27.2), 16.4, 18.0, fc="#f6f9fd", ec=EDGE_BLUE,
                         lw=1.1, linestyle=(0, (5, 3)), alpha=0.75, zorder=1))
    draw_cloud(10.3, 27.2, 1.0, 1.0)
    txt(10.3, 38.4, T["sample"], 12, DEEP, weight="bold")
    txt(10.3, 36.5, T["ncells"], 8.8, INK2)

    # 单细胞放大：向量条
    strip_y, strip_h = 15.6, 2.1
    src = pts_raw[len(pts_raw) // 2] + np.array([10.3, 27.2])
    ax.add_patch(FancyArrowPatch((src[0] + 0.4, src[1] - 0.4), (8.6, strip_y + strip_h + 1.3),
                                 arrowstyle="-", color=MUTED, lw=0.9,
                                 linestyle=(0, (2, 2)), zorder=3))
    ax.add_patch(Circle((src[0], src[1]), 0.48, fc="none", ec=INK2, lw=1.1, zorder=5))
    # STATE 段（蓝） + 谱系段（金）
    ax.add_patch(FancyBboxPatch((3.4, strip_y), 11.2, strip_h,
                                boxstyle="round,pad=0.02,rounding_size=0.25",
                                fc=C_CCE, ec="none", alpha=0.92, zorder=4))
    ax.add_patch(FancyBboxPatch((14.8, strip_y), 2.4, strip_h,
                                boxstyle="round,pad=0.02,rounding_size=0.25",
                                fc=C_STATEm, ec="none", alpha=0.95, zorder=4))
    txt(9.0, strip_y + strip_h / 2, T["state"], 8.6, "white", weight="bold")
    txt(16.0, strip_y - 1.35, T["lineage"], 8.2, INK2)
    txt(9.0, strip_y + strip_h + 1.55, T["onecell"] + "  ·  2,070", 9.2, INK, weight="bold")

    arrow(18.9, 27.2, 22.3, 27.2)
    dim(20.6, 28.9, T["d2070"])

    # =========================================================
    # 2) 线性投影
    # =========================================================
    softbox(22.5, 23.0, 4.9, 8.4, SURFACE, C_CCE, lw=1.3)
    txt(25.0, 29.4, T["linear"], 11, INK, weight="bold")
    txt(25.0, 25.4, "2070 → 256", 9.2, C_CCE)
    arrow(27.7, 27.2, 30.6, 27.2)
    dim(29.2, 28.9, T["d256"])

    # =========================================================
    # 3) GAT ×2：同一云 + kNN 边 + 注意力
    # =========================================================
    softbox(30.9, 15.8, 22.6, 20.6, FILL_BLUE, EDGE_BLUE, lw=1.0, rad=1.1, shadow=False)
    txt(42.2, 34.2, T["gat"], 12, DEEP, weight="bold")
    txt(42.2, 32.4, T["gatsub"], 8.4, INK2)
    # 点云副本
    gpts = pts_raw * np.array([1.0, 1.0]) + np.array([42.2, 26.4])
    d = np.linalg.norm(gpts[:, None] - gpts[None], axis=2)
    for i in range(len(gpts)):
        for j in np.argsort(d[i])[1:4]:
            if i < j:
                ax.plot([gpts[i][0], gpts[j][0]], [gpts[i][1], gpts[j][1]],
                        color="#a9bddc", lw=0.8, alpha=0.75, zorder=3)
    hub = int(np.argmin(np.linalg.norm(pts_raw, axis=1)))
    nbrs = np.argsort(d[hub])[1:4]
    for w, j in zip([1.4, 2.2, 3.0], nbrs):
        arrow(gpts[j][0], gpts[j][1], gpts[hub][0], gpts[hub][1],
              color=C_CCE, lw=w, ms=9, z=5,
              alpha=0.95 if w > 2 else (0.75 if w > 1.5 else 0.55))
    for k, (px, py) in enumerate(gpts):
        if k == hub:
            ax.add_patch(Circle((px, py), 0.62, fc=DEEP, ec=SURFACE, lw=1.0, zorder=6))
        else:
            li = cols_raw[k]
            ax.add_patch(Circle((px, py), 0.32, fc=LIN_COLORS[li], ec=SURFACE, lw=0.7, zorder=4))
    txt(42.2, 18.6, T["gatcap"], 8.2, INK2, style="italic")

    arrow(53.8, 27.2, 56.3, 27.2)
    # LayerNorm 微块
    ax.add_patch(FancyBboxPatch((56.4, 24.7), 3.0, 5.0,
                                boxstyle="round,pad=0.02,rounding_size=0.5",
                                fc="#f5f4f0", ec=MUTED, lw=1.0, zorder=2))
    txt(57.9, 27.2, "LN", 9.2, INK2, weight="bold")
    arrow(59.6, 27.2, 62.1, 27.2)
    dim(60.8, 28.9, T["d256"])

    # =========================================================
    # 4) 双池化：褪色云 → 双头 → concat
    # =========================================================
    softbox(62.4, 15.8, 20.4, 20.6, FILL_WARM, EDGE_WARM, lw=1.0, rad=1.1, shadow=False)
    txt(72.6, 34.2, T["pool"], 12, DEEP, weight="bold")
    txt(72.6, 32.4, T["poolsub"], 8.4, INK2)
    draw_cloud(66.3, 27.2, 0.58, 0.58, alpha=0.42, r=0.26)
    # 均值头
    ax.add_patch(Circle((71.6, 30.6), 1.42, fc=SURFACE, ec=C_STATEm, lw=1.5, zorder=5))
    txt(71.6, 30.62, "μ", 12, C_STATEm, weight="bold", family="DejaVu Sans")
    for j in [2, 7, 12, 17, 22, 27]:
        if j < len(pts_raw):
            p = pts_raw[j] * 0.58 + np.array([66.3, 27.2])
            ax.add_patch(FancyArrowPatch((p[0] + 0.3, p[1] - 0.15), (70.4, 30.2),
                                         arrowstyle="-", color=C_STATEm, lw=0.7,
                                         alpha=0.55, zorder=3))
    txt(71.6, 33.6, "", 8, INK2)
    txt(74.9, 30.6, T["mean"], 8.6, INK2, ha="left")
    # 查询头：4 菱形
    qy = [25.6, 24.2, 22.8, 21.4]
    for q in qy:
        ax.add_patch(Polygon([[70.7, q + 0.55], [71.25, q], [70.7, q - 0.55], [70.15, q]],
                             closed=True, fc=SURFACE, ec=C_VIOL, lw=1.3, zorder=5))
        ax.add_patch(FancyArrowPatch((69.2, 26.4), (70.1, q + 0.1),
                                     arrowstyle="-", color=C_VIOL, lw=0.7,
                                     linestyle=(0, (2, 2)), alpha=0.6, zorder=3))
    txt(74.9, 23.4, T["query"], 8.6, INK2, ha="left")
    # concat
    arrow(73.1, 30.4, 75.9, 28.8, color=C_STATEm, lw=1.2, ms=9)
    arrow(71.9, 22.6, 75.9, 25.6, color=C_VIOL, lw=1.2, ms=9)
    softbox(76.0, 24.6, 3.3, 5.2, SURFACE, DEEP, lw=1.2, rad=0.5, shadow=False)
    txt(77.65, 28.3, T["concat"], 7.6, INK2)
    txt(77.65, 26.2, T["d1280"], 10, DEEP, weight="bold")

    arrow(79.5, 27.2, 82.0, 27.2)

    # =========================================================
    # 5) MLP 漏斗
    # =========================================================
    ax.add_patch(Polygon([[82.2, 33.2], [87.8, 30.3], [87.8, 24.1], [82.2, 21.2]],
                         closed=True, fc=FILL_BLUE, ec=C_CCE, lw=1.3, zorder=3))
    txt(85.0, 27.2, "MLP", 10.5, DEEP, weight="bold")
    txt(85.0, 19.4, T["mlp"], 10.5, INK, weight="bold")
    txt(85.0, 17.6, T["mlpsub"], 8.0, INK2)
    arrow(88.0, 27.2, 89.1, 27.2, ms=9)

    # =========================================================
    # 6) 样本向量 z
    # =========================================================
    softbox(89.4, 20.2, 5.4, 14.0, SURFACE, DEEP, lw=1.5)
    rrng = np.random.default_rng(11)
    for i in range(10):
        w_i = rrng.uniform(1.6, 4.4)
        ax.add_patch(FancyBboxPatch((90.3, 31.9 - i * 1.18), w_i, 0.62,
                                    boxstyle="round,pad=0.02,rounding_size=0.2",
                                    fc=DEEP, ec="none", alpha=1 - i * 0.085, zorder=4))
    txt(92.1, 18.7, "z ∈ ℝ²⁵⁶", 10.5, DEEP, weight="bold", family="DejaVu Sans")
    txt(92.1, 17.1, T["zlab"], 8.6, INK2)

    # 训练头（上，红虚线）
    arrow(92.1, 34.6, 92.1, 37.6, color=C_FULL, lw=1.3, ls=(0, (4, 3)), ms=9)
    ax.add_patch(FancyBboxPatch((85.2, 37.9), 13.0, 5.6,
                                boxstyle="round,pad=0.25,rounding_size=0.7",
                                fc=SURFACE, ec=C_FULL, lw=1.2,
                                linestyle=(0, (4, 3)), zorder=3))
    txt(91.7, 41.9, T["trainonly"], 7.8, C_FULL, style="italic")
    txt(91.7, 40.4, T["trainhead"], 9.6, INK, weight="bold")
    txt(91.7, 38.9, T["celoss"], 8.6, C_FULL)

    # 部署出口（下，青）
    arrow(92.1, 16.1, 92.1, 13.6, color=C_AQUA, lw=1.5, ms=10)
    txt(92.1, 12.3, T["deploy"], 9.6, C_AQUA, weight="bold")
    txt(92.1, 10.6, T["deploysub"], 8.2, INK2)

    # 溯源脚注
    txt(0.8, 1.2, T["prov"], 7.4, MUTED, ha="left")

    fig.savefig(out_png, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("saved:", out_png)


if __name__ == "__main__":
    render("cn", ROOT / "figures_cn_20261106" / "panels" / "architecture_main_v2.png")
    render("en", ROOT / "figures_en_20261106" / "panels" / "architecture_main_en_v2.png")
