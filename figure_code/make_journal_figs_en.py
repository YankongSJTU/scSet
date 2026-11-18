# -*- coding: utf-8 -*-
"""期刊格式组合图（EN）：每个 Fig 2–11 由 3 个面板组合成单幅多面板图。
- 布局自动选择（a-top / a-left / 单行 / 单列），取组合纵横比最接近 1.45 者；
- 面板左上角加期刊式粗体小写字母（a/b/c，合并面板 c, d）；
- 白底、留白，输出 manuscript/figures_word/journal/figN.png。
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
PANELS = ROOT / "figures_en_20261106" / "panels"
OUT = ROOT / "manuscript" / "figures_word" / "journal"
OUT.mkdir(parents=True, exist_ok=True)

# 每图面板列表：(文件名, 字母标签)
FIGS = {
    2: [("fig1a_petal.png", "a"), ("fig1b_paired.png", "b"),
        ("fig1cd_retrieval_gap.png", "c,d")],
    3: [("fig2a_comparator.png", "a"), ("fig2b_scpoli.png", "b"),
        ("fig2c_scvi.png", "c")],
    4: [("fig3a_arch.png", "a"), ("fig3b_paired.png", "b"),
        ("fig3c_input.png", "c")],
    5: [("fig4a_loss.png", "a"), ("fig4b_planb.png", "b"),
        ("fig4c_2048.png", "c")],
    6: [("fig5a_cellsat.png", "a"), ("fig5b_delta.png", "b"),
        ("fig5c_baseline.png", "c")],
    7: [("fig6a_acc.png", "a"), ("fig6b_p1.png", "b"),
        ("fig6c_cosself.png", "c")],
    8: [("fig7a_dimsat.png", "a"), ("fig7b_width.png", "b"),
        ("fig7c_bagging.png", "c")],
    9: [("fig8a_radar.png", "a"), ("fig8c_jiang.png", "b"),
        ("fig8d_deficit.png", "c")],
    10: [("fig9a_icb.png", "a"), ("fig9b_law.png", "b"),
         ("fig9c_frozen.png", "c")],
    11: [("fig10a_ssl.png", "a"), ("fig10b_identity.png", "b"),
         ("fig10c_geom.png", "c")],
}

BAND = 104      # 面板上方字母带高
GAP = 44        # 面板间距
MARGIN = 30     # 左右/下边距
TARGET_AR = 1.45
FONT = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 66)


def load(panels):
    return [(Image.open(PANELS / fn).convert("RGB"), lab)
            for fn, lab in panels]


def layout_size(imgs, kind):
    """返回 (canvas_w, canvas_h, 摆放函数位置表)。"""
    (a, _), (b, _), (c, _) = imgs
    wa, ha = a.size
    wb, hb = b.size
    wc, hc = c.size
    if kind == "a-top":           # a 顶部居中，b/c 底行
        w = max(wa, wb + GAP + wc)
        h = BAND + ha + GAP + BAND + max(hb, hc)
        pos = [((w - wa) // 2, BAND),
               ((w - (wb + GAP + wc)) // 2, BAND + ha + GAP + BAND),
               ((w - (wb + GAP + wc)) // 2 + wb + GAP,
                BAND + ha + GAP + BAND)]
    elif kind == "a-left":        # a 左列，b/c 右列上下
        w = wa + GAP + BAND + max(wb, wc)
        h = max(BAND + ha, BAND + hb + GAP + BAND + hc)
        pos = [(0, BAND),
               (wa + GAP + BAND, BAND),
               (wa + GAP + BAND, BAND + hb + GAP + BAND)]
    elif kind == "row3":
        w = wa + GAP + BAND + wb + GAP + BAND + wc
        h = BAND + max(ha, hb, hc)
        pos = [(0, BAND), (wa + GAP + BAND, BAND),
               (wa + GAP + BAND + wb + GAP + BAND, BAND)]
    else:                          # col3
        w = max(wa, wb, wc)
        h = (BAND + ha + GAP + BAND + hb + GAP + BAND + hc)
        pos = [((w - wa) // 2, BAND), ((w - wb) // 2, BAND + ha + GAP + BAND),
               ((w - wc) // 2, BAND + ha + GAP + BAND + hb + GAP + BAND)]
    return w, h, pos


def compose(fig_n, panels):
    imgs = load(panels)
    best = None
    for kind in ("a-top", "a-left", "row3", "col3"):
        w, h, pos = layout_size(imgs, kind)
        ar = (w + 2 * MARGIN) / (h + MARGIN)
        score = abs(ar - TARGET_AR)
        if best is None or score < best[0]:
            best = (score, kind, w, h, pos)
    _, kind, w, h, pos = best
    W, H = w + 2 * MARGIN, h + MARGIN
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    for (img, lab), (x, y) in zip(imgs, pos):
        px, py = x + MARGIN, y
        canvas.paste(img, (px, py))
        draw.text((px - 8, py - BAND + 18), lab, font=FONT, fill=(26, 26, 26))
    out = OUT / f"fig{fig_n}.png"
    canvas.save(out)
    print(f"fig{fig_n}: {kind}  {W}x{H}  ar={W/H:.2f}")


if __name__ == "__main__":
    for n, panels in FIGS.items():
        compose(n, panels)
