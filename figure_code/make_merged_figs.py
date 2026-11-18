# -*- coding: utf-8 -*-
"""CNS 风格合并大图：12 图 → 6 图（2×3 / 3×3 密排），子图字母统一排布。
输出 manuscript/figures_word/journal_merged/figN.png（论文 v5 用）+
paper_figures_final/EN|CN/figures 同步（fig01-06 PNG+PDF）。
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent

MERGED = {  # figN -> (rows, cols, [(panel_file, letter), ...])
    2: (2, 3, [("fig1a_petal.png", "a"), ("fig1b_paired.png", "b"),
               ("fig1cd_retrieval_gap.png", "c, d"),
               ("fig2a_comparator.png", "e"), ("fig2b_scpoli.png", "f"),
               ("fig2c_scvi.png", "g")]),
    3: (2, 3, [("fig3a_arch.png", "a"), ("fig3b_paired.png", "b"),
               ("fig3c_input.png", "c"),
               ("fig4a_loss.png", "d"), ("fig4b_planb.png", "e"),
               ("fig4c_2048.png", "f")]),
    4: (3, 3, [("fig5a_cellsat.png", "a"), ("fig5b_delta.png", "b"),
               ("fig5c_baseline.png", "c"),
               ("fig6a_acc.png", "d"), ("fig6b_p1.png", "e"),
               ("fig6c_cosself.png", "f"),
               ("fig7a_dimsat.png", "g"), ("fig7b_width.png", "h"),
               ("fig7c_bagging.png", "i")]),
    5: (2, 3, [("fig8a_radar.png", "a"), ("fig8c_jiang.png", "b"),
               ("fig8d_deficit.png", "c"),
               ("fig9a_icb.png", "d"), ("fig9b_law.png", "e"),
               ("fig9c_frozen.png", "f")]),
    6: (2, 3, [("fig10a_ssl.png", "a"), ("fig10b_identity.png", "b"),
               ("fig10c_geom.png", "c"),
               ("fig12a_backbone.png", "d"), ("fig12b_backbone.png", "e"),
               ("fig12c_backbone.png", "f")]),
}
FONT = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 64)


def compose(panel_dir: Path, rows, cols, panels, out: Path, W=2500):
    imgs = [(Image.open(panel_dir / f).convert("RGB"), lab)
            for f, lab in panels]
    cell_w = W // cols
    BAND, GAP, MARGIN = 96, 34, 26
    # 每格按宽度等比缩放，行高取该行最大
    scaled = []
    for im, lab in imgs:
        w = cell_w - 2 * MARGIN
        im2 = im.resize((w, round(im.height * w / im.width)),
                        Image.LANCZOS)
        scaled.append((im2, lab))
    row_h = []
    for r in range(rows):
        hs = [scaled[r * cols + c][0].height for c in range(cols)]
        row_h.append(max(hs))
    H = BAND * rows + sum(row_h) + GAP * (rows - 1) + 2 * MARGIN
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    y = MARGIN
    for r in range(rows):
        for c in range(cols):
            im, lab = scaled[r * cols + c]
            x = c * cell_w + MARGIN
            # 垂直居中于行高
            yy = y + (row_h[r] - im.height) // 2
            canvas.paste(im, (x, yy + BAND // 2))
            draw.text((x - 8, y + 6), lab, font=FONT, fill=(26, 26, 26))
        y += row_h[r] + BAND + GAP
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return canvas.size


def png_pdf(png: Path, pdf: Path):
    import fitz
    img = Image.open(png)
    doc = fitz.open()
    page = doc.new_page(width=img.width * 72 / 300, height=img.height * 72 / 300)
    page.insert_image(page.rect, filename=str(png))
    doc.save(str(pdf))
    doc.close()


def build(lang: str, panel_dir: Path):
    outdir = ROOT / "manuscript" / "figures_word" / "journal_merged" / lang
    outdir.mkdir(parents=True, exist_ok=True)
    # fig1 = 架构图整幅
    arch = "architecture_main_en_v2.png" if lang == "en" \
        else "architecture_main_v2.png"
    compose(panel_dir, 1, 1, [(arch, "a")], outdir / "fig1.png", W=2500)
    for n, (rows, cols, panels) in MERGED.items():
        size = compose(panel_dir, rows, cols, panels, outdir / f"fig{n}.png")
        print(f"[{lang}] fig{n}: {rows}x{cols} {size}")
    return outdir


if __name__ == "__main__":
    for lang, pdir in [("en", ROOT / "figures_en_20261106" / "panels"),
                       ("cn", ROOT / "figures_cn_20261106" / "panels")]:
        outdir = build(lang, pdir)
        # 同步到 paper_figures_final
        fin = ROOT / "paper_figures_final" / lang.upper() / "figures"
        fin.mkdir(parents=True, exist_ok=True)
        # 清掉旧 fig01-12，写新 fig01-06
        for f in list(fin.glob("fig*.png")) + list(fin.glob("fig*.pdf")):
            f.unlink()
        for n in range(1, 7):
            shutil_src = outdir / f"fig{n}.png"
            shutil_src.replace(fin / f"fig{n:02d}.png") \
                if False else None
            import shutil
            shutil.copy2(shutil_src, fin / f"fig{n:02d}.png")
            png_pdf(fin / f"fig{n:02d}.png", fin / f"fig{n:02d}.pdf")
        print(f"[{lang}] paper_figures_final/{lang.upper()}/figures -> 6 figs")
