# -*- coding: utf-8 -*-
"""Backbone (scGPT) orthogonal validation: paired analysis, 42 cohorts.
输入：cohort_rows.csv（backbone_scgpt）+ identity CSV。
输出：终端表格 + 保存 summary_backbone_scgpt.csv。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

src = Path(sys.argv[1] if len(sys.argv) > 1 else "cohort_rows.csv")
df = pd.read_csv(src)
df = df[df.cohort.notna()]

wide = (df.groupby(["method", "cohort"])["cohort_balanced_accuracy"]
        .mean().unstack(0))
print("cohorts per method:\n", wide.notna().sum(), "\n")

METHODS = ["scgpt_fm_mean", "scgpt_fm_sup_mean_mlp",
           "scgpt_scset_contrast", "scgpt_scset"]
rows = []
base = "scgpt_fm_sup_mean_mlp"
for m in METHODS:
    sub = wide[[m, base]].dropna()
    mean = sub[m].mean()
    d = (sub[m] - sub[base]).mean()
    if m == base:
        p, dmean = np.nan, 0.0
    else:
        try:
            p = wilcoxon(sub[m], sub[base]).pvalue
        except ValueError:
            p = np.nan
        dmean = (sub[m] - sub[base]).mean()
    rows.append({"method": m, "n_cohorts": len(sub), "mean": round(mean, 4),
                 "delta_vs_sup": round(dmean, 4),
                 "wilcoxon_p": round(float(p), 4) if p == p else None})
    print(f"{m:24s} mean {mean:.4f}  Δvs-sup {dmean:+.4f}  p={p if p==p else float('nan'):.4f}  n={len(sub)}")

# 关键排序检查：scSet > fm_sup_mean_mlp ?  contrast 最差？
m_ = {r["method"]: r for r in rows}
print("\norder:", " > ".join(sorted([r["method"] for r in rows],
      key=lambda k: -m_[k]["mean"])))

out = src.parent / "summary_backbone_scgpt.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("saved", out)

# identity
idc = src.parent / "identity_fm_mean.csv"
if idc.exists():
    print("\nidentity (fm_mean):\n", pd.read_csv(idc).to_string(index=False))
