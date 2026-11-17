# -*- coding: utf-8 -*-
"""Orthogonal-backbone validation, step 3 (bonus, minutes of compute):
identity–semantics antinomy check on the new backbone, training-free part.

For fm_mean on the new backbone: 64-cell query vs 256-cell library
self-retrieval P@1 (exclude the sample itself) and disease-neighbour
retrieval — the antinomy's identity half. Expected if the antinomy is
backbone-general: self-retrieval ≈ 1.0, disease-neighbour far lower.

  CELLS_CACHE=.../cell_inputs_scgpt.npz python id_eval_backbone.py --tag scgpt
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark.evaluate import load_cells  # noqa: E402


def cos(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(a @ b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    assert os.environ.get("CELLS_CACHE"), "set CELLS_CACHE to the backbone cache"

    cache = Path(os.environ['CELLS_CACHE'])
    reg = pd.read_csv(cache.parent / 'samples.csv')
    reg['cancer_code'] = pd.Categorical(
        reg.cancer, categories=sorted(reg.cancer.unique())).codes
    y = reg.cancer_code.to_numpy()
    cells256 = load_cells(cache, reg, 0, args.seed, 256)
    cells64 = [c[np.random.default_rng(7000 + i).choice(len(c), 64, replace=False)]
               for i, c in enumerate(cells256)]
    Z256 = np.stack([c.mean(axis=0) for c in cells256])
    Z64 = np.stack([c.mean(axis=0) for c in cells64])
    Zn = Z256 / (np.linalg.norm(Z256, axis=1, keepdims=True) + 1e-12)
    Qn = Z64 / (np.linalg.norm(Z64, axis=1, keepdims=True) + 1e-12)
    sims = Qn @ Zn.T
    np.fill_diagonal(sims, -2.0)  # exclude self

    order = np.argsort(-sims, axis=1)
    top1 = order[:, 0]
    self_p1 = float((top1 == np.arange(len(reg))).mean())
    disease_p1 = float((y[top1] == y).mean())
    cos_self = float(np.mean([cos(Z64[i], Z256[i]) for i in range(len(reg))]))

    rows = [{
        "backbone": args.tag, "method": "fm_mean",
        "self_retrieval_P1": self_p1,
        "disease_neighbour_P1": disease_p1,
        "cos_64_vs_256": cos_self,
        "n": len(reg),
    }]
    outdir = Path("results") / f"backbone_{args.tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / "identity_fm_mean.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("saved", outdir / "identity_fm_mean.csv")


if __name__ == "__main__":
    main()
