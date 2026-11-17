# -*- coding: utf-8 -*-
"""Full pre-registered benchmark CLI: baselines + scSet variants under the
cohort-held-out rotation protocol with shared cells and probes.

  python scripts/run_benchmark.py --cache data/demo/cell_inputs_demo.npz \
      --samples data/demo/samples.csv --rounds 0,1,2,3,4,5 --seeds 42,43,44

Outputs results/benchmark_cohort_rows.csv + a printed summary. Methods:
  fm_mean            direct mean pooling of the cell embeddings
  fm_sup_mean_mlp    parameter-matched supervised projection (control)
  scSet              terminal recipe (proj-GAT-dual pooling, CE)
  scSet-Contrastive  initial prototype (full arch + Transformer, SupCon)
  scSet-CE           full architecture + cross-entropy (pre-registered winner)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scset.train import fm_mean_features, train_encoder, train_sup_mean_mlp
from benchmark.evaluate import (build_rounds, cohort_rows, load_cells,
                                probe_and_score, retrieval_scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--rounds", default="0,1,2,3,4,5")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--cells", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--methods", default="fm_mean,fm_sup_mean_mlp,scSet,"
                                         "scSet-Contrastive,scSet-CE")
    ap.add_argument("--out", default="results/benchmark_cohort_rows.csv")
    args = ap.parse_args()
    methods = args.methods.split(",")

    cache = Path(args.cache)
    samples_csv = Path(args.samples)
    # load_cells expects samples.csv next to the cache; honour an explicit path
    if samples_csv.parent != cache.parent:
        import shutil
        shutil.copy(samples_csv, cache.parent / "samples.csv")

    registry = pd.read_csv(samples_csv)
    registry["cancer_code"] = pd.Categorical(
        registry.cancer, categories=sorted(registry.cancer.unique())).codes
    y = registry.cancer_code.to_numpy()
    all_rounds = build_rounds(registry, n_rounds=6)
    folds = {f"round_{r}": held for r, held in enumerate(all_rounds)}
    Path("results").mkdir(exist_ok=True)
    Path("results/folds.json").write_text(json.dumps(folds, indent=1))

    rows, t0 = [], time.time()
    for r in [int(x) for x in args.rounds.split(",")]:
        held = all_rounds[r]
        test_mask = registry.cohort.isin(held).to_numpy()
        tr, te = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
        for seed in [int(s) for s in args.seeds.split(",")]:
            cells = load_cells(cache, registry, r, seed, args.cells)
            feats = {"fm_mean": fm_mean_features(cells)}
            if "fm_sup_mean_mlp" in methods:
                feats["fm_sup_mean_mlp"] = train_sup_mean_mlp(
                    feats["fm_mean"], y, tr, seed)
            if "scSet-CE" in methods:
                feats["scSet-CE"] = train_encoder(
                    cells, y, tr, seed, "ce", use_transformer=True,
                    epochs=args.epochs)
            if "scSet-Contrastive" in methods:
                feats["scSet-Contrastive"] = train_encoder(
                    cells, y, tr, seed, "supcon", use_transformer=True,
                    epochs=args.epochs)
            if "scSet" in methods:
                feats["scSet"] = train_encoder(
                    cells, y, tr, seed, "ce", use_transformer=False,
                    epochs=args.epochs)
            for method, Z in feats.items():
                pred, sc = probe_and_score(Z[tr], y[tr], Z[te], y[te])
                ret = retrieval_scores(Z[tr], y[tr], Z[te], y[te])
                base = {"method": method, "round": r, "seed": seed,
                        "cells": args.cells, **sc, **ret}
                rows.extend(cohort_rows(registry.iloc[te], y[te], pred, base))
            print(f"round {r} seed {seed} done ({time.time() - t0:.0f}s)",
                  flush=True)

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows)
    summary = (df.groupby("method")["cohort_balanced_accuracy"]
               .agg(["mean", "std", "count"]).round(4))
    print("\n=== Benchmark summary (cohort-level balanced accuracy) ===")
    print(summary.to_string())
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
