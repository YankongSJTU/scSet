# -*- coding: utf-8 -*-
"""One-click demo: scSet vs pooling on a small cohort-held-out benchmark.

Runs the four decisive configurations of the paper on the bundled demo data
(real 2058-d STATE-embedded cell subsets when available, otherwise a
synthetic cache with cluster structure), under the pre-registered protocol:
shared cell subsampling, identical probe, cohort-level balanced accuracy.

  python run_demo.py                 # uses data/demo/, writes results/
  python run_demo.py --real data/demo/cell_inputs_demo.npz
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))

from scset.train import (fm_mean_features, train_encoder,
                         train_sup_mean_mlp)
from benchmark.evaluate import (build_rounds, cohort_rows, load_cells,
                                probe_and_score, retrieval_scores)

HERE = Path(__file__).parent
DATA = HERE / "data" / "demo"
OUT = HERE / "results"


def make_synthetic_cache(n_cancers=3, cohorts_per_cancer=3, samples_per_cohort=4,
                         cells=256, dim=512, seed=0):
    """Synthetic cache with per-cancer cluster structure (fallback demo)."""
    rng = np.random.default_rng(seed)
    rows, cache = [], {}
    uid = 0
    for ci in range(n_cancers):
        center = rng.normal(0, 1, dim)
        for co in range(cohorts_per_cancer):
            shift = rng.normal(0, 0.6, dim)  # cohort batch effect
            for si in range(samples_per_cohort):
                X = center + shift + rng.normal(0, 0.55, (cells, dim))
                cache[str(uid)] = X.astype(np.float32)
                rows.append({"uid": f"s{uid:04d}",
                             "cancer": f"Cancer{ci}",
                             "cohort": f"SynC{co}_Cancer{ci}",
                             "sample_id": f"s{uid:04d}"})
                uid += 1
    return cache, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(DATA / "cell_inputs_demo.npz"))
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--cells", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    cache_path = Path(args.cache)
    if cache_path.exists():
        cache_file = cache_path
        registry = pd.read_csv(cache_path.parent / "samples.csv")
        tag = cache_path.stem
        print(f"[demo] real cache: {cache_file.name} "
              f"({len(registry)} samples)")
    else:
        cache, registry = make_synthetic_cache()
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(DATA / "cell_inputs_synth.npz", **cache)
        registry.to_csv(DATA / "samples.csv", index=False)
        cache_file = DATA / "cell_inputs_synth.npz"
        tag = "synth"
        print(f"[demo] synthetic cache generated ({len(registry)} samples)")

    registry["cancer_code"] = pd.Categorical(
        registry.cancer, categories=sorted(registry.cancer.unique())).codes
    y = registry.cancer_code.to_numpy()
    rounds = build_rounds(registry, n_rounds=max(args.rounds,
                                                 registry.cohort.nunique()))
    rounds = rounds[: args.rounds]

    t0 = time.time()
    rows = []
    for r, held in enumerate(rounds):
        test_mask = registry.cohort.isin(held).to_numpy()
        tr, te = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
        for seed in seeds:
            cells = load_cells(cache_file, registry, r, seed, args.cells)
            feats = {"fm_mean": fm_mean_features(cells)}
            feats["fm_sup_mean_mlp"] = train_sup_mean_mlp(
                feats["fm_mean"], y, tr, seed)
            feats["scSet"] = train_encoder(
                cells, y, tr, seed, "ce", use_transformer=False,
                epochs=args.epochs)
            feats["scSet-Contrastive"] = train_encoder(
                cells, y, tr, seed, "supcon", use_transformer=True,
                epochs=args.epochs)
            for method, Z in feats.items():
                pred, sc = probe_and_score(Z[tr], y[tr], Z[te], y[te])
                ret = retrieval_scores(Z[tr], y[tr], Z[te], y[te])
                base = {"method": method, "round": r, "seed": seed,
                        "cells": args.cells, **sc, **ret}
                rows.extend(cohort_rows(registry.iloc[te], y[te], pred, base))
            print(f"  round {r} seed {seed} done "
                  f"({time.time() - t0:.0f}s)", flush=True)

    OUT.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "demo_cohort_rows.csv", index=False)
    summary = (df.groupby("method")["cohort_balanced_accuracy"]
               .agg(["mean", "std", "count"]).round(4))
    print("\n=== Demo results (cohort-level balanced accuracy) ===")
    print(summary.to_string())
    print(f"\nsaved: {OUT / 'demo_cohort_rows.csv'}")
    return summary


if __name__ == "__main__":
    main()
