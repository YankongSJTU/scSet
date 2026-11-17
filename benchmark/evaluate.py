# -*- coding: utf-8 -*-
"""Benchmark protocol: fold registry, shared subsampling, probe, metrics.

Mirrors the pre-registered protocol of the paper:
  - cohort-held-out rotation rounds; every cohort tested exactly once
  - one shared RNG per (round, seed) draws the inference-time cell subsample,
    identical for every method
  - probe = StandardScaler + class-balanced LogisticRegression(C=1), fitted
    on training cohorts only
  - endpoints: cohort-level balanced accuracy (primary), macro-F1,
    same-cancer retrieval P@1/P@5
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------ fold logic
def build_rounds(registry: pd.DataFrame, n_rounds: int = 6):
    """round r holds out the r-th cohort (lexicographic) of every cancer."""
    held_by_cancer = {c: sorted(g.cohort.unique())
                      for c, g in registry.groupby("cancer")}
    rounds = []
    for r in range(n_rounds):
        held = [held_by_cancer[c][r] for c in sorted(held_by_cancer)
                if r < len(held_by_cancer[c])]
        rounds.append(held)
    return rounds


def load_folds(path):
    return json.loads(Path(path).read_text())


def subsample_rng(round_: int, seed: int):
    """The pre-registered shared RNG (round/seed keyed)."""
    return np.random.default_rng(100000 + 1000 * int(round_) + int(seed))


def load_cells(cache: Path, registry: pd.DataFrame, round_: int, seed: int,
               max_cells: int = 256):
    """Shared deterministic subsample for one (round, seed); every method
    consumes exactly these cells."""
    f = np.load(cache)
    base = pd.read_csv(cache.parent / "samples.csv")
    idx = base.set_index("uid").index.get_indexer(registry.uid)
    if np.any(idx < 0):
        raise ValueError("registry uid missing from cache registry")
    rng = subsample_rng(round_, seed)
    return [f[str(i)][rng.choice(len(f[str(i)]), min(max_cells, len(f[str(i)])),
                                 replace=False)] for i in idx]


# ---------------------------------------------------------- probe & score
def probe_and_score(Z_train, y_train, Z_test, y_test):
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1, max_iter=2000, class_weight="balanced"))
    probe.fit(Z_train, y_train)
    pred = probe.predict(Z_test)
    return pred, {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }


def retrieval_scores(Z_train, y_train, Z_test, y_test, ks=(1, 5)):
    Ztr = Z_train / (np.linalg.norm(Z_train, axis=1, keepdims=True) + 1e-12)
    Zte = Z_test / (np.linalg.norm(Z_test, axis=1, keepdims=True) + 1e-12)
    sims = Zte @ Ztr.T
    out = {}
    for k in ks:
        top = np.argsort(-sims, axis=1)[:, :k]
        out[f"precision_at_{k}"] = float(
            (y_train[top] == y_test[:, None]).mean())
    return out


def cohort_rows(registry_test, y_test, pred, base):
    rows = []
    for coh in np.unique(registry_test.cohort):
        m = registry_test.cohort.to_numpy() == coh
        yt, yp = y_test[m], pred[m]
        rows.append({
            **base, "cohort": coh,
            "cancer": registry_test.cancer.to_numpy()[m][0],
            "n_test": int(m.sum()),
            "cohort_accuracy": float(accuracy_score(yt, yp)),
            "cohort_balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "cohort_macro_f1": float(f1_score(yt, yp, average="macro",
                                               labels=np.unique(yt),
                                               zero_division=0))})
    return rows
