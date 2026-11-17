# -*- coding: utf-8 -*-
"""Training and inference helpers for scSet.

train_encoder(): one fold of cohort-held-out training (scSet or
scSet-Contrastive). eval_features(): baseline feature builders
(fm_mean, fm_sup_mean_mlp) for the same fold. All consumers share the
per-(round, seed) cell subsample so comparisons are exact on inputs.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import scSet, knn_edge_index, supcon_loss


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_encoder(cells_list, y, train_idx, seed, objective="ce",
                  use_transformer=False, d=256, epochs=20, accum=8,
                  lr=1e-3, wd=1e-4, n_classes=None):
    """Train on train_idx sample sets; embed ALL samples; returns (N, d)."""
    dev = _dev()
    torch.manual_seed(seed)
    n_cls = int(y.max()) + 1 if n_classes is None else n_classes
    enc = scSet(cells_list[0].shape[1], d=d, use_transformer=use_transformer,
                n_classes=n_cls).to(dev)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=wd)
    samples = [torch.as_tensor(c, dtype=torch.float32) for c in cells_list]
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        perm = rng.permutation(len(train_idx))
        for off in range(0, len(train_idx), accum):
            batch = perm[off:off + accum]
            if len(batch) < 2:
                continue
            zs, logits, yb = [], [], []
            for si in train_idx[batch]:
                x = samples[si].to(dev)
                z, lg = enc(x)
                zs.append(z.squeeze(0))
                logits.append(lg.squeeze(0))
                yb.append(int(y[si]))
            yb_t = torch.as_tensor(yb, device=dev)
            if objective == "ce":
                loss = F.cross_entropy(torch.stack(logits), yb_t)
            else:  # supervised contrastive across the sample batch
                loss = supcon_loss(torch.stack(zs), yb_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
            opt.step()
            opt.zero_grad()
    enc.eval()
    out = []
    with torch.inference_mode():
        for c in samples:
            out.append(enc.embed_sample(c.to(dev)).cpu().numpy())
    return np.stack(out)


class MeanMLP(nn.Module):
    """Parameter-matched supervised projection of the pooled mean (control)."""

    def __init__(self, input_dim, hidden=512, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Dropout(0.1), nn.Linear(hidden, output_dim), nn.GELU(),
            nn.LayerNorm(output_dim), nn.Linear(output_dim, output_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def train_sup_mean_mlp(Z, y, train_idx, seed, epochs=100, batch=32,
                       lr=1e-3, wd=1e-4):
    dev = _dev()
    torch.manual_seed(seed)
    m = MeanMLP(Z.shape[1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    X = torch.as_tensor(Z[train_idx], dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y[train_idx], dtype=torch.long, device=dev)
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=dev)
        for off in range(0, len(X), batch):
            idx = perm[off:off + batch]
            if len(idx) < 2:
                continue
            loss = F.cross_entropy(m(X[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    with torch.inference_mode():
        return m(torch.as_tensor(Z, dtype=torch.float32, device=dev)).cpu().numpy()


def fm_mean_features(cells_list):
    return np.stack([np.asarray(c, dtype=np.float32).mean(axis=0)
                     for c in cells_list])
