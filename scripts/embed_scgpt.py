# -*- coding: utf-8 -*-
"""Backbone validation step 1 (server): extract scGPT whole-human cell
embeddings for the benchmark samples into a cell_inputs-compatible cache.

Sample selection mirrors extract_cells.py exactly:
  manifest['inputs'] entries {cohort, sample_key, path}; obs[sample_key] is
  categorical (codes/categories); key = f"{cohort}::{sample_id}" -> row of
  cce_nm_20261017/samples.csv; per-sample subsample rng(1000 + row).
Only samples present in the ORIGINAL cell_inputs.npz are extracted (exact
sample-set parity with the pre-registered benchmark).

Run (GPU):  CUDA_VISIBLE_DEVICES=7 ~/run_venv.sh embed_scgpt_server.py \
                --out cell_inputs_scgpt --max-cells 2048 --batch 32
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

BASE = Path("/export/home/kongyan/project/singlecellsamplecluster")
NM = BASE / "cce_nm_20261017"
BENCH = BASE / "cce_nmi_benchmark_20261028"
MODEL_DIR = BENCH / "scGPT_human"


def load_model(device):
    from scgpt.model import TransformerModel

    vocab = json.loads((MODEL_DIR / "vocab.json").read_text())
    args = json.loads((MODEL_DIR / "args.json").read_text())
    for tok in ("<pad>", "<cls>"):
        assert tok in vocab, f"vocab missing {tok}"
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=int(args["embsize"]),
        nhead=int(args["nheads"]),
        d_hid=int(args["d_hid"]),
        nlayers=int(args["nlayers"]),
        nlayers_cls=int(args["n_layers_cls"]),
        n_cls=1,
        vocab=vocab,
        do_mvc=True,
        dropout=float(args["dropout"]),
        pad_token="<pad>",
        pad_value=-2,
        input_emb_style=args["input_emb_style"],
        n_input_bins=int(args["n_bins"]),
        cell_emb_style="cls",
        use_fast_transformer=False,
    )
    sd = torch.load(MODEL_DIR / "best_model.pt", map_location="cpu")
    # checkpoint saved with flash-attn layers: remap Wqkv -> in_proj (same layout)
    remapped = {}
    for k, v in sd.items():
        if k == "flag_encoder.weight":
            continue  # generative flag, unused in embedding
        nk = k.replace("self_attn.Wqkv.weight", "self_attn.in_proj_weight")
        nk = nk.replace("self_attn.Wqkv.bias", "self_attn.in_proj_bias")
        remapped[nk] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    assert not unexpected, f"unexpected: {unexpected[:3]}"
    assert all("cls_decoder" in k for k in missing), f"missing: {missing[:3]}"
    model.to(device).eval()
    return model, vocab, int(args["n_bins"]), int(args["max_seq_len"])


def stage_with_retry(src: Path, dst: Path, tries: int = 6, wait: int = 30):
    """NFS intermittently returns Errno 5; copy locally with verify+retry."""
    import shutil
    import time as _t
    for attempt in range(tries):
        try:
            if dst.exists():
                dst.unlink()
            shutil.copyfile(src, dst)
            if dst.stat().st_size == src.stat().st_size:
                return dst
            print(f"  size mismatch staging {src.name} "
                  f"({dst.stat().st_size} vs {src.stat().st_size}), retry", flush=True)
        except OSError as e:
            print(f"  staging error attempt {attempt+1}: {e}; retry in {wait}s",
                  flush=True)
        _t.sleep(wait)
    raise RuntimeError(f"failed to stage {src} after {tries} attempts")


def read_x(f):
    """X may be a dense dataset or a sparse CSR group across cohorts."""
    X = f["X"]
    if hasattr(X, "shape"):
        return X                                   # dense dataset
    from scipy.sparse import csr_matrix
    data = np.asarray(X["data"])
    indices = np.asarray(X["indices"])
    indptr = np.asarray(X["indptr"])
    shape = tuple(np.asarray(X.attrs["shape"]))
    return csr_matrix((data, indices, indptr), shape=shape)


@torch.inference_mode()
def embed_matrix(model, X, gene_ids, in_vocab, n_bins, max_len, device,
                 batch=32, cls_id=60695, pad_id=60694):
    """X: (n_cells, n_genes) raw counts; returns (n_cells, d) CLS embeddings."""
    from scgpt.preprocess import binning

    n_cells = X.shape[0]
    Xb = binning(X, n_bins=n_bins)  # (n_cells, n_genes), zeros stay 0

    embs = []
    for off in range(0, n_cells, batch):
        XB = Xb[off:off + batch]
        seqs = []
        for row in XB:
            nz = np.flatnonzero(row)
            if len(nz) > max_len - 1:
                nz = nz[np.argsort(row[nz])[::-1][: max_len - 1]]
            ids = [gene_ids[j] for j in nz if in_vocab[j]]
            vals = np.asarray([row[j] for j in nz if in_vocab[j]],
                              dtype=np.float32)
            seqs.append((ids, vals))
        L = max(len(s[0]) for s in seqs) + 1  # +1 for <cls>
        B = len(seqs)
        tok = torch.full((B, L), pad_id, dtype=torch.long)
        val = torch.zeros((B, L), dtype=torch.float32)
        src = torch.zeros((B, L), dtype=torch.bool)
        for bi, (ids, vals) in enumerate(seqs):
            tok[bi, 0] = cls_id
            src[bi, 0] = True
            for gi, (g, v) in enumerate(zip(ids, vals)):
                tok[bi, gi + 1] = int(g)
                val[bi, gi + 1] = float(v)
                src[bi, gi + 1] = True
        out = model._encode(tok.to(device), val.to(device),
                            ~src.to(device))
        embs.append(out[:, 0, :].cpu().numpy())  # CLS position
    return np.concatenate(embs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cell_inputs_scgpt")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: BENCH); use a LOCAL disk "
                         "e.g. /data1/... if NFS is unreliable)")
    ap.add_argument("--max-cells", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit-cohorts", type=int, default=0,
                    help="smoke test: only first N cohorts")
    ap.add_argument("--checkpoint-every", type=int, default=8,
                    help="write partial npz every N cohorts")
    args = ap.parse_args()
    OUTDIR = Path(args.out_dir) if args.out_dir else BENCH
    OUTDIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, n_bins, max_len = load_model(device)
    print(f"model loaded on {device}; vocab {len(vocab)}", flush=True)

    samples = pd.read_csv(NM / "samples.csv")
    uid_pos = {f"{r.cohort}::{str(r.sample_id)}": i
               for i, r in enumerate(samples.itertuples())}
    orig_keys = set(np.load(NM / "cell_inputs.npz").files)
    manifest = json.loads((NM / "manifest.json").read_text())

    out = {}
    cohort_out = {}
    n_kept = 0
    inputs = manifest["inputs"][: args.limit_cohorts] if args.limit_cohorts \
        else manifest["inputs"]
    for ci, entry in enumerate(inputs):
        cohort, skey = entry["cohort"], entry["sample_key"]
        path = Path(entry["path"])
        if not path.exists():
            print(f"[{cohort}] MISSING {path}", flush=True)
            continue
        shard_dir = OUTDIR / f"{args.out}_shards"
        shard_file = shard_dir / f"shard_{ci:02d}_{cohort}.npz"
        if shard_file.exists():
            z = np.load(shard_file)
            for k in z.files:
                out[k] = z[k]
                cohort_out[k] = z[k]
            n_kept += len(z.files)
            print(f"[{cohort}] shard exists, resumed ({len(z.files)} samples)",
                  flush=True)
            continue
        stage_dir = Path("/data1/TumorGroup/tmp/scgpt_stage_kongyan")
        stage_dir.mkdir(parents=True, exist_ok=True)
        staged = stage_with_retry(path, stage_dir / path.name)
        with h5py.File(staged, "r") as f:
            obs = f["obs"]
            codes = np.asarray(obs[skey]["codes"])
            cats = [c.decode() for c in np.asarray(obs[skey]["categories"])]
            sample_of_cell = np.asarray([cats[c] for c in codes])
            var_key = next((k for k in ("gene", "_index", "Index")
                             if k in f["var"]), None) or list(f["var"].keys())[0]
            genes = [g.decode() if isinstance(g, bytes) else g
                     for g in np.asarray(f["var"][var_key])]
            print(f"  [{cohort}] var key: {var_key}", flush=True)
            # vocab lookup per gene (once per file)
            in_vocab = np.array([g in vocab for g in genes])
            gene_ids = np.array([vocab[g] if g in vocab else 0 for g in genes],
                                dtype=np.int64)
            n_in = int(in_vocab.sum())
            Xmat = read_x(f)
            for sid in np.unique(sample_of_cell):
                key = f"{cohort}::{sid}"
                if key not in uid_pos or str(uid_pos[key]) not in orig_keys:
                    continue
                row_cells = np.flatnonzero(sample_of_cell == sid)
                rng = np.random.default_rng(1000 + uid_pos[key])
                idx = rng.choice(len(row_cells),
                                 min(args.max_cells, len(row_cells)),
                                 replace=False)
                Xsl = Xmat[sorted(row_cells[idx]), :]
                X = (Xsl.toarray() if hasattr(Xsl, "toarray")
                     else np.asarray(Xsl, dtype=np.float32)).astype(np.float32)
                E = embed_matrix(model, X, gene_ids, in_vocab, n_bins,
                                 max_len, device, batch=args.batch)
                out[str(uid_pos[key])] = E.astype(np.float32)
                cohort_out[str(uid_pos[key])] = out[str(uid_pos[key])]
                n_kept += 1
        staged.unlink(missing_ok=True)
        print(f"[{cohort}] genes-in-vocab {n_in}/{len(genes)}; "
              f"samples kept {n_kept}", flush=True)
        shard_dir.mkdir(parents=True, exist_ok=True)
        if cohort_out:
            np.savez_compressed(shard_dir / f"shard_{ci:02d}_{cohort}.npz",
                                **cohort_out)
            print(f"  shard {ci:02d} written ({len(cohort_out)} samples)",
                  flush=True)
        cohort_out = {}

    np.savez_compressed(OUTDIR / f"{args.out}.npz", **out)
    print(f"WROTE {args.out}.npz: {len(out)} samples, "
          f"mean cells {np.mean([len(v) for v in out.values()]):.0f}, "
          f"dim {next(iter(out.values())).shape[1]}", flush=True)


if __name__ == "__main__":
    main()
