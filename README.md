# scSet — single-cell set encoder & cohort-held-out benchmark

**When do set encoders beat pooling? A pre-registered cohort-held-out benchmark
for transferable single-cell sample representations.**

scSet is a minimal permutation-invariant set encoder over foundation-model cell
embeddings: `Linear → GAT×2 (kNN graph, 4 heads) → LayerNorm → dual pooling
(mean ‖ 4 learnable queries) → MLP → L2`, trained with plain cross-entropy
(~3.5M parameters, one forward pass at deployment). This repository releases:

- the **scSet** encoder (terminal recipe) and the **scSet-Contrastive** initial
  prototype (with the Transformer block and the contrastive objective),
- the **pre-registered benchmark protocol** (cohort-held-out rotation folds,
  shared cell subsampling, shared probe, cohort-level endpoints),
- the **orthogonal backbone check** (scGPT) scripts,
- a bundled **demo** that runs end-to-end on a laptop CPU in about a minute.

```
python run_demo.py        # or: bash run_demo.sh
```

## Installation

```bash
pip install -r requirements.txt     # torch, numpy, pandas, scikit-learn, scipy
```

`torch_geometric` is **optional**: the GAT layer has a built-in pure-torch
implementation with identical semantics; if PyG is installed the original
`GATConv` of the paper runs is used instead.

## Demo (bundled, ~1 minute, CPU)

The demo bundles 36 real tumour samples (3 cancers × 3 cohorts, ≤256 cells per
sample, 2,058-d STATE foundation embeddings, float16) and runs the four
decisive configurations under the paper protocol on 2 rotation rounds:

```bash
bash run_demo.sh                      # checks deps + runs
# or directly:
python run_demo.py --rounds 2 --seeds 42,43 --cells 128 --epochs 8
```

Expected output (cohort-level balanced accuracy; n = 12 cohort-rounds, so
expect noise — the paper's pre-registered evaluation uses 42 cohorts × 6
rounds × 3 seeds):

```
                     mean
fm_mean            ~0.70
fm_sup_mean_mlp    ~0.60
scSet              ~0.60
scSet-Contrastive  ~0.55   (bottom — the contrastive term is toxic)
```

Without the bundled cache the demo falls back to a synthetic cache with
cluster structure, so the pipeline always runs.

## Full benchmark

```bash
python scripts/run_benchmark.py \
    --cache /path/to/cell_inputs.npz \
    --samples /path/to/samples.csv \
    --rounds 0,1,2,3,4,5 --seeds 42,43,44 --cells 256
```

Input contract: `cell_inputs.npz` keys = row indices of `samples.csv`; values
are `(n_cells_i × dim)` float arrays (STATE 2,058-d + 12 lineage covariates in
the paper). `samples.csv` needs `uid, cancer, cohort, sample_id`. The fold
registry (which cohort is held out in which round) is deterministic
lexicographic rotation and is written to `results/folds.json` for the record.

## Orthogonal backbone check (scGPT)

```bash
python scripts/embed_scgpt.py --model-dir /path/to/scGPT_human \
    --sample-manifest samples.csv --raw-h5ad-dir /path/to/h5ad \
    --out cell_inputs_scgpt.npz
CELLS_CACHE=cell_inputs_scgpt.npz python scripts/run_benchmark.py --tag scgpt ...
python scripts/id_eval_backbone.py --tag scgpt   # identity vs semantics
```

## Repository layout

```
scset/            model.py (encoder + GAT + losses), train.py (fold training)
benchmark/        evaluate.py (folds, shared subsampling, probe, metrics)
scripts/          run_benchmark.py, embed_scgpt.py, id_eval_backbone.py
data/demo/        36-sample real demo cache (float16) + registry
run_demo.py       one-command demo
run_demo.sh       dependency check + demo
results/          demo output (demo_cohort_rows.csv)
```

## Key findings (see the paper)

1. **Protocol decides ranking** — in-atlas diagnostics inflate learned
   encoders; under cohort hold-out the ordering inverts (scPoli 0.988 → 0.458).
2. **Fix the objective** — the contrastive term is toxic in every
   configuration; cross-entropy is the repair (+0.060 over the
   parameter-matched pooling control, Holm p = 0.0138).
3. **One mixing mechanism suffices** — a single GAT over the cell kNN graph;
   stacking a Transformer over-smooths.
4. **Two-scale saturation** — encoder information intake saturates at ~10³
   cells per sample, pooling at ~10².
5. **Identity–semantics antinomy** — pooled means solve identity (0.999) with
   no semantics; supervised encoders the mirror image; no objective family we
   tested preserves both.
6. **What transfers across backbones** — contrastive toxicity replicates on
   scGPT (−0.097, p = 0.0044); the encoder-over-pooling advantage is
   STATE-specific; the antinomy inverts (scGPT means: semantics 0.824,
   identity 0.0).

## Citation

```bibtex
@article{scset2026,
  title   = {When do set encoders beat pooling? A pre-registered
             cohort-held-out benchmark for transferable single-cell
             sample representations},
  author  = {Yan Kong and Hui Lu},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Artefacts & registry

- `registry/folds.json` — the **pre-registered fold registry** (frozen before any
  benchmark run; 42 cohorts, 6 rotation rounds, every cohort held out exactly
  once).
- `artefacts/` — per-run **cohort-level metrics and sample-level predictions**
  for every experiment family: pre-registered benchmark (results_full,
  results_screening), comparator (results_scvi, backbone_scgpt), ablations and
  sweeps (countsweep, dimsat, cells1024/2048/all, bagging), plus
  benchmark_registry.csv (the sample registry).
- `analysis/` — paired statistical analysis of the backbone check.

**Large per-run artefacts** (sample embeddings `.npz`, training histories,
per-run metrics/predictions for all runs, 925 MB tar.gz) are deposited at:
[Google Drive — scset_run_artefacts_full.tgz](https://drive.google.com/file/d/1vyQ6UJjs-70eNZ7HBCWnFUUIc-AI66am/view?usp=sharing)
(md5 `7c11376d2ac30a86bfcf48821a94fe31`, 3,928 entries).
