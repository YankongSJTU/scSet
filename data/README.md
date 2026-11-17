# Data

## Demo (bundled)

`demo/cell_inputs_demo.npz` + `demo/samples.csv`: 36 real tumour single-cell
samples (3 cancers — Breast, CRC, Esophageal — × 3 cohorts × 4 samples), ≤256
cells per sample, 2,070-d vectors (2,058-d STATE foundation embedding + 12
lineage covariates), stored float16 (~33 MB). Extracted from the paper's
benchmark cache with a fixed seed; keys are row indices of `samples.csv`.

The demo runs the full protocol (cohort-held-out rotation, shared subsampling,
shared probe, cohort-level endpoints) on 2 rounds — enough to exercise every
code path in about a minute on CPU. Numbers are illustrative, not the paper's
pre-registered evaluation (42 cohorts × 6 rounds × 3 seeds).

## Full benchmark cache

Build `cell_inputs.npz` for your own data with the same contract:
- `samples.csv`: `uid, cancer, cohort, sample_id` (one row per sample)
- npz keys `0..N-1` = row indices of `samples.csv`; values `(n_cells, dim)`
  float32/16 cell-embedding matrices.

Any foundation-model cell embedding works (STATE, scGPT — see
`scripts/embed_scgpt.py`). The fold registry is deterministic and written to
`results/folds.json` on every run.
