#!/usr/bin/env bash
# scSet one-click demo: check deps -> run demo benchmark -> print results
set -e
cd "$(dirname "$0")"

echo '[1/3] checking python + dependencies ...'
python -c 'import numpy, pandas, sklearn, scipy, torch' 2>/dev/null ||   pip install -r requirements.txt

echo '[2/3] running scSet vs pooling demo (CPU, ~1 min) ...'
python run_demo.py --rounds 2 --seeds 42,43 --cells 128 --epochs 8

echo '[3/3] done. results in results/demo_cohort_rows.csv'
