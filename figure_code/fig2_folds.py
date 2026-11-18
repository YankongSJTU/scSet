# -*- coding: utf-8 -*-
"""Fig. 2b: fold-design matrix (42 cohorts x 6 rounds; held-out vs train)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent.parent
RES = HERE / 'results'
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)

# palette roles (dataviz reference, light mode)
INK = '#0b0b0b'
MUTED = '#898781'
GRID = '#e1e0d9'
TRAIN = '#e1e0d9'      # neutral surface gray: in training
HELD = '#2a78d6'       # slot-1 blue: held out
SURFACE = '#fcfcfb'

reg = pd.read_csv(RES / 'benchmark_registry.csv')
design = json.loads((RES / 'folds.json').read_text())

cohorts_by_cancer = {c: sorted(g.cohort.unique()) for c, g in reg.groupby('cancer')}
cancers = sorted(cohorts_by_cancer, key=lambda c: -len(cohorts_by_cancer[c]))
rows = [(c, ch) for c in cancers for ch in cohorts_by_cancer[c]]
n_cohorts = len(rows)
rounds = design['rounds']

M = np.zeros((n_cohorts, len(rounds)))
for i, (c, ch) in enumerate(rows):
    n_c = len(cohorts_by_cancer[c])
    for r in rounds:
        M[i, r] = 1 if r < n_c and cohorts_by_cancer[c][r] == ch else 0

fig, ax = plt.subplots(figsize=(4.2, 7.0), dpi=300)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.imshow(M, aspect='auto', cmap=matplotlib.colors.ListedColormap([TRAIN, HELD]),
          vmin=0, vmax=1, interpolation='nearest')

# hairline grid between cells
for x in np.arange(-0.5, len(rounds), 1):
    ax.axvline(x, color=SURFACE, lw=1.2)
for y in np.arange(-0.5, n_cohorts, 1):
    ax.axhline(y, color=SURFACE, lw=1.2)

# cancer group brackets on the left
bounds, start = [], 0
for c in cancers:
    n = len(cohorts_by_cancer[c])
    bounds.append((c, start, start + n - 1))
    start += n
for c, a, b in bounds:
    ax.plot([-0.5 - 0.15, -0.5 - 0.15], [a - 0.5, b + 0.5], color=INK, lw=1.0,
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.text(-0.5 - 0.28, (a + b) / 2, f'{c} ({b - a + 1})', ha='right', va='center',
            fontsize=6.5, color=INK, transform=ax.get_yaxis_transform(), clip_on=False)

ax.set_xticks(range(len(rounds)))
ax.set_xticklabels([f'R{r}' for r in rounds], fontsize=7, color=INK)
ax.set_yticks([])
ax.set_xlabel('Round', fontsize=8, color=INK)
ax.set_ylabel('')
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)

# legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=HELD, label='held out (test)'),
                   Patch(facecolor=TRAIN, label='in training')],
          loc='upper center', bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False,
          fontsize=7)

fig.suptitle('Pre-registered fold design: each cohort held out exactly once',
             fontsize=8.5, color=INK, y=0.995)
fig.tight_layout(rect=[0.18, 0, 1, 0.98])
fig.savefig(FIG / 'fig2b_fold_matrix.pdf')
Fig2 = FIG / 'fig2b_fold_matrix.png'
fig.savefig(Fig2, dpi=300)
print('saved', FIG / 'fig2b_fold_matrix.pdf')
