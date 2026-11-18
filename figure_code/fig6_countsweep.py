# -*- coding: utf-8 -*-
"""Fig. 6: cell-count robustness (PLAN_weekend Step 2 results).

a  probe balanced accuracy vs cell count n (deployment: fit at 256, infer at n)
b  cross-count self-retrieval P@1 (query n cells vs index 256 cells)
c  cosine to own 256-cell reference

Run after pulling results_countsweep/ from the server:
  scp -r faceba-gpu6:.../results_countsweep results_countsweep_pulled
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as st

st.apply()
HERE = Path(__file__).resolve().parent.parent
PULL = HERE / 'results_countsweep_pulled'
FIG = HERE / 'figures'
SD = FIG / 'source_data'
SD.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(PULL / 'summary.csv')
coh = pd.read_csv(PULL / 'cohort_rows.csv')
ret = pd.read_csv(PULL / 'retrieval_rows.csv')

METHOD_STYLE = {
    'cce_ce': (st.BLUE, 'CCE-CE'),
    'cce_full': (st.BLUE, 'CCE (reference)'),
    'sup_mean_mlp': (st.ORANGE, 'sup_mean_mlp'),
    'pca256': (st.DEEMPH, 'PCA-256'),
    'state_mean': (st.DEEMPH, 'STATE mean'),
    'state_moments': (st.DEEMPH, 'STATE mean+var'),
    'composition': (st.DEEMPH, 'lineage composition'),
}
ORDER = ['cce_ce', 'cce_full', 'sup_mean_mlp', 'pca256', 'state_mean',
         'state_moments', 'composition']

fig = plt.figure(figsize=(7.3, 2.6), dpi=300)
ax_a = fig.add_axes((0.085, 0.22, 0.26, 0.66))
ax_b = fig.add_axes((0.435, 0.22, 0.24, 0.66))
ax_c = fig.add_axes((0.795, 0.22, 0.185, 0.66))

for ax, col, group_col, df, ylab in (
        (ax_a, 'cohort_bal_acc', 'method', summary, 'balanced accuracy'),
        (ax_b, 'P1', 'method', summary, 'self-retrieval P@1'),
        (ax_c, 'cos_self', 'method', summary, 'cosine to 256-cell ref')):
    for m in ORDER:
        sub = df[df[group_col] == m].sort_values('n')
        if not len(sub):
            continue
        c, label = METHOD_STYLE[m]
        ls = '-' if m.startswith('cce') else '--'
        ax.plot(sub.n, sub[col], color=c, lw=1.6, ls=ls, marker='o', ms=3,
                label=label, zorder=3)
    ax.set_xscale('log', base=2)
    ax.set_xticks([8, 16, 32, 64, 128, 256])
    ax.set_xticklabels(['8', '16', '32', '64', '128', '256'], fontsize=6)
    ax.xaxis.grid(True, color=st.GRID, lw=0.6)
    ax.yaxis.grid(True, color=st.GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('cells per sample', fontsize=7)
    ax.set_ylabel(ylab, fontsize=7)
    st.clean_axes(ax)

ax_a.set_title('a   Probe accuracy vs count', loc='left', fontsize=7.5,
               fontweight='bold', pad=5)
ax_b.set_title('b   Cross-count retrieval', loc='left', fontsize=7.5,
               fontweight='bold', pad=5)
ax_c.set_title('c   Embedding stability', loc='left', fontsize=7.5,
               fontweight='bold', pad=5)
handles = [plt.Line2D([], [], color=METHOD_STYLE[m][0], lw=1.6,
                      ls='-' if m.startswith('cce') else '--', label=METHOD_STYLE[m][1])
           for m in ORDER]
fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.085, -0.08),
           ncol=4, frameon=False, fontsize=6, handletextpad=0.3,
           columnspacing=1.0)

fig.savefig(FIG / 'fig6_countsweep.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig6_countsweep.png', dpi=300, bbox_inches='tight')
print('saved', FIG / 'fig6_countsweep.png')

summary.to_csv(SD / 'fig6_countsweep_summary.csv', index=False)
coh.to_csv(SD / 'fig6_countsweep_cohort_rows.csv', index=False)
ret.to_csv(SD / 'fig6_countsweep_retrieval_rows.csv', index=False)
print('source data written to', SD)
