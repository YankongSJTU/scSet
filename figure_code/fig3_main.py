# -*- coding: utf-8 -*-
"""Fig. 3: main cohort-held-out results (full stage, 42 cohorts).

a  per-cohort balanced accuracy, 9 methods, sorted by mean
b  paired per-cohort delta vs state_mean (beeswarm) + Wilcoxon p
c  retrieval P@1 / P@5 per cohort
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import figstyle as st
from figstyle import SURFACE

st.apply()
HERE = Path(__file__).resolve().parent.parent
PULL = HERE / 'results_full_pulled'
FIG = HERE / 'figures'
SD = FIG / 'source_data'
SD.mkdir(parents=True, exist_ok=True)

ORDER = ['cce_ce', 'cce_long', 'cce_full', 'sup_mean_mlp', 'pseudobulk',
         'pca256', 'state_mean', 'state_moments', 'composition']

wide = pd.read_csv(PULL / 'cohort_level_balanced_accuracy_full.csv',
                   index_col=[0, 1])
wide = wide[ORDER]
rows = wide.mean()
order = sorted(ORDER, key=lambda m: -rows[m])
summary = json.loads((PULL / 'summary_full.json').read_text())

allrows = pd.read_csv(PULL / 'all_rows_full.csv')
pk = (allrows.groupby(['method', 'cohort'])
      .agg(P1=('precision_at_1', 'mean'), P5=('precision_at_5', 'mean'))
      .reset_index())

fig = plt.figure(figsize=(7.3, 3.6), dpi=300)
ax_a = fig.add_axes((0.145, 0.335, 0.255, 0.59))
ax_b = fig.add_axes((0.475, 0.335, 0.225, 0.59))
ax_c = fig.add_axes((0.775, 0.335, 0.22, 0.59))
dot_kw = dict(s=16, linewidths=0.7, edgecolors=SURFACE, zorder=3)

# ---------- panel a: per-cohort balanced accuracy ----------
ax = ax_a
for yi, m in enumerate(order):
    vals = wide[m].dropna().values
    ax.scatter(vals, np.full(len(vals), yi), color=st.ROLE[m], alpha=0.85,
               **dot_kw)
    mu = np.mean(vals)
    ax.plot([mu, mu], [yi - 0.28, yi + 0.28], color=st.INK, lw=1.6, zorder=4)
    ax.text(1.025, yi, f'{mu:.3f}', va='center', ha='left', fontsize=6,
            color=st.SECONDARY, transform=st.blend(ax), clip_on=False)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([st.LABEL[m] for m in order], fontsize=6.5)
ax.invert_yaxis()
ax.set_xlim(0.12, 0.96)
ax.set_xticks([0.2, 0.4, 0.6, 0.8])
ax.xaxis.grid(True, color=st.GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_xlabel('balanced accuracy', fontsize=7)
st.clean_axes(ax)
ax.set_title('a   Cancer-label transfer, 42 cohorts', loc='left',
             fontsize=7.5, fontweight='bold', pad=5)

# ---------- panel b: paired delta vs state_mean ----------
ax = ax_b
delta_order = [m for m in order if m != 'state_mean']
for yi, m in enumerate(delta_order):
    d = (wide[m] - wide['state_mean']).dropna().values
    off = st.swarm_offsets(d, step=0.055, max_slot=14)
    ax.scatter(d, yi + off, color=st.ROLE[m], alpha=0.85, **dot_kw)
    mu = np.mean(d)
    ax.plot([mu, mu], [yi - 0.28, yi + 0.28], color=st.INK, lw=1.6, zorder=4)
    p = summary['deltas_vs_state_mean'][m]['wilcoxon_p']
    ptxt = 'p<0.001' if p < 0.001 else f'p={p:.3f}'
    ax.text(0.56, yi, ptxt, va='center', ha='left', fontsize=5.8,
            color=st.SECONDARY)
ax.axvline(0, color=st.BASELINE, lw=0.8, zorder=1)
ax.text(0.0, len(delta_order) - 0.45, 'state_mean', ha='center', fontsize=5.8,
        color=st.MUTED)
ax.set_yticks(range(len(delta_order)))
ax.set_yticklabels([])
ax.invert_yaxis()
ax.set_xlim(-0.36, 0.72)
ax.set_xticks([-0.3, -0.15, 0, 0.15, 0.3])
ax.set_xticklabels(['-0.30', '-0.15', '0', '0.15', '0.30'])
ax.xaxis.grid(True, color=st.GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_xlabel('Δ balanced accuracy vs state_mean', fontsize=7)
st.clean_axes(ax)
ax.set_title('b   Paired difference vs STATE mean', loc='left', fontsize=7.5,
             fontweight='bold', pad=5)

# ---------- panel c: retrieval precision@k ----------
ax = ax_c
pk_order = [m for m in order if m in set(pk.method.unique())]
for yi, m in enumerate(pk_order):
    p1 = pk.loc[pk.method == m, 'P1'].values
    p5 = pk.loc[pk.method == m, 'P5'].values
    off1 = st.swarm_offsets(p1, step=0.055, max_slot=8)
    off5 = st.swarm_offsets(p5, step=0.055, max_slot=8)
    c = st.ROLE[m]
    ax.scatter(p5, yi - 0.18 + off5, facecolors='none', edgecolors=c,
               linewidths=0.8, s=14, alpha=0.85, zorder=3)
    ax.scatter(p1, yi + 0.18 + off1, color=c, alpha=0.85, **dot_kw)
    ax.plot([np.mean(p1), np.mean(p1)], [yi + 0.02, yi + 0.34], color=st.INK,
            lw=1.4, zorder=4)
ax.set_yticks(range(len(pk_order)))
ax.set_yticklabels([])
ax.invert_yaxis()
ax.set_xlim(0.0, 0.94)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
ax.xaxis.grid(True, color=st.GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_xlabel('retrieval precision', fontsize=7)
st.clean_axes(ax)
ax.set_title('c   Retrieval precision@k', loc='left', fontsize=7.5,
             fontweight='bold', pad=5)

# ---------- figure-level legends (fixed positions, no layout coupling) ----------
handles_grp = [plt.Line2D([], [], marker='o', ls='', color=st.BLUE, ms=4.5,
                          label='CCE set encoders'),
               plt.Line2D([], [], marker='o', ls='', color=st.ORANGE, ms=4.5,
                          label='supervised pooled mean'),
               plt.Line2D([], [], marker='o', ls='', color=st.DEEMPH, ms=4.5,
                          label='simple baselines')]
fig.legend(handles=handles_grp, loc='lower left', bbox_to_anchor=(0.145, 0.02),
           ncol=3, frameon=False, fontsize=6, handletextpad=0.2,
           columnspacing=1.1)
handles_pk = [plt.Line2D([], [], marker='o', ls='', color=st.MUTED, ms=4.5,
                         label='P@1 (filled)'),
              plt.Line2D([], [], marker='o', ls='', markerfacecolor='none',
                         color=st.MUTED, ms=4.5, label='P@5 (open)')]
fig.legend(handles=handles_pk, loc='lower left', bbox_to_anchor=(0.745, 0.02),
           ncol=2, frameon=False, fontsize=6, handletextpad=0.2,
           columnspacing=1.1)

fig.savefig(FIG / 'fig3_main_results.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig3_main_results.png', dpi=300, bbox_inches='tight')
print('saved', FIG / 'fig3_main_results.png')

# source data
wide.reset_index().to_csv(SD / 'fig3a_cohort_balanced_accuracy.csv', index=False)
rows_a = []
for m in ORDER:
    d = (wide[m] - wide['state_mean']).dropna()
    nz = (d != 0).any()
    rows_a.append(dict(method=m, mean_delta=d.mean(),
                       wilcoxon_p=float(stats.wilcoxon(d).pvalue) if nz else 1.0,
                       n=len(d)))
pd.DataFrame(rows_a).to_csv(SD / 'fig3b_deltas_vs_state_mean.csv', index=False)
pk.to_csv(SD / 'fig3c_retrieval_precision_at_k.csv', index=False)
print('source data written to', SD)
