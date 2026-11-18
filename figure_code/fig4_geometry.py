# -*- coding: utf-8 -*-
"""Fig. 4: representation geometry under cohort-held-out training (18 runs).

a  effective rank (participation ratio)
b  within-class vs between-class cosine structure (dumbbell)
c  within-class variance ratio (collapse diagnostic)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as st
from figstyle import SURFACE

st.apply()
HERE = Path(__file__).resolve().parent.parent
PULL = HERE / 'results_full_pulled'
FIG = HERE / 'figures'
SD = FIG / 'source_data'
SD.mkdir(parents=True, exist_ok=True)

g = pd.read_csv(PULL / 'geometry_rows.csv')
order = ['cce_ce', 'cce_long', 'cce_full', 'pseudobulk', 'state_mean']

fig = plt.figure(figsize=(7.3, 2.4), dpi=300)
ax_a = fig.add_axes((0.145, 0.30, 0.24, 0.60))
ax_b = fig.add_axes((0.505, 0.30, 0.22, 0.60))
ax_c = fig.add_axes((0.815, 0.30, 0.18, 0.60))
dot_kw = dict(s=16, linewidths=0.7, edgecolors=SURFACE, zorder=3)

# ---------- panel a: effective rank ----------
ax = ax_a
for yi, m in enumerate(order):
    vals = g.loc[g.method == m, 'effective_rank'].values
    ax.scatter(vals, np.full(len(vals), yi), color=st.ROLE[m], alpha=0.85,
               **dot_kw)
    mu = np.mean(vals)
    ax.plot([mu, mu], [yi - 0.28, yi + 0.28], color=st.INK, lw=1.6, zorder=4)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([st.LABEL[m] for m in order], fontsize=6.5)
ax.invert_yaxis()
ax.set_xlim(0, 12)
ax.xaxis.grid(True, color=st.GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_xlabel('effective rank', fontsize=7)
st.clean_axes(ax)
ax.set_title('a   Effective rank', loc='left', fontsize=7.5,
             fontweight='bold', pad=5)

# ---------- panel b: within vs between class cosine (dumbbell) ----------
ax = ax_b
agg = g.groupby('method')[['within_class_cos', 'between_class_cos']].mean()
for yi, m in enumerate(order):
    w, b = agg.loc[m, 'within_class_cos'], agg.loc[m, 'between_class_cos']
    ax.plot([b, w], [yi, yi], color=st.GRID, lw=1.2, zorder=2)
    ax.scatter([w], [yi], color=st.ROLE[m], alpha=0.9, **dot_kw)
    ax.scatter([b], [yi], facecolors='none', edgecolors=st.ROLE[m],
               linewidths=0.9, s=16, alpha=0.9, zorder=3)
ax.axvline(0, color=st.BASELINE, lw=0.8, zorder=1)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([])
ax.invert_yaxis()
ax.set_xlim(-0.15, 1.0)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.xaxis.grid(True, color=st.GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_xlabel('mean cosine similarity', fontsize=7)
st.clean_axes(ax)
ax.set_title('b   Within- vs between-class cosine', loc='left', fontsize=7.5,
             fontweight='bold', pad=5)
handles = [plt.Line2D([], [], marker='o', ls='', color=st.MUTED, ms=4.5,
                      label='within-class (filled)'),
           plt.Line2D([], [], marker='o', ls='', markerfacecolor='none',
                      color=st.MUTED, ms=4.5, label='between-class (open)')]
fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.505, 0.03),
           ncol=1, frameon=False, fontsize=6, handletextpad=0.2)

# ---------- panel c: within-class variance ratio ----------
ax = ax_c
for yi, m in enumerate(order):
    vals = g.loc[g.method == m, 'wcr'].values
    ax.scatter(vals, np.full(len(vals), yi), color=st.ROLE[m], alpha=0.85,
               **dot_kw)
    mu = np.mean(vals)
    ax.plot([mu, mu], [yi - 0.28, yi + 0.28], color=st.INK, lw=1.6, zorder=4)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([])
ax.invert_yaxis()
ax.set_xlim(0, 1.0)
ax.xaxis.grid(True, color=st.GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_xlabel('within-class variance ratio', fontsize=7)
st.clean_axes(ax)
ax.set_title('c   Within-class variance ratio', loc='left', fontsize=7.5,
             fontweight='bold', pad=5)

fig.savefig(FIG / 'fig4_geometry.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig4_geometry.png', dpi=300, bbox_inches='tight')
print('saved', FIG / 'fig4_geometry.png')

agg.round(4).to_csv(SD / 'fig4_geometry_summary.csv')
g.to_csv(SD / 'fig4_geometry_runs.csv', index=False)
print('source data written to', SD)
