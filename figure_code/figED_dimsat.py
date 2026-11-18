# -*- coding: utf-8 -*-
"""ED figure: dimension-saturation of existing 256-d embeddings (Step 0).

PCA(k) -> LR probe, balanced accuracy vs k, for the three full-stage CCE
variants and the state_mean pooled baseline.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import figstyle as st

st.apply()
HERE = Path(__file__).resolve().parent.parent
PULL = HERE / 'results_dimsat_pulled'
FIG = HERE / 'figures'
SD = FIG / 'source_data'
SD.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PULL / 'dimsat_summary.csv')
df['k'] = df.method.str.extract(r'pca(\d+)').astype(int)
df['variant'] = df.method.str.replace(r'_pca\d+', '', regex=True)

STYLE = {
    'cce_ce': (st.BLUE, '-', 'CCE-CE'),
    'cce_long': (st.BLUE, '--', 'CCE-long'),
    'cce_full': (st.BLUE, ':', 'CCE (reference)'),
    'state_mean': (st.DEEMPH, '-', 'STATE mean'),
}

fig = plt.figure(figsize=(3.4, 2.6), dpi=300)
ax = fig.add_axes((0.175, 0.19, 0.79, 0.72))
for v, (c, ls, label) in STYLE.items():
    sub = df[df.variant == v].sort_values('k')
    ax.plot(sub.k, sub.bal, color=c, ls=ls, lw=1.8, marker='o', ms=3.2,
            label=label, zorder=3)
ax.set_xscale('log', base=2)
ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256])
ax.set_xticklabels(['2', '4', '8', '16', '32', '64', '128', '256'], fontsize=6)
ax.set_xlabel('PCA components kept (k)', fontsize=7)
ax.set_ylabel('balanced accuracy', fontsize=7)
ax.set_ylim(0.42, 0.63)
ax.xaxis.grid(True, color=st.GRID, lw=0.6)
ax.yaxis.grid(True, color=st.GRID, lw=0.6)
ax.set_axisbelow(True)
st.clean_axes(ax)
ax.legend(frameon=False, fontsize=6, loc='lower right', handletextpad=0.4)
ax.set_title('Transferable information saturates at low dimension',
             loc='left', fontsize=7.5, fontweight='bold', pad=5)

fig.savefig(FIG / 'figED_dimsat.pdf', bbox_inches='tight')
fig.savefig(FIG / 'figED_dimsat.png', dpi=300, bbox_inches='tight')
print('saved', FIG / 'figED_dimsat.png')
df.to_csv(SD / 'figED_dimsat.csv', index=False)
