# -*- coding: utf-8 -*-
"""Fig. 5: information increment and external-task consistency.

a  paired delta of concat[state_mean; encoder] vs state_mean probe, per
   encoder variant (18 runs each) - does the encoder carry complementary info?
b  published external-task results (audit dirs), one row per task;
   key methods annotated, full mapping in source data.
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

p = pd.read_csv(PULL / 'probe_concat_rows.csv')
ext = pd.read_csv(HERE / 'results' / 'external_tasks_summary.csv')

fig = plt.figure(figsize=(7.3, 5.0), dpi=300)
ax_a = fig.add_axes((0.335, 0.72, 0.35, 0.20))
ax_b = fig.add_axes((0.335, 0.11, 0.645, 0.44))
dot_kw = dict(s=16, linewidths=0.7, edgecolors=SURFACE, zorder=3)

# ---------- panel a: concat vs state_mean paired delta ----------
variants = ['cce_ce', 'cce_long', 'cce_full']
deltas = {}
for yi, v in enumerate(variants):
    d = p.loc[(p.variant == v) & (p.method == 'concat_mean_encoder'),
              'delta_concat_vs_mean'].dropna().values
    deltas[v] = d
    off = st.swarm_offsets(d, step=0.16, max_slot=10)
    ax_a.scatter(d, np.full(len(d), yi) + off, color=st.BLUE, alpha=0.85,
                 **dot_kw)
    mu = np.mean(d)
    ax_a.plot([mu, mu], [yi - 0.28, yi + 0.28], color=st.INK, lw=1.6,
              zorder=4)
    ax_a.text(1.035, yi, f'{mu:+.3f}', va='center', ha='left', fontsize=6,
              color=st.SECONDARY, transform=st.blend(ax_a), clip_on=False)
lo = min(min(v) for v in deltas.values())
ax_a.set_yticks(range(len(variants)))
ax_a.set_yticklabels([st.LABEL[v] for v in variants], fontsize=6.5)
ax_a.invert_yaxis()
ax_a.set_xlim(lo - 0.01, 0.125)
ax_a.axvline(0, color=st.BASELINE, lw=0.8, zorder=1)
ax_a.xaxis.grid(True, color=st.GRID, lw=0.6)
ax_a.set_axisbelow(True)
ax_a.xaxis.set_major_formatter(lambda x, _: f'{x:.2f}')
ax_a.set_xlabel('Δ balanced accuracy: concat vs state_mean', fontsize=6.8,
                labelpad=2)
st.clean_axes(ax_a)
ax_a.set_title('a   Complementary-information probe', loc='left',
               fontsize=7.5, fontweight='bold', pad=5)

# ---------- panel b: external tasks (prior results) ----------
rows = [
    ('ARTEMIS TNBC\npCR vs RD (AUROC)', 'ARTEMIS_TNBC', 'AUROC', True),
    ('GSE306201\nprimary vs met (AUROC)', 'GSE306201', 'AUROC', True),
    ('Jiang CRISPR\ngene-level retrieval', 'Jiang_CRISPR',
     ('AUROC', 'gene_level'), True),
    ('Jiang CRISPR\nguide-level retrieval', 'Jiang_CRISPR',
     ('AUROC', 'guide_level'), True),
    ('LOCO pilot, 4 cancers\nlabel transfer (bal. acc)', 'LOCO_pilot_4cancer',
     'balanced_accuracy', False),
]
# annotated short names; moment variants left unlabeled (context, in source data)
SHORT = {'CCE': 'CCE', 'STATE_mean': 'STATE',
         'lineage_composition': 'comp', 'pseudobulk': 'pbulk',
         'CCE_gene_level': 'CCE', 'STATE_mean_gene_level': 'STATE',
         'CCE_guide_level': 'CCE', 'STATE_mean_guide_level': 'STATE'}


def method_color(name):
    n = name.lower()
    if n.startswith('cce'):
        return st.BLUE
    if n == 'state_mean' or n.endswith('_gene_level') or n.endswith('_guide_level'):
        return st.ORANGE if n.startswith('state') else st.BLUE
    return st.DEEMPH


label_w = 1.08  # approx pt per character at 4.6pt font
for yi, (lab, task, metric, chance) in enumerate(rows):
    sub = ext[ext.task == task]
    if isinstance(metric, tuple):
        sub = sub[sub.metric == metric[0]]
        sub = sub[sub.method.str.contains(metric[1])]
    else:
        sub = sub[sub.metric == metric]
    sub = sub.assign(v=[0.5 if r.value == 'chance' else float(r.value)
                        for r in sub.itertuples()]).sort_values('v')
    placed = []   # (x0, x1, dy) of drawn labels, in data-x / pt-y
    for r in sub.itertuples():
        val = r.v
        ax_b.scatter(val, yi, color=method_color(r.method), alpha=0.9,
                     **dot_kw)
        name = SHORT.get(r.method)
        if name is None:
            continue
        w = len(name) * label_w
        dy = None
        for cand in (6.0, -8.5, 14.0, -16.5):
            # horizontal check only against labels on the same vertical band
            same_band = [(x0, x1) for x0, x1, d in placed
                         if abs(cand - d) < 5]
            if all(val + w * 0.0021 < x0 - 0.006 or val > x1 + 0.006
                   for x0, x1 in same_band):
                dy = cand
                break
        if dy is None:
            continue
        placed.append((val, val + w * 0.0021, dy))
        ax_b.annotate(name, (val, yi), textcoords='offset points',
                      xytext=(0, dy), ha='center', fontsize=4.8,
                      color=st.SECONDARY, zorder=5)
    if chance:
        ax_b.plot([0.5, 0.5], [yi - 0.32, yi + 0.32], color=st.BASELINE,
                  lw=0.9, zorder=1)
ax_b.text(0.5, -0.55, 'chance', ha='center', fontsize=5.5, color=st.MUTED)
ax_b.set_yticks(range(len(rows)))
ax_b.set_yticklabels([r[0] for r in rows], fontsize=6.3)
ax_b.invert_yaxis()
ax_b.set_xlim(0.25, 1.0)
ax_b.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax_b.xaxis.grid(True, color=st.GRID, lw=0.6)
ax_b.set_axisbelow(True)
ax_b.set_xlabel('external-task performance (per-row metric)', fontsize=7)
st.clean_axes(ax_b)
ax_b.set_title('b   External tasks (prior audit results, not rerun)',
               loc='left', fontsize=7.5, fontweight='bold', pad=5)
handles = [plt.Line2D([], [], marker='o', ls='', color=st.BLUE, ms=4.5,
                      label='CCE'),
           plt.Line2D([], [], marker='o', ls='', color=st.ORANGE, ms=4.5,
                      label='STATE mean'),
           plt.Line2D([], [], marker='o', ls='', color=st.DEEMPH, ms=4.5,
                      label='other baselines')]
fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.335, 0.005),
           ncol=3, frameon=False, fontsize=6, handletextpad=0.2,
           columnspacing=1.1)

fig.savefig(FIG / 'fig5_info_external.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig5_info_external.png', dpi=300, bbox_inches='tight')
print('saved', FIG / 'fig5_info_external.png')

rows_out = []
for v, d in deltas.items():
    rows_out.append(dict(variant=v, mean_delta=float(np.mean(d)), n=len(d)))
pd.DataFrame(rows_out).to_csv(SD / 'fig5a_concat_delta.csv', index=False)
ext.to_csv(SD / 'fig5b_external_tasks.csv', index=False)
print('source data written to', SD)
