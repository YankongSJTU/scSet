# -*- coding: utf-8 -*-
"""Shared style for paper figures (dataviz reference palette, light mode)."""
import matplotlib as mpl

mpl.use('Agg')

# palette roles (dataviz reference, light mode)
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
SECONDARY = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'
BLUE = '#2a78d6'      # slot 1: CCE encoder variants (the subject)
ORANGE = '#eb6834'    # slot 2: sup_mean_mlp (the supervised pooled-mean control)
DEEMPH = '#c3c2b7'    # de-emphasis gray: context baselines

ROLE = {
    'cce_full': BLUE, 'cce_long': BLUE, 'cce_ce': BLUE,
    'sup_mean_mlp': ORANGE,
    'state_mean': DEEMPH, 'state_moments': DEEMPH, 'composition': DEEMPH,
    'pseudobulk': DEEMPH, 'pca256': DEEMPH,
}

LABEL = {
    'cce_full': 'CCE (reference)',
    'cce_long': 'CCE-long (2x epochs)',
    'cce_ce': 'CCE-CE (CE loss)',
    'sup_mean_mlp': 'sup_mean_mlp',
    'state_mean': 'STATE mean',
    'state_moments': 'STATE mean+var',
    'composition': 'lineage composition',
    'pseudobulk': 'pseudobulk',
    'pca256': 'PCA-256',
}

MPL = dict(font_family=['Segoe UI', 'DejaVu Sans'], surface=SURFACE)


def apply():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'Segoe UI',
        'font.size': 7,
        'text.color': INK,
        'axes.edgecolor': BASELINE,
        'axes.labelcolor': SECONDARY,
        'axes.titlecolor': INK,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5,
        'axes.labelsize': 7,
        'axes.titlesize': 7.5,
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'svg.fonttype': 'none',
    })


def clean_axes(ax, keep_x=False, keep_y=False):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if not keep_x:
        ax.spines['bottom'].set_visible(False)
    else:
        ax.spines['bottom'].set_color(BASELINE)
        ax.spines['bottom'].set_linewidth(0.8)
    if not keep_y:
        ax.spines['left'].set_visible(False)
    else:
        ax.spines['left'].set_color(BASELINE)
        ax.spines['left'].set_linewidth(0.8)
    ax.tick_params(length=0)


def swarm_offsets(values, step=0.16, max_slot=12):
    """Deterministic greedy beeswarm: return per-point y offsets centered on 0.

    values: 1-D array-like of x values. Each point (processed in x order)
    takes the nearest free row (0, 1, -1, 2, -2, ...) among points that are
    horizontally within `step` of it.
    """
    import numpy as np
    v = np.asarray(values, dtype=float)
    order = np.argsort(v)
    placed = []                     # (x, slot) in placement order
    slots = {}
    for i in order:
        x = v[i]
        chosen = 0
        for k in [0] + [s * sign for s in range(1, max_slot + 1)
                        for sign in (1, -1)]:
            if all(abs(x - px) >= step or ps != k for px, ps in placed):
                chosen = k
                break
        placed.append((x, chosen))
        slots[i] = chosen
    off = np.array([slots[i] for i in range(len(v))], dtype=float) * step
    return off


def blend(ax):
    """Transform: x in axes fraction, y in data coordinates."""
    import matplotlib.transforms as mtransforms
    return mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
