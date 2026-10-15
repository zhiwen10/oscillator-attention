"""Three-panel overview figure: oscillators bind, scale, and can do attention.

    python experiments/make_overview_figure.py            # results/overview_figure.png/.pdf
    python experiments/make_overview_figure.py --lang zh  # Chinese labels

Needs: results/un0_dynamics.pt (experiments/un0_dynamics.py) and results/assoc_p8.json
(experiments/assoc_learning.py --tag p8). Panel A is recomputed (a few seconds on CPU).

A  Binding by synchrony: shuffled observations of 3 moving objects, grouped by oscillator phase.
B  Scale: Un-0 (16,384 coupled oscillators), run locally: the oscillator state (PCA) moves from random phases
   to class clusters; two example images at the ends of their trajectories. Also saved alone as un0_subspace.
C  Oscillator attention on associative recall (1 layer, ~58k parameters, 5 seeds, mean ± s.d.).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oscattn import make_task, continuity_coupling, simulate, same_object, binding_accuracy

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
OBJ = ['#2a78d6', '#eb6834', '#1baf7a']
INK, MUTED, GRID = '#222222', '#8a8a8a', '#dddddd'
OSC, SOFT = '#d1495b', '#9aa5b1'
# same class colours as experiments/un0_dynamics.py
DYN_COLORS = ['#e8547a', '#c9a227', '#b5651d', '#eb8a1e', '#d62728', '#5b8def', '#7a1f1f', '#17a2b8', '#4caf50',
              '#8c8c8c']

TEXT = {
    'en': dict(A='Oscillators bind', Anote='3 moving objects, 72 shuffled observations;\ncolour = phase group (%.0f%% correct)',
               B='Oscillators scale', pc='PC', start='random phases',
               Bnote='Un-0, 16,384 oscillators: random phases → class clusters\n(10 ImageNet classes; class decoding %.0f%% → %.0f%%)',
               C='Oscillators attend', Cy='associative recall accuracy',
               bars=['softmax attention', 'oscillator, no dynamics', 'oscillator, no lag', 'oscillator attention'],
               Cnote='1 layer each, ~58k parameters; chance 3%'),
    'zh': dict(A='振荡同步实现绑定', Anote='3个运动物体，72个打乱的观测；\n颜色 = 相位分组（准确率%.0f%%）',
               B='振荡网络可扩展', pc='主成分', start='随机相位',
               Bnote='Un-0（16,384个振荡器）：随机相位 → 各类别簇\n（10类ImageNet；类别解码准确率 %.0f%% → %.0f%%）',
               C='振荡实现注意力', Cy='联想回忆准确率',
               bars=['常规注意力', '振荡（无动力学）', '振荡（无相位滞后）', '振荡注意力'],
               Cnote='均为单层，约5.8万参数；随机水平3%'),
}


def clean(ax, axes=False):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if not axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)


def panel_binding(ax, fig, T):
    task = make_task(seed=0)
    theta, _ = simulate(continuity_coupling(task), steps=1200)
    acc = binding_accuracy(same_object(theta), task.label)
    th, L, F, P = theta[0].numpy(), task.label.numpy(), task.frame.numpy(), task.pos.numpy()
    # colour each token by its phase group (groups mapped to colours by majority label)
    centres = [np.angle(np.exp(1j * th[L == o]).mean()) for o in range(3)]
    group = np.array([np.argmin([abs(np.angle(np.exp(1j * (t - c)))) for c in centres]) for t in th])
    for o in range(3):
        m = group == o
        i = np.argsort(F[m])
        ax.plot(P[m][i, 0], P[m][i, 1], color=OBJ[o], lw=1.3, alpha=0.45, solid_capstyle='round')
        ax.scatter(P[m, 0], P[m, 1], s=5, color=OBJ[o], lw=0, zorder=3)
    ax.set_aspect('equal'); clean(ax)
    ax.set_title(T['A'], fontsize=8.5, pad=4)
    ax.text(0.5, -0.04, T['Anote'] % (100 * acc), transform=ax.transAxes, ha='center', va='top',
            fontsize=6, color=MUTED, linespacing=1.3)
    # inset: final phases on the unit circle
    bb = ax.get_position()
    ins = fig.add_axes([bb.x1 - 0.055, bb.y1 - 0.12, 0.06, 0.12], projection='polar')
    for o in range(3):
        m = group == o
        ins.scatter(th[m], np.ones(m.sum()), s=4, color=OBJ[o], lw=0)
    ins.set_ylim(0, 1.25); ins.set_xticks([]); ins.set_yticks([]); ins.grid(False)
    ins.spines['polar'].set_color(GRID)


def panel_subspace(ax, T, examples=(130, 323)):
    """Un-0 oscillator state (PCA, from un0_dynamics.py): random phases -> class clusters, with the images
    two of the trajectories end in."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    d = torch.load(RES / 'un0_dynamics.pt', weights_only=False)
    proj, lab, names = d['proj'].numpy(), d['labels'].numpy(), list(d['names'])
    for i, c in enumerate(names):
        m = lab == c
        for s_ in np.flatnonzero(m)[:4]:
            ax.plot(proj[:, s_, 0], proj[:, s_, 1], color=DYN_COLORS[i], lw=0.3, alpha=0.5)
        mu = proj[:, m].mean(1)
        ax.plot(mu[:, 0], mu[:, 1], color=DYN_COLORS[i], lw=1.2)
        ax.scatter(proj[-1, m, 0], proj[-1, m, 1], s=1.5, color=DYN_COLORS[i], lw=0, alpha=0.8)
    x0, y0 = proj[0, :, 0].mean(), proj[0, :, 1].mean()
    ax.scatter(x0, y0, s=14, color='k', marker='x', lw=0.9, zorder=5)
    ax.annotate(T['start'], (x0, y0), xytext=(0.36, 0.02), textcoords='axes fraction', fontsize=5.5,
                arrowprops={'arrowstyle': '-', 'lw': 0.4, 'color': INK})
    boxes = {130: (0.28, 0.86), 323: (0.9, 0.16)}               # thumbnail centres, axes fraction
    for c in examples:
        i = names.index(c)
        s_ = int(np.flatnonzero(lab == c)[0])                    # the sample decoded in un0_dynamics.py
        ax.plot(proj[:, s_, 0], proj[:, s_, 1], color=DYN_COLORS[i], lw=0.9, ls='--')
        img = d['strip'][c][-1].permute(1, 2, 0).numpy()
        ab = AnnotationBbox(OffsetImage(img, zoom=0.26), (proj[-1, s_, 0], proj[-1, s_, 1]),
                            xybox=boxes[c], boxcoords='axes fraction', pad=0.1,
                            bboxprops={'edgecolor': DYN_COLORS[i], 'lw': 1.0},
                            arrowprops={'arrowstyle': '-', 'color': DYN_COLORS[i], 'lw': 0.6})
        ax.add_artist(ab)
    ax.set_box_aspect(1)                                         # square plot
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(T['pc'] + ' 1', labelpad=1, fontsize=6); ax.set_ylabel(T['pc'] + ' 2', labelpad=1, fontsize=6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_title(T['B'], fontsize=8.5, pad=4)
    ax.text(0.5, -0.1, T['Bnote'] % (100 * d['decoding_acc'][0], 100 * d['decoding_acc'][-1]),
            transform=ax.transAxes, ha='center', va='top', fontsize=6, color=MUTED, linespacing=1.3)


def panel_attention(ax, T):
    import json
    runs = json.loads((RES / 'assoc_p8.json').read_text())['runs']
    names = ['softmax-1L', 'osc-1L-T0', 'osc-1L-T4-nolag', 'osc-1L-T4']
    means, sds, pts = [], [], []
    for n in names:                    # each model at its better lr (same rule as make_assoc_figure.py)
        by_lr = {}
        for r in runs:
            if r['model'] == n:
                by_lr.setdefault(r['lr'], []).append(r)
        lr = max(by_lr, key=lambda l: np.mean([np.mean([a for _, a, _ in r['curve']]) for r in by_lr[l]]))
        f = [r['final_acc'] for r in by_lr[lr]]
        means.append(np.mean(f)); sds.append(np.std(f)); pts.append(f)
    y = np.arange(len(names))[::-1]
    colors = [SOFT, '#e8c3c8', '#e3909b', OSC]
    ax.barh(y, means, xerr=sds, color=colors, height=0.62, capsize=2, error_kw={'lw': 0.7})
    for yi, f in zip(y, pts):
        ax.scatter(f, np.full(len(f), yi) + np.linspace(-0.13, 0.13, len(f)), s=4, color=INK, zorder=3, lw=0)
    ax.set_yticks(y, T['bars'], fontsize=6)
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel(T['Cy'], fontsize=6.5, labelpad=1)
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', length=0)
    clean(ax, axes=True)
    ax.set_title(T['C'], fontsize=8.5, pad=4)
    ax.text(1.0, -0.33, T['Cnote'], transform=ax.transAxes, ha='right', va='top', fontsize=6, color=MUTED)


def main(lang):
    T = TEXT[lang]
    plt.rcParams.update({'font.family': ['Arial', 'Microsoft YaHei'], 'font.size': 7, 'axes.linewidth': 0.6,
                         'axes.edgecolor': MUTED, 'axes.unicode_minus': False, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(18 / 2.54, 5.6 / 2.54), dpi=300)
    axA = fig.add_axes([0.02, 0.2, 0.2, 0.62])
    axB = fig.add_axes([0.3, 0.24, 0.32, 0.58])
    axC = fig.add_axes([0.835, 0.3, 0.155, 0.5])
    panel_binding(axA, fig, T)
    panel_subspace(axB, T)
    panel_attention(axC, T)
    for x, lt in [(0.01, 'A'), (0.25, 'B'), (0.69, 'C')]:
        fig.text(x, 0.97, lt, fontsize=10, fontweight='bold', va='top')
    suffix = '' if lang == 'en' else '_zh'
    for ext in ('png', 'pdf'):
        fig.savefig(RES / f'overview_figure{suffix}.{ext}', dpi=300, facecolor='white')
    print('saved', RES / f'overview_figure{suffix}.png')

    # panel B on its own
    fb = plt.figure(figsize=(7 / 2.54, 6.2 / 2.54), dpi=300)
    panel_subspace(fb.add_axes([0.08, 0.2, 0.88, 0.7]), T)
    for ext in ('png', 'pdf'):
        fb.savefig(RES / f'un0_subspace{suffix}.{ext}', dpi=300, facecolor='white')
    print('saved', RES / f'un0_subspace{suffix}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', default='en', choices=['en', 'zh'])
    main(ap.parse_args().lang)
