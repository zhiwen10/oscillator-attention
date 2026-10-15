"""Simple 4-panel figure for the moving-objects demo: input -> dynamics -> attention -> output.

    python experiments/make_simple_figure.py            # results/simple_figure.png/.pdf
    python experiments/make_simple_figure.py --lang zh  # Chinese labels

A  What the network sees: 72 shuffled, unlabelled tokens (position + velocity).
B  Each token is an oscillator; phases start random and pull into three in-phase groups.
C  Attention read from the final phases, A_ij = (1 + cos(θi − θj)) / 2, tokens sorted by object.
D  What the network outputs: tokens coloured by the phase group they ended in.
Also saves results/simple_wave.png: with a phase lag, phase rises along each object (travelling wave).
Colours in A-C show the hidden true object only for the reader; the network never sees them.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oscattn import make_task, continuity_coupling, simulate, phase_attention, same_object, binding_accuracy

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
OBJ = ['#2a78d6', '#eb6834', '#1baf7a']
INK, MUTED, GRID = '#222222', '#8a8a8a', '#dddddd'

TEXT = {
    'en': dict(A='Input: shuffled tokens', Anote='position + velocity, no labels',
               B='Oscillators synchronise', Bx='integration step', By='phase (rad)',
               C='Attention from phases', Cnote='tokens sorted by object',
               D='Output: recovered objects', Dnote='binding accuracy %.0f%%',
               W='Phase lag: each object becomes a wave', Wx='frame', Wy='phase (rad)'),
    'zh': dict(A='输入：打乱的词元', Anote='位置 + 速度，无标签',
               B='振荡器相位同步', Bx='演化步数', By='相位（rad）',
               C='由相位差读出注意力', Cnote='按物体排序',
               D='输出：分离出的物体', Dnote='绑定准确率 %.0f%%',
               W='相位滞后：每个物体形成行波', Wx='时间帧', Wy='相位（rad）'),
}


def clean(ax, keep_axes=False):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if not keep_axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)


def groups_from_phases(theta):
    """Connected components of the 'in phase' graph (phase difference < 60 degrees)."""
    same = same_object(theta[None]).numpy()
    n, lab = len(same), -np.ones(len(same), int)
    for i in range(n):
        if lab[i] < 0:
            stack, lab[i] = [i], lab.max() + 1
            while stack:
                j = stack.pop()
                for k in np.flatnonzero(same[j] & (lab < 0)):
                    lab[k] = lab[i]; stack.append(k)
    return lab


def main(lang):
    T = TEXT[lang]
    plt.rcParams.update({'font.family': ['Arial', 'Microsoft YaHei'], 'font.size': 8, 'axes.linewidth': 0.6,
                         'axes.edgecolor': MUTED, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'axes.unicode_minus': False, 'pdf.fonttype': 42})
    task = make_task(seed=0)
    K = continuity_coupling(task)
    steps = 1200
    rec = tuple(range(0, steps + 1, 4))
    final, snaps = simulate(K, steps=steps, record=rec)
    th_final = final[0]
    acc = binding_accuracy(same_object(final), task.label)
    L, F, P, V = (a.numpy() for a in (task.label, task.frame, task.pos, task.vel))

    fig, ax = plt.subplots(1, 4, figsize=(18 / 2.54, 4.8 / 2.54), dpi=300,
                           gridspec_kw={'wspace': 0.45})

    # A: input
    a = ax[0]
    a.quiver(P[:, 0], P[:, 1], V[:, 0], V[:, 1], color=MUTED, angles='xy', scale_units='xy', scale=0.6,
             width=0.004, headwidth=3, headlength=3)
    a.scatter(P[:, 0], P[:, 1], s=5, color=INK, lw=0, zorder=3)
    a.set_aspect('equal'); clean(a)
    a.set_title(T['A'], fontsize=8, pad=3)
    a.text(0.5, -0.05, T['Anote'], transform=a.transAxes, ha='center', va='top', fontsize=6.5, color=MUTED)

    # B: phases over integration time, unwrapped; each oscillator shifted by 2π so it ends near its group
    b = ax[1]
    th = np.unwrap(np.stack([snaps[s][0].numpy() for s in rec]), axis=0)          # (time, n)
    centre = {o: np.angle(np.exp(1j * th[-1, L == o]).mean()) for o in range(3)}
    for i in range(th.shape[1]):
        c = centre[L[i]]
        th[:, i] -= 2 * np.pi * np.round((th[-1, i] - c) / (2 * np.pi))
    for i in np.random.default_rng(0).permutation(th.shape[1]):
        b.plot(rec, th[:, i], color=OBJ[L[i]], lw=0.5, alpha=0.8)
    b.set_xlabel(T['Bx'], labelpad=1); b.set_ylabel(T['By'], labelpad=1)
    b.set_xlim(0, steps); b.set_xticks([0, 600, 1200])
    b.set_title(T['B'], fontsize=8, pad=3); clean(b, keep_axes=True)

    # C: phase attention matrix, sorted by object then frame
    c = ax[2]
    order = np.lexsort((F, L))
    A = phase_attention(th_final).numpy()[np.ix_(order, order)]
    c.imshow(A, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    counts, y0 = np.bincount(L), 0
    for o in range(3):
        c.add_patch(plt.Rectangle((-3.5, y0 - 0.5), 2.2, counts[o], fc=OBJ[o], ec='none', clip_on=False))
        c.add_patch(plt.Rectangle((y0 - 0.5, len(L) + 0.8), counts[o], 2.2, fc=OBJ[o], ec='none', clip_on=False))
        y0 += counts[o]
    clean(c)
    c.set_title(T['C'], fontsize=8, pad=3)
    c.text(0.5, -0.1, T['Cnote'], transform=c.transAxes, ha='center', va='top', fontsize=6.5, color=MUTED)

    # D: output, coloured by the phase group each token ended in (majority true label -> colour)
    d = ax[3]
    g = groups_from_phases(th_final)
    colour_of = {k: OBJ[np.bincount(L[g == k]).argmax()] for k in np.unique(g)}
    for k in np.unique(g):
        m = g == k
        i = np.argsort(F[m])
        d.plot(P[m][i, 0], P[m][i, 1], color=colour_of[k], lw=1.4, alpha=0.5, solid_capstyle='round')
        d.scatter(P[m, 0], P[m, 1], s=5, color=colour_of[k], lw=0, zorder=3)
    d.set_aspect('equal'); clean(d)
    d.set_title(T['D'], fontsize=8, pad=3)
    d.text(0.5, -0.05, T['Dnote'] % (100 * acc), transform=d.transAxes, ha='center', va='top', fontsize=6.5,
           color=MUTED)

    for axx, lt in zip(ax, 'ABCD'):
        axx.set_box_aspect(1)
        axx.text(-0.22 if axx is b else -0.1, 1.32, lt, transform=axx.transAxes, fontsize=10,
                 fontweight='bold', va='top')
    suffix = '' if lang == 'en' else '_zh'
    fig.savefig(RES / f'simple_figure{suffix}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(RES / f'simple_figure{suffix}.pdf', bbox_inches='tight', facecolor='white')

    # Optional extra panel: travelling wave with a phase lag
    theta_lag, _ = simulate(K, steps=1500, lag=0.10, frames=task.frame)
    fw, w = plt.subplots(figsize=(5.5 / 2.54, 4.8 / 2.54), dpi=300)
    for o in range(3):
        m = L == o
        i = np.argsort(F[m])
        ph = np.unwrap(theta_lag[0][m].numpy()[i])
        w.plot(F[m][i], ph - ph[0], 'o-', color=OBJ[o], lw=1, ms=2)
    w.set_xlabel(T['Wx'], labelpad=1); w.set_ylabel(T['Wy'], labelpad=1)
    w.set_title(T['W'], fontsize=8, pad=3); clean(w, keep_axes=True)
    fw.savefig(RES / f'simple_wave{suffix}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fw.savefig(RES / f'simple_wave{suffix}.pdf', bbox_inches='tight', facecolor='white')
    print('binding accuracy %.3f; groups found: %d' % (acc, len(np.unique(g))))
    print('saved', RES / f'simple_figure{suffix}.png', 'and', RES / f'simple_wave{suffix}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', default='en', choices=['en', 'zh'])
    main(ap.parse_args().lang)
