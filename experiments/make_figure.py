"""Draw the six-panel summary figure from results/results.json (run run_experiments.py first).

    python experiments/make_figure.py            # English labels -> results/figure.png
    python experiments/make_figure.py --lang zh  # Chinese labels  -> results/figure_zh.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from oscattn import continuity_coupling, phase_attention

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
INK, INK2, MUTED, BASE, GRID = '#0b0b0b', '#52514e', '#898781', '#c3c2b7', '#e1e0d9'
OBJ = ['#2a78d6', '#eb6834', '#1baf7a']
HEADS = {1: '#9ec5f4', 2: '#3987e5', 4: '#104281'}

TEXT = {
    'en': dict(A='Moving objects (tokens shuffled)', Anote='each dot = one token (position + velocity)',
               B='Phases synchronise into groups', t0='start', t1='settling', t2='converged',
               C1='1 local layer', C2='phase readout',
               Dx='attention layers', Dy='binding accuracy', Dosc='one oscillator network', Dstk='stacked local attention',
               E='Phase lag -> travelling wave', Ex='frame', Ey='phase (rad)', Enote='phase gradient encodes order',
               F='More heads, more capacity', Fx='number of objects', Fy='binding accuracy', head='%d head'),
    'zh': dict(A='连续运动的多个物体', Anote='每个点 = 一个词元（位置 + 速度），顺序打乱',
               B='振荡器相位自发同步为簇', t0='初始', t1='演化中', t2='收敛',
               C1='单层局部注意力', C2='相位差读出',
               Dx='注意力层数', Dy='绑定准确率', Dosc='一个振荡网络（共享耦合）', Dstk='堆叠局部注意力',
               E='相位滞后 → 物体内形成行波', Ex='时间帧', Ey='相位（rad）', Enote='相位梯度编码时间顺序',
               F='多头振荡提高绑定容量', Fx='物体数量', Fy='绑定准确率', head='%d 头'),
}


def tint(c, a):
    r = np.array(to_rgb(c))
    return tuple(1 - a * (1 - r))


def spines(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def main(lang):
    T = TEXT[lang]
    plt.rcParams.update({'font.family': ['Arial', 'Microsoft YaHei'], 'font.size': 7, 'axes.linewidth': 0.6,
                         'axes.edgecolor': MUTED, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.labelsize': 6.5,
                         'axes.unicode_minus': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    R = json.loads((RES / 'results.json').read_text())
    ex = torch.load(RES / 'exp1_example.pt', weights_only=False)
    task, snaps = ex['task'], ex['snaps']
    L, F, P = task.label.numpy(), task.frame.numpy(), task.pos.numpy()

    fig = plt.figure(figsize=(15 / 2.54, 10 / 2.54), dpi=300)

    # A: the task
    ax = fig.add_axes([0.045, 0.56, 0.24, 0.36])
    for o in range(3):
        m = L == o
        i = np.argsort(F[m])
        ax.plot(P[m][i, 0], P[m][i, 1], color=tint(OBJ[o], 0.35), lw=2.2, solid_capstyle='round')
    ax.scatter(P[:, 0], P[:, 1], s=4, c=INK2, lw=0, zorder=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_title(T['A'], fontsize=6.8, pad=3)
    ax.text(0.5, -0.06, T['Anote'], transform=ax.transAxes, ha='center', va='top', fontsize=5.6, color=INK2)

    # B: phase snapshots on the unit circle
    for k, (st, lab) in enumerate([(0, T['t0']), (150, T['t1']), (1200, T['t2'])]):
        axp = fig.add_axes([0.34 + k * 0.105, 0.60, 0.095, 0.30], projection='polar')
        th = snaps[st][0].numpy()
        for o in range(3):
            m = L == o
            axp.scatter(th[m], 1 + 0.06 * np.random.default_rng(o).standard_normal(m.sum()), s=3.2, color=OBJ[o], lw=0)
        axp.set_ylim(0, 1.3); axp.set_yticks([]); axp.set_xticks([]); axp.grid(False)
        axp.spines['polar'].set_color(BASE)
        axp.set_title(lab, fontsize=6.2, color=INK2, pad=2)
    fig.text(0.5, 0.955, T['B'], ha='center', va='top', fontsize=6.8)

    # C: local attention vs phase-difference attention (tokens sorted by object, then time)
    order = np.lexsort((F, L))
    K = continuity_coupling(task).numpy()
    A_loc = K / K.sum(1, keepdims=True).clip(1e-9)
    A_loc = (A_loc / A_loc.max(1, keepdims=True))[np.ix_(order, order)]
    A_osc = phase_attention(snaps[1200][0]).numpy()[np.ix_(order, order)]
    for k, (M, lab) in enumerate([(A_loc, T['C1']), (A_osc, T['C2'])]):
        axm = fig.add_axes([0.70 + k * 0.145, 0.60, 0.13, 0.30])
        axm.imshow(M, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
        axm.set_xticks([]); axm.set_yticks([])
        for s in axm.spines.values():
            s.set_color(BASE)
        axm.set_title(lab, fontsize=6.2, color=INK2, pad=2)
        counts = np.bincount(L)
        y0 = 0
        for o in range(3):
            axm.add_patch(Rectangle((-2.5, y0 - 0.5), 1.6, counts[o], fc=OBJ[o], ec='none', clip_on=False))
            y0 += counts[o]
    fig.text(0.845, 0.955, r'$A_{ij}=(1+\cos(\theta_i-\theta_j))/2$', ha='center', va='top', fontsize=6.5)

    # D: oscillator vs stacked local attention
    d = R['exp2_depth']
    axd = fig.add_axes([0.07, 0.10, 0.22, 0.32])
    lay = np.arange(1, len(d['stack_mean']) + 1)
    axd.plot(lay, d['stack_mean'], color=MUTED, lw=1.4, marker='o', ms=2.5)
    axd.axhline(d['oscillator_mean'], color=OBJ[0], lw=1.4)
    axd.text(lay[-1], d['oscillator_mean'] - 0.03, T['Dosc'], ha='right', va='top', fontsize=5.7)
    axd.text(4.6, 0.64, T['Dstk'], fontsize=5.7)
    axd.set_xlim(0.5, lay[-1] + 0.5); axd.set_ylim(0.5, 1.05); axd.set_xticks([1, 5, 10, 14]); axd.set_yticks([0.5, 0.75, 1.0])
    axd.set_xlabel(T['Dx'], labelpad=1); axd.set_ylabel(T['Dy'], labelpad=1); spines(axd)

    # E: travelling wave (lag 0.10)
    axe = fig.add_axes([0.40, 0.10, 0.22, 0.32])
    for o, obj in enumerate(R['exp3_order']['0.1']):
        axe.plot(obj['frames'], obj['phase'], 'o-', color=OBJ[o], lw=1.0, ms=2.2)
    axe.set_xlabel(T['Ex'], labelpad=1); axe.set_ylabel(T['Ey'], labelpad=1); axe.set_xticks([0, 10, 20])
    axe.set_title(T['E'], fontsize=6.3, color=INK2, pad=2)
    axe.text(0.97, 0.06, T['Enote'], transform=axe.transAxes, ha='right', fontsize=5.7); spines(axe)

    # F: capacity
    axf = fig.add_axes([0.74, 0.10, 0.22, 0.32])
    cap = R['exp4_capacity']
    for h, col in HEADS.items():
        rows = sorted([c for c in cap if c['heads'] == h], key=lambda c: c['objects'])
        x = [c['objects'] for c in rows]; y = [c['mean'] for c in rows]; e = [c['sd'] for c in rows]
        axf.errorbar(x, y, yerr=e, color=col, lw=1.3, ms=3, marker='o', capsize=1.5, elinewidth=0.6)
        axf.text(9.35, y[-1], T['head'] % h, fontsize=5.8, va='center')
    axf.set_xlim(2.5, 10.7); axf.set_ylim(0.7, 1.03); axf.set_xticks([3, 5, 7, 9]); axf.set_yticks([0.7, 0.8, 0.9, 1.0])
    axf.set_xlabel(T['Fx'], labelpad=1); axf.set_ylabel(T['Fy'], labelpad=1)
    axf.set_title(T['F'], fontsize=6.3, color=INK2, pad=2); spines(axf)

    for x, y, lt in [(0.005, 0.985, 'A'), (0.31, 0.985, 'B'), (0.66, 0.985, 'C'), (0.005, 0.47, 'D'),
                     (0.34, 0.47, 'E'), (0.675, 0.47, 'F')]:
        fig.text(x, y, lt, fontsize=10, fontweight='bold', va='top')
    name = 'figure' if lang == 'en' else 'figure_zh'
    fig.savefig(RES / f'{name}.png', dpi=600, facecolor='white')
    fig.savefig(RES / f'{name}.pdf', facecolor='white')
    print('saved', RES / f'{name}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', default='en', choices=['en', 'zh'])
    main(ap.parse_args().lang)
