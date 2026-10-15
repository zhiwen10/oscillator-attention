"""Final 3-panel figure (one row): A binding by synchrony, B oscillators at scale, C oscillator attention.

    python experiments/make_final_figure.py            # results/final_figure.png/.pdf
    python experiments/make_final_figure.py --lang zh

A = make_simple_figure.py panels A, C, D: input tokens -> attention from phases (on the arrow) -> recovered objects.
B, C = make_overview_figure.py panels B, C (Un-0 oscillator-state trajectories; associative recall).
Needs results/un0_dynamics.pt and results/assoc_p8.json (see those scripts).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_overview_figure as ov                 # noqa: E402  panel_subspace, panel_attention
import make_simple_figure as sf                   # noqa: E402  TEXT, clean, groups_from_phases
from oscattn import make_task, continuity_coupling, simulate, phase_attention, same_object, binding_accuracy  # noqa: E402

RES = ov.RES
NOTE_B = {'en': 'Un-0, 16,384 oscillators; 10 ImageNet classes' + chr(10) + 'class decoding %.0f%% → %.0f%%',
          'zh': 'Un-0（16,384个振荡器）；10类ImageNet' + chr(10) + '类别解码准确率 %.0f%% → %.0f%%'}
OBJ, INK, MUTED = sf.OBJ, sf.INK, sf.MUTED


def binding_panels(axA, axB, axC, T, small_matrix=False):
    task = make_task(seed=0)
    final, _ = simulate(continuity_coupling(task), steps=1200)
    th = final[0]
    acc = binding_accuracy(same_object(final), task.label)
    L, F, P, V = (a.numpy() for a in (task.label, task.frame, task.pos, task.vel))

    # input: shuffled, unlabelled tokens (position + velocity)
    axA.quiver(P[:, 0], P[:, 1], V[:, 0], V[:, 1], color=MUTED, angles='xy', scale_units='xy', scale=0.6,
               width=0.004, headwidth=3, headlength=3)
    axA.scatter(P[:, 0], P[:, 1], s=4, color=INK, lw=0, zorder=3)
    axA.set_aspect('equal'); sf.clean(axA)
    axA.set_title(T['A'], fontsize=8, pad=3)
    axA.text(0.5, -0.05, T['Anote'], transform=axA.transAxes, ha='center', va='top', fontsize=6, color=MUTED)

    # attention read from phase differences, tokens sorted by object then frame
    order = np.lexsort((F, L))
    # vector cells (pcolormesh), not a bitmap: tiny embedded images get lost when the PDF is edited
    M = phase_attention(th).numpy()[np.ix_(order, order)]
    n = len(M)
    axB.pcolormesh(np.arange(n + 1) - 0.5, np.arange(n + 1) - 0.5, M, cmap='Greys', vmin=0, vmax=1,
                   shading='flat', edgecolors='face', linewidth=0.05, antialiased=False, rasterized=False)
    axB.set_xlim(-0.5, n - 0.5); axB.set_ylim(n - 0.5, -0.5)
    counts, y0 = np.bincount(L), 0
    for o in range(3):
        w = 4.0 if small_matrix else 2.2
        axB.add_patch(plt.Rectangle((-1.5 - w, y0 - 0.5), w, counts[o], fc=OBJ[o], ec='none', clip_on=False))
        axB.add_patch(plt.Rectangle((y0 - 0.5, len(L) + 0.8), counts[o], w, fc=OBJ[o], ec='none', clip_on=False))
        y0 += counts[o]
    sf.clean(axB)
    fs = 5.8 if small_matrix else 8
    axB.set_title(T['C'], fontsize=fs, pad=2 if small_matrix else 3)
    axB.text(0.5, -0.16 if small_matrix else -0.1, T['Cnote'], transform=axB.transAxes, ha='center', va='top',
             fontsize=5 if small_matrix else 6, color=MUTED)

    # output: tokens coloured by the phase group they ended in
    g = sf.groups_from_phases(th)
    colour_of = {k: OBJ[np.bincount(L[g == k]).argmax()] for k in np.unique(g)}
    for k in np.unique(g):
        m = g == k
        i = np.argsort(F[m])
        axC.plot(P[m][i, 0], P[m][i, 1], color=colour_of[k], lw=1.3, alpha=0.5, solid_capstyle='round')
        axC.scatter(P[m, 0], P[m, 1], s=4, color=colour_of[k], lw=0, zorder=3)
    axC.set_aspect('equal'); sf.clean(axC)
    axC.set_title(T['D'], fontsize=8, pad=3)
    axC.text(0.5, -0.05, T['Dnote'] % (100 * acc), transform=axC.transAxes, ha='center', va='top', fontsize=6,
             color=MUTED)
    for ax in (axA, axB, axC):
        ax.set_box_aspect(1)


def main(lang):
    plt.rcParams.update({'font.family': ['Arial', 'Microsoft YaHei'], 'font.size': 7, 'axes.linewidth': 0.6,
                         'axes.edgecolor': MUTED, 'axes.unicode_minus': False, 'pdf.fonttype': 42})
    W, H = 18.0, 5.6                                      # cm
    fig = plt.figure(figsize=(W / 2.54, H / 2.54), dpi=300)
    cm = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])

    # A: input -> (attention matrix on the arrow) -> output
    axIn = cm(0.35, 1.3, 2.9, 2.9)
    axM = cm(3.75, 2.1, 1.3, 1.3)
    axOut = cm(5.55, 1.3, 2.9, 2.9)
    binding_panels(axIn, axM, axOut, sf.TEXT[lang], small_matrix=True)
    ya = (1.3 + 1.45) / H
    fig.add_artist(matplotlib.patches.FancyArrowPatch((3.3 / W, ya), (5.5 / W, ya), transform=fig.transFigure,
                                                      arrowstyle='-|>', mutation_scale=9, lw=1.0, color=INK,
                                                      zorder=0))
    fig.text(4.4 / W, 4.92 / H, ov.TEXT[lang]['A'], ha='center', va='bottom', fontsize=8.5)
    for ax in (axIn, axOut):
        ax.title.set_fontsize(7)

    # B: Un-0 oscillator state; C: associative recall (x-axis half the overview length)
    axB = cm(9.35, 1.55, 3.2, 3.2)
    ov.panel_subspace(axB, ov.TEXT[lang])
    acc = torch.load(RES / 'un0_dynamics.pt', weights_only=False)['decoding_acc']
    axB.texts[-1].set_text(NOTE_B[lang] % (100 * acc[0], 100 * acc[-1]))   # narrower note than the overview's
    axC = cm(15.55, 2.05, 1.4, 2.6)
    ov.panel_attention(axC, ov.TEXT[lang])
    note = axC.texts[-1]                                  # note below the x-axis label, two lines
    note.set_position((0.5, -0.45)); note.set_ha('center')
    note.set_text(note.get_text().replace('; ', '\n').replace('；', '\n'))

    for lt, x in [('A', 0.05), ('B', 8.75), ('C', 12.95)]:
        fig.text(x / W, 5.5 / H, lt, fontsize=10, fontweight='bold', va='top')
    suffix = '' if lang == 'en' else '_zh'
    for ext in ('png', 'pdf'):
        fig.savefig(RES / f'final_figure{suffix}.{ext}', dpi=300, facecolor='white')
    print('saved', RES / f'final_figure{suffix}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', default='en', choices=['en', 'zh'])
    main(ap.parse_args().lang)
