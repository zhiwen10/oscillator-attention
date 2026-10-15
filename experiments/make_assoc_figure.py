"""Figure + summary table for the associative-recall comparison.

    python experiments/make_assoc_figure.py                 # reads results/assoc_p8.json
    python experiments/make_assoc_figure.py --tag p16 --lang zh

For each model the learning rate is chosen by one rule (best mean accuracy over the whole learning
curve, averaged across seeds), so every model gets the same tuning.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

LABELS = {
    'en': {'softmax-1L': 'Softmax attention, 1 layer', 'softmax-2L': 'Softmax attention, 2 layers',
           'softmax-1Lx4': 'Softmax, 1 layer looped 4x', 'osc-1L-T4': 'Oscillator attention, 1 layer',
           'osc-1L-T0': 'Oscillator, no dynamics', 'osc-1L-T4-nolag': 'Oscillator, no phase lag',
           'osc-1L-T1x4': 'Oscillator, 1 layer looped 4x', 'osc-1L-T4-qk': 'Oscillator, separate q/k phases',
           'osc-1L-T4-staticq': 'Oscillator, static query phases',
           'osc-1L-T1x4-staticq': 'Oscillator looped 4x, static query',
           'a': 'Learning curve', 'b': 'Training sequences to 95% accuracy', 'c': 'Test on more pairs than trained',
           'x_seen': 'Training sequences seen (thousands)', 'y_acc': 'Test accuracy', 'x_pairs': 'Pairs in sequence',
           'never': 'not reached', 'trained': 'trained'},
    'zh': {'softmax-1L': 'Softmax 注意力（1层）', 'softmax-2L': 'Softmax 注意力（2层）',
           'softmax-1Lx4': 'Softmax 1层循环4次', 'osc-1L-T4': '振荡注意力（1层）',
           'osc-1L-T0': '振荡（无动力学）', 'osc-1L-T4-nolag': '振荡（无相位滞后）',
           'osc-1L-T1x4': '振荡 1层循环4次', 'osc-1L-T4-qk': '振荡（分离q/k相位）',
           'osc-1L-T4-staticq': '振荡（静态查询相位）', 'osc-1L-T1x4-staticq': '振荡循环4次（静态查询相位）',
           'a': '学习曲线', 'b': '达到95%准确率所需训练序列数', 'c': '在多于训练对数的序列上测试',
           'x_seen': '已训练序列数（千）', 'y_acc': '测试准确率', 'x_pairs': '序列中的配对数',
           'never': '未达到', 'trained': '训练'},
}
COLORS = {'softmax-1L': '#9aa5b1', 'softmax-2L': '#3b6fb6', 'softmax-1Lx4': '#6c4fa3',
          'osc-1L-T4': '#d1495b', 'osc-1L-T0': '#edae49', 'osc-1L-T4-nolag': '#e8a3ad',
          'osc-1L-T1x4': '#c46f2e', 'osc-1L-T4-qk': '#b0406f',
          'osc-1L-T4-staticq': '#d98992', 'osc-1L-T1x4-staticq': '#e07a5f'}
_PALETTE = ['#4c9f70', '#7e6bc4', '#4fa3a5', '#8f7a45', '#a05fc4']


def color_for(model):
    if model not in COLORS:
        COLORS[model] = _PALETTE[len(COLORS) % len(_PALETTE)]
    return COLORS[model]


def best_lr_runs(runs, model):
    by_lr = {}
    for r in runs:
        if r['model'] == model:
            by_lr.setdefault(r['lr'], []).append(r)
    score = {lr: np.mean([np.mean([a for _, a, _ in r['curve']]) for r in rs]) for lr, rs in by_lr.items()}
    lr = max(score, key=score.get)
    return lr, by_lr[lr]


def summarise(data):
    runs, rows = data['runs'], []
    for m in dict.fromkeys(r['model'] for r in runs):
        lr, rs = best_lr_runs(runs, m)
        ex = [r['examples_to_criterion'] for r in rs]
        reached = [e for e in ex if e is not None]
        gen = {n: np.mean([r['generalization'][n] for r in rs]) for n in rs[0]['generalization']}
        rows.append({'model': m, 'lr': lr, 'params': rs[0]['params'], 'n': len(rs), 'reached': len(reached),
                     'examples_median': float(np.median(reached)) if len(reached) == len(rs) else None,
                     'examples_all': ex, 'final_mean': float(np.mean([r['final_acc'] for r in rs])),
                     'generalization': gen, 'runs': rs})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='p8')
    ap.add_argument('--lang', default='en', choices=['en', 'zh'])
    args = ap.parse_args()
    data = json.loads((ROOT / 'results' / f'assoc_{args.tag}.json').read_text())
    rows = summarise(data)
    T = LABELS[args.lang]

    print('%-16s %7s %6s %9s %22s %s' % ('model', 'params', 'lr', 'final', 'examples to 95% (med)', 'generalization'))
    for r in rows:
        med = '%d (%d/%d)' % (r['examples_median'], r['reached'], r['n']) if r['examples_median'] else \
              'not all (%d/%d)' % (r['reached'], r['n'])
        gen = ' '.join('%s:%.2f' % (n, a) for n, a in r['generalization'].items())
        print('%-16s %7d %6g %9.3f %22s  %s' % (r['model'], r['params'], r['lr'], r['final_mean'], med, gen))

    if args.lang == 'zh':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))

    for r in rows:
        curves = np.array([[a for _, a, _ in run['curve']] for run in r['runs']])
        x = np.array([e for e, _, _ in r['runs'][0]['curve']]) / 1000
        mu, sd = curves.mean(0), curves.std(0)
        ax[0].plot(x, mu, color=color_for(r['model']), label=T.get(r['model'], r['model']), lw=1.6)
        ax[0].fill_between(x, mu - sd, mu + sd, color=color_for(r['model']), alpha=0.15, lw=0)
    ax[0].set(xlabel=T['x_seen'], ylabel=T['y_acc'], title=T['a'], ylim=(0, 1.02))
    ax[0].legend(frameon=False, fontsize=7, loc='lower right')

    xmax = data['args']['steps'] * data['args']['batch'] / 1000
    for i, r in enumerate(rows):
        vals = [e / 1000 if e is not None else np.nan for e in r['examples_all']]
        if all(np.isnan(vals)):
            ax[1].barh(i, xmax, color='none', edgecolor=color_for(r['model']), hatch='//', lw=0.8)
            ax[1].text(xmax * 0.5, i, T['never'], va='center', ha='center', fontsize=7)
        else:
            ax[1].barh(i, np.nanmedian(vals), color=color_for(r['model']), alpha=0.85)
            ax[1].scatter(vals, np.full(len(vals), i) + np.linspace(-0.2, 0.2, len(vals)), s=8, color='k', zorder=3)
    ax[1].set_yticks(range(len(rows)), [T.get(r['model'], r['model']) for r in rows], fontsize=7)
    ax[1].invert_yaxis()
    ax[1].set(xlabel=T['x_seen'], title=T['b'])

    for r in rows:
        ns = [int(n) for n in r['generalization']]
        ax[2].plot(ns, list(r['generalization'].values()), 'o-', color=color_for(r['model']), ms=3, lw=1.4)
    ax[2].axvline(data['args']['pairs'], color='k', ls=':', lw=0.8)
    ax[2].text(data['args']['pairs'], 1.03, ' ' + T['trained'], fontsize=7)
    ax[2].set(xlabel=T['x_pairs'], ylabel=T['y_acc'], title=T['c'], ylim=(0, 1.08))

    fig.tight_layout()
    suffix = '' if args.lang == 'en' else '_zh'
    for ext in ('png', 'pdf'):
        fig.savefig(ROOT / 'results' / f'assoc_{args.tag}{suffix}.{ext}', dpi=200)
    print('saved', ROOT / 'results' / f'assoc_{args.tag}{suffix}.png')


if __name__ == '__main__':
    main()
