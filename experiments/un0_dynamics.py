"""What the oscillator dynamics do inside Un-0 (after the blog's Figs 0, 5b and 6), recomputed locally.

    python experiments/un0_dynamics.py            # ~1 min on GPU -> results/un0_dynamics.pt
    python experiments/un0_dynamics.py --plot-only --lang zh

Un-0 generates an image by integrating 16,385 coupled Kuramoto oscillators (10 Euler steps, t = 0 -> 1)
from random phases, with a class-specific one-way drive, then decoding the final phases. Here we keep the
state at every step and ask:

  (a) what does the decoder see at each step?        -> decode the intermediate phases
  (b) how does the oscillator state move?             -> PCA of the readout features (fit at t = 1, train half)
  (c) when does the state carry the class?            -> nearest-centroid class decoding at each step,
                                                         on held-out samples (chance = 1 / n_classes)
Needs checkpoints/imagenet64_n16384.pt (see un0_samples.py).
"""
import argparse
import os

os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')

from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
CLASSES = {130: 'flamingo', 207: 'golden retriever', 291: 'lion', 323: 'monarch', 417: 'balloon',
           437: 'lighthouse', 980: 'volcano', 973: 'coral reef', 948: 'Granny Smith', 511: 'convertible'}
SHOW = [130, 323, 980]             # rows of the image strip


def load_model(dev):
    from un0.model import build_from_config, build_imagenet64_model
    state = torch.load(ROOT / 'checkpoints' / 'imagenet64_n16384.pt', map_location='cpu', weights_only=True)
    model = build_from_config(build_imagenet64_model, state['config'])
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state['model'].items()})
    return model.to(dev).eval()


@torch.no_grad()
def trajectories(model, class_ids, generator):
    """All integration states (steps+1, B, n+n_cond), exactly as Un-0's forward computes them."""
    from torchdiffeq import odeint
    p = next(model.parameters())
    x0 = model._sample_initial_state(len(class_ids), device=p.device, dtype=p.dtype, generator=generator)
    drive = model._class_drive(class_ids)
    grid = model._time_grid(p.device, p.dtype)
    return odeint(lambda t, s: model.dynamics(s, t, drive), x0, grid, method=model.solver,
                  options={'step_size': model.integration_time / model.num_steps}), grid


@torch.no_grad()
def compute(per_class=40, seed=0):
    dev = 'cuda'
    model = load_model(dev)
    n = model.dynamics.n
    g = torch.Generator(device=dev).manual_seed(seed)
    feats, labels, strip = [], [], {}
    for c in CLASSES:
        ids = torch.full((per_class,), c, device=dev)
        states, grid = trajectories(model, ids, g)                      # (T, B, n+1)
        phases = states[:, :, :n]
        f = torch.stack([model.readout(phases[t]) for t in range(len(grid))])   # (T, B, 2n)
        feats.append(f.half().cpu()); labels.append(torch.full((per_class,), c))
        if c in SHOW:            # decode sample 0 at every step
            strip[c] = torch.stack([((model.decoder(f[t, :1]).view(1, 3, 64, 64) + 1) / 2).clamp(0, 1)[0].cpu()
                                    for t in range(len(grid))])
    feats = torch.cat(feats, 1).float()                                 # (T, N, 2n)
    labels = torch.cat(labels)
    T, N, D = feats.shape

    # split first: half the samples of each class build the PCA basis and class centroids, half are tested
    rng = np.random.default_rng(0)
    train = torch.zeros(N, dtype=torch.bool)
    for c in CLASSES:
        idx = np.flatnonzero(labels.numpy() == c)
        train[rng.choice(idx, len(idx) // 2, replace=False)] = True

    # (b) PCA fit on the TRAINING samples' t = 1 features only; every step of every sample projected onto it
    mu1 = feats[-1, train].mean(0)
    X1 = feats[-1, train] - mu1
    U, S, V = torch.pca_lowrank(X1, q=10, center=False)
    proj = (feats - mu1) @ V[:, :3]                                     # (T, N, 3)
    explained = (S[:3] ** 2 / (X1 ** 2).sum()).tolist()

    # (c) nearest-centroid decoding in the 10-PC space, tested on held-out samples
    cls = torch.tensor(list(CLASSES))
    acc = []
    for t in range(T):
        Z = (feats[t] - mu1) @ V                                         # 10 PCs
        cent = torch.stack([Z[train & (labels == c)].mean(0) for c in cls])
        pred = cls[torch.cdist(Z[~train], cent).argmin(1)]
        acc.append((pred == labels[~train]).float().mean().item())
    out = {'time': grid.cpu().tolist(), 'proj': proj, 'labels': labels, 'names': CLASSES, 'strip': strip,
           'decoding_acc': acc, 'chance': 1 / len(CLASSES), 'explained_var_pc123': explained,
           'per_class': per_class, 'n_oscillators': n}
    torch.save(out, RES / 'un0_dynamics.pt')
    print('decoding accuracy by step:', [round(a, 2) for a in acc])
    return out


TEXT = {
    'en': dict(a='Images emerge as oscillators evolve', b='Oscillator state: random → class clusters',
               c='Class readable from oscillator state', t='t', pc='PC', acc='class decoding accuracy',
               chance='chance', start='start (random phases)', colour='colour = class; bold = class mean',
               names={130: 'flamingo', 323: 'monarch', 980: 'volcano'}, note='Un-0, 16,384 oscillators; %d classes × %d samples'),
    'zh': dict(a='振荡器演化中图像逐步形成', b='振荡器状态：随机 → 按类别聚集',
               c='振荡器状态中可读出类别', t='t', pc='主成分', acc='类别解码准确率',
               chance='随机', start='起点（随机相位）', colour='颜色 = 类别；粗线 = 类别均值',
               names={130: '火烈鸟', 323: '帝王蝶', 980: '火山'}, note='Un-0，16,384个振荡器；%d类 × 每类%d个样本'),
}
COLORS = ['#e8547a', '#c9a227', '#b5651d', '#eb8a1e', '#d62728', '#5b8def', '#7a1f1f', '#17a2b8', '#4caf50', '#8c8c8c']


def plot(d, lang):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = TEXT[lang]
    plt.rcParams.update({'font.family': ['Arial', 'Microsoft YaHei'], 'font.size': 7, 'axes.linewidth': 0.6,
                         'axes.edgecolor': '#8a8a8a', 'axes.unicode_minus': False, 'pdf.fonttype': 42})
    time, proj, lab = np.array(d['time']), d['proj'].numpy(), d['labels'].numpy()
    show_t = [0, 2, 4, 6, 8, 10]
    fig = plt.figure(figsize=(18 / 2.54, 6.2 / 2.54), dpi=300)

    # (a) image strip: rows = classes, columns = time
    rows = list(d['strip'])
    for r, c in enumerate(rows):
        for k, t in enumerate(show_t):
            ax = fig.add_axes([0.035 + k * 0.057, 0.62 - r * 0.24, 0.054, 0.2])
            ax.imshow(d['strip'][c][t].permute(1, 2, 0).numpy()); ax.set_axis_off()
            if k == 0:
                ax.text(-0.08, 0.5, T['names'][c], transform=ax.transAxes, rotation=90, ha='right',
                        va='center', fontsize=5.8)
            if r == len(rows) - 1:
                ax.text(0.5, -0.12, '%s=%.1f' % (T['t'], time[t]), transform=ax.transAxes, ha='center',
                        va='top', fontsize=6)
    fig.text(0.205, 0.93, T['a'], ha='center', fontsize=8)

    # (b) PCA trajectories: class-mean path (bold) + a few single samples (thin)
    ax = fig.add_axes([0.42, 0.14, 0.26, 0.72])
    names = list(d['names'])
    for i, c in enumerate(names):
        m = lab == c
        for s in np.flatnonzero(m)[:4]:
            ax.plot(proj[:, s, 0], proj[:, s, 1], color=COLORS[i], lw=0.35, alpha=0.5)
        mu = proj[:, m].mean(1)
        ax.plot(mu[:, 0], mu[:, 1], color=COLORS[i], lw=1.4)
        ax.scatter(proj[-1, m, 0], proj[-1, m, 1], s=2, color=COLORS[i], lw=0, alpha=0.8)
    ax.scatter(proj[0, :, 0].mean(), proj[0, :, 1].mean(), s=18, color='k', marker='x', lw=1, zorder=5)
    ax.annotate(T['start'], (proj[0, :, 0].mean(), proj[0, :, 1].mean()), xytext=(0.62, 0.12),
                textcoords='axes fraction', fontsize=5.5, arrowprops={'arrowstyle': '-', 'lw': 0.5, 'color': 'k'})
    ax.text(0.02, 0.98, T['colour'], transform=ax.transAxes, fontsize=5.5, color='#8a8a8a', va='top')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(T['pc'] + ' 1', labelpad=2); ax.set_ylabel(T['pc'] + ' 2', labelpad=2)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.set_title(T['b'], fontsize=8, pad=4)

    # (c) decoding accuracy over time
    ax = fig.add_axes([0.76, 0.2, 0.22, 0.62])
    ax.plot(time, d['decoding_acc'], 'o-', color='#d1495b', ms=3, lw=1.4)
    ax.axhline(d['chance'], color='#8a8a8a', ls=':', lw=0.8)
    ax.text(time[-1], d['chance'] + 0.02, T['chance'], ha='right', va='bottom', fontsize=5.5, color='#8a8a8a')
    ax.set_ylim(0, 1.05); ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel(T['t'], labelpad=1); ax.set_ylabel(T['acc'], labelpad=1)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.set_title(T['c'], fontsize=8, pad=4)
    ax.text(1.0, -0.3, T['note'] % (len(names), d['per_class']), transform=ax.transAxes, ha='right',
            fontsize=5.5, color='#8a8a8a')

    for x, lt in [(0.005, 'A'), (0.38, 'B'), (0.72, 'C')]:
        fig.text(x, 0.98, lt, fontsize=10, fontweight='bold', va='top')
    suffix = '' if lang == 'en' else '_zh'
    for ext in ('png', 'pdf'):
        fig.savefig(RES / f'un0_dynamics{suffix}.{ext}', dpi=300, facecolor='white')
    print('saved', RES / f'un0_dynamics{suffix}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot-only', action='store_true')
    ap.add_argument('--lang', default='en', choices=['en', 'zh'])
    args = ap.parse_args()
    d = torch.load(RES / 'un0_dynamics.pt', weights_only=False) if args.plot_only else compute()
    plot(d, args.lang)
