"""Open up a trained oscillator-attention model: how does the query find its associated response?

    python experiments/assoc_mechanism.py                       # base model (evolving receiver phases)
    python experiments/assoc_mechanism.py --variant staticq     # frozen receiver/query phases

For held-out sequences it measures, at every dynamics step, the phase coupling similarity of the query
token to (a) the matching stimulus, (b) the response that followed it (the answer), (c) the other
responses. It also reports where the query's final attention lands. Saves results/assoc_mechanism*.json/.png.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from oscattn.assoc import make_batch
from oscattn.models import Model

ROOT = Path(__file__).resolve().parents[1]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
P, NK, NV = 8, 32, 32


def train(steps=4000, lr=3e-3, seed=0, variant='base'):
    torch.manual_seed(seed)
    m = Model(NK + NV, kind='osc', steps=4, dt=1.0,
              **({'static_query': True} if variant == 'staticq' else {})).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200))
    g = torch.Generator().manual_seed(1000 + seed)
    for _ in range(steps):
        x, y = make_batch(64, P, NK, NV, generator=g, device=dev)
        loss = F.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='base', choices=['base', 'staticq'])
    args = ap.parse_args()
    m = train(variant=args.variant)
    x, y = make_batch(2048, P, NK, NV, generator=torch.Generator().manual_seed(7), device=dev)
    with torch.no_grad():
        acc = (m(x).argmax(-1) == y).float().mean().item()
        layer = m.blocks[0].attn
        h = m.blocks[0].ln1(m.emb(x))
        _, A, phases = layer(h, return_attn=True, return_phases=True)     # phases: (T+1, B, H, L, m)
    B, L = x.shape
    q = L - 1
    key_pos = (x[:, :-1:2] == x[:, -1:]).float().argmax(1) * 2          # where the queried stimulus sat
    ans_pos = key_pos + 1                                                # the response that followed it
    rows = torch.arange(B)
    w = torch.softmax(layer.w, -1)[None, :, 0, :]                       # (1, H, m)
    lag = layer.lag[None, :, 0, :]

    def sim(t, j):          # coupling similarity the query sees toward token j at step t, mean over heads
        th_r = phases[0] if layer.static_query else phases[t]            # receiver phase (static or evolved)
        d = th_r[rows, :, q, :] - phases[t][rows, :, j, :] - lag
        return (w * torch.cos(d)).sum(-1).mean(1)                        # (B,)

    T = phases.shape[0]
    traj = {'stimulus': [], 'answer': [], 'other responses': []}
    for t in range(T):
        traj['stimulus'].append(sim(t, key_pos).mean().item())
        traj['answer'].append(sim(t, ans_pos).mean().item())
        others = [sim(t, torch.full_like(ans_pos, j)) for j in range(1, L - 1, 2)]
        others = torch.stack(others, 1)                                  # (B, P)
        mask = torch.arange(1, L - 1, 2, device=dev)[None] != ans_pos[:, None]
        traj['other responses'].append((others * mask).sum(1).div(mask.sum(1)).mean().item())

    attn_q = A[:, :, q, :].mean(1)                                       # (B, L) averaged over heads
    to_answer = attn_q[rows, ans_pos].mean().item()
    to_stim = attn_q[rows, key_pos].mean().item()
    to_self = attn_q[:, q].mean().item()
    per_head_top = (A[:, :, q, :].argmax(-1) == ans_pos[:, None]).float().mean(0).tolist()
    out = {'variant': args.variant, 'test_accuracy': acc, 'similarity_by_step': traj, 'query_attention': {
        'answer': to_answer, 'matching_stimulus': to_stim, 'self': to_self,
        'everything_else': 1 - to_answer - to_stim - to_self},
        'fraction_of_sequences_where_head_attends_most_to_answer': per_head_top,
        'learned_lag_rad': layer.lag[:, 0].tolist(), 'kappa': layer.log_kappa.exp().flatten().tolist(),
        'beta': layer.log_beta.exp().flatten().tolist()}
    suffix = '' if args.variant == 'base' else '_' + args.variant
    (ROOT / 'results' / ('assoc_mechanism%s.json' % suffix)).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ('learned_lag_rad',)}, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    for k, c in zip(traj, ['#edae49', '#d1495b', '#9aa5b1']):
        ax[0].plot(range(T), traj[k], 'o-', color=c, label=k)
    ax[0].set(xlabel='Oscillator dynamics step', ylabel='Phase coupling to query', title='Query locks onto the answer')
    ax[0].legend(frameon=False, fontsize=8)
    i = 0
    ax[1].bar(range(L), attn_q[i].cpu(), color=['#d1495b' if j == ans_pos[i] else '#edae49' if j == key_pos[i]
                                                else '#9aa5b1' for j in range(L)])
    ax[1].set(xlabel='Position in sequence', ylabel='Query attention (final phases)',
              title='Example sequence')
    for a in ax: a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(ROOT / 'results' / ('assoc_mechanism%s.png' % suffix), dpi=200)


if __name__ == '__main__':
    main()
