"""How fast do oscillator attention and softmax attention learn in-context association?
After training, each model is also tested, without retraining, on sequences with more or fewer pairs.

    python experiments/assoc_learning.py                       # default comparison -> results/assoc.json
    python experiments/assoc_learning.py --pairs 16 --tag p16  # harder: 16 pairs per sequence

Every model sees the same stream of fresh training sequences (same seed -> same data) and is scored on
the same held-out test set every `--eval-every` steps. "Examples to criterion" = training sequences seen
before test accuracy first reaches the criterion (default 95%).
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from oscattn.assoc import make_batch
from oscattn.models import Model, n_params

ROOT = Path(__file__).resolve().parents[1]

# name: (attention kind, layers, dynamics steps, loops, oscillator options)
MODELS = {
    'softmax-1L': ('softmax', 1, 0, 1, {}),
    'softmax-2L': ('softmax', 2, 0, 1, {}),
    'softmax-1Lx4': ('softmax', 1, 0, 4, {}),     # control: one layer re-applied 4x with shared weights
    'osc-1L-T4': ('osc', 1, 4, 1, {'dt': 1.0}),
    'osc-1L-T0': ('osc', 1, 0, 1, {}),            # ablation: dynamics switched off
    'osc-1L-T4-nolag': ('osc', 1, 4, 1, {'dt': 1.0, 'learn_lag': False}),   # ablation: symmetric coupling
    'osc-1L-T1x4': ('osc', 1, 1, 4, {'dt': 1.0}),  # osc block looped 4x, the looped-softmax analog
    'osc-1L-T4-qk': ('osc', 1, 4, 1, {'dt': 1.0, 'separate_qk': True}),      # separate query/key phases
    'osc-1L-T4-staticq': ('osc', 1, 4, 1, {'dt': 1.0, 'static_query': True}),  # receiver phases frozen
}


def evaluate(model, test):
    model.eval()
    with torch.no_grad():
        acc = (model(test[0]).argmax(-1) == test[1]).float().mean().item()
    model.train()
    return acc


def train(name, seed, lr, args, test, device):
    kind, layers, steps, loops, osc = MODELS[name]
    torch.manual_seed(seed)
    vocab = args.keys + args.vals
    model = Model(vocab, d=args.d, heads=args.heads, n_layers=layers, kind=kind, steps=steps, loops=loops, **osc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200))   # short warm-up
    g = torch.Generator().manual_seed(1000 + seed)                  # data stream: same for every model
    curve, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        x, y = make_batch(args.batch, args.pairs, args.keys, args.vals, generator=g, device=device)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % args.eval_every == 0:
            curve.append((step * args.batch, evaluate(model, test), loss.item()))
    seen = [e for e, a, _ in curve if a >= args.criterion]
    gen_g = torch.Generator().manual_seed(11)
    generalization = {n: evaluate(model, make_batch(2048, n, args.keys, args.vals, generator=gen_g, device=device))
                      for n in args.gen_pairs}
    return {'generalization': generalization, 'model': name, 'seed': seed, 'lr': lr, 'params': n_params(model), 'curve': curve,
            'final_acc': curve[-1][1], 'examples_to_criterion': seen[0] if seen else None,
            'seconds': time.time() - t0}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=list(MODELS))
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lrs', type=float, nargs='+', default=[1e-3, 3e-3])
    ap.add_argument('--pairs', type=int, default=8)
    ap.add_argument('--keys', type=int, default=32)
    ap.add_argument('--vals', type=int, default=32)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--criterion', type=float, default=0.95)
    ap.add_argument('--gen-pairs', type=int, nargs='+', default=[4, 8, 12, 16, 24],
                    help='after training, test (no retraining) on sequences with this many pairs')
    ap.add_argument('--tag', default='')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    test = make_batch(4096, args.pairs, args.keys, args.vals, generator=torch.Generator().manual_seed(7),
                      device=args.device)
    runs = []
    for name in args.models:
        for lr, seed in [(lr, s) for lr in args.lrs for s in range(args.seeds)]:
            r = train(name, seed, lr, args, test, args.device)
            runs.append(r)
            print('%-15s lr %g seed %d  params %6d  final %.3f  examples to %.0f%%: %s  (%.0fs)'
                  % (name, lr, seed, r['params'], r['final_acc'], 100 * args.criterion,
                     r['examples_to_criterion'], r['seconds']), flush=True)
    out = ROOT / 'results' / ('assoc%s.json' % ('_' + args.tag if args.tag else ''))
    out.write_text(json.dumps({'args': vars(args), 'runs': runs}))
    print('saved', out)
