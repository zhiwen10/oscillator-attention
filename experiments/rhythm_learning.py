"""Rhythmic-world comparison: does oscillator attention learn periodic association faster?

    python experiments/rhythm_learning.py --tag r1

Same rules as assoc_learning.py: every model sees the same data stream (same seed), same lr grid, same
seeds; the looped softmax control is always included. Task: cycle of C tokens repeated R=3 times, query
one cycle token, target its cyclic successor (chance 1/C). Trained with C ~ {5..8} drawn per batch;
generalised (no retraining) to R in {1,2,4,6} and to unseen C in {4,10,12}.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from oscattn.models import Model, n_params
from oscattn.rhythm import make_rhythm_batch, draw_c

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    'softmax-1L': ('softmax', 1, 0, 1, {}),
    'softmax-2L': ('softmax', 2, 0, 1, {}),
    'softmax-1Lx4': ('softmax', 1, 0, 4, {}),
    'osc-1L-T4': ('osc', 1, 4, 1, {'dt': 1.0}),
    'osc-1L-T4-staticq': ('osc', 1, 4, 1, {'dt': 1.0, 'static_query': True}),
}


def make_eval(n, c, reps, device, seed=7):
    return make_rhythm_batch(n, c=c, reps=reps, generator=torch.Generator().manual_seed(seed), device=device)


def evaluate(model, tests):
    model.eval()
    with torch.no_grad():
        acc = sum((model(x).argmax(-1) == y).float().mean().item() for x, y in tests) / len(tests)
    model.train()
    return acc


def train(name, seed, lr, args, tests, device):
    kind, layers, steps, loops, osc = MODELS[name]
    torch.manual_seed(seed)
    model = Model(args.vocab, d=args.d, heads=args.heads, n_layers=layers, kind=kind, steps=steps,
                  loops=loops, **osc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200))
    g = torch.Generator().manual_seed(1000 + seed)
    curve, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        c = draw_c(g, args.cmin, args.cmax)
        x, y = make_rhythm_batch(args.batch, c=c, reps=args.reps, vocab=args.vocab, generator=g, device=device)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % args.eval_every == 0:
            curve.append((step * args.batch, evaluate(model, tests), loss.item()))
    seen = [e for e, a, _ in curve if a >= args.criterion]
    gen = {}
    for reps in args.gen_reps:                                     # more/fewer rhythm repetitions
        gen['r%d' % reps] = evaluate(model, [make_eval(1024, c, reps, device, seed=11 + c) for c in
                                             range(args.cmin, args.cmax + 1)])
    for c in args.gen_cs:                                          # unseen cycle lengths
        gen['c%d' % c] = evaluate(model, [make_eval(1024, c, args.reps, device, seed=21 + c)])
    return {'generalization': gen, 'model': name, 'seed': seed, 'lr': lr, 'params': n_params(model),
            'curve': curve, 'final_acc': curve[-1][1],
            'examples_to_criterion': seen[0] if seen else None, 'seconds': time.time() - t0}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=list(MODELS))
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lrs', type=float, nargs='+', default=[1e-3, 3e-3])
    ap.add_argument('--vocab', type=int, default=64)
    ap.add_argument('--cmin', type=int, default=5)
    ap.add_argument('--cmax', type=int, default=8)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--criterion', type=float, default=0.95)
    ap.add_argument('--gen-reps', type=int, nargs='+', default=[1, 2, 4, 6])
    ap.add_argument('--gen-cs', type=int, nargs='+', default=[4, 10, 12])
    ap.add_argument('--tag', default='')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    tests = [make_eval(1024, c, args.reps, args.device) for c in range(args.cmin, args.cmax + 1)]
    runs = []
    for name in args.models:
        for lr, seed in [(lr, s) for lr in args.lrs for s in range(args.seeds)]:
            r = train(name, seed, lr, args, tests, args.device)
            runs.append(r)
            print('%-17s lr %g seed %d  params %6d  final %.3f  examples to %.0f%%: %s  (%.0fs)'
                  % (name, lr, seed, r['params'], r['final_acc'], 100 * args.criterion,
                     r['examples_to_criterion'], r['seconds']), flush=True)
    out = ROOT / 'results' / ('rhythm%s.json' % ('_' + args.tag if args.tag else ''))
    out.write_text(json.dumps({'args': vars(args), 'runs': runs}))
    print('saved', out)
