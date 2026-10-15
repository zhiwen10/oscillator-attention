"""Run the four experiments and save every number to results/results.json.

    python experiments/run_experiments.py            # all four, ~1-2 min on CPU
    python experiments/run_experiments.py --device cuda

Exp 1  Binding: do phase differences group tokens by object? (accuracy over integration steps)
Exp 2  Depth:   one oscillator network vs a stack of local-attention layers
Exp 3  Order:   with a phase lag, does phase inside each object encode time (a travelling wave)?
Exp 4  Capacity: accuracy vs number of objects, with 1 / 2 / 4 independent phase sets (heads)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from oscattn import (make_task, continuity_coupling, simulate, phase_attention, same_object,
                     binding_accuracy, stacked_local_attention)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'


def exp1_binding(device):
    """Binding accuracy of the phase readout at several points during integration (one example task)."""
    task = make_task(seed=0)
    K = continuity_coupling(task)
    record = (0, 50, 150, 300, 600, 1200)
    _, snaps = simulate(K, steps=1200, record=record, device=device)
    acc = {s: binding_accuracy(same_object(snaps[s]), task.label) for s in record}
    torch.save({'task': task, 'snaps': snaps}, OUT / 'exp1_example.pt')     # used by the figure
    return {'steps': list(record), 'accuracy': [acc[s] for s in record]}


def exp2_depth(device, seeds=8):
    """Same local coupling for both: oscillators (one shared matrix, run to convergence) vs k stacked
    local-attention layers."""
    osc, stack = [], []
    for seed in range(seeds):
        task = make_task(seed=seed)
        K = continuity_coupling(task, window=3)
        theta, _ = simulate(K, steps=1200, seed=seed, device=device)
        osc.append(binding_accuracy(same_object(theta), task.label))
        stack.append(stacked_local_attention(K, task.label, max_layers=14))
    stack = np.array(stack)
    layers_to_match = int(np.argmax(stack.mean(0) >= np.mean(osc)) + 1)
    return {'oscillator_mean': float(np.mean(osc)), 'oscillator_sd': float(np.std(osc)),
            'stack_mean': stack.mean(0).tolist(), 'stack_sd': stack.std(0).tolist(),
            'layers_to_match_oscillator': layers_to_match, 'seeds': seeds}


def exp3_order(device, lags=(0.05, 0.10, 0.15)):
    """With a forward phase lag, phase should rise steadily with frame inside each object."""
    task = make_task(seed=0)
    K = continuity_coupling(task)
    out = {}
    for lag in lags:
        theta, _ = simulate(K, steps=1500, lag=lag, frames=task.frame, device=device)
        per_object = []
        for o in task.label.unique():
            m = task.label == o
            idx = torch.argsort(task.frame[m])
            ph = np.unwrap(theta[0][m][idx].numpy())
            fr = task.frame[m][idx].numpy()
            per_object.append({'r': float(np.corrcoef(fr, ph)[0, 1]), 'span_rad': float(np.ptp(ph)),
                               'frames': fr.tolist(), 'phase': (ph - ph[0]).tolist()})
        out[str(lag)] = per_object
    # does the lag cost binding? score binding with the lag = 0.10 run
    theta, _ = simulate(K, steps=1500, lag=0.10, frames=task.frame, device=device)
    out['binding_with_lag_0.10'] = binding_accuracy(same_object(theta, threshold=0.5), task.label)
    return out


def exp4_capacity(device, n_objects=(3, 5, 7, 9), heads=(1, 2, 4), seeds=8):
    res = {}
    for n in n_objects:
        for h in heads:
            accs = []
            for seed in range(seeds):
                task = make_task(n_objects=n, seed=seed, noise=0.008, drop=0.1)
                K = continuity_coupling(task, window=3)
                theta, _ = simulate(K, steps=1200, heads=h, seed=seed, device=device)
                accs.append(binding_accuracy(same_object(theta), task.label))
            res[f'{n}_{h}'] = {'objects': n, 'heads': h, 'mean': float(np.mean(accs)), 'sd': float(np.std(accs))}
            print(f'  capacity: {n} objects, {h} head(s): {np.mean(accs):.3f}')
    return list(res.values())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    results = {}
    print('Exp 1 binding'); results['exp1_binding'] = exp1_binding(args.device); print(' ', results['exp1_binding'])
    print('Exp 2 depth'); results['exp2_depth'] = exp2_depth(args.device)
    print('  oscillator %.3f, layers to match %d' % (results['exp2_depth']['oscillator_mean'],
                                                     results['exp2_depth']['layers_to_match_oscillator']))
    print('Exp 3 order'); results['exp3_order'] = exp3_order(args.device)
    for lag in ('0.05', '0.1', '0.15'):
        print('  lag', lag, [round(o['r'], 3) for o in results['exp3_order'][lag]])
    print('  binding with lag 0.10:', round(results['exp3_order']['binding_with_lag_0.10'], 3))
    print('Exp 4 capacity'); results['exp4_capacity'] = exp4_capacity(args.device)
    (OUT / 'results.json').write_text(json.dumps(results, indent=1))
    print('saved', OUT / 'results.json')
