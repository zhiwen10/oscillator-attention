"""Screening for the option-B variants (3000 steps, seed 0, both lrs) — decides which variants earn a
full run. Compares against the looped-softmax control on the same data stream.

    python experiments/screen_variants.py

Every result, pass or fail, gets recorded in AGENTS.md section 5.
"""
import math
import time

import torch
import torch.nn.functional as F

from oscattn.assoc import make_batch
from oscattn.models import Model

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
test = make_batch(4096, 8, 32, 32, generator=torch.Generator().manual_seed(7), device=dev)

CONFIGS = {
    'osc-T1-x4':        dict(kind='osc', steps=1, loops=4, osc=dict(dt=1.0)),
    'osc-T2-x2':        dict(kind='osc', steps=2, loops=2, osc=dict(dt=1.0)),
    'osc-T4-cons.25':   dict(kind='osc', steps=4, loops=1, osc=dict(dt=1.0, value_consensus=True),
                             post=lambda m: [b.attn.log_kappa_v.data.fill_(math.log(0.25)) for b in m.blocks]),
    'osc-T1-x4-qk':     dict(kind='osc', steps=1, loops=4, osc=dict(dt=1.0, separate_qk=True)),
}
LRS = [1e-3, 3e-3]


def run(cfg, lr, n_steps=3000, seed=0):
    torch.manual_seed(seed)
    m = Model(64, kind=cfg['kind'], steps=cfg['steps'], loops=cfg['loops'], **cfg['osc']).to(dev)
    if cfg.get('post'):
        cfg['post'](m)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200))
    g = torch.Generator().manual_seed(1000 + seed)
    accs = []
    for s in range(1, n_steps + 1):
        x, y = make_batch(64, 8, 32, 32, generator=g, device=dev)
        loss = F.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        if s % 500 == 0:
            m.eval()
            with torch.no_grad():
                accs.append(round((m(test[0]).argmax(-1) == test[1]).float().mean().item(), 3))
            m.train()
    return accs


if __name__ == '__main__':
    for name, cfg in CONFIGS.items():
        for lr in LRS:
            t0 = time.time()
            accs = run(cfg, lr)
            print('%-17s lr %g  %s  (%.0fs)' % (name, lr, accs, time.time() - t0), flush=True)
