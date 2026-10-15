"""Ad-hoc hyperparameter sweep used to pick the oscillator settings (T=4, dt=1.0). 3000 steps, seed 0.

    python experiments/sweep_osc.py

Results are recorded in AGENTS.md ("Tuning history"). Edit `configs` to try other settings.
"""
import sys, time, itertools, json
import torch, torch.nn.functional as F
from oscattn.assoc import make_batch
from oscattn.models import Model

dev = 'cuda'
test = make_batch(4096, 8, 32, 32, generator=torch.Generator().manual_seed(7), device=dev)


def run(kind, steps_dyn, dt, lr, loops=1, n_steps=3000, seed=0, beta=4.0):
    torch.manual_seed(seed)
    m = Model(64, kind=kind, steps=steps_dyn, loops=loops).to(dev)
    if kind == 'osc':
        for b in m.blocks:
            b.attn.dt = dt
            b.attn.log_beta.data.fill_(torch.log(torch.tensor(beta)))
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


configs = [("osc", T, dt, lr, 4.0) for T, dt, lr in [(4, 1.0, 1e-3), (8, 0.5, 3e-3), (8, 0.5, 1e-3), (4, 1.0, 3e-3)]]


for kind, T, dt, lr, beta in configs:
    t0 = time.time()
    if kind == 'softmax-loop4':
        a = run('softmax', 0, 0, lr, loops=4)
    else:
        a = run(kind, T, dt, lr, beta=beta)
    print(kind, 'T=%d dt=%.1f lr=%g beta=%g' % (T, dt, lr, beta), a, '%.0fs' % (time.time() - t0), flush=True)
