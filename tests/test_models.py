"""OscAttention's vectorised update equals the Kuramoto–Sakaguchi sum written out term by term."""
import math

import torch
import torch.nn.functional as F

from oscattn.models import OscAttention, Model
from oscattn.assoc import make_batch


def test_sakaguchi_step_matches_explicit_sum():
    torch.manual_seed(0)
    layer = OscAttention(d=16, heads=2, steps=1, dt=0.3)
    with torch.no_grad():
        layer.lag.normal_(0, 0.5)
        layer.w.normal_(0, 0.5)
        layer.log_kappa.fill_(math.log(1.7))
    x = torch.randn(1, 5, 16)
    _, phases = layer(x, return_phases=True)
    th0, th1 = phases[0][0], phases[1][0]                     # (H, L, m)
    H, L, m = th0.shape
    w = torch.softmax(layer.w, -1)[:, 0]                      # (H, m)
    lag = layer.lag[:, 0]                                     # (H, m)
    beta = layer.log_beta.exp()[:, 0, 0]
    expected = th0.clone()
    for h in range(H):
        S = torch.tensor([[sum(w[h, k] * math.cos(th0[h, i, k] - th0[h, j, k] - lag[h, k]) for k in range(m))
                           for j in range(L)] for i in range(L)])
        S = S.masked_fill(torch.ones(L, L).tril() == 0, float('-inf'))
        A = torch.softmax(beta[h] * S, -1)
        for i in range(L):
            for k in range(m):
                drive = sum(A[i, j] * math.sin(th0[h, j, k] + lag[h, k] - th0[h, i, k]) for j in range(L))
                expected[h, i, k] = th0[h, i, k] + 0.3 * 1.7 * drive
    assert torch.allclose(th1, expected, atol=1e-5)


def test_models_run_and_match_parameter_counts():
    tokens, target = make_batch(4, n_pairs=3)
    soft, osc = Model(64, kind='softmax'), Model(64, kind='osc', steps=2)
    assert soft(tokens).shape == osc(tokens).shape == (4, 64)
    n = lambda mdl: sum(p.numel() for p in mdl.parameters())
    assert abs(n(soft) - n(osc)) < 0.01 * n(soft)             # only a few per-head scalars differ


VARIANTS = [dict(value_consensus=True), dict(pos='low'), dict(pos='none'), dict(separate_qk=True),
            dict(value_consensus=True, pos='low'), dict(value_consensus=True, separate_qk=True),
            dict(static_query=True)]


def test_receiver_phases_evolve_over_steps():
    """Regression: with the default settings the receiver/query phases must track the evolved phases at
    every dynamics step (they were once accidentally frozen at their initial values)."""
    torch.manual_seed(0)
    layer = OscAttention(d=16, heads=2, steps=2, dt=0.5)
    with torch.no_grad():
        layer.lag.normal_(0, 0.5)
        layer.w.normal_(0, 0.5)
        layer.log_kappa.fill_(math.log(1.3))
    x = torch.randn(1, 5, 16)
    _, phases = layer(x, return_phases=True)
    th0, th1, th2 = phases[:, 0, 0]                               # (L, m) head 0
    w = torch.softmax(layer.w, -1)[0, 0]
    lag = layer.lag[0, 0]
    beta = layer.log_beta.exp()[0, 0, 0]

    def step(th_receiver, th_sender):
        S = torch.tensor([[sum(w[k] * math.cos(th_receiver[i, k] - th_sender[j, k] - lag[k]) for k in range(th0.shape[1]))
                           for j in range(5)] for i in range(5)])
        S = S.masked_fill(torch.ones(5, 5).tril() == 0, float('-inf'))
        A = torch.softmax(beta * S, -1)
        return th_sender + 0.5 * 1.3 * torch.tensor(
            [[sum(A[i, j] * math.sin(th_sender[j, k] + lag[k] - th_sender[i, k]) for j in range(5))
              for k in range(th0.shape[1])] for i in range(5)])

    assert torch.allclose(th1, step(th0, th0), atol=1e-5)
    assert torch.allclose(th2, step(th1, th1), atol=1e-5)         # receiver = EVOLVED phases
    assert not torch.allclose(th2, step(th0, th1), atol=1e-3)     # ...not the frozen initial ones


def test_variants_match_parameter_counts_and_run():
    tokens, target = make_batch(4, n_pairs=3)
    n = lambda mdl: sum(p.numel() for p in mdl.parameters())
    base = n(Model(64, kind='softmax'))
    for kw in VARIANTS:
        m = Model(64, kind='osc', steps=4, dt=1.0, **kw)
        assert abs(n(m) - base) < 0.01 * base, (kw, n(m), base)
        loss = F.cross_entropy(m(tokens), target)
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters()
                                            if p.grad is not None), kw
