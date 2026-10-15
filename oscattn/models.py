"""Two attention layers with the same interface and parameter count, and a small model around them.

SoftmaxAttention   standard Transformer attention with rotary position encoding (RoPE).
OscAttention       each token is a set of phase oscillators; attention is phase synchrony.

How OscAttention works, per head:

    1. Each token i gets m phases  θ_i = atan2(W_s h_i, W_c h_i)        (learned, like q/k projections)
       plus a positional phase  f_k * i  ("time since the input arrived", a travelling wave; this is
       the same rotation RoPE applies, so both models get the same position information).
    2. Coupling = lagged phase similarity:  S_ij = Σ_k w_k cos(θ_ik − θ_jk − α_k),
       A = softmax_j(β S_ij), causal. The learned phase lag α (Kuramoto–Sakaguchi) makes coupling
       directional: i couples most to a j that trails it by α. Without it (learn_lag=False) coupling is
       symmetric and the model cannot express "the response follows the stimulus" well.
    3. Kuramoto–Sakaguchi dynamics for T steps:  dθ_ik/dt = κ Σ_j A_ij sin(θ_jk + α_k − θ_ik).
       Tokens that attend to each other lock (at lag α), so A is recomputed from the new phases
       and synchrony can spread (i locks to j, j to l  ->  i ends up coupled to l).
    4. Read out: out_i = Σ_j A_ij(T) v_j   with A from the final phases (values are read once, at the end).

With T = 0 and α = 0 this is RoPE attention with unit-amplitude, tied queries and keys; T > 0 adds the
dynamics.
Parameters: W_c, W_s replace W_q, W_k one for one, so both layers have 4 d^2 weights (+ a few scalars).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(L, device):
    return torch.ones(L, L, dtype=torch.bool, device=device).tril()


class SoftmaxAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.register_buffer('freq', 10000 ** (-torch.arange(self.dh // 2) / (self.dh // 2)), persistent=False)

    def rope(self, x):                                   # x: (B, H, L, dh)
        L = x.shape[2]
        ang = torch.arange(L, device=x.device)[:, None] * self.freq[None]
        cos, sin = ang.cos(), ang.sin()
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)

    def forward(self, x, return_attn=False):
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = self.rope(q), self.rope(k)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        A = logits.masked_fill(~causal_mask(L, x.device), float('-inf')).softmax(-1)
        y = (A @ v).transpose(1, 2).reshape(B, L, d)
        return (self.out(y), A) if return_attn else self.out(y)


class OscAttention(nn.Module):
    """Options (all default off = the original layer):
    value_consensus  also propagate the value vectors along the attention graph each dynamics step,
                     r ← r + dt κ_v (A r − r), mirroring how a looped-softmax block refines its residual
                     stream each iteration (B.1). Values are then read with the final A.
    pos              'full' positional phase on all oscillators (original); 'low' only on the slow half
                     (less phase wrapping at long lengths, B.2); 'none' no positional phase (ablation).
    separate_qk      query and key get their own phase projections (two d×d maps instead of one d×2d map,
                     so the parameter count is unchanged; m halves to d/(2·heads) per head, B.3). The
                     query phases are static by construction.
    static_query     freeze the receiver ("query") phases at their initial values while the key phases
                     evolve; the default is that both evolve together (Kuramoto: each oscillator's phase
                     is both sender and receiver).
    """

    def __init__(self, d, heads, steps=8, dt=0.5, learn_lag=True, value_consensus=False, pos='full',
                 separate_qk=False, static_query=False):
        super().__init__()
        self.h, self.dh, self.steps, self.dt = heads, d // heads, steps, dt
        self.m = self.dh // 2 if separate_qk else self.dh   # oscillators per head
        self.value_consensus, self.separate_qk = value_consensus, separate_qk
        self.static_query = static_query or separate_qk     # separate q/k phases are static by design
        if separate_qk:
            self.phase_q, self.phase_k = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        else:
            self.phase = nn.Linear(d, 2 * d, bias=False)    # -> (cos part, sin part) for d phases
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.log_beta = nn.Parameter(torch.full((heads, 1, 1), math.log(4.0)))
        self.log_kappa = nn.Parameter(torch.zeros(heads, 1, 1))
        if value_consensus:
            self.log_kappa_v = nn.Parameter(torch.zeros(heads, 1, 1))
        self.w = nn.Parameter(torch.zeros(heads, 1, self.m))      # per-oscillator coupling strength
        self.lag = nn.Parameter(torch.zeros(heads, 1, self.m), requires_grad=learn_lag)   # phase lag α
        freq = 10000 ** (-torch.arange(self.m, dtype=torch.float) / self.m)
        if pos == 'low':                                      # positional phase only on the slow half
            freq = freq * (torch.arange(self.m) >= self.m // 2).float()
        elif pos == 'none':
            freq = torch.zeros(self.m)
        self.register_buffer('freq', freq, persistent=False)

    def phases(self, x):
        """Content phases (optionally separate for query/key roles) plus the positional phase ramp."""
        B, L, _ = x.shape
        pos = torch.arange(L, device=x.device)[:, None] * self.freq[None]          # (L, m)
        if self.separate_qk:
            cq, sq = self.phase_q(x).view(B, L, self.h, 2, self.m).permute(3, 0, 2, 1, 4)
            ck, sk = self.phase_k(x).view(B, L, self.h, 2, self.m).permute(3, 0, 2, 1, 4)
            return torch.atan2(sq, cq) + pos, torch.atan2(sk, ck) + pos
        c, s = self.phase(x).view(B, L, 2, self.h, self.m).permute(2, 0, 3, 1, 4)  # (B, H, L, m)
        th = torch.atan2(s, c) + pos
        return th, th

    def attention(self, th_q, th_k, mask):
        """A_ij = softmax_j(β Σ_k w_k cos(θ^q_ik − θ^k_jk − α_k)): i couples most to j lagging it by α."""
        w = F.softmax(self.w, -1)                           # weights over oscillators, sum to 1
        q = th_q - self.lag                                 # receiver phase, shifted by the lag
        S = (q.cos() * w) @ th_k.cos().transpose(-1, -2) + (q.sin() * w) @ th_k.sin().transpose(-1, -2)
        return (self.log_beta.exp() * S).masked_fill(~mask, float('-inf')).softmax(-1), q

    def forward(self, x, return_attn=False, return_phases=False):
        B, L, d = x.shape
        th_q, th_k = self.phases(x)
        mask = causal_mask(L, x.device)
        kappa = self.log_kappa.exp()[None]
        vv = self.v(x).view(B, L, self.h, self.dh).transpose(1, 2)          # (B, H, L, dh)
        if self.value_consensus:
            kappa_v = self.log_kappa_v.exp()[None]
        phases = [th_k]
        for _ in range(self.steps):
            A, _ = self.attention(th_q, th_k, mask)
            rc, rs = A @ th_k.cos(), A @ th_k.sin()     # mean field  r_i = Σ_j A_ij e^{iθ_j}
            # Kuramoto–Sakaguchi: dθ_i = κ Σ_j A_ij sin(θ_j + α − θ_i); settles at θ_i = θ_j + α
            qk = th_k - self.lag                        # receiver side of the dynamics
            th_k = th_k + self.dt * kappa * (rs * qk.cos() - rc * qk.sin())
            if not self.static_query:
                th_q = th_k                             # the receiver tracks its evolved phases
            if self.value_consensus:
                vv = vv + self.dt * kappa_v * (A @ vv - vv)   # values spread along the same graph
            phases.append(th_k)
        A, _ = self.attention(th_q, th_k, mask)
        y = self.out((A @ vv).transpose(1, 2).reshape(B, L, d))
        extra = []
        if return_attn: extra.append(A)
        if return_phases: extra.append(torch.stack(phases))
        return (y, *extra) if extra else y


class Block(nn.Module):
    def __init__(self, d, heads, kind, steps, **osc):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = SoftmaxAttention(d, heads) if kind == 'softmax' else OscAttention(d, heads, steps, **osc)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class Model(nn.Module):
    """Embedding -> n_layers blocks (repeated `loops` times) -> answer read from the last (query) position."""

    def __init__(self, vocab, d=64, heads=4, n_layers=1, kind='softmax', steps=8, loops=1, **osc):
        super().__init__()
        self.loops = loops                                  # >1: re-apply the blocks with shared weights
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, heads, kind, steps, **osc) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, tokens):
        x = self.emb(tokens)
        for _ in range(self.loops):
            for b in self.blocks:
                x = b(x)
        return self.head(self.ln(x[:, -1]))


def n_params(model):
    return sum(p.numel() for p in model.parameters())
