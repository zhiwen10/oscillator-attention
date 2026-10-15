"""Oscillator dynamics, using Un-0's Kuramoto update.

Every token is one phase oscillator. We integrate

    dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j − θ_i)

with Un-0's own vectorised implementation (`un0.model._kuramoto_velocity`, MIT licence). Two additions
of ours:

* normalised coupling plus a weak global repulsion, so separate objects settle at different phases
  instead of all merging into one cluster;
* an optional phase lag α (the Kuramoto–Sakaguchi model): Σ_j K_ij sin(θ_j − θ_i − α_ij). With α_ij
  pointing along time, each bound object becomes a travelling wave. Un-0 has no lag term, so the lagged
  update is written out here; with α = 0 it is identical to Un-0's (checked in tests/).

Several independent phase sets can run at once ("heads"): they share the coupling but start from
different random phases, like the heads of multi-head attention.
"""
import torch
from un0.model import _kuramoto_velocity


def coupling_matrix(K, gain=4.0, repel=0.08):
    """Row-normalise K (each token's total pull = gain) and add a weak repulsion between all pairs."""
    n = K.shape[0]
    Kc = gain * K / K.sum(1, keepdim=True).clamp(min=1e-6) - repel / n
    Kc.fill_diagonal_(0.0)
    return Kc


def lagged_velocity(theta, omega, K, alpha):
    """Kuramoto–Sakaguchi: ω_i + Σ_j K_ij sin(θ_j − θ_i − α_ij), for theta of shape (heads, n)."""
    diff = theta[:, None, :] - theta[:, :, None] - alpha[None]          # [h, i, j] = θ_j − θ_i − α_ij
    return omega + (K[None] * torch.sin(diff)).sum(-1)


def simulate(K, steps=1200, dt=0.05, heads=1, seed=0, lag=0.0, frames=None, record=(), device='cpu'):
    """Integrate from random phases with Euler steps (as in Un-0 training). Returns final phases
    (heads, n) in [0, 2π) and a dict of recorded snapshots {step: phases}."""
    g = torch.Generator().manual_seed(seed)
    Kc = coupling_matrix(K).to(device)
    n = Kc.shape[0]
    theta = (2 * torch.pi * torch.rand(heads, n, generator=g)).to(device)
    omega = torch.zeros(1, n, device=device)
    alpha = None
    if lag:
        f = frames.to(device).float()
        alpha = lag * torch.sign(f[None, :] - f[:, None])               # lag points forward in time
    snaps = {}
    for s in range(steps + 1):
        if s in record:
            snaps[s] = theta.remainder(2 * torch.pi).cpu()
        v = _kuramoto_velocity(theta, omega, Kc) if alpha is None else lagged_velocity(theta, omega, Kc, alpha)
        theta = theta + dt * v
    return theta.remainder(2 * torch.pi).cpu(), snaps
