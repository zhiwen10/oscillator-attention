"""Coupled-oscillator attention on a spatiotemporally continuous binding task.

Each observation (position + velocity of some object at some frame) is one Kuramoto oscillator.
Coupling encodes spatiotemporal continuity: token j is coupled to token i if j lies where i's motion
predicts it should be a few frames later/earlier. After integrating the dynamics, the attention/binding
matrix is read directly from phase differences: A_ij = (1 + cos(theta_i - theta_j)) / 2.

The Kuramoto velocity is ported from Un-0 (MIT licence, (c) 2026 Unconventional, Inc.):
    un0/model.py::_kuramoto_velocity  dθ/dt = ω + cos θ·(sin θ Kᵀ) − sin θ·(cos θ Kᵀ)
"""
import numpy as np


def kuramoto_velocity(theta, omega, K, alpha=None):
    """Un-0's vectorised Kuramoto term, sum_j K_ij sin(theta_j - theta_i), optionally with a
    Sakaguchi phase lag alpha_ij (sum_j K_ij sin(theta_j - theta_i - alpha_ij))."""
    s, c = np.sin(theta), np.cos(theta)
    if alpha is None:
        return omega + c * (K @ s) - s * (K @ c)
    ca, sa = K * np.cos(alpha), K * np.sin(alpha)
    # sin(tj - ti - a) = sin(tj - ti)cos a - cos(tj - ti) sin a
    return omega + (c * (ca @ s) - s * (ca @ c)) - (c * (sa @ c) + s * (sa @ s))


def make_task(n_obj=3, T=24, seed=0, noise=0.006, speed=0.035):
    """Smooth, crossing 2-D trajectories. Returns token features and object labels (shuffled)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    P, V, L, F = [], [], [], []
    for o in range(n_obj):
        a = rng.uniform(0, 2 * np.pi)
        w = rng.uniform(0.10, 0.22) * rng.choice([-1, 1])
        heading = a + w * t                                  # smoothly turning heading
        vel = speed * np.stack([np.cos(heading), np.sin(heading)], 1)
        start = np.array([0.5, 0.5]) + 0.22 * np.array([np.cos(2 * np.pi * o / n_obj), np.sin(2 * np.pi * o / n_obj)])
        pos = start + np.cumsum(vel, 0) - vel[0]
        P.append(pos + noise * rng.standard_normal(pos.shape)); V.append(vel + noise * rng.standard_normal(vel.shape))
        L.append(np.full(T, o)); F.append(t)
    P, V, L, F = map(np.concatenate, (P, V, L, F))
    perm = rng.permutation(len(L))
    return P[perm], V[perm], L[perm], F[perm]


def continuity_kernel(P, V, F, sigma=0.03, window=2):
    """K_ij high when token j sits where token i's motion predicts, within +-window frames."""
    dt = F[None, :] - F[:, None]
    pred = P[:, None, :] + V[:, None, :] * dt[..., None]
    d2 = ((pred - P[None, :, :]) ** 2).sum(-1)
    vsim = (V[:, None, :] * V[None, :, :]).sum(-1) / (np.linalg.norm(V, axis=1)[:, None] * np.linalg.norm(V, axis=1)[None, :])
    K = np.exp(-d2 / (2 * sigma ** 2)) * np.clip(vsim, 0, 1) ** 4
    K[(np.abs(dt) == 0) | (np.abs(dt) > window)] = 0
    return np.maximum(K, K.T)


def run_oscillators(K, steps=600, dt=0.05, gain=4.0, repel=0.08, seed=1, alpha_dir=None, F=None, record=()):
    """Integrate Kuramoto dynamics (Euler, as in Un-0 training). Global weak repulsion pushes unbound
    groups apart. alpha_dir: Sakaguchi lag applied along time (makes a travelling wave inside each object)."""
    rng = np.random.default_rng(seed)
    n = K.shape[0]
    Kc = gain * K / K.sum(1, keepdims=True).clip(1e-6) - repel / n
    np.fill_diagonal(Kc, 0)
    alpha = None
    if alpha_dir is not None:
        alpha = alpha_dir * np.sign(F[None, :] - F[:, None])
    theta = rng.uniform(0, 2 * np.pi, n)
    omega = np.zeros(n)
    snaps = {}
    for s in range(steps + 1):
        if s in record:
            snaps[s] = theta.copy()
        theta = theta + dt * kuramoto_velocity(theta, omega, Kc, alpha)
    return theta % (2 * np.pi), snaps


def phase_attention(theta):
    return 0.5 * (1 + np.cos(theta[:, None] - theta[None, :]))


def pairwise_binding_accuracy(same_pred, L):
    """Balanced accuracy over all token pairs: are two tokens correctly judged same/different object?"""
    gt = L[:, None] == L[None, :]
    iu = np.triu_indices(len(L), 1)
    g, p = gt[iu], same_pred[iu]
    return 0.5 * ((p & g).sum() / g.sum() + (~p & ~g).sum() / (~g).sum())


def stacked_local_attention_accuracy(K, L, layers, thr=0.05):
    """k layers of local attention = information can travel k hops along the continuity graph."""
    A = K / K.sum(1, keepdims=True).clip(1e-9)
    A = (A > thr).astype(float) + np.eye(len(L))
    R = np.eye(len(L))
    out = []
    for k in range(1, layers + 1):
        R = np.clip(R @ A, 0, 1)
        out.append(pairwise_binding_accuracy(R > 0, L))
    return out


if __name__ == '__main__':
    P, V, L, F = make_task()
    K = continuity_kernel(P, V, F)
    theta, snaps = run_oscillators(K, record=(0, 50, 150, 300, 600))
    for s, th in snaps.items():
        print('step', s, 'acc', round(pairwise_binding_accuracy(phase_attention(th) > 0.75, L), 3))
    print('stacked local attention', [round(a, 3) for a in stacked_local_attention_accuracy(K, L, 14)])
    # travelling-wave variant
    th2, _ = run_oscillators(K, alpha_dir=0.12, F=F)
    for o in range(3):
        m = L == o
        ph = np.unwrap(th2[m][np.argsort(F[m])])
        print('obj', o, 'phase-vs-time corr', round(np.corrcoef(np.sort(F[m]), ph)[0, 1], 3), 'span', round(np.ptp(ph), 2))
