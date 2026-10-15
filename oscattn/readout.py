"""Reading attention out of phases, scoring, and the stacked-attention baseline."""
import torch


def phase_attention(theta):
    """A_ij = (1 + cos(θ_i − θ_j)) / 2 for phases of shape (n,): 1 = in phase, 0 = opposite."""
    return 0.5 * (1 + torch.cos(theta[:, None] - theta[None, :]))


def same_object(theta_heads, threshold=0.75):
    """Two tokens are judged 'same object' if they are in phase in every head.
    threshold 0.75 means a phase difference under 60 degrees."""
    same = None
    for th in theta_heads:
        s = phase_attention(th) > threshold
        same = s if same is None else same & s
    return same


def binding_accuracy(same_pred, label):
    """Balanced accuracy over all token pairs (same-object pairs and different-object pairs count equally)."""
    truth = label[:, None] == label[None, :]
    iu = torch.triu_indices(len(label), len(label), 1)
    t, p = truth[iu[0], iu[1]], same_pred[iu[0], iu[1]]
    hit = (p & t).sum() / t.sum()
    correct_reject = (~p & ~t).sum() / (~t).sum()
    return float(0.5 * (hit + correct_reject))


def stacked_local_attention(K, label, max_layers=14, threshold=0.05):
    """Baseline: each attention layer can only mix a token with its coupled neighbours (the same local
    links the oscillators use), so after k layers information has travelled at most k links. Returns the
    binding accuracy after 1..max_layers layers, judging 'same object' = reachable within k layers."""
    A = K / K.sum(1, keepdim=True).clamp(min=1e-9)
    step = ((A > threshold).float() + torch.eye(len(label))).clamp(max=1)
    reach = torch.eye(len(label))
    out = []
    for _ in range(max_layers):
        reach = (reach @ step).clamp(max=1)
        out.append(binding_accuracy(reach > 0, label))
    return out
