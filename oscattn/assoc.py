"""In-context associative recall: learn to associate within one sequence.

A sequence lists n stimulus->response pairs, then repeats one stimulus as a query:

    k3 v17  k9 v2  k14 v30 ...  k9   ->  answer v2

The pairings are drawn fresh for every sequence, so no fixed mapping can be memorised: the network has
to learn the mechanism "find where this stimulus appeared, report what followed it". This is the
standard associative-recall probe for attention (Ba et al. 2016; Olsson et al. 2022; Arora et al. 2024).
"""
import torch


def make_batch(batch, n_pairs=8, n_keys=32, n_vals=32, generator=None, device='cpu'):
    """Returns tokens (batch, 2*n_pairs+1) and targets (batch,).

    Token ids: keys 0..n_keys-1, values n_keys..n_keys+n_vals-1. Keys within a sequence are distinct.
    """
    g = generator
    keys = torch.rand(batch, n_keys, generator=g).argsort(dim=1)[:, :n_pairs]
    vals = torch.randint(0, n_vals, (batch, n_pairs), generator=g) + n_keys
    q = torch.randint(0, n_pairs, (batch,), generator=g)
    seq = torch.stack([keys, vals], dim=2).reshape(batch, 2 * n_pairs)
    rows = torch.arange(batch)
    tokens = torch.cat([seq, keys[rows, q][:, None]], dim=1)
    return tokens.to(device), vals[rows, q].to(device)
