"""Periodic association: the stimulus->response mapping repeats with a rhythm.

A cycle of C distinct tokens is repeated R times, then one cycle token is appended as the query; the
target is its successor within the cycle (cyclically). The cycle is drawn fresh for every sequence, so
the mapping must be read in-context, and the (stimulus, response) pair always sits one step apart at
every phase of the rhythm:

    c0 c1 c2 c3 c4 | c0 c1 c2 c3 c4 | c0 c1 c2 c3 c4 | c3 -> answer c4

Chance = 1/C. Tests whether an oscillator's phase representation exploits periodic input structure.
C is fixed within a batch (variable lengths cannot stack); the experiment draws it per step.
"""
import torch


def make_rhythm_batch(batch, c=6, reps=3, vocab=64, generator=None, device='cpu'):
    """Returns tokens (batch, reps*c+1) and targets (batch,)."""
    g = generator
    cycle = torch.rand(batch, vocab, generator=g).argsort(dim=1)[:, :c]        # c distinct tokens per row
    seq = cycle.repeat(1, reps + 1)[:, :reps * c]                              # R repeats of the cycle
    qi = torch.randint(0, c, (batch,), generator=g)                            # queried cycle position
    rows = torch.arange(batch)
    tokens = torch.cat([seq, cycle[rows, qi][:, None]], dim=1)
    return tokens.to(device), cycle[rows, (qi + 1) % c].to(device)


def draw_c(g, cmin=5, cmax=8):
    """Cycle length for this training batch, from the data-stream generator (same for every model)."""
    return int(torch.randint(cmin, cmax + 1, (1,), generator=g))
