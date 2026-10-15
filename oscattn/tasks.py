"""The binding task: several objects move along smooth, crossing paths.

Each observation ("token") is one object's position and velocity at one frame. Tokens are shuffled and
their object labels hidden; the network has to work out which tokens belong to the same object.
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Task:
    pos: torch.Tensor      # (n, 2) observed position
    vel: torch.Tensor      # (n, 2) observed velocity
    frame: torch.Tensor    # (n,)   frame index (time)
    label: torch.Tensor    # (n,)   hidden ground-truth object id (used only for scoring)


def make_task(n_objects=3, n_frames=24, seed=0, noise=0.006, speed=0.035, drop=0.0):
    """Smooth trajectories whose headings turn slowly, so paths curve and cross.

    drop: fraction of observations removed at random (simulates missed frames / occlusion).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames)
    pos, vel, frame, label = [], [], [], []
    for o in range(n_objects):
        heading0 = rng.uniform(0, 2 * np.pi)
        turn = rng.uniform(0.10, 0.22) * rng.choice([-1, 1])            # radians per frame
        heading = heading0 + turn * t
        v = speed * np.stack([np.cos(heading), np.sin(heading)], axis=1)
        angle = 2 * np.pi * o / n_objects                              # objects start spread on a circle
        start = np.array([0.5, 0.5]) + 0.22 * np.array([np.cos(angle), np.sin(angle)])
        p = start + np.cumsum(v, axis=0) - v[0]
        pos.append(p + noise * rng.standard_normal(p.shape))
        vel.append(v + noise * rng.standard_normal(v.shape))
        frame.append(t)
        label.append(np.full(n_frames, o))
    pos, vel, frame, label = (np.concatenate(a) for a in (pos, vel, frame, label))
    keep = rng.random(len(label)) >= drop
    order = rng.permutation(np.flatnonzero(keep))                     # shuffle: no ordering cue
    as_t = lambda a, dt: torch.as_tensor(a[order], dtype=dt)
    return Task(as_t(pos, torch.float32), as_t(vel, torch.float32), as_t(frame, torch.long), as_t(label, torch.long))
