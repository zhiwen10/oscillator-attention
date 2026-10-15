"""Coupling rule: which oscillators pull on each other.

Token j is coupled to token i when j sits where i's own motion predicts it should be |dt| frames later
(or earlier), and both move in a similar direction. Only nearby frames are coupled (|dt| <= window), so
every link is local in time; linking an object's first and last frame has to happen through the dynamics.
"""
import torch


def continuity_coupling(task, sigma=0.03, window=2, direction_power=4):
    dt = (task.frame[None, :] - task.frame[:, None]).float()                     # (n, n) frame difference
    predicted = task.pos[:, None, :] + task.vel[:, None, :] * dt[..., None]      # where i expects j
    miss = ((predicted - task.pos[None, :, :]) ** 2).sum(-1)                     # squared prediction error
    unit_v = task.vel / task.vel.norm(dim=1, keepdim=True)
    same_direction = (unit_v @ unit_v.T).clamp(min=0) ** direction_power
    K = torch.exp(-miss / (2 * sigma ** 2)) * same_direction
    K[(dt == 0) | (dt.abs() > window)] = 0
    return torch.maximum(K, K.T)                                                 # symmetric
