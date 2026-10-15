"""Checks that our update rules match the textbook Kuramoto sum and Un-0's implementation."""
import torch
from un0.model import _kuramoto_velocity

from oscattn.dynamics import lagged_velocity


def naive(theta, omega, K, alpha):
    h, n = theta.shape
    out = omega.expand(h, n).clone()
    for b in range(h):
        for i in range(n):
            out[b, i] += sum(K[i, j] * torch.sin(theta[b, j] - theta[b, i] - alpha[i, j]) for j in range(n))
    return out


def test_un0_matches_textbook_kuramoto():
    torch.manual_seed(0)
    theta, omega, K = torch.rand(2, 6) * 6.28, torch.randn(1, 6), torch.randn(6, 6)
    assert torch.allclose(_kuramoto_velocity(theta, omega, K), naive(theta, omega, K, torch.zeros(6, 6)), atol=1e-5)


def test_lag_zero_equals_un0():
    torch.manual_seed(1)
    theta, omega, K = torch.rand(3, 7) * 6.28, torch.randn(1, 7), torch.randn(7, 7)
    assert torch.allclose(lagged_velocity(theta, omega, K, torch.zeros(7, 7)), _kuramoto_velocity(theta, omega, K), atol=1e-5)


def test_lagged_matches_textbook():
    torch.manual_seed(2)
    theta, omega, K, alpha = torch.rand(1, 5) * 6.28, torch.randn(1, 5), torch.randn(5, 5), torch.randn(5, 5)
    assert torch.allclose(lagged_velocity(theta, omega, K, alpha), naive(theta, omega, K, alpha), atol=1e-5)
