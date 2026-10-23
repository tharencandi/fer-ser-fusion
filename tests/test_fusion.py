"""Temperature scaling: always positive and at the validation cross-entropy optimum.

An unconstrained fit (T directly, L-BFGS without line search) can diverge to T < 0,
which would invert the face probabilities.
"""
import numpy as np
import pytest
import torch

from cv.fusion import TemperatureScaler, calibrated_sum_fusion


def _ce(z, y, T):
    return torch.nn.functional.cross_entropy(torch.tensor(z, dtype=torch.float64) / T, torch.tensor(y)).item()


def _margins(seed, n=240, C=7, signal=1.0):
    """XGBRF-like margins: small range, weakly informative."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, C, n)
    z = rng.uniform(-0.6, 0.6, (n, C))
    z[np.arange(n), y] += signal * rng.uniform(0, 3, n)
    return z.astype(np.float32), y


@pytest.mark.parametrize("seed,signal", [(0, 1.0), (1, 0.4), (2, 2.0), (3, 0.1), (4, 0.0)])
def test_temperature_positive_and_optimal(seed, signal):
    z, y = _margins(seed, signal=signal)
    T = TemperatureScaler().fit(z, y).temperature.item()
    assert T > 0
    grid = np.exp(np.linspace(np.log(0.01), np.log(100), 4000))
    best = min(_ce(z, y, g) for g in grid)
    assert _ce(z, y, T) <= best + 1e-6


def test_fusion_returns_positive_temperatures_and_probabilities():
    za, y = _margins(5, signal=2.0)
    zv, _ = _margins(6, signal=0.5)
    probs, ta, tv = calibrated_sum_fusion(za, zv, y, za, zv)
    assert ta > 0 and tv > 0
    assert np.allclose(probs.sum(1), 2.0, atol=1e-5)
