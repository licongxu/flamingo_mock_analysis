"""Tests for NPIPE noise handling and the prepared split maps."""

from __future__ import annotations

import numpy as np
import pytest

from flamingo_mock.ilc.noise import load_noise_map, noise_map_path
from flamingo_mock.ilc.paths import ILCPaths

PATHS = ILCPaths()


def test_noise_map_path_template():
    p = noise_map_path(PATHS.noise_dir, 100, "A", mc=200)
    assert p.name == "npipe6v20_noise_100_A_mc_00200.fits"
    assert p.parent.name == "A"
    assert p.parent.parent.name == "100GHz"
    with pytest.raises(ValueError):
        noise_map_path(PATHS.noise_dir, 100, "C")


def test_noise_rejects_mjysr_channels():
    with pytest.raises(ValueError, match="K_CMB"):
        load_noise_map(PATHS.noise_dir, 545, "A")


def test_npipe_splits_present_and_independent():
    """A/B detector-set splits: nonzero, weakly correlated noise."""
    try:
        n_a = load_noise_map(PATHS.noise_dir, 100, "A")
        n_b = load_noise_map(PATHS.noise_dir, 100, "B")
    except FileNotFoundError:
        pytest.skip("NPIPE noise maps not on disk")
    assert n_a.std() > 1e-6
    assert n_b.std() > 1e-6
    step = max(1, n_a.size // 500_000)
    c = float(np.corrcoef(n_a[::step], n_b[::step])[0, 1])
    assert abs(c) < 0.2


def test_noise_residual_order_of_magnitude_when_prepared():
    """map − signal should look like NPIPE noise, not zero (proves noise added)."""
    import healpy as hp

    sig = PATHS.signal_map(100)
    sk_a = PATHS.split_map(100, "A")
    sk_b = PATHS.split_map(100, "B")
    if not (sig.is_file() and sk_a.is_file() and sk_b.is_file()):
        pytest.skip("prepared inputs missing (flamingo-ilc prepare)")
    s = hp.read_map(str(sig), dtype=np.float64)
    m_a = hp.read_map(str(sk_a), dtype=np.float64)
    m_b = hp.read_map(str(sk_b), dtype=np.float64)
    n_a = m_a - s
    n_b = m_b - s
    assert n_a.std() > 1e-6
    assert n_b.std() > 1e-6
    # independent splits: residual correlation should be low
    step = max(1, n_a.size // 500_000)
    c = float(np.corrcoef(n_a[::step], n_b[::step])[0, 1])
    assert abs(c) < 0.2
    # noise should be a non-negligible fraction of sky variance at 100 GHz
    assert n_a.std() > 0.05 * m_a.std()
