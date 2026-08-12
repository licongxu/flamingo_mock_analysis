"""Tests for NPIPE noise handling and the prepared split maps."""

from __future__ import annotations

import numpy as np
import pytest

from flamingo_mock.ilc.noise import load_noise_map, mjy_sr_to_K_cmb, noise_map_path
from flamingo_mock.ilc.paths import ILC_FREQUENCIES_GHZ, ILCPaths

PATHS = ILCPaths()


def test_noise_map_path_template():
    p = noise_map_path(PATHS.noise_dir, 100, "A", mc=200)
    assert p.name == "npipe6v20_noise_100_A_mc_00200.fits"
    assert p.parent.name == "A"
    assert p.parent.parent.name == "100GHz"
    with pytest.raises(ValueError):
        noise_map_path(PATHS.noise_dir, 100, "C")


def test_mjy_to_k_conversion_matches_known_planck_factors():
    # Planck 2015-ish thermodynamic conversion (noise_description.md §7).
    assert mjy_sr_to_K_cmb(545) == pytest.approx(0.01751, rel=1e-2)
    assert mjy_sr_to_K_cmb(857) == pytest.approx(0.6965, rel=1e-2)


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


def test_545_857_noise_converted_to_k_cmb():
    """545/857 load path must convert MJy/sr → K_CMB (not raw MJy)."""
    try:
        n545 = load_noise_map(PATHS.noise_dir, 545, "A")
        n100 = load_noise_map(PATHS.noise_dir, 100, "A")
    except FileNotFoundError:
        pytest.skip("NPIPE noise maps not on disk")
    # Converted 545 noise std should be O(1e-5–1e-4) K, not O(1e-3) MJy raw.
    assert 1e-6 < n545.std() < 5e-3
    # Same order of magnitude as 100 GHz K_CMB noise (not 50× larger from unit bug).
    assert n545.std() < 50 * n100.std()


def test_noise_residual_order_of_magnitude_when_prepared():
    """map − signal should look like NPIPE noise, not zero (proves noise added)."""
    import healpy as hp

    # Check all prepared ILC channels exist and carry nonzero noise residual.
    for freq in ILC_FREQUENCIES_GHZ:
        sig = PATHS.signal_map(freq)
        sk_a = PATHS.split_map(freq, "A")
        sk_b = PATHS.split_map(freq, "B")
        if not (sig.is_file() and sk_a.is_file() and sk_b.is_file()):
            pytest.skip("prepared inputs missing (flamingo-ilc prepare)")
        s = hp.read_map(str(sig), dtype=np.float64)
        m_a = hp.read_map(str(sk_a), dtype=np.float64)
        m_b = hp.read_map(str(sk_b), dtype=np.float64)
        n_a = m_a - s
        n_b = m_b - s
        assert n_a.std() > 1e-8, freq
        assert n_b.std() > 1e-8, freq
        assert n_a.std() > 0.01 * max(m_a.std(), 1e-12), freq
    # A/B independence: lower-HFI channels are the cleanest null test.
    for freq in (100, 143):
        s = hp.read_map(str(PATHS.signal_map(freq)), dtype=np.float64)
        n_a = hp.read_map(str(PATHS.split_map(freq, "A")), dtype=np.float64) - s
        n_b = hp.read_map(str(PATHS.split_map(freq, "B")), dtype=np.float64) - s
        step = max(1, n_a.size // 500_000)
        c = float(np.corrcoef(n_a[::step], n_b[::step])[0, 1])
        assert abs(c) < 0.25, (freq, c)
