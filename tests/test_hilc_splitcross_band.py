"""Split-cross Gaussian is not Knox; q>5 trispectrum uses the ILC f_sky."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flamingo_mock.powerspectra import sigma_dl_cross_binned

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "cov_trispectrum_L1_m9_ilc_q5_metadata.json"
QFROMMAP_FSKY = 0.8595492784996496


def test_splitcross_not_knox_when_signal_equals_noise():
    """C11=C22=2 C12 ⇒ Var = 5 S²/N_modes, not Knox-on-cross 2 S²/N_modes."""
    s = 1.0
    cl_s = np.full(20, s)
    cl_auto = np.full(20, 2.0 * s)
    ell_min = np.array([5])
    ell_max = np.array([6])  # exclusive: ℓ=5 only
    sig = sigma_dl_cross_binned(cl_auto, cl_auto, cl_s, ell_min, ell_max, fsky=1.0)[0]
    ell = 5.0
    nmodes = 2.0 * ell + 1.0
    fac = ell * (ell + 1.0) / (2.0 * np.pi)
    want_split = fac * np.sqrt(5.0 * s**2 / nmodes)
    want_knox = fac * np.sqrt(2.0 * s**2 / nmodes)
    assert np.isclose(sig, want_split)
    assert not np.isclose(sig, want_knox)
    assert sig / want_knox == np.sqrt(5.0 / 2.0)


def test_q5_trispectrum_metadata_uses_ilc_fsky():
    assert META.is_file(), "run scripts/compute_ilc_q5_trispectrum.py"
    meta = json.loads(META.read_text())
    assert abs(meta["f_sky"] - QFROMMAP_FSKY) > 0.005
    assert 0.84 < meta["f_sky"] < 0.86
    assert meta["qfrommap_f_sky_not_used"] == QFROMMAP_FSKY
    cov = np.load(ROOT / "data" / "cov_trispectrum_L1_m9_ilc_q5_Dl_yy_binned_18.npy")
    assert cov.shape == (18, 18)
