"""Masked-HILC inputs: common beam before masking, mask-weighted monopole removed."""
from __future__ import annotations

import sys
from pathlib import Path

import healpy as hp
import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import build_hilc_q5prep_inputs as prep  # noqa: E402
from hilc_prescriptions import DEPROJ_CIB_DBETA_CMB  # noqa: E402


def test_prep_makes_masked_channels_identical(monkeypatch):
    nside, lmax = 64, 128
    monkeypatch.setattr(prep, "LMAX", lmax)
    monkeypatch.setattr(prep, "FWHM_COMMON", 240.0)
    ell = np.arange(lmax + 1)
    np.random.seed(1)
    alm = hp.synalm(1.0 / (ell + 10.0), lmax=lmax)
    fwhm = (90.0, 180.0)
    maps = [5.0 + hp.alm2map(hp.almxfl(alm, hp.gauss_beam(np.radians(f / 60), lmax=lmax)), nside)
            for f in fwhm]
    mask = np.ones(hp.nside2npix(nside))
    for i, vec in enumerate(np.random.default_rng(0).normal(size=(30, 3))):
        mask[hp.query_disc(nside, vec / np.linalg.norm(vec), np.radians(6.0))] = 0.0

    def masked_alm(m):
        return hp.map2alm(mask * m, lmax=lmax, iter=3)

    out = [prep.to_common_beam_monosub(m, f, mask) for m, f in zip(maps, fwhm)]
    assert abs(np.sum(mask * out[0]) / np.sum(mask)) < 1e-12
    b10 = hp.gauss_beam(np.radians(240.0 / 60), lmax=lmax)
    old = [hp.almxfl(masked_alm(m), b10 / hp.gauss_beam(np.radians(f / 60), lmax=lmax))
           for m, f in zip(maps, fwhm)]
    new = [masked_alm(m) for m in out]
    d_old = hp.alm2cl(old[0] - old[1]).sum() / hp.alm2cl(old[0]).sum()
    d_new = hp.alm2cl(new[0] - new[1]).sum() / hp.alm2cl(new[0]).sum()
    assert d_old > 1e-6
    assert d_new < 1e-4 * d_old


def test_masked_yaml_reads_q5prep_at_common_beam(tmp_path, monkeypatch):
    import hilc_prescriptions as hpmod

    monkeypatch.setattr(hpmod, "REPO", tmp_path)
    t = hpmod.write_hilc_yaml("L1_m9", masked=True, real=2, deproj=DEPROJ_CIB_DBETA_CMB).read_text()
    assert "inputs_nside2048_homog_r2_q5prep/" in t
    assert "beam_FWHM_arcmin: [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]" in t
    assert "wavelet_beam_criterion: 1.0e-6" in t
    full = hpmod.write_hilc_yaml("L1_m9", masked=False, real=2).read_text()
    assert "q5prep" not in full and "wavelet_beam_criterion" not in full
    assert "[9.66, 7.22, 4.92, 4.9, 4.67, 4.22]" in full
