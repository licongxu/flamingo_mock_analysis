"""Fiducial and hydro variants must share CMB seed 42 and homog noise r1/r2.

LS8 is allowed a different CMB. Do not trust scripts: compare on-disk maps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import healpy as hp

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from hilc_prescriptions import SYNTH, cmb_path, FREQS_ILC, total_maps_dir  # noqa: E402

HYDRO = ("L1_m9", "fgas-8sigma", "Mstar-1sigma")
NOISE = SYNTH / "planck_noise" / "homogeneous"
SEED_R1 = 42
SEED_R2 = 1042


def _header(path: Path):
    with fits.open(path) as hdul:
        return hdul[1].header if len(hdul) > 1 else hdul[0].header


def _map(path: Path) -> np.ndarray:
    return np.asarray(hp.read_map(str(path), dtype=np.float64))


def test_hydro_cmb_pixel_identical_seed42():
    fid = cmb_path("L1_m9")
    assert fid.is_file()
    hdr = _header(fid)
    if "SEED" in hdr:
        assert int(hdr["SEED"]) == SEED_R1
    m0 = _map(fid)
    for name in HYDRO[1:]:
        p = cmb_path(name)
        assert p.is_file(), name
        m = _map(p)
        assert m.shape == m0.shape, name
        assert np.array_equal(m, m0), name


def test_ls8_cmb_is_allowed_to_differ():
    p = cmb_path("LS8")
    if not p.is_file():
        return
    m_fid = _map(cmb_path("L1_m9"))
    m_ls8 = _map(p)
    assert m_ls8.shape == m_fid.shape
    assert not np.array_equal(m_ls8, m_fid)


def test_shared_homog_noise_r1_r2_seeds_and_pixels():
    for nu in FREQS_ILC:
        p1 = NOISE / f"{nu}GHz" / f"white_noise_{nu}GHz_nside2048_uK.fits"
        p2 = NOISE / f"{nu}GHz" / f"white_noise_{nu}GHz_nside2048_uK_r2.fits"
        assert p1.is_file() and p2.is_file(), nu
        h1, h2 = _header(p1), _header(p2)
        if "SEED" in h1:
            assert int(h1["SEED"]) == SEED_R1 + int(nu), nu
        if "SEED" in h2:
            assert int(h2["SEED"]) == SEED_R2 + int(nu), nu
        n1, n2 = _map(p1), _map(p2)
        assert n1.shape == n2.shape
        assert not np.array_equal(n1, n2), f"r1/r2 identical at {nu}"
        # Same files for every hydro run (not per-prescription copies).
        for name in HYDRO:
            tot = total_maps_dir(name) / f"sky_CMB_tSZ_CIB_homog_{nu}GHz_nside2048_uK.fits"
            tot2 = total_maps_dir(name) / f"sky_CMB_tSZ_CIB_homog_{nu}GHz_nside2048_uK_r2.fits"
            if tot.is_file() and tot2.is_file():
                # Totals are float32 FITS; 857 GHz CIB is large, so allow
                # rounding vs an independent noise realisation (rms ~ 10^4).
                recovered = _map(tot2) - (_map(tot) - n1)
                err = float(np.max(np.abs(recovered - n2)))
                rms = float(np.std(n2))
                assert err < 1e-3 * rms + 1.0, (name, nu, err, rms)
