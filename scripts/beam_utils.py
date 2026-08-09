"""Gaussian beam helpers for pyILC mock validation (beam-deconvolved C_ell)."""
from __future__ import annotations

import numpy as np


def gaussian_beam_bl(fwhm_arcmin: float, lmax: int) -> np.ndarray:
    """Return B_ell for a Gaussian beam, length lmax+1, B_0 = 1.

    Uses the standard healpy convention:
        sigma = FWHM_rad / sqrt(8 ln 2)
        B_ell = exp(-0.5 * ell*(ell+1) * sigma^2)
    """
    if fwhm_arcmin is None or fwhm_arcmin <= 0:
        return np.ones(lmax + 1, dtype=np.float64)
    fwhm_rad = float(fwhm_arcmin) * np.pi / (180.0 * 60.0)
    sigma = fwhm_rad / np.sqrt(8.0 * np.log(2.0))
    ell = np.arange(lmax + 1, dtype=np.float64)
    return np.exp(-0.5 * ell * (ell + 1.0) * sigma * sigma)


def deconvolve_cl_beam(
    cl: np.ndarray,
    fwhm_arcmin: float,
    *,
    bl_floor: float = 1e-6,
) -> np.ndarray:
    """Return C_ell / B_ell^2 with a floor on B_ell to avoid high-ell blow-up.

    Multipoles where B_ell < bl_floor are left NaN (unreliable deconvolution).
    """
    cl = np.asarray(cl, dtype=np.float64)
    lmax = len(cl) - 1
    bl = gaussian_beam_bl(fwhm_arcmin, lmax)
    out = np.full_like(cl, np.nan)
    good = bl >= bl_floor
    out[good] = cl[good] / (bl[good] ** 2)
    return out


def apply_beam_to_map(m: np.ndarray, fwhm_arcmin: float, lmax: int | None = None) -> np.ndarray:
    """Smooth a HEALPix map with a Gaussian beam of given FWHM (arcmin)."""
    import healpy as hp

    if fwhm_arcmin is None or fwhm_arcmin <= 0:
        return np.asarray(m, dtype=np.float64)
    nside = hp.npix2nside(len(m))
    if lmax is None:
        lmax = 3 * nside - 1
    alm = hp.map2alm(np.asarray(m, dtype=np.float64), lmax=lmax)
    bl = gaussian_beam_bl(fwhm_arcmin, lmax)
    alm = hp.almxfl(alm, bl)
    return hp.alm2map(alm, nside=nside, lmax=lmax)
