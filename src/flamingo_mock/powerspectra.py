"""Angular power-spectrum estimation for full-sky HEALPix maps.

Used to reproduce the Yang et al. (2026) power-spectrum figures from the
FLAMINGO integrated maps. All estimators subtract the monopole and deconvolve
the HEALPix pixel window by default.
"""

from __future__ import annotations

import healpy as hp
import numpy as np
from scipy.signal import savgol_filter


def pixel_window(nside: int, lmax: int) -> np.ndarray:
    """HEALPix temperature pixel window w_ell of length lmax+1 (zero-padded)."""
    wl = np.asarray(hp.pixwin(nside, lmax=None, pol=False), dtype=np.float64)
    if wl.size >= lmax + 1:
        return wl[: lmax + 1]
    pad = np.full(lmax + 1 - wl.size, wl[-1] if wl.size else 1.0)
    return np.concatenate([wl, pad])


def compute_cl(
    map_a: np.ndarray,
    map_b: np.ndarray | None = None,
    lmax: int | None = None,
    iter: int = 0,
    deconv_pixel_window: bool = True,
) -> np.ndarray:
    """Auto- or cross-spectrum with mean subtraction and pixel-window deconvolution.

    ``lmax`` defaults to 3*Nside - 1. Multipoles where the pixel window is
    numerically tiny are returned as NaN when deconvolving.
    """
    a = np.asarray(map_a, dtype=np.float64)
    a = a - np.mean(a[np.isfinite(a)])
    nside = hp.get_nside(a)
    if lmax is None:
        lmax = 3 * nside - 1

    if map_b is None:
        cl = hp.anafast(a, lmax=lmax, iter=iter)
    else:
        b = np.asarray(map_b, dtype=np.float64)
        b = b - np.mean(b[np.isfinite(b)])
        cl = hp.anafast(a, b, lmax=lmax, iter=iter)

    if deconv_pixel_window:
        wl = pixel_window(nside, lmax)
        good = wl > 1e-6
        cl_corr = np.full_like(cl, np.nan)
        cl_corr[good] = cl[good] / wl[good] ** 2
        cl = cl_corr
    return cl


def bin_cl(
    cl: np.ndarray, delta_ell: int = 7, lmin: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Bin C_ell into equal-width multipole bands (Yang et al. use Delta_ell=7)."""
    ell = np.arange(cl.size)
    n_bins = (cl.size - lmin) // delta_ell
    ell_b = np.empty(n_bins)
    cl_b = np.empty(n_bins)
    for i in range(n_bins):
        lo = lmin + i * delta_ell
        hi = lo + delta_ell
        sl = slice(lo, hi)
        w = np.isfinite(cl[sl])
        ell_b[i] = ell[sl][w].mean() if w.any() else np.nan
        cl_b[i] = cl[sl][w].mean() if w.any() else np.nan
    return ell_b, cl_b


def dl_from_cl(ell: np.ndarray, cl: np.ndarray) -> np.ndarray:
    """D_ell = ell(ell+1) C_ell / (2 pi)."""
    return ell * (ell + 1.0) * cl / (2.0 * np.pi)


def smooth_for_display(
    y: np.ndarray, window: int = 15, order: int = 3
) -> np.ndarray:
    """Savitzky-Golay smoothing used in Yang et al. for visualisation only."""
    if y.size < window:
        return y
    finite = np.isfinite(y)
    if np.all(y[finite] > 0):
        out = np.full_like(y, np.nan)
        out[finite] = np.exp(savgol_filter(np.log(y[finite]), window, order))
        return out
    return savgol_filter(y, window, order)


def decorrelation(
    cl_x: np.ndarray,
    cl_a: np.ndarray,
    cl_b: np.ndarray,
    ell_lo: int = 150,
    ell_hi: int = 1000,
) -> tuple[float, float]:
    """Mean and std of the cross-correlation coefficient over ell_lo < ell < ell_hi."""
    ell = np.arange(cl_x.size)
    m = (ell > ell_lo) & (ell < ell_hi) & np.isfinite(cl_x) & (cl_a > 0) & (cl_b > 0)
    r = cl_x[m] / np.sqrt(cl_a[m] * cl_b[m])
    return float(r.mean()), float(r.std())
