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


def n_modes_tophat_hilc(
    lmax: int, bin_size: int, fsky: float = 1.0
) -> np.ndarray:
    """Modes used to estimate the HILC covariance at each ℓ.

    pyILC ``TopHatHarmonic`` with ``BinSize`` estimates one covariance per
    ℓ-bin.  McCarthy & Hill (2024) after Eq. (35): in a harmonic domain,
    N_modes = f_sky Σ_ℓ∈D (2ℓ+1).  The returned array is piecewise constant
    on those bins.
    """
    ell = np.arange(lmax + 1, dtype=np.float64)
    edges = np.arange(0, lmax + 1, bin_size)
    n_ell = np.empty(lmax + 1, dtype=np.float64)
    for i in range(len(edges) - 1):
        lo, hi = int(edges[i]), int(edges[i + 1])
        n_ell[lo:hi] = fsky * np.sum(2.0 * ell[lo:hi] + 1.0)
    if int(edges[-1]) <= lmax:
        lo = int(edges[-1])
        n_ell[lo:] = fsky * np.sum(2.0 * ell[lo:] + 1.0)
    return n_ell


def ilc_bias_fraction(
    n_deproj: int, n_freq: int, n_modes: np.ndarray
) -> np.ndarray:
    """Fractional ILC bias of the reconstructed auto, Delabrouille et al. (2009).

        ΔC_ℓ^{ss} / C_ℓ^{ss} = (1 − N_ν + N_deproj) / N_eff

    ``N_eff`` is the number of modes used to estimate the covariance (for
    full-sky HILC, Σ_ℓ∈bin (2ℓ+1)).  The bias is negative; this function
    returns the absolute value.  ``s`` is the preserved component (tSZ y).
    """
    return np.abs(1.0 - n_freq + n_deproj) / np.asarray(n_modes, dtype=np.float64)


def sigma_dl_cross_binned(
    cl_11: np.ndarray,
    cl_22: np.ndarray,
    cl_12: np.ndarray,
    ell_min: np.ndarray,
    ell_max: np.ndarray,
    fsky: float = 1.0,
) -> np.ndarray:
    """Gaussian error on binned mean D_l of a split cross-spectrum.

    Per multipole (noise in measured autos, N^{12}=0 in the cross mean):

        Var(C_hat_l^{12}) = (C_l^{11} C_l^{22} + (C_l^{12})^2) / ((2l+1) f_sky)

    For bandpower D_b = (1/N) sum_{l in bin} D_l with delta_{ll'} covariance,

        sigma(D_b) = sqrt( (1/N^2) sum_l [l(l+1)/(2pi)]^2 Var(C_hat_l^{12}) ).
    """
    ell = np.arange(cl_12.size, dtype=np.float64)
    fac = ell * (ell + 1.0) / (2.0 * np.pi)
    sig = np.empty(len(ell_min), dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(ell_min, ell_max)):
        m = (
            np.isfinite(cl_11[lo:hi])
            & np.isfinite(cl_22[lo:hi])
            & np.isfinite(cl_12[lo:hi])
        )
        if not m.any():
            sig[i] = np.nan
            continue
        els = ell[lo:hi][m]
        n = els.size
        var_c = (cl_11[lo:hi][m] * cl_22[lo:hi][m] + cl_12[lo:hi][m] ** 2) / (
            (2.0 * els + 1.0) * fsky
        )
        sig[i] = float(np.sqrt(np.sum((fac[lo:hi][m] ** 2) * var_c) / n**2))
    return sig


def sigma_dl_auto_binned(
    cl: np.ndarray,
    ell_min: np.ndarray,
    ell_max: np.ndarray,
    fsky: float = 1.0,
) -> np.ndarray:
    """Gaussian error on binned mean D_l of a full-sky auto-spectrum.

    This is *not* the split-cross formula.  McCarthy & Hill (2024) Eq. (55):

        Var(C_hat_l^{yy}) = 2 (C_l^{yy})^2 / ((2l+1) f_sky)

    ``cl`` must be the **total** auto of the map whose bandpowers are plotted
    (signal + residual foregrounds + noise).  Using the tSZ-only or split-cross
    C_l here understates the error, especially on noise-dominated scales.

    For bandpower D_b = (1/N) sum_{l in bin} D_l with delta_{ll'} covariance
    (same per-ℓ accumulation as ``sigma_dl_cross_binned``, no Knox-at-ℓ_eff
    shortcut):

        sigma(D_b) = sqrt( (1/N^2) sum_l [l(l+1)/(2pi)]^2 Var(C_hat_l) ).
    """
    cl = np.asarray(cl, dtype=np.float64)
    ell = np.arange(cl.size, dtype=np.float64)
    fac = ell * (ell + 1.0) / (2.0 * np.pi)
    sig = np.empty(len(ell_min), dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(ell_min, ell_max)):
        c = cl[lo:hi]
        m = np.isfinite(c)
        if not m.any():
            sig[i] = np.nan
            continue
        els = ell[lo:hi][m]
        n = els.size
        var_c = 2.0 * c[m] ** 2 / ((2.0 * els + 1.0) * fsky)
        sig[i] = float(np.sqrt(np.sum((fac[lo:hi][m] ** 2) * var_c) / n**2))
    return sig


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
