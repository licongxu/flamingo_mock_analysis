"""HILC y auto-spectrum of one CMB+tSZ+CIB+noise map, by deprojection.

Not r1×r2.  The measured spectrum is the auto of the r1 HILC y-map.
tSZ / CMB / CIB / noise curves are that map's HILC-weighted components
(same 10' rebeam + taper as pyILC).

Bandpower errors are the Gaussian auto formula (McCarthy & Hill 2024 Eq. 55),
using the *total* measured auto C_ℓ^{yy} (signal + residuals + noise),
accumulated per ℓ then averaged — not the split-cross variance, and not
a Knox evaluation at ℓ_eff.
"""
from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
import sys

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from flamingo_mock.powerspectra import dl_from_cl, sigma_dl_auto_binned

ROOT = Path("/scratch/scratch-lxu/flamingo_mock_analysis")
_DIAG = ROOT / "scripts" / "plot_hilc_homog_r1xr2_split_diagnostics.py"
_spec = importlib.util.spec_from_file_location("hilc_r1xr2_diag", _DIAG)
diag = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["hilc_r1xr2_diag"] = diag
_spec.loader.exec_module(diag)

NSIDE = diag.NSIDE
LMAX = diag.LMAX
FREQS = diag.FREQS
FSKY = 1.0
ELL_MIN = diag.ELL_MIN
ELL_MAX = diag.ELL_MAX
ELL_EFF = diag.ELL_EFF
ELL_PLOT_MAX = diag.ELL_PLOT_MAX
NOISE_DIR = diag.NOISE_DIR
FIG_DIR = diag.FIG_DIR
YLIM = (3.0e-16, 3.0e-9)


def _auto_cache(case: diag.Case) -> Path:
    return case.wdir_r1 / "hilc_homog_auto_cl_unbinned.npz"


@lru_cache(maxsize=1)
def noise_alms_r1() -> tuple[np.ndarray, ...]:
    print("map2alm noise r1 ...", flush=True)
    alms = []
    for f in FREQS:
        m = diag.load_uk_to_k(
            NOISE_DIR / f"{f}GHz" / f"white_noise_{f}GHz_nside{NSIDE}_uK.fits"
        )
        alms.append(hp.map2alm(m, lmax=LMAX, iter=0))
        print(f"  noise {f} GHz", flush=True)
    return tuple(alms)


def y_alm_noise_r1(w: np.ndarray) -> np.ndarray:
    n = noise_alms_r1()
    y = None
    for a in range(len(FREQS)):
        c = hp.almxfl(n[a], diag._noise_filt(w[a], a))
        y = c if y is None else y + c
    return y


def bin_mean_dl(cl: np.ndarray) -> np.ndarray:
    out = np.empty(len(ELL_MIN), dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(ELL_MIN, ELL_MAX)):
        ell = np.arange(lo, hi, dtype=np.float64)
        out[i] = float(np.nanmean(dl_from_cl(ell, cl[lo:hi])))
    return out


def yerr_log(y: np.ndarray, sig: np.ndarray):
    y = np.asarray(y, dtype=np.float64)
    sig = np.asarray(sig, dtype=np.float64)
    mag = np.maximum(np.abs(y), 1e-40)
    lo = np.minimum(sig, mag * 0.99)
    return [lo, sig]


def load_or_compute(case: diag.Case, bl: np.ndarray, good: np.ndarray) -> dict:
    if not case.cl_cache.is_file():
        raise FileNotFoundError(
            f"need r1×r2 cache for signal C_ℓ: {case.cl_cache}"
        )
    z = np.load(case.cl_cache)
    cache = {k: z[k] for k in z.files}
    print("loaded", case.cl_cache)

    auto_path = _auto_cache(case)
    if auto_path.is_file():
        a = np.load(auto_path)
        cache.update({k: a[k] for k in a.files})
        print("loaded", auto_path)

    if "cl_n_auto" not in cache:
        if not case.ymap_r1.is_file():
            raise FileNotFoundError(case.ymap_r1)
        print(f"  noise auto y-alm ({case.label}) ...", flush=True)
        w1 = diag.hilc_weights(case.wdir_r1, case.wtag, LMAX)
        cache["cl_n_auto"] = hp.alm2cl(y_alm_noise_r1(w1))
        auto_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(auto_path, cl_n_auto=cache["cl_n_auto"])
        print("wrote", auto_path)

    cl_yy = diag.deconv_auto(cache["cl_11"], bl, good)
    cl_tsz = diag.deconv_auto(cache["cl_tsz_w"], bl, good)
    cl_cib = diag.deconv_auto(cache["cl_cib_w"], bl, good)
    cl_cmb = diag.deconv_auto(cache["cl_cmb_w"], bl, good)
    cl_n = diag.deconv_auto(cache["cl_n_auto"], bl, good)
    cl_tt = np.asarray(cache["cl_tt"], dtype=np.float64)

    pack = {
        "cl_yy": cl_yy,
        "cl_tsz": cl_tsz,
        "cl_cib": cl_cib,
        "cl_cmb": cl_cmb,
        "cl_n": cl_n,
        "cl_tt": cl_tt,
        "dl_yy": bin_mean_dl(cl_yy),
        "dl_tsz": bin_mean_dl(cl_tsz),
        "dl_cib": bin_mean_dl(cl_cib),
        "dl_cmb": bin_mean_dl(cl_cmb),
        "dl_n": bin_mean_dl(cl_n),
        "dl_tt": bin_mean_dl(cl_tt),
        "dl_yy_sigma": sigma_dl_auto_binned(cl_yy, ELL_MIN, ELL_MAX, FSKY),
    }

    print(f"\n{case.label}  (r1 auto, full sky)")
    print(
        f"{'ell':>7}  {'D_yy':>10}  {'sigma':>10}  {'frac':>7}  "
        f"{'D_tsz':>10}  {'D_cib':>10}  {'D_cmb':>10}  {'D_n':>10}"
    )
    for i, L in enumerate(ELL_EFF):
        yy, sig = pack["dl_yy"][i], pack["dl_yy_sigma"][i]
        print(
            f"{L:7.1f}  {yy:10.3e}  {sig:10.3e}  {sig / abs(yy):7.3f}  "
            f"{pack['dl_tsz'][i]:10.3e}  {pack['dl_cib'][i]:10.3e}  "
            f"{pack['dl_cmb'][i]:10.3e}  {pack['dl_n'][i]:10.3e}"
        )
    return pack


def _curve_and_bins(ax, ells, sl, cl, dl, *, color, marker, label, lw=1.3):
    mag_u = np.abs(dl_from_cl(ells, cl)[sl])
    ax.plot(ells[sl], mag_u, color=color, lw=lw, alpha=0.55, zorder=2)
    mag_b = np.abs(dl)
    pos = np.asarray(dl) >= 0
    ax.plot(
        ELL_EFF[pos], mag_b[pos], marker, color=color, ms=5.0, ls="none",
        zorder=4, label=label,
    )
    if np.any(~pos):
        ax.plot(
            ELL_EFF[~pos], mag_b[~pos], marker, color=color, ms=5.5, ls="none",
            mfc="none", mew=1.5, zorder=4,
        )


def plot_residuals(cases_data, ells, sl) -> Path:
    n = len(cases_data)
    fig, axes = plt.subplots(n, 1, figsize=(8.6, 3.35 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (case, d) in zip(axes, cases_data):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ELL_EFF[0], ELL_PLOT_MAX)
        ax.set_ylim(*YLIM)
        ax.plot(
            ells[sl], dl_from_cl(ells, d["cl_tt"])[sl],
            color="k", lw=1.7, label=r"input truth $y$", zorder=3,
        )
        ax.plot(
            ells[sl], np.abs(dl_from_cl(ells, d["cl_yy"]))[sl],
            color="C1", lw=1.8, alpha=0.45, zorder=2,
        )
        y = np.abs(d["dl_yy"])
        sig = np.asarray(d["dl_yy_sigma"], dtype=np.float64)
        lo, hi = yerr_log(d["dl_yy"], sig)
        ax.errorbar(
            ELL_EFF, y, yerr=[lo, hi],
            fmt="o", color="C1", ms=6.0, elinewidth=1.8, capsize=3.5, capthick=1.4,
            zorder=6, label=r"HILC $y$ auto (total)",
        )
        _curve_and_bins(
            ax, ells, sl, d["cl_cib"], d["dl_cib"],
            color="C2", marker="^", label=r"CIB residual",
        )
        _curve_and_bins(
            ax, ells, sl, d["cl_cmb"], d["dl_cmb"],
            color="C4", marker="v", label=r"CMB residual",
        )
        _curve_and_bins(
            ax, ells, sl, d["cl_n"], d["dl_n"],
            color="0.35", marker="+", label=r"noise residual", lw=1.0,
        )
        ax.set_ylabel(r"$D_\ell$")
        ax.set_title(case.label)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    axes[-1].set_xlabel(r"$\ell$")
    fig.suptitle(
        r"Full-sky HILC $y$ auto on one CMB+tSZ+CIB+noise map"
        "\n"
        r"lines: unbinned $D_\ell$; points: Planck 2015 XXII bins; "
        r"error bars on the total auto only "
        r"($\mathrm{Var}\,\hat C_\ell^{yy}=2(\hat C_\ell^{yy})^2/[(2\ell+1)f_{\mathrm{sky}}]$)",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / "hilc_homog_auto_residuals_all_prescriptions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def plot_deproj_overlay(cases_data, ells, sl) -> Path:
    """pyILC Fig. 9-style: total auto for each deprojection, with auto errors."""
    colors = ("C0", "C1", "C2", "C3", "C4")
    markers = ("o", "s", "^", "v", "D")
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(
        ells[sl], dl_from_cl(ells, cases_data[0][1]["cl_tt"])[sl],
        color="k", lw=1.8, label=r"input truth $y$", zorder=3,
    )
    n = len(cases_data)
    for i, ((case, d), c, m) in enumerate(zip(cases_data, colors, markers)):
        ax.plot(
            ells[sl], np.abs(dl_from_cl(ells, d["cl_yy"]))[sl],
            color=c, lw=0.9, alpha=0.45, zorder=2,
        )
        x = ELL_EFF * (1.025 ** (i - 0.5 * (n - 1)))
        ax.errorbar(
            x, np.abs(d["dl_yy"]),
            yerr=yerr_log(d["dl_yy"], d["dl_yy_sigma"]),
            fmt=m, color=c, ms=5.5, elinewidth=1.1, capsize=2.5, zorder=5,
            label=case.label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ELL_EFF[0], ELL_PLOT_MAX)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell$")
    ax.set_title(
        r"HILC $y$ auto vs deprojection (one combined map)"
        "\n"
        r"Gaussian auto errors from total $\hat C_\ell^{yy}=C+N$, not $C^{\mathrm{tSZ}}$"
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    out = FIG_DIR / "hilc_homog_auto_deproj_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bl = diag.bl10()
    good = bl >= 1e-3
    ells = np.arange(LMAX + 1, dtype=np.float64)
    sl = slice(2, ELL_PLOT_MAX + 1)

    cases_data = [(c, load_or_compute(c, bl, good)) for c in diag.CASES]
    plot_residuals(cases_data, ells, sl)


if __name__ == "__main__":
    main()
