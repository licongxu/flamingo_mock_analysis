"""Fig. 9 layout for the homog HILC y r1×r2 split-cross.

r1 and r2 share CMB+tSZ+CIB; only the homogeneous instrumental noise differs.
Bandpower errors are the Gaussian split-cross formula:

    Var(Ĉ_ℓ^{12}) = (C_ℓ^{11} C_ℓ^{22} + (C_ℓ^{12})²) / ((2ℓ+1) f_sky)

ILC bias uses the same leading formula as the auto of the shared signal:
    ΔC^{y1 y2}/C^{yy} = (1 − N_ν + N_deproj) / N_eff
Independent noise does not cancel the covariance–signal correlation in each split.

r2 products exist for five prescriptions (no CMB-only or CIB+CMB r2 HILC).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from flamingo_mock.powerspectra import dl_from_cl, ilc_bias_fraction, n_modes_tophat_hilc

ROOT = Path("/scratch/scratch-lxu/flamingo_mock_analysis")
_DIAG = ROOT / "scripts" / "plot_hilc_homog_r1xr2_split_diagnostics.py"
_spec = importlib.util.spec_from_file_location("hilc_r1xr2_diag", _DIAG)
diag = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["hilc_r1xr2_diag"] = diag
_spec.loader.exec_module(diag)

LMAX = diag.LMAX
FSKY = diag.FSKY
N_FREQ = 6
HILC_BINSIZE = 50
ELL_PLOT_MAX = diag.ELL_PLOT_MAX
ELL_EFF = diag.ELL_EFF
FIG_DIR = diag.FIG_DIR
YLIM = (1.0e-17, 3.0e-9)

N_DEPROJ = {
    "": 0,
    "_deproject_CIB": 1,
    "_deproject_CIB_CIB_dbeta": 2,
    "_deproject_CIB_CIB_dbeta_CMB": 3,
    "_deproject_CIB_CIB_dbeta_CIB_dT": 3,
}


def ilc_bias_curves(cl_tt: np.ndarray, n_deproj: int) -> tuple[np.ndarray, np.ndarray]:
    n_modes = n_modes_tophat_hilc(cl_tt.size - 1, HILC_BINSIZE, FSKY)
    frac = ilc_bias_fraction(n_deproj, N_FREQ, n_modes)
    ells = np.arange(cl_tt.size, dtype=np.float64)
    d_bias = frac * np.abs(dl_from_cl(ells, cl_tt))
    return frac, d_bias


def print_ilc_bias(cases_data) -> None:
    cl_tt = cases_data[0][1]["cl_tt"]
    print("\nILC bias  ΔC^{y1 y2}/C^{yy} = (1−N_ν+N_deproj)/N_eff")
    print("  HILC BinSize=50, N_ν=6, fsky=1; independent noise does not cancel this.")
    print(
        f"{'case':<28}  {'N_d':>3}  {'1-Nν+Nd':>8}  "
        f"{'|ΔC/C| ℓ=10':>12}  {'|ΔC/C| ℓ=335':>13}"
    )
    for case, _ in cases_data:
        nd = N_DEPROJ[case.wtag]
        frac, _ = ilc_bias_curves(cl_tt, nd)
        print(
            f"{case.label:<28}  {nd:3d}  {1 - N_FREQ + nd:8d}  "
            f"{frac[10]:12.2e}  {frac[336]:13.2e}"
        )


def print_sigma_check(cases_data) -> None:
    print("\nError-bar check  (Gaussian split-cross, fsky=1)")
    print(
        "  Var(Ĉ^{12}) = (C^{11} C^{22} + (C^{12})²) / ((2ℓ+1) f_sky)  "
        "then binned like the auto."
    )
    print(f"{'ell':>7}  {'σ/D none':>10}  {'σ_mom/σ_none':>13}  {'D_mom/D_none':>13}")
    none, mom = cases_data[0][1], cases_data[-1][1]
    for i, L in enumerate(ELL_EFF):
        if L not in (10.0, 52.5, 335.5, 959.5, 2108.5):
            continue
        print(
            f"{L:7.1f}  {none['dl_cross_sigma'][i] / abs(none['dl_cross'][i]):10.3f}  "
            f"{mom['dl_cross_sigma'][i] / none['dl_cross_sigma'][i]:13.3f}  "
            f"{mom['dl_cross'][i] / none['dl_cross'][i]:13.3f}"
        )


def _curve_and_bins(ax, ells, sl, cl, dl, *, color, marker, label, lw=1.3):
    ax.plot(
        ells[sl], np.abs(dl_from_cl(ells, cl))[sl],
        color=color, lw=lw, alpha=0.55, zorder=2,
    )
    mag = np.abs(dl)
    pos = np.asarray(dl) >= 0
    ax.plot(
        ELL_EFF[pos], mag[pos], marker, color=color, ms=5.0, ls="none",
        zorder=4, label=label,
    )
    if np.any(~pos):
        ax.plot(
            ELL_EFF[~pos], mag[~pos], marker, color=color, ms=5.5, ls="none",
            mfc="none", mew=1.5, zorder=4,
        )


def plot_fig9(cases_data) -> Path:
    n = len(cases_data)
    fig, axes = plt.subplots(n, 1, figsize=(8.6, 3.15 * n), sharex=True)
    ells = np.arange(LMAX + 1, dtype=np.float64)
    sl = slice(2, ELL_PLOT_MAX + 1)
    for ax, (case, d) in zip(axes, cases_data):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ELL_EFF[0], ELL_PLOT_MAX)
        ax.set_ylim(*YLIM)
        nd = N_DEPROJ[case.wtag]
        ax.plot(
            ells[sl], dl_from_cl(ells, d["cl_tt"])[sl],
            color="k", lw=1.7, label=r"input truth $y$", zorder=3,
        )
        _, d_bias = ilc_bias_curves(d["cl_tt"], nd)
        ax.plot(
            ells[sl], d_bias[sl],
            color="crimson", lw=1.6, ls="--", zorder=5,
            label=r"$|\Delta C_\ell^{y_1 y_2}|$",
        )
        ax.plot(
            ells[sl], np.abs(dl_from_cl(ells, d["cl_12_d"]))[sl],
            color="C1", lw=1.8, alpha=0.45, zorder=2,
        )
        y = np.abs(d["dl_cross"])
        sig = np.asarray(d["dl_cross_sigma"], dtype=np.float64)
        lo = np.maximum(y - sig, y * 1.0e-3)
        hi = y + sig
        ax.fill_between(ELL_EFF, lo, hi, color="C1", alpha=0.28, zorder=1, lw=0)
        ax.errorbar(
            ELL_EFF, y, yerr=[y - lo, sig],
            fmt="o", color="C1", ms=6.5, mfc="C1", mec="C1",
            elinewidth=2.2, capsize=4.5, capthick=1.8, ecolor="C1",
            zorder=6, label=r"HILC $y$ $r_1\times r_2$",
        )
        _curve_and_bins(
            ax, ells, sl, d["cl_cib_d"], d["dl_cib"],
            color="C2", marker="^", label=r"CIB residual",
        )
        _curve_and_bins(
            ax, ells, sl, d["cl_cmb_d"], d["dl_cmb"],
            color="C4", marker="v", label=r"CMB residual",
        )
        _curve_and_bins(
            ax, ells, sl, d["cl_n_d"], d["dl_n"],
            color="0.35", marker="+", label=r"noise residual", lw=1.0,
        )
        ax.set_ylabel(r"$D_\ell$")
        ax.set_title(
            case.label
            + rf"  ($N_{{\mathrm{{deproj}}}}={nd}$, "
            + rf"$\Delta C/C=(1-{N_FREQ}+{nd})/N_{{\mathrm{{eff}}}}$)"
        )
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    axes[-1].set_xlabel(r"$\ell$")
    fig.suptitle(
        r"HILC $y$ $r_1\times r_2$ (independent noise, shared CMB+tSZ+CIB)"
        "\n"
        r"lines: unbinned $D_\ell$; points: Planck 2015 XXII bins; "
        r"error bars on the split-cross only; "
        r"dashed crimson: $|\Delta C_\ell^{y_1 y_2}|="
        r"|(1-N_\nu+N_{\mathrm{deproj}})/N_{\mathrm{eff}}|\,D_\ell^{yy,\mathrm{true}}$",
        y=1.01,
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / "hilc_homog_r1xr2_fig9_deproj.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bl = diag.bl10()
    good = bl >= 1e-3
    cases_data = [(c, diag.load_or_compute(c, bl, good)) for c in diag.CASES]
    print_sigma_check(cases_data)
    print_ilc_bias(cases_data)
    plot_fig9(cases_data)


if __name__ == "__main__":
    main()
