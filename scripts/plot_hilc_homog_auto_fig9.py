"""Test McCarthy & Hill (2024) Fig. 9 claims on the homog HILC y autos.

One combined CMB+tSZ+CIB+noise map (r1).  Bandpower errors are the Gaussian
auto formula using the *total* measured C_ℓ^{yy}=C+N (their Eq. 55):

    Var(C_hat) = 2 C^2 / ((2ℓ+1) f_sky)

CIB / CMB / noise curves are HILC-weighted component autos of that same map.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from flamingo_mock.powerspectra import (
    compute_cl,
    dl_from_cl,
    ilc_bias_fraction,
    n_modes_tophat_hilc,
    sigma_dl_auto_binned,
)

ROOT = Path("/scratch/scratch-lxu/flamingo_mock_analysis")
_DIAG = ROOT / "scripts" / "plot_hilc_homog_r1xr2_split_diagnostics.py"
_spec = importlib.util.spec_from_file_location("hilc_r1xr2_diag", _DIAG)
diag = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["hilc_r1xr2_diag"] = diag
_spec.loader.exec_module(diag)

NSIDE = 2048
LMAX = 4096
FWHM_ILC = 10.0
FSKY = 1.0
N_FREQ = 6
HILC_BINSIZE = 50
ELL_PLOT_MAX = 3000
FREQS = diag.FREQS
NOISE_DIR = diag.NOISE_DIR

ELL_MIN = np.array(
    [9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494,
     642, 835, 1085, 1411, 1834, 2384],
    dtype=int,
)
ELL_MAX = np.array(
    [12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642,
     835, 1085, 1411, 1834, 2384, 3001],
    dtype=int,
)
ELL_EFF = np.array(
    [10.0, 13.5, 18.0, 23.5, 30.5, 40.0, 52.5, 68.5, 89.5, 117.0, 152.5,
     198.0, 257.5, 335.5, 436.5, 567.5, 738.0, 959.5, 1247.5, 1622.0, 2108.5, 2692.0],
)

ILC = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc")
FIG_DIR = Path("/scratch/scratch-lxu/flamingo_mock_analysis/figures")
CL_CACHE = ILC / "hilc_output_homog" / "hilc_homog_auto_fig9_cl.npz"
TRUTH = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/tsz/test"
    "/compton_y_nside4096.fits"
)
YLIM = (1.0e-17, 3.0e-9)

# Constrained-ILC deprojections (pyILC N_deproj).  Preserved component is always tSZ.
N_DEPROJ = {
    "none": 0,
    "cmb": 1,
    "cib": 1,
    "cib_cmb": 2,
    "dbeta": 2,
    "dbeta_cmb": 3,
    "moments": 3,
}


@dataclass(frozen=True)
class Case:
    key: str
    label: str
    ymap: Path
    wdir: Path
    wtag: str
    color: str
    marker: str
    ls: str
    offset: float


CASES = (
    Case(
        "none", "no deprojection",
        ILC / "hilc_output_homog"
        / "flamingo_needletILCmap_component_tSZ_hilc_y_homog_fullsky.fits",
        ILC / "hilc_output_homog", "",
        "C0", "o", "-", 1.0,
    ),
    Case(
        "cmb", "CMB",
        ILC / "hilc_output_homog_deproj_CMB"
        / "flamingo_needletILCmap_component_tSZ_deproject_CMB"
        "_hilc_y_homog_fullsky_deproj_CMB.fits",
        ILC / "hilc_output_homog_deproj_CMB", "_deproject_CMB",
        "C1", "s", "--", 1.04,
    ),
    Case(
        "cib", "CIB",
        ILC / "hilc_output_homog_deproj_CIB"
        / "flamingo_needletILCmap_component_tSZ_deproject_CIB"
        "_hilc_y_homog_fullsky_deproj_CIB.fits",
        ILC / "hilc_output_homog_deproj_CIB", "_deproject_CIB",
        "C2", "^", "-", 1.0,
    ),
    Case(
        "cib_cmb", "CIB + CMB",
        ILC / "hilc_output_homog_deproj_CIB_CMB"
        / "flamingo_needletILCmap_component_tSZ_deproject_CIB_CMB"
        "_hilc_y_homog_fullsky_deproj_CIB_CMB.fits",
        ILC / "hilc_output_homog_deproj_CIB_CMB", "_deproject_CIB_CMB",
        "C3", "v", "--", 1.04,
    ),
    Case(
        "dbeta", r"CIB + $\delta\beta$",
        ILC / "hilc_output_homog_deproj_CIB_CIB_dbeta"
        / "flamingo_needletILCmap_component_tSZ_deproject_CIB_CIB_dbeta"
        "_hilc_y_homog_fullsky_deproj_CIB_CIB_dbeta.fits",
        ILC / "hilc_output_homog_deproj_CIB_CIB_dbeta", "_deproject_CIB_CIB_dbeta",
        "C4", "D", "-", 1.0,
    ),
    Case(
        "dbeta_cmb", r"CIB + $\delta\beta$ + CMB",
        ILC / "hilc_output_homog_deproj_CIB_CIB_dbeta_CMB"
        / "flamingo_needletILCmap_component_tSZ_deproject_CIB_CIB_dbeta_CMB"
        "_hilc_y_homog_fullsky_deproj_CIB_CIB_dbeta_CMB.fits",
        ILC / "hilc_output_homog_deproj_CIB_CIB_dbeta_CMB",
        "_deproject_CIB_CIB_dbeta_CMB",
        "0.35", "P", "--", 1.04,
    ),
    Case(
        "moments", r"CIB + $\delta\beta$ + $\delta T$",
        ILC / "hilc_output_homog_deproj_CIB_CIB_dbeta_CIB_dT"
        / "flamingo_needletILCmap_component_tSZ_deproject_CIB_CIB_dbeta_CIB_dT"
        "_hilc_y_homog_fullsky_deproj_CIB_CIB_dbeta_CIB_dT.fits",
        ILC / "hilc_output_homog_deproj_CIB_CIB_dbeta_CIB_dT",
        "_deproject_CIB_CIB_dbeta_CIB_dT",
        "C5", "X", "-", 1.0,
    ),
)


def find_ymap(case: Case) -> Path:
    if case.ymap.is_file():
        return case.ymap
    matches = sorted(case.ymap.parent.glob("*needletILCmap*.fits"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(case.ymap)


def bin_mean_dl(cl: np.ndarray) -> np.ndarray:
    out = np.empty(len(ELL_MIN), dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(ELL_MIN, ELL_MAX)):
        ell = np.arange(lo, hi, dtype=np.float64)
        out[i] = float(np.nanmean(dl_from_cl(ell, cl[lo:hi])))
    return out


def deconv_auto(cl: np.ndarray, bl: np.ndarray, good: np.ndarray) -> np.ndarray:
    out = np.full_like(cl, np.nan, dtype=np.float64)
    out[good] = cl[good] / bl[good] ** 2
    return out


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


def _weighted_auto(alms: tuple[np.ndarray, ...], w: np.ndarray, noise: bool) -> np.ndarray:
    y = None
    same = len(alms) == 1
    for a in range(len(FREQS)):
        alm = alms[0] if same else alms[a]
        filt = diag._noise_filt(w[a], a) if noise else diag._signal_filt(w[a])
        c = hp.almxfl(alm, filt)
        y = c if y is None else y + c
    return hp.alm2cl(y)


def load_or_compute() -> dict[str, dict]:
    bl = hp.gauss_beam(np.deg2rad(FWHM_ILC / 60.0), lmax=LMAX)
    good = bl >= 1e-3
    stored: dict[str, np.ndarray] = {}
    if CL_CACHE.is_file():
        z = np.load(CL_CACHE)
        stored = {k: z[k] for k in z.files}

    dirty = False
    if "cl_tt" not in stored:
        print("anafast truth y ...", flush=True)
        yt = np.asarray(hp.read_map(str(TRUTH), field=0, dtype=np.float64))
        if hp.get_nside(yt) != NSIDE:
            yt = hp.ud_grade(yt, NSIDE)
        stored["cl_tt"] = compute_cl(yt, lmax=LMAX, deconv_pixel_window=False)
        dirty = True
    cl_tt = np.asarray(stored["cl_tt"], dtype=np.float64)

    out: dict[str, dict] = {}
    sig = None
    nalm = None
    for case in CASES:
        need = [f"yy_{case.key}", f"cib_{case.key}", f"cmb_{case.key}", f"n_{case.key}"]
        if any(k not in stored for k in need):
            if f"yy_{case.key}" not in stored:
                path = find_ymap(case)
                print(f"anafast {case.label}: {path.name}", flush=True)
                m = np.asarray(hp.read_map(str(path), field=0, dtype=np.float64))
                if hp.get_nside(m) != NSIDE:
                    m = hp.ud_grade(m, NSIDE)
                stored[f"yy_{case.key}"] = compute_cl(m, lmax=LMAX, deconv_pixel_window=False)
            if any(f"{p}_{case.key}" not in stored for p in ("cib", "cmb", "n")):
                print(f"  weighted residuals ({case.label}) ...", flush=True)
                w = diag.hilc_weights(case.wdir, case.wtag, LMAX)
                if sig is None:
                    sig = diag.signal_alms()
                if nalm is None:
                    nalm = noise_alms_r1()
                stored[f"cib_{case.key}"] = _weighted_auto(sig["cib"], w, noise=False)
                stored[f"cmb_{case.key}"] = _weighted_auto(sig["cmb"], w, noise=False)
                stored[f"n_{case.key}"] = _weighted_auto(nalm, w, noise=True)
            dirty = True

        pack = {}
        for name, raw in (
            ("yy", stored[f"yy_{case.key}"]),
            ("cib", stored[f"cib_{case.key}"]),
            ("cmb", stored[f"cmb_{case.key}"]),
            ("n", stored[f"n_{case.key}"]),
        ):
            cl = deconv_auto(raw, bl, good)
            pack[f"cl_{name}"] = cl
            pack[f"dl_{name}"] = bin_mean_dl(cl)
        pack["sig"] = sigma_dl_auto_binned(pack["cl_yy"], ELL_MIN, ELL_MAX, FSKY)
        pack["cl_tt"] = cl_tt
        out[case.key] = pack

    if dirty:
        CL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CL_CACHE, **stored)
        print("wrote", CL_CACHE)
    return out


def median_ratio(a: np.ndarray, b: np.ndarray, lo: float, hi: float) -> float:
    m = (ELL_EFF >= lo) & (ELL_EFF < hi) & np.isfinite(a) & np.isfinite(b) & (b > 0)
    return float(np.median(a[m] / b[m]))


def print_errorbar_check(data: dict[str, dict]) -> None:
    """σ/D is only N_modes; deproj grows absolute σ in proportion to C^{yy}."""
    print("\nError-bar check  (Gaussian auto, fsky=1)")
    print("  σ/D = sqrt(2 / Σ_ℓ(2ℓ+1)) in each Planck bin — independent of deprojection.")
    print(f"{'ell':>7}  {'σ/D none':>10}  {'σ_mom/σ_none':>13}  {'D_mom/D_none':>13}")
    none, mom = data["none"], data["moments"]
    for i, L in enumerate(ELL_EFF):
        if L not in (10.0, 52.5, 335.5, 959.5, 2108.5):
            continue
        print(
            f"{L:7.1f}  {none['sig'][i] / none['dl_yy'][i]:10.3f}  "
            f"{mom['sig'][i] / none['sig'][i]:13.3f}  "
            f"{mom['dl_yy'][i] / none['dl_yy'][i]:13.3f}"
        )


def ilc_bias_curves(cl_tt: np.ndarray, n_deproj: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (|ΔC^{yy}/C^{yy}|, |ΔD_ℓ^{yy}|) with ΔC/C = (1−N_ν+N_deproj)/N_eff."""
    n_modes = n_modes_tophat_hilc(cl_tt.size - 1, HILC_BINSIZE, FSKY)
    frac = ilc_bias_fraction(n_deproj, N_FREQ, n_modes)
    ells = np.arange(cl_tt.size, dtype=np.float64)
    d_bias = frac * np.abs(dl_from_cl(ells, cl_tt))
    return frac, d_bias


def print_ilc_bias(data: dict[str, dict]) -> None:
    cl_tt = data["none"]["cl_tt"]
    print("\nILC bias  ΔC^{yy}/C^{yy} = (1−N_ν+N_deproj)/N_eff")
    print("  HILC BinSize=50, N_ν=6, fsky=1; N_eff = Σ(2ℓ+1) in each harmonic bin.")
    print("  Bias is negative (signal under-estimate); table shows |ΔC/C|.")
    print(
        f"{'case':<28}  {'N_d':>3}  {'1-Nν+Nd':>8}  "
        f"{'|ΔC/C| ℓ=10':>12}  {'|ΔC/C| ℓ=335':>13}"
    )
    for case in CASES:
        nd = N_DEPROJ[case.key]
        frac, _ = ilc_bias_curves(cl_tt, nd)
        print(
            f"{case.label:<28}  {nd:3d}  {1 - N_FREQ + nd:8d}  "
            f"{frac[10]:12.2e}  {frac[336]:13.2e}"
        )


def print_claims(data: dict[str, dict]) -> None:
    none = data["none"]["dl_yy"]
    bands = (("low 10–100", 10, 100), ("mid 100–1000", 100, 1000), ("high 1000–3000", 1000, 3000))
    print("\nD_ℓ / D_none  (median in Planck bins)")
    print(f"{'case':<28}  {'low 10-100':>12}  {'mid 100-1000':>12}  {'high 1000-3000':>14}")
    for case in CASES:
        row = [case.label]
        for _, lo, hi in bands:
            row.append(f"{median_ratio(data[case.key]['dl_yy'], none, lo, hi):12.3f}")
        print(f"{row[0]:<28}  {row[1]}  {row[2]}  {row[3]}")

    cib = data["cib"]["dl_yy"]
    dbeta = data["dbeta"]["dl_yy"]
    dbeta_cmb = data["dbeta_cmb"]["dl_yy"]
    moments = data["moments"]["dl_yy"]
    cmb = data["cmb"]["dl_yy"]
    cib_cmb = data["cib_cmb"]["dl_yy"]

    print("\nFig. 9 claims vs this mock (HILC, 6 HFI including 857 GHz, full sky)")
    r_cmb = median_ratio(cmb, none, 10, 1000)
    r_cib = median_ratio(cib, none, 10, 1000)
    r_cib_cmb = median_ratio(cib_cmb, cib, 10, 1000)
    r_dbeta = median_ratio(dbeta, none, 10, 1000)
    r_dbeta_cmb = median_ratio(dbeta_cmb, dbeta, 10, 1000)
    r_dbeta_cmb_hi = median_ratio(dbeta_cmb, dbeta, 1000, 3000)
    r_mom_lo = median_ratio(moments, none, 10, 100)
    r_mom_hi_vs_dbeta = median_ratio(moments, dbeta, 1000, 3000)
    print(f"  CMB vs none (paper: almost free):                 {r_cmb:.3f}")
    print(f"  CIB vs none (paper: ~1.10):                       {r_cib:.3f}")
    print(f"  CIB+CMB vs CIB (paper: negligible):               {r_cib_cmb:.3f}")
    print(f"  CIB+δβ vs none (paper Fig.9 ~5, Fig.10 smaller):  {r_dbeta:.3f}")
    print(f"  CIB+δβ+CMB vs CIB+δβ at ℓ<1000:                   {r_dbeta_cmb:.3f}")
    print(f"  CIB+δβ+CMB vs CIB+δβ at ℓ>1000:                   {r_dbeta_cmb_hi:.3f}")
    print(f"  δβ+δT vs none at ℓ<100 (paper: orders of mag.):   {r_mom_lo:.3f}")
    print(f"  δβ+δT vs CIB+δβ at ℓ>1000 (paper: decreases):     {r_mom_hi_vs_dbeta:.3f}")
    print_errorbar_check(data)
    print_ilc_bias(data)


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


def plot_fig9(data: dict[str, dict]) -> Path:
    n = len(CASES)
    fig, axes = plt.subplots(n, 1, figsize=(8.6, 3.15 * n), sharex=True)
    ells = np.arange(LMAX + 1, dtype=np.float64)
    sl = slice(2, ELL_PLOT_MAX + 1)
    for ax, case in zip(axes, CASES):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ELL_EFF[0], ELL_PLOT_MAX)
        ax.set_ylim(*YLIM)
        d = data[case.key]
        ax.plot(
            ells[sl], dl_from_cl(ells, d["cl_tt"])[sl],
            color="k", lw=1.7, label=r"input truth $y$", zorder=3,
        )
        _, d_bias = ilc_bias_curves(d["cl_tt"], N_DEPROJ[case.key])
        ax.plot(
            ells[sl], d_bias[sl],
            color="crimson", lw=1.6, ls="--", zorder=5,
            label=r"$|\Delta C_\ell^{yy}|$",
        )
        ax.plot(
            ells[sl], np.abs(dl_from_cl(ells, d["cl_yy"]))[sl],
            color="C1", lw=1.8, alpha=0.45, zorder=2,
        )
        y = np.abs(d["dl_yy"])
        sig = np.asarray(d["sig"], dtype=np.float64)
        lo = np.maximum(y - sig, y * 1.0e-3)
        hi = y + sig
        ax.fill_between(ELL_EFF, lo, hi, color="C1", alpha=0.28, zorder=1, lw=0)
        ax.errorbar(
            ELL_EFF, y, yerr=[y - lo, sig],
            fmt="o", color="C1", ms=6.5, mfc="C1", mec="C1",
            elinewidth=2.2, capsize=4.5, capthick=1.8, ecolor="C1",
            zorder=6, label=r"HILC $y$ auto",
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
        nd = N_DEPROJ[case.key]
        ax.set_title(
            case.label
            + rf"  ($N_{{\mathrm{{deproj}}}}={nd}$, "
            + rf"$\Delta C/C=(1-{N_FREQ}+{nd})/N_{{\mathrm{{eff}}}}$)"
        )
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    axes[-1].set_xlabel(r"$\ell$")
    fig.suptitle(
        r"HILC $y$ auto and component residuals on one CMB+tSZ+CIB+noise map"
        "\n"
        r"lines: unbinned $D_\ell$; points: Planck 2015 XXII bins; "
        r"error bars on HILC $y$ only; "
        r"dashed crimson: $|\Delta C_\ell^{yy}|="
        r"|(1-N_\nu+N_{\mathrm{deproj}})/N_{\mathrm{eff}}|\,D_\ell^{yy,\mathrm{true}}$ "
        r"($N_{\mathrm{eff}}=\sum(2\ell+1)$ in each HILC $\Delta\ell=50$ bin)",
        y=1.01,
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / "hilc_homog_auto_fig9_deproj.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_or_compute()
    print_claims(data)
    plot_fig9(data)


if __name__ == "__main__":
    main()
