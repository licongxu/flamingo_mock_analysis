#!/usr/bin/env python3
"""L1_m9 prescriptions vs Yang et al. (2026) CIB crosses.

Four Yang26 L1_m9 lightcones (z<=3, 1 Gpc):
  L1_m9         fiducial hydro, D3A cosmology
  fgas-8sigma   feedback variant
  Mstar-1sigma  feedback variant
  LS8           cosmology variant (S8=0.766)

Paper Table 3 / Fig. 8 / Fig. 10 are L2p8 HYDRO_FIDUCIAL (2.8 Gpc, 8
lightcones). This script overlays each L1 prescription on that L2p8
lightcone0 measurement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from flamingo_mock.io import load_flamingo_map
from flamingo_mock.powerspectra import (
    bin_cl,
    decorrelation,
    dl_from_cl,
    pixel_window,
    smooth_for_display,
)
from pub_style import apply_pub_style, no_grid, savefig

COMP = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/components")
L1_ROOT = COMP / "L1_m9"
L2 = Path("/rds/flamingo/L2800N5040/HYDRO_FIDUCIAL/lightcone0_shells")
CACHE_DIR = COMP.parent / "spectra_yang26"
OLD_CACHE = CACHE_DIR / "l1_vs_l2p8_nside4096_lmax4000.npz"
FIG = ROOT / "figures"

CIB_FREQS = (217, 353, 545, 857)
LMAX = 4000
DELTA_ELL = 7
SG_WINDOW = 15
SG_ORDER = 3

# Folder name -> (cache prefix, legend)
PRESCRIPTIONS = {
    "L1_m9": ("fid_", r"L1\_m9 fiducial"),
    "fgas-8sigma": ("fgas_", r"fgas$-8\sigma$"),
    "Mstar-1sigma": ("Mstar_", r"Mstar$-1\sigma$"),
    "LS8": ("LS8_", r"LS8"),
}
RUN_COLOR = {
    "L1_m9": "#000000",
    "fgas-8sigma": "#D55E00",
    "Mstar-1sigma": "#0072B2",
    "LS8": "#009E73",
}
L2_COLOR = "#888888"

PAPER_R = {
    (857, 545): (0.959, 0.004),
    (857, 353): (0.895, 0.009),
    (857, 217): (0.841, 0.012),
    (545, 353): (0.983, 0.001),
    (545, 217): (0.956, 0.003),
    (353, 217): (0.993, 0.001),
}


def _paths_l1(run: str) -> dict[str, Path]:
    d = L1_ROOT / run
    out = {
        "y": d / "lensed_tSZ_rot_same_rot.hdf5",
        "kappa": d / "CMB_lensing_rot_same_rot.hdf5",
    }
    for nu in CIB_FREQS:
        out[f"cib{nu}"] = d / f"lensed_CIB_rot_BANDPASS_F{nu}_three_params_same_rot.hdf5"
    return out


def _paths_l2() -> dict[str, Path]:
    out = {
        "y": L2 / "lensed_tSZ_rot.fits",
        "kappa": L2 / "kappa_rot.fits",
    }
    for nu in CIB_FREQS:
        out[f"cib{nu}"] = L2 / f"lensed_CIB_rot_BANDPASS_F{nu}_three_params.fits"
    return out


def _alms(paths: dict[str, Path], lmax: int) -> dict[str, np.ndarray]:
    alms = {}
    for k, p in paths.items():
        print(f"  map2alm {k}  {p}", flush=True)
        x = load_flamingo_map(p)
        x = np.asarray(x, dtype=np.float64)
        x -= np.mean(x)
        alms[k] = hp.map2alm(x, lmax=lmax, iter=0)
        del x
    return alms


def _cl_from_alms(a, b=None, lmax=LMAX, nside=4096) -> np.ndarray:
    cl = hp.alm2cl(a, b, lmax=lmax) if b is not None else hp.alm2cl(a, lmax=lmax)
    wl = pixel_window(nside, lmax)
    good = wl > 1e-6
    out = np.full_like(cl, np.nan)
    out[good] = cl[good] / wl[good] ** 2
    return out


def spectra_from_alms(alms: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    cls: dict[str, np.ndarray] = {}
    cls[f"{prefix}tsz_auto"] = _cl_from_alms(alms["y"])
    cls[f"{prefix}kappa_auto"] = _cl_from_alms(alms["kappa"])
    for nu in CIB_FREQS:
        cls[f"{prefix}cib_auto_{nu}"] = _cl_from_alms(alms[f"cib{nu}"])
        cls[f"{prefix}cib_x_tsz_{nu}"] = _cl_from_alms(alms[f"cib{nu}"], alms["y"])
        cls[f"{prefix}cib_x_kappa_{nu}"] = _cl_from_alms(alms[f"cib{nu}"], alms["kappa"])
    for i, nu1 in enumerate(CIB_FREQS):
        for nu2 in CIB_FREQS[i + 1 :]:
            cls[f"{prefix}cib_x_{nu1}_{nu2}"] = _cl_from_alms(
                alms[f"cib{nu1}"], alms[f"cib{nu2}"]
            )
    return cls


def r_coeff(cl_x, cl_a, cl_b) -> tuple[float, float]:
    return decorrelation(cl_x, cl_a, cl_b, ell_lo=150, ell_hi=1000)


def _cache_run(run: str) -> Path:
    return CACHE_DIR / f"{run}_nside4096_lmax4000.npz"


def _strip_prefix(d: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    n = len(prefix)
    return {k[n:]: v for k, v in d.items() if k.startswith(prefix)}


def load_spectra() -> dict[str, np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cls: dict[str, np.ndarray] = {}

    l2_cache = _cache_run("L2p8")
    if l2_cache.is_file():
        z = np.load(l2_cache)
        cls.update({k: z[k] for k in z.files})
    elif OLD_CACHE.is_file():
        z = np.load(OLD_CACHE)
        cls.update({k: z[k] for k in z.files if k.startswith("l2_")})
        np.savez_compressed(l2_cache, **{k: cls[k] for k in cls if k.startswith("l2_")})
        print(f"wrote {l2_cache} from old cache", flush=True)
    else:
        print("L2p8 HYDRO_FIDUCIAL lightcone0 ...", flush=True)
        alms = _alms(_paths_l2(), LMAX)
        part = spectra_from_alms(alms, "l2_")
        del alms
        np.savez_compressed(l2_cache, **part)
        cls.update(part)

    for run, (prefix, _) in PRESCRIPTIONS.items():
        p = _cache_run(run)
        if p.is_file():
            z = np.load(p)
            cls.update({f"{prefix}{k}": z[k] for k in z.files})
            print(f"loading {p}", flush=True)
            continue
        if run == "L1_m9" and OLD_CACHE.is_file():
            z = np.load(OLD_CACHE)
            stripped = _strip_prefix({k: z[k] for k in z.files}, "l1_")
            np.savez_compressed(p, **stripped)
            cls.update({f"{prefix}{k}": v for k, v in stripped.items()})
            print(f"wrote {p} from old L1_m9 fiducial cache", flush=True)
            continue
        print(f"{run} maps ...", flush=True)
        alms = _alms(_paths_l1(run), LMAX)
        part = spectra_from_alms(alms, "")
        del alms
        np.savez_compressed(p, **part)
        cls.update({f"{prefix}{k}": v for k, v in part.items()})
        print(f"cached -> {p}", flush=True)
    return cls


def main() -> None:
    cls = load_spectra()
    runs = list(PRESCRIPTIONS)

    print("\n=== Table 3  CIB r_nu,nu'  (150 < ell < 1000) ===")
    hdr = f"{'pair':>10s}  {'paper L2p8':>12s}  {'L2p8 lc0':>12s}"
    for run, (_, lab) in PRESCRIPTIONS.items():
        hdr += f"  {run:>14s}"
    print(hdr)
    for (nu_hi, nu_lo), (p_m, p_s) in PAPER_R.items():
        key = f"cib_x_{min(nu_lo, nu_hi)}_{max(nu_lo, nu_hi)}"
        r2, s2 = r_coeff(cls[f"l2_{key}"], cls[f"l2_cib_auto_{nu_hi}"], cls[f"l2_cib_auto_{nu_lo}"])
        line = f"{nu_hi:4d}x{nu_lo:<4d}  {p_m:.3f}±{p_s:.3f}   {r2:.3f}±{s2:.3f}"
        for run, (prefix, _) in PRESCRIPTIONS.items():
            r, s = r_coeff(
                cls[f"{prefix}{key}"],
                cls[f"{prefix}cib_auto_{nu_hi}"],
                cls[f"{prefix}cib_auto_{nu_lo}"],
            )
            line += f"  {r:6.3f}±{s:.3f}  "
        print(line)

    print("\n=== r(CIB, y)  (150 < ell < 1000) ===")
    print(f"{'nu':>5s}  {'L2p8 lc0':>12s}" + "".join(f"  {run:>14s}" for run in runs))
    for nu in CIB_FREQS:
        ry2, sy2 = r_coeff(cls[f"l2_cib_x_tsz_{nu}"], cls[f"l2_cib_auto_{nu}"], cls["l2_tsz_auto"])
        line = f"{nu:5d}  {ry2:6.3f}±{sy2:.3f}"
        for run, (prefix, _) in PRESCRIPTIONS.items():
            r, s = r_coeff(
                cls[f"{prefix}cib_x_tsz_{nu}"],
                cls[f"{prefix}cib_auto_{nu}"],
                cls[f"{prefix}tsz_auto"],
            )
            line += f"  {r:6.3f}±{s:.3f}  "
        print(line)

    print("\n=== r(CIB, kappa)  (150 < ell < 1000) ===")
    print(f"{'nu':>5s}  {'L2p8 lc0':>12s}" + "".join(f"  {run:>14s}" for run in runs))
    for nu in CIB_FREQS:
        rk2, sk2 = r_coeff(
            cls[f"l2_cib_x_kappa_{nu}"], cls[f"l2_cib_auto_{nu}"], cls["l2_kappa_auto"]
        )
        line = f"{nu:5d}  {rk2:6.3f}±{sk2:.3f}"
        for run, (prefix, _) in PRESCRIPTIONS.items():
            r, s = r_coeff(
                cls[f"{prefix}cib_x_kappa_{nu}"],
                cls[f"{prefix}cib_auto_{nu}"],
                cls[f"{prefix}kappa_auto"],
            )
            line += f"  {r:6.3f}±{s:.3f}  "
        print(line)

    print("\n=== Auto D_ell / L2p8  (300 < ell < 1000) ===")
    ell = np.arange(cls["l2_tsz_auto"].size)
    m = (ell > 300) & (ell < 1000)
    d2y = dl_from_cl(ell, cls["l2_tsz_auto"])
    d2k = dl_from_cl(ell, cls["l2_kappa_auto"])
    print(f"{'field':>10s}" + "".join(f"  {run:>14s}" for run in runs))
    for label, l2cl, key in [
        ("tSZ y", d2y, "tsz_auto"),
        ("kappa", d2k, "kappa_auto"),
    ]:
        line = f"{label:>10s}"
        for run, (prefix, _) in PRESCRIPTIONS.items():
            d1 = dl_from_cl(ell, cls[f"{prefix}{key}"])
            line += f"  {np.nanmean(d1[m] / l2cl[m]):14.3f}"
        print(line)
    for nu in CIB_FREQS:
        d2 = dl_from_cl(ell, cls[f"l2_cib_auto_{nu}"])
        line = f"{'CIB '+str(nu):>10s}"
        for run, (prefix, _) in PRESCRIPTIONS.items():
            d1 = dl_from_cl(ell, cls[f"{prefix}cib_auto_{nu}"])
            line += f"  {np.nanmean(d1[m] / d2[m]):14.3f}"
        print(line)

    try:
        apply_pub_style()
    except Exception:
        mpl.rcParams.update({"text.usetex": False, "font.size": 11})

    # Fig. 10 at 353 GHz, all prescriptions
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True)
    nu = 353
    e2, c2y = bin_cl(cls[f"l2_cib_x_tsz_{nu}"], DELTA_ELL)
    e2k, c2k = bin_cl(cls[f"l2_cib_x_kappa_{nu}"], DELTA_ELL)
    axes[0].loglog(
        e2,
        smooth_for_display(np.abs(c2y) * 1e6, SG_WINDOW, SG_ORDER),
        color=L2_COLOR,
        lw=1.6,
        ls="--",
        label=r"L2p8 HYDRO\_FIDUCIAL",
    )
    axes[1].loglog(
        e2k,
        e2k * smooth_for_display(np.abs(c2k), SG_WINDOW, SG_ORDER),
        color=L2_COLOR,
        lw=1.6,
        ls="--",
    )
    for run, (prefix, lab) in PRESCRIPTIONS.items():
        ey, cy = bin_cl(cls[f"{prefix}cib_x_tsz_{nu}"], DELTA_ELL)
        ek, ck = bin_cl(cls[f"{prefix}cib_x_kappa_{nu}"], DELTA_ELL)
        axes[0].loglog(
            ey,
            smooth_for_display(np.abs(cy) * 1e6, SG_WINDOW, SG_ORDER),
            color=RUN_COLOR[run],
            lw=1.8,
            label=lab,
        )
        axes[1].loglog(
            ek,
            ek * smooth_for_display(np.abs(ck), SG_WINDOW, SG_ORDER),
            color=RUN_COLOR[run],
            lw=1.8,
        )
    axes[0].set_xlabel(r"Multipole $\ell$")
    axes[1].set_xlabel(r"Multipole $\ell$")
    axes[0].set_ylabel(r"$|C_\ell^{\mathrm{CIB}\times y}|\;[10^{-6}\,\mathrm{Jy}\,\mathrm{sr}^{-1}]$")
    axes[1].set_ylabel(r"$\ell\,|C_\ell^{\mathrm{CIB}\times\kappa}|\;[\mathrm{Jy}\,\mathrm{sr}^{-1}]$")
    axes[0].set_xlim(10, LMAX)
    axes[0].set_title(r"CIB--tSZ at $353\,\mathrm{GHz}$ (Yang26 Fig.~10 left)")
    axes[1].set_title(r"CIB--$\kappa$ at $353\,\mathrm{GHz}$ (Yang26 Fig.~10 right)")
    axes[0].legend(loc="upper right", fontsize=8)
    for ax in axes:
        no_grid(ax)
    fig.tight_layout()
    savefig(fig, "l1_prescriptions_vs_yang26_fig10_353", fig_dir=FIG)
    plt.close(fig)

    # CIB auto at 353 GHz
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    e2, c2 = bin_cl(cls["l2_cib_auto_353"], DELTA_ELL)
    ax.loglog(
        e2,
        smooth_for_display(dl_from_cl(e2, c2), SG_WINDOW, SG_ORDER),
        color=L2_COLOR,
        lw=1.6,
        ls="--",
        label=r"L2p8 HYDRO\_FIDUCIAL",
    )
    for run, (prefix, lab) in PRESCRIPTIONS.items():
        e, c = bin_cl(cls[f"{prefix}cib_auto_353"], DELTA_ELL)
        ax.loglog(
            e,
            smooth_for_display(dl_from_cl(e, c), SG_WINDOW, SG_ORDER),
            color=RUN_COLOR[run],
            lw=1.8,
            label=lab,
        )
    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{\mathrm{CIB}}/(2\pi)\;[\mathrm{Jy}\,\mathrm{sr}^{-1}]^{2}$")
    ax.set_xlim(200, LMAX)
    ax.set_title(r"CIB auto at $353\,\mathrm{GHz}$ (Yang26 Fig.~8)")
    ax.legend(fontsize=8, loc="lower left")
    no_grid(ax)
    fig.tight_layout()
    savefig(fig, "l1_prescriptions_vs_yang26_fig8_353", fig_dir=FIG)
    plt.close(fig)

    print("done.")


if __name__ == "__main__":
    main()
