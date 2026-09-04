"""q>5 hole-masked HILC y r1×r2, Fig. 9 layout (residuals + ILC bias).

Prescriptions already on disk: no deprojection, CIB, CIB+δβ+δT.
Spectra use the C2 0.25° apodized cluster-hole mask.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flamingo_mock.powerspectra import (
    dl_from_cl,
    ilc_bias_fraction,
    n_modes_tophat_hilc,
    sigma_dl_cross_binned,
)

ROOT = Path(__file__).resolve().parents[1]
_DIAG = ROOT / "scripts" / "plot_hilc_homog_r1xr2_split_diagnostics.py"
_spec = importlib.util.spec_from_file_location("hilc_r1xr2_diag", _DIAG)
diag = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["hilc_r1xr2_diag"] = diag
_spec.loader.exec_module(diag)

sys.path.insert(0, str(ROOT / "scripts"))
from hilc_prescriptions import cache_stale, cluster_mask_apo  # noqa: E402

NSIDE = diag.NSIDE
LMAX = diag.LMAX
N_FREQ = 6
HILC_BINSIZE = 50
ELL_PLOT_MAX = diag.ELL_PLOT_MAX
ELL_MIN = diag.ELL_MIN
ELL_MAX = diag.ELL_MAX
ELL_EFF = diag.ELL_EFF
FIG_DIR = ROOT / "figures" / "hilc" / "l1_m9_deproj_suite"
ILC = diag.ILC
YLIM = (1.0e-17, 3.0e-9)
MASK_APO = cluster_mask_apo("L1_m9")

N_DEPROJ = {
    "": 0,
    "_deproject_CIB": 1,
    "_deproject_CIB_CIB_dbeta_CIB_dT": 3,
}


@dataclass(frozen=True)
class Case:
    label: str
    wtag: str
    ymap_r1: Path
    ymap_r2: Path
    wdir_r1: Path
    wdir_r2: Path
    cl_cache: Path
    res_cache: Path


CASES = (
    Case(
        "no deprojection",
        "",
        ILC / "hilc_output_homog_q5masked"
        / "flamingo_needletILCmap_component_tSZ_hilc_y_homog_q5masked.fits",
        ILC / "hilc_output_homog_q5masked_r2"
        / "flamingo_needletILCmap_component_tSZ_hilc_y_homog_q5masked_r2.fits",
        ILC / "hilc_output_homog_q5masked",
        ILC / "hilc_output_homog_q5masked_r2",
        ILC / "hilc_output_homog_q5masked_r2" / "hilc_homog_q5masked_r1xr2_cl_unbinned_l1_m9.npz",
        ILC / "hilc_output_homog_q5masked_r2" / "hilc_homog_q5masked_r1xr2_fig9_cl_l1_m9.npz",
    ),
    Case(
        "CIB deprojection",
        "_deproject_CIB",
        ILC / "hilc_output_homog_q5masked_deproj_CIB"
        / "flamingo_needletILCmap_component_tSZ_deproject_CIB_hilc_y_homog_q5masked_deproj_CIB.fits",
        ILC / "hilc_output_homog_q5masked_r2_deproj_CIB"
        / "flamingo_needletILCmap_component_tSZ_deproject_CIB_hilc_y_homog_q5masked_r2_deproj_CIB.fits",
        ILC / "hilc_output_homog_q5masked_deproj_CIB",
        ILC / "hilc_output_homog_q5masked_r2_deproj_CIB",
        ILC / "hilc_output_homog_q5masked_r2_deproj_CIB"
        / "hilc_homog_q5masked_deproj_cib_r1xr2_cl_unbinned_l1_m9.npz",
        ILC / "hilc_output_homog_q5masked_r2_deproj_CIB"
        / "hilc_homog_q5masked_deproj_cib_fig9_cl_l1_m9.npz",
    ),
    Case(
        r"CIB + $\delta\beta$ + $\delta T$",
        "_deproject_CIB_CIB_dbeta_CIB_dT",
        ILC / "hilc_output_homog_q5masked_deproj_CIB_CIB_dbeta_CIB_dT"
        / "flamingo_needletILCmap_component_tSZ_deproject_CIB_CIB_dbeta_CIB_dT"
        "_hilc_y_homog_q5masked_deproj_CIB_CIB_dbeta_CIB_dT.fits",
        ILC / "hilc_output_homog_q5masked_r2_deproj_CIB_CIB_dbeta_CIB_dT"
        / "flamingo_needletILCmap_component_tSZ_deproject_CIB_CIB_dbeta_CIB_dT"
        "_hilc_y_homog_q5masked_r2_deproj_CIB_CIB_dbeta_CIB_dT.fits",
        ILC / "hilc_output_homog_q5masked_deproj_CIB_CIB_dbeta_CIB_dT",
        ILC / "hilc_output_homog_q5masked_r2_deproj_CIB_CIB_dbeta_CIB_dT",
        ILC / "hilc_output_homog_q5masked_r2_deproj_CIB_CIB_dbeta_CIB_dT"
        / "hilc_homog_q5masked_deproj_cib_moments_r1xr2_cl_unbinned_l1_m9.npz",
        ILC / "hilc_output_homog_q5masked_r2_deproj_CIB_CIB_dbeta_CIB_dT"
        / "hilc_homog_q5masked_deproj_cib_moments_fig9_cl_l1_m9.npz",
    ),
)


def masked_cl(a: np.ndarray, b: np.ndarray, w: np.ndarray, lmax: int) -> np.ndarray:
    a0 = (a - np.sum(w * a) / np.sum(w)) * w
    b0 = (b - np.sum(w * b) / np.sum(w)) * w
    return hp.anafast(a0, b0, lmax=lmax, iter=0) / float(np.mean(w**2))


def alm_pair_to_masked_cl(y1, y2, w: np.ndarray) -> np.ndarray:
    m1 = hp.alm2map(y1, nside=NSIDE, lmax=LMAX)
    m2 = hp.alm2map(y2, nside=NSIDE, lmax=LMAX)
    return masked_cl(m1, m2, w, LMAX)


def ilc_bias_curves(cl_tt: np.ndarray, n_deproj: int, fsky: float):
    n_modes = n_modes_tophat_hilc(cl_tt.size - 1, HILC_BINSIZE, fsky)
    frac = ilc_bias_fraction(n_deproj, N_FREQ, n_modes)
    ells = np.arange(cl_tt.size, dtype=np.float64)
    return frac, frac * np.abs(dl_from_cl(ells, cl_tt))


def load_or_compute(case: Case, bl, good, w_apo: np.ndarray, fsky: float) -> dict:
    stored: dict[str, np.ndarray] = {}
    cl_sources = [MASK_APO, case.ymap_r1, case.ymap_r2, diag.TRUTH]
    if not cache_stale(case.cl_cache, cl_sources):
        z = np.load(case.cl_cache)
        stored.update({k: z[k] for k in z.files})
        print("loaded", case.cl_cache)
    weight_pattern = f"flamingo_weightvector_scale*_component_tSZ{case.wtag}.txt"
    weight_sources = sorted(case.wdir_r1.glob(weight_pattern))
    weight_sources += sorted(case.wdir_r2.glob(weight_pattern))
    signal_sources = [diag.CMB_MAP]
    signal_sources += [
        diag.CIB_DIR / f"CIB_deltaT_{f}GHz_nside4096.fits" for f in diag.FREQS
    ]
    noise_sources = [
        diag.NOISE_DIR / f"{f}GHz" / f"white_noise_{f}GHz_nside{NSIDE}_uK{tag}.fits"
        for f in diag.FREQS
        for tag in ("", "_r2")
    ]
    res_sources = [MASK_APO, *weight_sources, *signal_sources, *noise_sources]
    if not cache_stale(case.res_cache, res_sources):
        z = np.load(case.res_cache)
        stored.update({k: z[k] for k in z.files})
        print("loaded", case.res_cache)

    need_yy = any(k not in stored for k in ("cl_11", "cl_22", "cl_12", "cl_tt"))
    if need_yy:
        y1, y2, yt = diag.load_map(case.ymap_r1), diag.load_map(case.ymap_r2), diag.load_map(diag.TRUTH)
        stored["cl_11"] = masked_cl(y1, y1, w_apo, LMAX)
        stored["cl_22"] = masked_cl(y2, y2, w_apo, LMAX)
        stored["cl_12"] = masked_cl(y1, y2, w_apo, LMAX)
        stored["cl_tt"] = masked_cl(yt, yt, w_apo, LMAX)
        case.cl_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            case.cl_cache,
            cl_11=stored["cl_11"],
            cl_22=stored["cl_22"],
            cl_12=stored["cl_12"],
            cl_tt=stored["cl_tt"],
        )
        print("wrote", case.cl_cache)

    need_res = any(k not in stored for k in ("cl_cib_w", "cl_cmb_w", "cl_n_w"))
    if need_res:
        print(f"  weighted masked residuals ({case.label}) ...", flush=True)
        w1 = diag.hilc_weights(case.wdir_r1, case.wtag, LMAX)
        w2 = diag.hilc_weights(case.wdir_r2, case.wtag, LMAX)
        sig = diag.signal_alms()
        y_cib1, y_cib2 = diag.y_alms_signal(w1, w2, sig["cib"], same_all_freq=False)
        y_cmb1, y_cmb2 = diag.y_alms_signal(w1, w2, sig["cmb"], same_all_freq=True)
        y_n1, y_n2 = diag.y_alms_noise(w1, w2)
        stored["cl_cib_w"] = alm_pair_to_masked_cl(y_cib1, y_cib2, w_apo)
        stored["cl_cmb_w"] = alm_pair_to_masked_cl(y_cmb1, y_cmb2, w_apo)
        stored["cl_n_w"] = alm_pair_to_masked_cl(y_n1, y_n2, w_apo)
        case.res_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            case.res_cache,
            cl_cib_w=stored["cl_cib_w"],
            cl_cmb_w=stored["cl_cmb_w"],
            cl_n_w=stored["cl_n_w"],
        )
        print("wrote", case.res_cache)

    pack = {
        "cl_tt": np.asarray(stored["cl_tt"], dtype=np.float64),
        "cl_12_d": diag.deconv_auto(stored["cl_12"], bl, good),
        "cl_11_d": diag.deconv_auto(stored["cl_11"], bl, good),
        "cl_22_d": diag.deconv_auto(stored["cl_22"], bl, good),
        "cl_cib_d": diag.deconv_auto(stored["cl_cib_w"], bl, good),
        "cl_cmb_d": diag.deconv_auto(stored["cl_cmb_w"], bl, good),
        "cl_n_d": diag.deconv_auto(stored["cl_n_w"], bl, good),
        "fsky": fsky,
    }
    pack["dl_cross"] = diag.bin_mean_dl(pack["cl_12_d"])
    pack["dl_cib"] = diag.bin_mean_dl(pack["cl_cib_d"])
    pack["dl_cmb"] = diag.bin_mean_dl(pack["cl_cmb_d"])
    pack["dl_n"] = diag.bin_mean_dl(pack["cl_n_d"])
    pack["dl_cross_sigma"] = sigma_dl_cross_binned(
        pack["cl_11_d"], pack["cl_22_d"], pack["cl_12_d"], ELL_MIN, ELL_MAX, fsky
    )
    return pack


def _curve_and_bins(ax, ells, sl, cl, dl, *, color, marker, label, lw=1.3):
    ax.plot(ells[sl], np.abs(dl_from_cl(ells, cl))[sl], color=color, lw=lw, alpha=0.55, zorder=2)
    mag = np.abs(dl)
    pos = np.asarray(dl) >= 0
    ax.plot(ELL_EFF[pos], mag[pos], marker, color=color, ms=5.0, ls="none", zorder=4, label=label)
    if np.any(~pos):
        ax.plot(
            ELL_EFF[~pos], mag[~pos], marker, color=color, ms=5.5, ls="none",
            mfc="none", mew=1.5, zorder=4,
        )


def plot_fig9(cases_data, fsky: float) -> Path:
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
            ells[sl], np.abs(dl_from_cl(ells, d["cl_tt"]))[sl],
            color="k", lw=1.7, label=r"input truth $y$", zorder=3,
        )
        _, d_bias = ilc_bias_curves(d["cl_tt"], nd, fsky)
        ax.plot(
            ells[sl], d_bias[sl], color="crimson", lw=1.6, ls="--", zorder=5,
            label=r"$|\Delta C_\ell^{y_1 y_2}|$",
        )
        ax.plot(
            ells[sl], np.abs(dl_from_cl(ells, d["cl_12_d"]))[sl],
            color="C1", lw=1.8, alpha=0.45, zorder=2,
        )
        y = np.abs(d["dl_cross"])
        sig = np.asarray(d["dl_cross_sigma"], dtype=np.float64)
        lo = np.maximum(y - sig, y * 1.0e-3)
        ax.fill_between(ELL_EFF, lo, y + sig, color="C1", alpha=0.28, zorder=1, lw=0)
        ax.errorbar(
            ELL_EFF, y, yerr=[y - lo, sig],
            fmt="o", color="C1", ms=6.5, elinewidth=2.2, capsize=4.5, capthick=1.8,
            zorder=6, label=r"HILC $y$ $r_1\times r_2$",
        )
        _curve_and_bins(ax, ells, sl, d["cl_cib_d"], d["dl_cib"], color="C2", marker="^", label=r"CIB residual")
        _curve_and_bins(ax, ells, sl, d["cl_cmb_d"], d["dl_cmb"], color="C4", marker="v", label=r"CMB residual")
        _curve_and_bins(
            ax, ells, sl, d["cl_n_d"], d["dl_n"], color="0.35", marker="+",
            label=r"noise residual", lw=1.0,
        )
        ax.set_ylabel(r"$D_\ell$")
        ax.set_title(case.label + rf"  ($q>5$ holes, $N_{{\mathrm{{deproj}}}}={nd}$)")
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    axes[-1].set_xlabel(r"$\ell$")
    fig.suptitle(
        r"HILC $y$ $r_1\times r_2$ with $q>5$ cluster holes"
        "\n"
        r"C2 $0.25^\circ$ apodised mask; split-cross errors; "
        r"open markers: $D_\ell<0$ plotted as $|D_\ell|$",
        y=1.01, fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / "hilc_homog_q5masked_r1xr2_fig9_deproj.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bl = diag.bl10()
    good = bl >= 1e-3
    w_apo = diag.load_map(MASK_APO)
    fsky = float(np.mean(w_apo**2))
    print(f"C2 apodised hole mask <W^2>={fsky:.4f}")
    cases_data = [(c, load_or_compute(c, bl, good, w_apo, fsky)) for c in CASES]
    plot_fig9(cases_data, fsky)


if __name__ == "__main__":
    main()
