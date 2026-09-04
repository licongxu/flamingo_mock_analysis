"""Split-spectrum diagnostics for all HILC prescriptions (full sky, homog r1×r2).

Orange = measured C_ℓ^{r1 r2} from the HILC y-maps.  Every other curve is the
same operation on one sky component: apply the HILC weights (pyILC 10' rebeam +
taper) to that component's frequency maps, then the r1×r2 split-cross spectrum.
No subtraction remainder.  No absolute value.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from flamingo_mock.config import BEAM_FWHM_ARCMIN
from flamingo_mock.powerspectra import compute_cl, dl_from_cl, sigma_dl_cross_binned

NSIDE = 2048
LMAX = 4096
FWHM_ILC = 10.0
TAPER_WIDTH = 200
BINSIZE, BEAM_CRIT = 50, 1.0e-3
FREQS = (100, 143, 353, 217, 545, 857)
FSKY = 1.0

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
ELL_PLOT_MAX = 3000

ILC = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc")
TSZ_DIR = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/tsz/L1_m9")
CIB_DIR = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/cib/L1_m9")
TRUTH = TSZ_DIR / "compton_y_nside4096.fits"
CMB_MAP = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/cmb"
    "/primary_CMB_T_lensed_nside4096_seed42.fits"
)
NOISE_DIR = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/planck_noise/homogeneous")
FIG_DIR = Path("/scratch/scratch-lxu/flamingo_mock_analysis/figures")


@dataclass(frozen=True)
class Case:
    label: str
    wtag: str
    ymap_r1: Path
    ymap_r2: Path
    wdir_r1: Path
    wdir_r2: Path
    cl_cache: Path


CASES = (
    Case(
        "no deprojection",
        "",
        ILC / "hilc_output_homog"
        / "flamingo_needletILCmap_component_tSZ_hilc_y_homog_fullsky.fits",
        ILC / "hilc_output_homog_r2"
        / "flamingo_needletILCmap_component_tSZ_hilc_y_homog_fullsky_r2.fits",
        ILC / "hilc_output_homog",
        ILC / "hilc_output_homog_r2",
        ILC / "hilc_output_homog_r2" / "hilc_homog_r1xr2_cl_unbinned.npz",
    ),
    Case(
        "CIB deprojection",
        "_deproject_CIB",
        ILC / "hilc_output_homog_deproj_CIB"
        / "flamingo_needletILCmap_component_tSZ_deproject_CIB_hilc_y_homog_fullsky_deproj_CIB.fits",
        ILC / "hilc_output_homog_r2_deproj_CIB"
        / "flamingo_needletILCmap_component_tSZ_deproject_CIB_hilc_y_homog_fullsky_r2_deproj_CIB.fits",
        ILC / "hilc_output_homog_deproj_CIB",
        ILC / "hilc_output_homog_r2_deproj_CIB",
        ILC / "hilc_output_homog_r2_deproj_CIB" / "hilc_homog_deproj_cib_r1xr2_cl_unbinned.npz",
    ),
    Case(
        r"CIB + $\delta\beta$",
        "_deproject_CIB_CIB_dbeta",
        ILC / "hilc_output_homog_deproj_CIB_CIB_dbeta"
        / "flamingo_needletILCmap_component_tSZ_deproject_CIB_CIB_dbeta"
        "_hilc_y_homog_fullsky_deproj_CIB_CIB_dbeta.fits",
        ILC / "hilc_output_homog_r2_deproj_CIB_CIB_dbeta"
        / "flamingo_needletILCmap_component_tSZ_deproject_CIB_CIB_dbeta"
        "_hilc_y_homog_fullsky_r2_deproj_CIB_CIB_dbeta.fits",
        ILC / "hilc_output_homog_deproj_CIB_CIB_dbeta",
        ILC / "hilc_output_homog_r2_deproj_CIB_CIB_dbeta",
        ILC / "hilc_output_homog_r2_deproj_CIB_CIB_dbeta"
        / "hilc_homog_deproj_cib_dbeta_r1xr2_cl_unbinned.npz",
    ),
    Case(
        r"CIB + $\delta\beta$ + CMB",
        "_deproject_CIB_CIB_dbeta_CMB",
        ILC / "hilc_output_homog_deproj_CIB_CIB_dbeta_CMB"
        / "flamingo_needletILCmap_component_tSZ_deproject_CIB_CIB_dbeta_CMB"
        "_hilc_y_homog_fullsky_deproj_CIB_CIB_dbeta_CMB.fits",
        ILC / "hilc_output_homog_r2_deproj_CIB_CIB_dbeta_CMB"
        / "flamingo_needletILCmap_component_tSZ_deproject_CIB_CIB_dbeta_CMB"
        "_hilc_y_homog_fullsky_r2_deproj_CIB_CIB_dbeta_CMB.fits",
        ILC / "hilc_output_homog_deproj_CIB_CIB_dbeta_CMB",
        ILC / "hilc_output_homog_r2_deproj_CIB_CIB_dbeta_CMB",
        ILC / "hilc_output_homog_r2_deproj_CIB_CIB_dbeta_CMB"
        / "hilc_homog_deproj_cib_dbeta_cmb_r1xr2_cl_unbinned.npz",
    ),
    Case(
        r"CIB + $\delta\beta$ + $\delta T$",
        "_deproject_CIB_CIB_dbeta_CIB_dT",
        ILC / "hilc_output_homog_deproj_CIB_CIB_dbeta_CIB_dT"
        / "flamingo_needletILCmap_component_tSZ_deproject_CIB_CIB_dbeta_CIB_dT"
        "_hilc_y_homog_fullsky_deproj_CIB_CIB_dbeta_CIB_dT.fits",
        ILC / "hilc_output_homog_r2_deproj_CIB_CIB_dbeta_CIB_dT"
        / "flamingo_needletILCmap_component_tSZ_deproject_CIB_CIB_dbeta_CIB_dT"
        "_hilc_y_homog_fullsky_r2_deproj_CIB_CIB_dbeta_CIB_dT.fits",
        ILC / "hilc_output_homog_deproj_CIB_CIB_dbeta_CIB_dT",
        ILC / "hilc_output_homog_r2_deproj_CIB_CIB_dbeta_CIB_dT",
        ILC / "hilc_output_homog_r2_deproj_CIB_CIB_dbeta_CIB_dT"
        / "hilc_homog_deproj_cib_moments_r1xr2_cl_unbinned.npz",
    ),
)


def load_map(path: Path) -> np.ndarray:
    m = np.asarray(hp.read_map(str(path), field=0, dtype=np.float64))
    if hp.get_nside(m) != NSIDE:
        m = hp.ud_grade(m, NSIDE)
    return m


def load_uk_to_k(path: Path) -> np.ndarray:
    m = load_map(path) * 1e-6
    return m - np.mean(m)


@lru_cache(maxsize=1)
def bl10() -> np.ndarray:
    return hp.gauss_beam(np.deg2rad(FWHM_ILC / 60.0), lmax=LMAX)


@lru_cache(maxsize=1)
def taper() -> np.ndarray:
    ell = np.arange(LMAX + 1, dtype=np.float64)
    return 1.0 - 0.5 * (np.tanh(0.025 * (ell - (LMAX - TAPER_WIDTH))) + 1.0)


@lru_cache(maxsize=1)
def bnu() -> np.ndarray:
    out = np.zeros((len(FREQS), LMAX + 1))
    for a, f in enumerate(FREQS):
        out[a] = hp.gauss_beam(np.deg2rad(BEAM_FWHM_ARCMIN[int(f)] / 60.0), lmax=LMAX)
    return out


def hilc_weights(wdir: Path, wtag: str, lmax: int) -> np.ndarray:
    fwhm_ch = [BEAM_FWHM_ARCMIN[int(f)] for f in FREQS]
    ellbins = np.arange(0, lmax + 1, BINSIZE)
    n_scales = len(ellbins) - 1
    filts = np.zeros((n_scales, lmax + 1))
    for i in range(n_scales - 1):
        filts[i, ellbins[i] : ellbins[i + 1]] = 1.0
    filts[-1, ellbins[-1] :] = 1.0
    inp_beams = [hp.gauss_beam(np.deg2rad(f / 60.0), lmax=lmax) for f in fwhm_ch]
    ell_B = np.array([int(np.argmin(np.abs(b - BEAM_CRIT))) for b in inp_beams])
    ell_F = np.zeros(n_scales)
    for i in range(n_scales - 1):
        peak = int(np.argmax(filts[i]))
        ell_F[i] = min(lmax, peak + int(np.argmin(np.abs(filts[i][peak:] - BEAM_CRIT))))
    ell_F[-1] = ell_F[-2]
    w_ell = np.zeros((len(FREQS), lmax + 1))
    for j in range(n_scales):
        use = [ell_F[j] <= ell_B[a] for a in range(len(FREQS))]
        wfile = wdir / f"flamingo_weightvector_scale{j}_component_tSZ{wtag}.txt"
        wraw = np.atleast_1d(np.loadtxt(wfile)).ravel()
        sl = filts[j] > 0
        count = 0
        for a, ok in enumerate(use):
            if not ok:
                continue
            w_ell[a, sl] = wraw[count]
            count += 1
        if count != wraw.size:
            raise RuntimeError(f"{wfile.name}: used {count} channels, file has {wraw.size}")
    return w_ell


def _map2alm_list(maps: list[np.ndarray]) -> list[np.ndarray]:
    return [hp.map2alm(m, lmax=LMAX, iter=0) for m in maps]


@lru_cache(maxsize=1)
def signal_alms() -> dict[str, tuple[np.ndarray, ...]]:
    """Unbeamed component alms (K_CMB). Signal is the same in r1 and r2."""
    print("map2alm CMB / tSZ / CIB ...", flush=True)
    cmb = load_uk_to_k(CMB_MAP)
    tsz = [load_uk_to_k(TSZ_DIR / f"tSZ_deltaT_{f}GHz_nside4096.fits") for f in FREQS]
    cib = [load_uk_to_k(CIB_DIR / f"CIB_deltaT_{f}GHz_nside4096.fits") for f in FREQS]
    cmb_alm = hp.map2alm(cmb, lmax=LMAX, iter=0)
    return {
        "cmb": (cmb_alm,),
        "tsz": tuple(_map2alm_list(tsz)),
        "cib": tuple(_map2alm_list(cib)),
    }


@lru_cache(maxsize=1)
def noise_alms() -> dict[str, tuple[np.ndarray, ...]]:
    print("map2alm noise r1 / r2 ...", flush=True)
    n1, n2 = [], []
    for f in FREQS:
        n1.append(load_uk_to_k(NOISE_DIR / f"{f}GHz" / f"white_noise_{f}GHz_nside{NSIDE}_uK.fits"))
        n2.append(load_uk_to_k(NOISE_DIR / f"{f}GHz" / f"white_noise_{f}GHz_nside{NSIDE}_uK_r2.fits"))
    return {"r1": tuple(_map2alm_list(n1)), "r2": tuple(_map2alm_list(n2))}


def _signal_filt(w_nu: np.ndarray) -> np.ndarray:
    """pyILC: beamed signal × (B_10 / B_ν) × taper × w → net B_10 × taper × w."""
    return w_nu * bl10() * taper()


def _noise_filt(w_nu: np.ndarray, a: int) -> np.ndarray:
    """Noise is unbeamed; pyILC still multiplies by B_10 / B_ν."""
    return w_nu * np.divide(bl10(), np.maximum(bnu()[a], 1e-30)) * taper()


def y_alms_signal(w1: np.ndarray, w2: np.ndarray, alms: tuple[np.ndarray, ...], same_all_freq: bool):
    y1 = y2 = None
    for a in range(len(FREQS)):
        alm = alms[0] if same_all_freq else alms[a]
        c1 = hp.almxfl(alm, _signal_filt(w1[a]))
        c2 = hp.almxfl(alm, _signal_filt(w2[a]))
        y1 = c1 if y1 is None else y1 + c1
        y2 = c2 if y2 is None else y2 + c2
    return y1, y2


def y_alms_noise(w1: np.ndarray, w2: np.ndarray):
    n = noise_alms()
    y1 = y2 = None
    for a in range(len(FREQS)):
        c1 = hp.almxfl(n["r1"][a], _noise_filt(w1[a], a))
        c2 = hp.almxfl(n["r2"][a], _noise_filt(w2[a], a))
        y1 = c1 if y1 is None else y1 + c1
        y2 = c2 if y2 is None else y2 + c2
    return y1, y2


def deconv_auto(cl: np.ndarray, bl: np.ndarray, good: np.ndarray) -> np.ndarray:
    out = np.full_like(cl, np.nan, dtype=np.float64)
    out[good] = cl[good] / bl[good] ** 2
    return out


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


def load_or_compute(case: Case, bl: np.ndarray, good: np.ndarray) -> dict:
    cache: dict = {}
    if case.cl_cache.is_file():
        z = np.load(case.cl_cache)
        cache = {k: z[k] for k in z.files}
        print("loaded", case.cl_cache, sorted(cache))

    if "cl_12" not in cache:
        for p in (case.ymap_r1, case.ymap_r2):
            if not p.is_file():
                raise FileNotFoundError(p)
        y1, y2, yt = load_map(case.ymap_r1), load_map(case.ymap_r2), load_map(TRUTH)
        print(
            f"{case.label}: r1 rms={y1.std():.4g}  r2 rms={y2.std():.4g}  "
            f"truth rms={yt.std():.4g}"
        )
        cache["cl_11"] = compute_cl(y1, lmax=LMAX, deconv_pixel_window=False)
        cache["cl_22"] = compute_cl(y2, lmax=LMAX, deconv_pixel_window=False)
        cache["cl_12"] = compute_cl(y1, y2, lmax=LMAX, deconv_pixel_window=False)
        cache["cl_tt"] = compute_cl(yt, lmax=LMAX, deconv_pixel_window=False)
        del y1, y2, yt

    w1 = hilc_weights(case.wdir_r1, case.wtag, LMAX)
    w2 = hilc_weights(case.wdir_r2, case.wtag, LMAX)

    need = (
        "cl_tsz_w", "cl_cib_w", "cl_cmb_w", "cl_n_w", "cl_recon_w",
        "cl_x_tsz_cib_w", "cl_x_tsz_cmb_w", "cl_x_cib_cmb_w",
    )
    if any(k not in cache for k in need):
        print(f"  weighted component y-alms ({case.label}) ...", flush=True)
        sig = signal_alms()
        y_tsz1, y_tsz2 = y_alms_signal(w1, w2, sig["tsz"], same_all_freq=False)
        y_cib1, y_cib2 = y_alms_signal(w1, w2, sig["cib"], same_all_freq=False)
        y_cmb1, y_cmb2 = y_alms_signal(w1, w2, sig["cmb"], same_all_freq=True)
        y_n1, y_n2 = y_alms_noise(w1, w2)
        cache["cl_tsz_w"] = hp.alm2cl(y_tsz1, y_tsz2)
        cache["cl_cib_w"] = hp.alm2cl(y_cib1, y_cib2)
        cache["cl_cmb_w"] = hp.alm2cl(y_cmb1, y_cmb2)
        cache["cl_n_w"] = hp.alm2cl(y_n1, y_n2)
        cache["cl_x_tsz_cib_w"] = hp.alm2cl(y_tsz1, y_cib2) + hp.alm2cl(y_cib1, y_tsz2)
        cache["cl_x_tsz_cmb_w"] = hp.alm2cl(y_tsz1, y_cmb2) + hp.alm2cl(y_cmb1, y_tsz2)
        cache["cl_x_cib_cmb_w"] = hp.alm2cl(y_cib1, y_cmb2) + hp.alm2cl(y_cmb1, y_cib2)
        cache["cl_recon_w"] = hp.alm2cl(
            y_tsz1 + y_cib1 + y_cmb1 + y_n1,
            y_tsz2 + y_cib2 + y_cmb2 + y_n2,
        )
        case.cl_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(case.cl_cache, **cache)
        print("wrote", case.cl_cache)
        del y_tsz1, y_tsz2, y_cib1, y_cib2, y_cmb1, y_cmb2, y_n1, y_n2

    cl_11_d = deconv_auto(cache["cl_11"], bl, good)
    cl_22_d = deconv_auto(cache["cl_22"], bl, good)
    cl_12_d = deconv_auto(cache["cl_12"], bl, good)
    cl_tsz_d = deconv_auto(cache["cl_tsz_w"], bl, good)
    cl_cib_d = deconv_auto(cache["cl_cib_w"], bl, good)
    cl_cmb_d = deconv_auto(cache["cl_cmb_w"], bl, good)
    cl_n_d = deconv_auto(cache["cl_n_w"], bl, good)
    cl_recon_d = deconv_auto(cache["cl_recon_w"], bl, good)
    cl_x_tsz_cib_d = deconv_auto(cache["cl_x_tsz_cib_w"], bl, good)
    cl_x_tsz_cmb_d = deconv_auto(cache["cl_x_tsz_cmb_w"], bl, good)
    cl_x_cib_cmb_d = deconv_auto(cache["cl_x_cib_cmb_w"], bl, good)
    # Truth is unbeamed — do not divide by B_10^2.
    cl_tt = np.asarray(cache["cl_tt"], dtype=np.float64)
    cl_sum_d = (
        cl_tsz_d + cl_cib_d + cl_cmb_d + cl_n_d
        + cl_x_tsz_cib_d + cl_x_tsz_cmb_d + cl_x_cib_cmb_d
    )

    pack = {
        "cl_11_d": cl_11_d,
        "cl_22_d": cl_22_d,
        "cl_12_d": cl_12_d,
        "cl_tsz_d": cl_tsz_d,
        "cl_cib_d": cl_cib_d,
        "cl_cmb_d": cl_cmb_d,
        "cl_n_d": cl_n_d,
        "cl_x_tsz_cib_d": cl_x_tsz_cib_d,
        "cl_x_tsz_cmb_d": cl_x_tsz_cmb_d,
        "cl_x_cib_cmb_d": cl_x_cib_cmb_d,
        "cl_sum_d": cl_sum_d,
        "cl_recon_d": cl_recon_d,
        "cl_tt": cl_tt,
        "dl_cross": bin_mean_dl(cl_12_d),
        "dl_tsz": bin_mean_dl(cl_tsz_d),
        "dl_cib": bin_mean_dl(cl_cib_d),
        "dl_cmb": bin_mean_dl(cl_cmb_d),
        "dl_n": bin_mean_dl(cl_n_d),
        "dl_x_tsz_cib": bin_mean_dl(cl_x_tsz_cib_d),
        "dl_x_tsz_cmb": bin_mean_dl(cl_x_tsz_cmb_d),
        "dl_x_cib_cmb": bin_mean_dl(cl_x_cib_cmb_d),
        "dl_sum": bin_mean_dl(cl_sum_d),
        "dl_recon": bin_mean_dl(cl_recon_d),
        "dl_tt": bin_mean_dl(cl_tt),
        "dl_cross_sigma": sigma_dl_cross_binned(
            cl_11_d, cl_22_d, cl_12_d, ELL_MIN, ELL_MAX, FSKY
        ),
    }

    print(f"\n{case.label}  (full sky, signed D_ell)")
    print(
        f"{'ell':>7}  {'D_x':>10}  {'D_recon':>10}  {'D_tsz':>10}  {'D_cib':>10}  "
        f"{'D_cmb':>10}  {'D_n':>10}  {'D_tSZxCIB':>10}"
    )
    for i, L in enumerate(ELL_EFF):
        print(
            f"{L:7.1f}  {pack['dl_cross'][i]:10.3e}  {pack['dl_recon'][i]:10.3e}  "
            f"{pack['dl_tsz'][i]:10.3e}  {pack['dl_cib'][i]:10.3e}  "
            f"{pack['dl_cmb'][i]:10.3e}  {pack['dl_n'][i]:10.3e}  "
            f"{pack['dl_x_tsz_cib'][i]:10.3e}"
        )
    return pack


YLIM = (3.0e-16, 2.0e-10)
NEG_NOTE = r"open markers: $D_\ell<0$, plotted as $|D_\ell|$"


def _style_log(ax) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ELL_EFF[0], ELL_PLOT_MAX)
    ax.set_ylim(*YLIM)
    ax.set_ylabel(r"$D_\ell$")


def plot_abs_signed(ax, x, y, *, color, marker, label, lw=1.3) -> None:
    """Log-log: plot |D|; filled markers if D≥0, open if D<0."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mag = np.abs(y)
    pos = y >= 0
    ax.plot(x, mag, color=color, lw=lw, ls="-", zorder=3)
    ax.plot(
        x[pos], mag[pos], marker, color=color, ms=5.0, ls="none",
        zorder=4, label=label,
    )
    if np.any(~pos):
        ax.plot(
            x[~pos], mag[~pos], marker, color=color, ms=5.5, ls="none",
            mfc="none", mew=1.5, zorder=4,
        )


def plot_split_spectra(cases_data: list[tuple[Case, dict]], ells: np.ndarray, sl: slice) -> Path:
    n = len(cases_data)
    fig, axes = plt.subplots(n, 1, figsize=(8.2, 3.15 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (case, d) in zip(axes, cases_data):
        ax.plot(
            ells[sl], dl_from_cl(ells, d["cl_tt"])[sl],
            color="k", lw=1.6, label=r"input truth $y$",
        )
        plot_abs_signed(
            ax, ELL_EFF, d["dl_cross"],
            color="C1", marker="o", label=r"measured $r_1\times r_2$ (total)",
        )
        ax.errorbar(
            ELL_EFF, np.abs(d["dl_cross"]),
            yerr=yerr_log(d["dl_cross"], d["dl_cross_sigma"]),
            fmt="none", ecolor="C1", elinewidth=1.0, capsize=2.0, zorder=2,
        )
        _style_log(ax)
        ax.set_title(case.label)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    axes[-1].set_xlabel(r"$\ell$")
    fig.suptitle(r"Full-sky HILC $y$: measured split-cross total", y=1.02, fontsize=11)
    fig.text(0.5, -0.01, NEG_NOTE, ha="center", fontsize=8, color="0.25")
    fig.tight_layout()
    out = FIG_DIR / "hilc_homog_r1xr2_split_spectra_all_prescriptions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def plot_component_spectra(cases_data: list[tuple[Case, dict]], ells: np.ndarray, sl: slice) -> Path:
    n = len(cases_data)
    fig, axes = plt.subplots(n, 1, figsize=(8.4, 3.25 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (case, d) in zip(axes, cases_data):
        plot_abs_signed(ax, ELL_EFF, d["dl_cross"], color="C1", marker="o",
                        label=r"measured $r_1\times r_2$ (total)", lw=1.8)
        plot_abs_signed(ax, ELL_EFF, d["dl_tsz"], color="C3", marker="s",
                        label=r"tSZ signal")
        plot_abs_signed(ax, ELL_EFF, d["dl_cib"], color="C2", marker="^",
                        label=r"CIB residual")
        plot_abs_signed(ax, ELL_EFF, d["dl_cmb"], color="C4", marker="v",
                        label=r"CMB residual")
        plot_abs_signed(ax, ELL_EFF, d["dl_n"], color="0.4", marker="+",
                        label=r"noise residual", lw=1.0)
        plot_abs_signed(ax, ELL_EFF, d["dl_x_tsz_cib"], color="C8", marker="P",
                        label=r"tSZ$\times$CIB", lw=1.0)
        _style_log(ax)
        ax.set_title(case.label)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    axes[-1].set_xlabel(r"$\ell$")
    fig.suptitle(
        r"Measured $r_1\times r_2$ (total) vs HILC-weighted component split-crosses",
        y=1.02, fontsize=11,
    )
    fig.text(0.5, -0.01, NEG_NOTE, ha="center", fontsize=8, color="0.25")
    fig.tight_layout()
    out = FIG_DIR / "hilc_homog_r1xr2_cross_residual_decomposition_all_prescriptions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def plot_residual_breakdown(cases_data: list[tuple[Case, dict]]) -> Path:
    n = len(cases_data)
    fig, axes = plt.subplots(n, 1, figsize=(8.4, 3.25 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (case, d) in zip(axes, cases_data):
        plot_abs_signed(ax, ELL_EFF, d["dl_cross"], color="C1", marker="o",
                        label=r"measured $r_1\times r_2$ (total)", lw=1.8)
        plot_abs_signed(ax, ELL_EFF, d["dl_tsz"], color="C3", marker="s",
                        label=r"tSZ signal")
        plot_abs_signed(ax, ELL_EFF, d["dl_cib"], color="C2", marker="^",
                        label=r"CIB residual")
        plot_abs_signed(ax, ELL_EFF, d["dl_cmb"], color="C4", marker="v",
                        label=r"CMB residual")
        plot_abs_signed(ax, ELL_EFF, d["dl_n"], color="0.4", marker="+",
                        label=r"noise residual", lw=1.0)
        plot_abs_signed(ax, ELL_EFF, d["dl_x_tsz_cib"], color="C8", marker="P",
                        label=r"tSZ$\times$CIB", lw=1.0)
        _style_log(ax)
        ax.set_title(case.label)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    axes[-1].set_xlabel(r"$\ell$")
    fig.suptitle(
        r"Binned $D_\ell$: measured total vs weighted component split-crosses",
        y=1.02, fontsize=11,
    )
    fig.text(0.5, -0.01, NEG_NOTE, ha="center", fontsize=8, color="0.25")
    fig.tight_layout()
    out = FIG_DIR / "hilc_homog_r1xr2_cross_residual_binned_all_prescriptions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def plot_cross_sigma_vs_deproj(cases_data: list[tuple[Case, dict]]) -> Path:
    """Show that extra cILC constraints enlarge the Gaussian split-cross error.

    On the log–log D_ℓ panels the bars are tiny: full-sky bins have many modes,
    and CIB-only deprojection only raises σ by a few percent.  Plotting σ
    itself (and σ / σ_none) makes the deprojection cost visible.
    """
    colors = ("C0", "C1", "C2", "C3", "C4")
    markers = ("o", "s", "^", "v", "D")
    sig0 = cases_data[0][1]["dl_cross_sigma"]

    fig, (ax_s, ax_r, ax_d) = plt.subplots(3, 1, figsize=(8.4, 9.6), sharex=False)
    ax_s.sharex(ax_r)

    for (case, d), c, m in zip(cases_data, colors, markers):
        ax_s.plot(
            ELL_EFF, d["dl_cross_sigma"], color=c, marker=m, ms=5.5, lw=1.4,
            label=case.label,
        )
        ax_r.plot(
            ELL_EFF, d["dl_cross_sigma"] / sig0, color=c, marker=m, ms=5.5, lw=1.4,
            label=case.label,
        )

    ax_s.set_yscale("log")
    ax_s.set_ylabel(r"$\sigma(D_\ell^{r_1 r_2})$")
    ax_s.set_title(
        r"Gaussian error on binned split-cross $D_\ell$"
        "\n"
        r"$\mathrm{Var}(\hat C_\ell^{12})=(C^{11}C^{22}+(C^{12})^2)/((2\ell+1)f_\mathrm{sky})$"
    )
    ax_s.legend(frameon=False, fontsize=8, loc="upper left")

    ax_r.axhline(1.0, color="0.5", lw=0.8, ls="--")
    ax_r.set_ylabel(r"$\sigma/\sigma_{\mathrm{no\,deproj}}$")
    ax_r.set_ylim(0.8, 8.0)
    ax_r.set_title("Deprojection cost relative to unconstrained HILC")

    # Linear D±σ at low ℓ, where cosmic-variance bars are large enough to see.
    # Offset in ℓ so caps do not sit on top of each other.
    lo = ELL_EFF <= 200
    x0 = ELL_EFF[lo]
    n = len(cases_data)
    for i, ((case, d), c, m) in enumerate(zip(cases_data, colors, markers)):
        x = x0 * (1.03 ** (i - 0.5 * (n - 1)))
        ax_d.errorbar(
            x, d["dl_cross"][lo], yerr=d["dl_cross_sigma"][lo],
            fmt=m, color=c, ms=5.0, elinewidth=1.3, capsize=3.0, capthick=1.1,
            label=case.label,
        )
    ax_d.set_ylabel(r"$D_\ell^{r_1 r_2}$")
    ax_d.set_title(r"Same $D_\ell\pm\sigma$ on a linear $y$-scale, $\ell\leq 200$")
    ax_d.axhline(0.0, color="0.6", lw=0.6)
    ax_d.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)

    ax_s.set_xscale("log")
    ax_s.set_xlim(ELL_EFF[0], ELL_PLOT_MAX)
    ax_r.set_xlabel(r"$\ell$")
    ax_d.set_xscale("log")
    ax_d.set_xlim(8.0, 220.0)
    ax_d.set_xlabel(r"$\ell$")
    fig.tight_layout()
    out = FIG_DIR / "hilc_homog_r1xr2_cross_sigma_vs_deproj.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bl = bl10()
    good = bl >= 1e-3
    ells = np.arange(LMAX + 1, dtype=np.float64)
    sl = slice(2, ELL_PLOT_MAX + 1)

    cases_data = [(c, load_or_compute(c, bl, good)) for c in CASES]
    plot_split_spectra(cases_data, ells, sl)
    plot_component_spectra(cases_data, ells, sl)
    plot_residual_breakdown(cases_data)
    plot_cross_sigma_vs_deproj(cases_data)


if __name__ == "__main__":
    main()
