"""HILC y vs truth after pyILC beam deconvolution (B_10 / B_nu, then / B_10).

Observed maps are d = B_nu s + n (noise unbeamed). pyILC multiplies each
channel by B_10 / B_nu (McCarthy & Hill 2024). The y-map is at 10'; we
deconvolve that common beam so spectra match unbeamed truth. Noise is
left as n / B_nu, not B_10 n.
"""
from __future__ import annotations

from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from pyilc.fg import get_mix

from flamingo_mock.config import BEAM_FWHM_ARCMIN
from flamingo_mock.powerspectra import bin_cl, compute_cl, dl_from_cl
from flamingo_mock.spectral import dB_dT_Jy_per_sr_per_K

NSIDE = 2048
LMAX = 4096
FWHM_ILC = 10.0
DELTA_ELL = 21
BINSIZE, BEAM_CRIT = 50, 1.0e-3

FREQS = (100, 143, 353, 217, 545, 857)  # YAML / weight order
NELL_UK2 = {100: 5.07e-4, 143: 9.21e-5, 353: 2.00e-3, 217: 1.85e-4, 545: 5.51e-2, 857: 30.9}

YMAP = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc/hilc_output_homog"
    "/flamingo_needletILCmap_component_tSZ_hilc_y_homog_fullsky.fits"
)
TRUTH = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/tsz"
    "/compton_y_nside4096.fits"
)
WDIR = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc/hilc_output_homog")
CIB_DIR = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/cib")
SED_YML = "/scratch/scratch-lxu/agent_dev/auto_research_agent/pyilc/input/fg_SEDs_default_params.yml"
FIG_DIR = Path("/scratch/scratch-lxu/flamingo_mock_analysis/figures")


def hilc_weights(lmax: int) -> np.ndarray:
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
        wraw = np.atleast_1d(
            np.loadtxt(WDIR / f"flamingo_weightvector_scale{j}_component_tSZ.txt")
        ).ravel()
        sl = filts[j] > 0
        count = 0
        for a, ok in enumerate(use):
            if not ok:
                continue
            w_ell[a, sl] = wraw[count]
            count += 1
    return w_ell, inp_beams


def p15_cib_cl_jy(ells: np.ndarray) -> np.ndarray:
    idx = {f: i for i, f in enumerate(FREQS)}
    tab_ell = np.array([187.0, 320.0, 502.0, 684.0, 890.0, 1158.0, 1505.0, 1956.0, 2649.0])
    tab = {
        857: np.array([2.87e5, 1.34e5, 7.20e4, 4.38e4, 3.23e4, 2.40e4, 1.83e4, 1.46e4, 1.16e4]),
        545: np.array([6.63e4, 3.34e4, 1.91e4, 1.25e4, 9.17e3, 6.83e3, 5.34e3, 4.24e3, 3.42e3]),
        353: np.array([7.88e3, 4.35e3, 2.60e3, 1.74e3, 1.29e3, 9.35e2, 7.45e2, 6.08e2, np.nan]),
        217: np.array([4.17e2, 2.62e2, 1.75e2, 1.17e2, 8.82e1, 6.42e1, 3.34e1, 4.74e1, np.nan]),
        143: np.array([3.64e1, 3.23e1, 2.81e1, 2.27e1, 1.84e1, 1.58e1, 1.25e1, np.nan, np.nan]),
    }
    shot = {100: 8.62, 143: 7.25, 217: 19.12, 353: 228.28, 545: 1456.86, 857: 5632.28}

    def interp_auto(nu, ell):
        y = tab[nu]
        m = np.isfinite(y)
        le, ye = tab_ell[m], y[m]
        out = np.empty_like(ell, dtype=float)
        for i, L in enumerate(ell):
            if L < 2:
                out[i] = 0.0
            elif L < le[0]:
                out[i] = ye[0] * (L / le[0]) ** -1.2
            elif L > le[-1]:
                clust = max(ye[-1] - shot[nu], 0.0) * (L / le[-1]) ** -1.2
                out[i] = shot[nu] + clust
            else:
                out[i] = np.exp(np.interp(np.log(L), np.log(le), np.log(ye)))
        return out

    cl_jy = np.zeros((len(FREQS), ells.size))
    for nu in (143, 217, 353, 545, 857):
        cl_jy[idx[nu]] = interp_auto(nu, ells)
    clust143 = np.maximum(cl_jy[idx[143]] - shot[143], 0.0)
    cl_jy[idx[100]] = shot[100] + 0.2 * clust143
    cl_jy[:, :2] = 0.0
    return cl_jy


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bl10 = hp.gauss_beam(np.deg2rad(FWHM_ILC / 60.0), lmax=LMAX)
    ells = np.arange(LMAX + 1, dtype=np.float64)

    y_ilc = np.asarray(hp.read_map(str(YMAP), field=0, dtype=np.float64))
    y_true = np.asarray(hp.read_map(str(TRUTH), field=0, dtype=np.float64))
    if hp.get_nside(y_ilc) != NSIDE:
        y_ilc = hp.ud_grade(y_ilc, NSIDE)
    if hp.get_nside(y_true) != NSIDE:
        y_true = hp.ud_grade(y_true, NSIDE)
    print(f"ILC rms={y_ilc.std():.4g}  truth rms={y_true.std():.4g}")

    cl_yy = compute_cl(y_ilc, lmax=LMAX, deconv_pixel_window=False)
    cl_tt = compute_cl(y_true, lmax=LMAX, deconv_pixel_window=False)
    cl_yt = compute_cl(y_ilc, y_true, lmax=LMAX, deconv_pixel_window=False)
    good = bl10 >= 1e-3
    cl_yy_dec = np.full_like(cl_yy, np.nan)
    cl_yy_dec[good] = cl_yy[good] / bl10[good] ** 2
    cl_yt_dec = np.full_like(cl_yt, np.nan)
    cl_yt_dec[good] = cl_yt[good] / bl10[good]
    cl_tt_10 = cl_tt * bl10**2

    ell_b, yy_b = bin_cl(cl_yy, delta_ell=DELTA_ELL)
    _, tt_b = bin_cl(cl_tt, delta_ell=DELTA_ELL)
    _, yt_b = bin_cl(cl_yt, delta_ell=DELTA_ELL)
    _, yy_dec_b = bin_cl(cl_yy_dec, delta_ell=DELTA_ELL)
    _, yt_dec_b = bin_cl(cl_yt_dec, delta_ell=DELTA_ELL)
    _, tt10_b = bin_cl(cl_tt_10, delta_ell=DELTA_ELL)
    with np.errstate(divide="ignore", invalid="ignore"):
        transfer_b = yt_dec_b / tt_b
        rho_b = yt_dec_b / np.sqrt(np.abs(yy_dec_b * tt_b))
    band = (ell_b >= 50) & (ell_b <= 500)
    print(f"median transfer C_yt/(B_10 C_tt) 50-500: {np.nanmedian(transfer_b[band]):.3f}")
    print(f"median rho (B_10-deconv) 50-500: {np.nanmedian(rho_b[band]):.3f}")

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.2), sharex=True)
    ax = axes[0]
    ax.loglog(ell_b, dl_from_cl(ell_b, tt_b), color="k", lw=2, label=r"truth $y$ (unsmoothed)")
    ax.loglog(
        ell_b, dl_from_cl(ell_b, tt10_b), color="0.55", lw=1.6,
        label=r"truth $\times B_{10'}^2$",
    )
    ax.loglog(ell_b, dl_from_cl(ell_b, yy_b), color="C0", lw=1.4, ls=":", label=r"HILC $y$ (raw, at $10'$)")
    ax.loglog(
        ell_b, dl_from_cl(ell_b, yy_dec_b), color="C0", lw=1.8,
        label=r"HILC $y$ / $B_{10'}^2$",
    )
    ax.loglog(ell_b, np.abs(dl_from_cl(ell_b, yt_dec_b)), color="C1", lw=1.4, label=r"$|C_\ell^{yt}|/B_{10'}$")
    ax.set_ylabel(r"$D_\ell=\ell(\ell+1)C_\ell/2\pi$")
    ax.set_title(r"Full-sky HILC $y$, pyILC $B_{10'}$ deconvolved, vs FLAMINGO truth")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(2, LMAX)
    ax = axes[1]
    ax.axhline(1.0, color="k", lw=0.8)
    ax.semilogx(ell_b, transfer_b, color="C1", lw=1.8, label=r"$C_\ell^{yt}/(B_{10'}\,C_\ell^{tt})$")
    ax.semilogx(ell_b, rho_b, color="C0", lw=1.5, ls="--", label=r"$\rho_\ell$ (beam-deconv)")
    ax.set_ylim(0.0, 1.6)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel("transfer / correlation")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out1 = FIG_DIR / "hilc_homog_fullsky_y_vs_truth_ps.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print("wrote", out1)

    w_ell, inp_beams = hilc_weights(LMAX)
    idx = {f: i for i, f in enumerate(FREQS)}
    cl_jy = p15_cib_cl_jy(ells)
    R = np.eye(len(FREQS))
    for (a, b), rij in {
        (143, 217): 0.78, (143, 353): 0.54, (143, 545): 0.51, (143, 857): 0.45,
        (217, 353): 0.91, (217, 545): 0.90, (217, 857): 0.85,
        (353, 545): 0.983, (353, 857): 0.911, (545, 857): 0.949,
        (100, 143): 0.70, (100, 217): 0.50, (100, 353): 0.40, (100, 545): 0.35, (100, 857): 0.30,
    }.items():
        i, j = idx[a], idx[b]
        R[i, j] = R[j, i] = rij
    k_per_jy = np.array([1.0 / dB_dT_Jy_per_sr_per_K(f) for f in FREQS])

    cl_cib_p15 = np.full(LMAX + 1, np.nan)
    cl_n = np.full(LMAX + 1, np.nan)
    for L in np.where(good)[0]:
        if L < 2:
            continue
        cj = cl_jy[:, L]
        Ck = np.outer(k_per_jy, k_per_jy) * (R * np.sqrt(np.outer(cj, cj)))
        w = w_ell[:, L]
        cl_cib_p15[L] = w @ Ck @ w
        cl_n[L] = sum(
            (w[a] / max(inp_beams[a][L], 1e-30)) ** 2 * (NELL_UK2[f] * 1e-12)
            for a, f in enumerate(FREQS)
        )

    print("loading CIB maps ...")
    R_YANG = np.eye(len(FREQS))
    for (a, b), rij in {
        (217, 353): 0.993, (217, 545): 0.956, (217, 857): 0.841,
        (353, 545): 0.983, (353, 857): 0.895, (545, 857): 0.959,
        (100, 143): 0.95, (100, 217): 0.95, (143, 217): 0.95,
        (100, 353): 0.993, (143, 353): 0.993,
        (100, 545): 0.956, (143, 545): 0.956,
        (100, 857): 0.841, (143, 857): 0.841,
    }.items():
        i, j = idx[a], idx[b]
        R_YANG[i, j] = R_YANG[j, i] = rij

    cl_cib_maps = np.zeros((len(FREQS), LMAX + 1))
    y_alm = None
    for a, f in enumerate(FREQS):
        m = np.asarray(hp.read_map(str(CIB_DIR / f"CIB_deltaT_{f}GHz_nside4096.fits"), dtype=np.float64))
        m = hp.ud_grade(m, NSIDE) * 1e-6
        m -= np.mean(m)
        alm = hp.map2alm(m, lmax=LMAX, iter=0)
        cl_cib_maps[a] = hp.alm2cl(alm)
        contrib = hp.almxfl(alm, w_ell[a])
        y_alm = contrib if y_alm is None else y_alm + contrib
        print(f"  CIB {f} GHz  rms={m.std():.3e} K")
        del m
    cl_map = hp.alm2cl(y_alm)
    del y_alm

    cl_yang = np.zeros(LMAX + 1)
    for L in range(2, LMAX + 1):
        cj = np.maximum(cl_cib_maps[:, L], 0.0)
        Ck = R_YANG * np.sqrt(np.outer(cj, cj))
        cl_yang[L] = w_ell[:, L] @ Ck @ w_ell[:, L]

    a_tsz = 1e-6 * np.array(
        [get_mix([float(f)], "tSZ", param_dict_file=SED_YML)[0] for f in FREQS]
    )
    g_tsz = a_tsz @ w_ell
    cl_th = (g_tsz ** 2) * cl_tt
    print(f"g=sum w a_tSZ: ell50={g_tsz[50]:.4f}  ell300={g_tsz[300]:.4f}  ell1500={g_tsz[1500]:.4f}")

    _, cib_b = bin_cl(cl_cib_p15, delta_ell=DELTA_ELL)
    _, n_b = bin_cl(cl_n, delta_ell=DELTA_ELL)
    _, yang_b = bin_cl(cl_yang, delta_ell=DELTA_ELL)
    _, map_b = bin_cl(cl_map, delta_ell=DELTA_ELL)
    _, th_b = bin_cl(cl_th, delta_ell=DELTA_ELL)
    dl_tot = dl_from_cl(ell_b, yy_dec_b)
    dl_cib = dl_from_cl(ell_b, cib_b)
    dl_n = dl_from_cl(ell_b, n_b)
    dl_yang = dl_from_cl(ell_b, yang_b)
    dl_map = dl_from_cl(ell_b, map_b)
    dl_th = dl_from_cl(ell_b, th_b)
    dl_sum = dl_th + dl_map + dl_n

    print("ell   D_HILC       D_th         D_CIB_map    D_noise")
    for L0 in (50, 100, 300, 500, 1000, 1500, 2000):
        i = int(np.nanargmin(np.abs(ell_b - L0)))
        print(
            f"{ell_b[i]:5.0f}  {dl_tot[i]:10.3e}  {dl_th[i]:10.3e}  "
            f"{dl_map[i]:10.3e}  {dl_n[i]:10.3e}"
        )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.loglog(ell_b, dl_tot, "C0", lw=2.2, label=r"HILC $y$ / $B_{10'}^2$")
    ax.loglog(ell_b, dl_th, color="k", lw=1.8, ls="-.", label=r"tSZ ($\sum w a^{\mathrm{tSZ}}\times$ truth)")
    ax.loglog(ell_b, dl_cib, "C3", lw=2.0, label=r"CIB P15 (weights, unbeamed)")
    ax.loglog(ell_b, dl_yang, "C2", lw=1.8, ls="--", label=r"CIB Yang 2026 (weights, unbeamed)")
    ax.loglog(ell_b, dl_map, "C1", lw=1.8, label=r"CIB from maps (weights)")
    ax.loglog(ell_b, dl_n, color="0.35", lw=1.6, ls=":", label=r"noise $\sum (w_\nu/B_\nu)^2 N_\ell$")
    ax.loglog(ell_b, dl_sum, color="C4", lw=1.5, ls="--", label=r"$y_{\mathrm{th}}+\mathrm{CIB}_{\mathrm{map}}+N$")
    ax.set_xlim(10, LMAX)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell=\ell(\ell+1)C_\ell/2\pi$")
    ax.set_title(r"Beam-deconvolved HILC $y$: tSZ, CIB, noise ($n/B_\nu$)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out2 = FIG_DIR / "hilc_homog_fullsky_y_cib_noise_residual.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print("wrote", out2)


if __name__ == "__main__":
    main()
