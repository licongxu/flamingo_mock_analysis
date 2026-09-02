"""HILC y (q>5 cluster-masked) vs truth, same beam convention as the full-sky plots.

HILC is run on the nside=2048 homog observed maps. Spectra use the C2 0.25°
apodized cluster mask (tsz_cnc_paper_plots recipe) and NaMaster.
"""
from __future__ import annotations

from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt
from pyilc.fg import get_mix

from flamingo_mock.config import BEAM_FWHM_ARCMIN
from flamingo_mock.powerspectra import bin_cl, dl_from_cl
from flamingo_mock.spectral import dB_dT_Jy_per_sr_per_K

NSIDE = 2048
LMAX = 4096
FWHM_ILC = 10.0
DELTA_ELL = 21
BINSIZE, BEAM_CRIT = 50, 1.0e-3

FREQS = (100, 143, 353, 217, 545, 857)
NELL_UK2 = {100: 5.07e-4, 143: 9.21e-5, 353: 2.00e-3, 217: 1.85e-4, 545: 5.51e-2, 857: 30.9}

YMAP = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc/hilc_output_homog_q5masked"
    "/flamingo_needletILCmap_component_tSZ_hilc_y_homog_q5masked.fits"
)
TRUTH = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/tsz/L1_m9"
    "/compton_y_nside4096.fits"
)
MASK = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc"
    "/szifi_immf_q5_cluster_mask_c2_025deg_nside2048.fits"
)
WDIR = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc/hilc_output_homog_q5masked")
WDIR_FULL = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc/hilc_output_homog")
CIB_DIR = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/cib/L1_m9")
SED_YML = "/scratch/scratch-lxu/agent_dev/auto_research_agent/pyilc/input/fg_SEDs_default_params.yml"
FIG_DIR = Path("/scratch/scratch-lxu/flamingo_mock_analysis/figures")


def hilc_weights(wdir: Path, lmax: int) -> tuple[np.ndarray, list]:
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
            np.loadtxt(wdir / f"flamingo_weightvector_scale{j}_component_tSZ.txt")
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


def nmt_field(m: np.ndarray, w: np.ndarray, lmax: int) -> nmt.NmtField:
    m = np.asarray(m, dtype=np.float64)
    m = m - np.sum(w * m) / np.sum(w)
    return nmt.NmtField(w, [m], lmax=lmax)


def namaster_cls(
    maps: list[np.ndarray],
    w: np.ndarray,
    lmax: int,
    nlb: int,
    wsp: nmt.NmtWorkspace | None = None,
    bins: nmt.NmtBin | None = None,
):
    """Decoupled C_ell at nside=2048. Weighted monopole removed."""
    fields = [nmt_field(m, w, lmax) for m in maps]
    if bins is None:
        bins = nmt.NmtBin.from_lmax_linear(lmax, nlb)
    if wsp is None:
        print("computing NaMaster coupling matrix ...", flush=True)
        wsp = nmt.NmtWorkspace.from_fields(fields[0], fields[0], bins)
    out = {}
    for i, fi in enumerate(fields):
        for j, fj in enumerate(fields):
            if j < i:
                continue
            out[(i, j)] = wsp.decouple_cell(nmt.compute_coupled_cell(fi, fj))[0]
    return bins.get_effective_ells(), out, wsp, bins


def deconv(cl: np.ndarray, bl: np.ndarray, power: int) -> np.ndarray:
    good = bl >= 1e-3
    out = np.full_like(cl, np.nan)
    out[good] = cl[good] / bl[good] ** power
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bl10 = hp.gauss_beam(np.deg2rad(FWHM_ILC / 60.0), lmax=LMAX)
    ells = np.arange(LMAX + 1, dtype=np.float64)
    good = bl10 >= 1e-3

    w = np.asarray(hp.read_map(str(MASK), field=0, dtype=np.float64))
    if hp.get_nside(w) != NSIDE:
        w = hp.ud_grade(w, NSIDE)
    print(f"mask fsky_raw={(w > 0).mean():.4f}  <W^2>={np.mean(w**2):.4f}")

    y_ilc = np.asarray(hp.read_map(str(YMAP), field=0, dtype=np.float64))
    y_true = np.asarray(hp.read_map(str(TRUTH), field=0, dtype=np.float64))
    if hp.get_nside(y_ilc) != NSIDE:
        y_ilc = hp.ud_grade(y_ilc, NSIDE)
    if hp.get_nside(y_true) != NSIDE:
        y_true = hp.ud_grade(y_true, NSIDE)
    print(f"ILC rms={y_ilc.std():.4g}  truth rms={y_true.std():.4g}")

    print("NaMaster decoupling ...", flush=True)
    ell_b, cls, wsp, bins = namaster_cls([y_ilc, y_true], w, LMAX, DELTA_ELL)
    yy_b = cls[(0, 0)]
    tt_b = cls[(1, 1)]
    yt_b = cls[(0, 1)]
    bl10_b = np.interp(ell_b, ells, bl10)
    yy_dec_b = deconv(yy_b, bl10_b, 2)
    yt_dec_b = deconv(yt_b, bl10_b, 1)
    tt10_b = tt_b * bl10_b**2
    with np.errstate(divide="ignore", invalid="ignore"):
        transfer_b = yt_dec_b / tt_b
        rho_b = yt_dec_b / np.sqrt(np.abs(yy_dec_b * tt_b))
    band = (ell_b >= 50) & (ell_b <= 500)
    print(f"median transfer 50-500: {np.nanmedian(transfer_b[band]):.3f}")
    print(f"median rho 50-500: {np.nanmedian(rho_b[band]):.3f}")

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.2), sharex=True)
    ax = axes[0]
    ax.loglog(ell_b, dl_from_cl(ell_b, tt_b), color="k", lw=2, label=r"truth $y$ (unsmoothed)")
    ax.loglog(ell_b, dl_from_cl(ell_b, tt10_b), color="0.55", lw=1.6, label=r"truth $\times B_{10'}^2$")
    ax.loglog(ell_b, dl_from_cl(ell_b, yy_b), color="C0", lw=1.4, ls=":", label=r"HILC $y$ (raw, at $10'$)")
    ax.loglog(ell_b, dl_from_cl(ell_b, yy_dec_b), color="C0", lw=1.8, label=r"HILC $y$ / $B_{10'}^2$")
    ax.loglog(ell_b, np.abs(dl_from_cl(ell_b, yt_dec_b)), color="C1", lw=1.4, label=r"$|C_\ell^{yt}|/B_{10'}$")
    ax.set_ylabel(r"$D_\ell=\ell(\ell+1)C_\ell/2\pi$")
    ax.set_title(r"HILC $y$, $q>5$ clusters masked, vs FLAMINGO truth")
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
    out1 = FIG_DIR / "hilc_homog_q5masked_y_vs_truth_ps.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print("wrote", out1)

    w_ell, inp_beams = hilc_weights(WDIR, LMAX)
    w_full, _ = hilc_weights(WDIR_FULL, LMAX)
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
        ww = w_ell[:, L]
        cl_cib_p15[L] = ww @ Ck @ ww
        cl_n[L] = sum(
            (ww[a] / max(inp_beams[a][L], 1e-30)) ** 2 * (NELL_UK2[f] * 1e-12)
            for a, f in enumerate(FREQS)
        )

    print("CIB maps at nside=2048 ...", flush=True)
    y_alm = None
    y_alm_full = None
    for a, f in enumerate(FREQS):
        m0 = np.asarray(hp.read_map(str(CIB_DIR / f"CIB_deltaT_{f}GHz_nside4096.fits"), dtype=np.float64))
        m0 = hp.ud_grade(m0, NSIDE) * 1e-6
        m_full = m0 - np.mean(m0)
        alm_full = hp.map2alm(m_full, lmax=LMAX, iter=0)
        contrib_full = hp.almxfl(alm_full, w_full[a])
        y_alm_full = contrib_full if y_alm_full is None else y_alm_full + contrib_full
        m = m0 * w
        m -= np.sum(w * m) / np.sum(w)
        alm = hp.map2alm(m, lmax=LMAX, iter=0)
        contrib = hp.almxfl(alm, w_ell[a])
        y_alm = contrib if y_alm is None else y_alm + contrib
        print(f"  CIB {f} GHz  rms={m.std():.3e} K")
        del m, m0, m_full
    y_cib = hp.alm2map(y_alm, nside=NSIDE, lmax=LMAX)
    cl_map_full = hp.alm2cl(y_alm_full)
    del y_alm, y_alm_full
    _, cls_cib, _, _ = namaster_cls([y_cib], w, LMAX, DELTA_ELL, wsp=wsp, bins=bins)
    map_b = cls_cib[(0, 0)]
    _, map_full_b = bin_cl(cl_map_full, delta_ell=DELTA_ELL)

    a_tsz = 1e-6 * np.array(
        [get_mix([float(f)], "tSZ", param_dict_file=SED_YML)[0] for f in FREQS]
    )
    g_tsz = a_tsz @ w_ell
    # Interpolate g(ell) onto NaMaster bins; tSZ uses masked truth auto.
    g_b = np.interp(ell_b, ells, g_tsz)
    th_b = (g_b**2) * tt_b
    print(f"g=sum w a_tSZ: ell50={g_tsz[50]:.4f}  ell300={g_tsz[300]:.4f}")

    _, cib_b = bin_cl(cl_cib_p15, delta_ell=DELTA_ELL)
    _, n_b = bin_cl(cl_n, delta_ell=DELTA_ELL)
    ell_th, _ = bin_cl(cl_cib_p15, delta_ell=DELTA_ELL)
    cib_b = np.interp(ell_b, ell_th, cib_b)
    n_b = np.interp(ell_b, ell_th, n_b)
    map_full_b = np.interp(ell_b, ell_th, map_full_b)

    dl_tot = dl_from_cl(ell_b, yy_dec_b)
    dl_cib = dl_from_cl(ell_b, cib_b)
    dl_n = dl_from_cl(ell_b, n_b)
    dl_map = dl_from_cl(ell_b, map_b)
    dl_map_full = dl_from_cl(ell_b, map_full_b)
    dl_th = dl_from_cl(ell_b, th_b)
    dl_sum = dl_th + dl_map + dl_n

    print("ell   D_HILC       D_th         D_CIB_full   D_CIB_q5     D_noise")
    for L0 in (50, 100, 300, 500, 1000, 1500, 2000):
        i = int(np.nanargmin(np.abs(ell_b - L0)))
        print(
            f"{ell_b[i]:5.0f}  {dl_tot[i]:10.3e}  {dl_th[i]:10.3e}  "
            f"{dl_map_full[i]:10.3e}  {dl_map[i]:10.3e}  {dl_n[i]:10.3e}"
        )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.loglog(ell_b, dl_tot, "C0", lw=2.2, label=r"HILC $y$ / $B_{10'}^2$")
    ax.loglog(ell_b, dl_th, color="k", lw=1.8, ls="-.", label=r"tSZ ($\sum w a^{\mathrm{tSZ}}\times$ truth)")
    ax.loglog(ell_b, dl_cib, "C3", lw=2.0, label=r"CIB P15 (weights, unbeamed)")
    ax.loglog(ell_b, dl_map_full, "C1", lw=2.0, label=r"CIB from maps (full sky)")
    ax.loglog(ell_b, dl_map, "C1", lw=1.8, ls="--", label=r"CIB from maps ($q>5$ masked)")
    ax.loglog(ell_b, dl_n, color="0.35", lw=1.6, ls=":", label=r"noise $\sum (w_\nu/B_\nu)^2 N_\ell$")
    ax.loglog(ell_b, dl_sum, color="C4", lw=1.5, ls="--", label=r"$y_{\mathrm{th}}+\mathrm{CIB}_{\mathrm{map}}+N$")
    ax.set_xlim(10, LMAX)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell=\ell(\ell+1)C_\ell/2\pi$")
    ax.set_title(r"HILC $y$ ($q>5$ masked): tSZ, CIB, noise")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out2 = FIG_DIR / "hilc_homog_q5masked_y_cib_noise_residual.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print("wrote", out2)


if __name__ == "__main__":
    main()
