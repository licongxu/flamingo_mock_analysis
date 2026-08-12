#!/usr/bin/env python3
"""Regenerate paper figures with publication style (usetex, readable fonts, no grids).

Reuses on-disk maps / catalogue metrics — does not re-run full MMF or ILC.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pub_style import FIG_DIR, apply_pub_style, no_grid, savefig  # noqa: E402

DATA = Path("/rds/flamingo/L2800N5040/HYDRO_FIDUCIAL/lightcone0_shells")
COMP = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/components")
ILC = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc")
SZIFI_CAT = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues"
)
NOISE = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/planck_noise/npipe"
)
CACHE = ROOT / "paper" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

DELTA_ELL = 7
SG_WINDOW = 11
SG_ORDER = 3
LMAX_PLOT = 4000
T_CMB = 2.7255


def _smooth(y: np.ndarray) -> np.ndarray:
    from scipy.signal import savgol_filter

    y = np.asarray(y, dtype=np.float64)
    if y.size < SG_WINDOW:
        return y
    m = np.isfinite(y)
    out = y.copy()
    if m.sum() >= SG_WINDOW:
        out[m] = savgol_filter(y[m], SG_WINDOW, SG_ORDER)
    return out


def _bin_cl(cl: np.ndarray, delta_ell: int = DELTA_ELL, lmin: int = 2):
    from flamingo_mock.powerspectra import bin_cl

    return bin_cl(cl, delta_ell=delta_ell, lmin=lmin)


def _dl(ell, cl):
    from flamingo_mock.powerspectra import dl_from_cl

    return dl_from_cl(ell, cl)


def load_or_compute_cls() -> dict[str, np.ndarray]:
    """Load cached C_ell or compute from full-sky maps (native Nside)."""
    cache_path = CACHE / "yang_cls_nside4096.npz"
    if cache_path.is_file():
        print(f"Loading cached spectra {cache_path}")
        z = np.load(cache_path)
        return {k: z[k] for k in z.files}

    from flamingo_mock.powerspectra import compute_cl

    print("Computing full-sky power spectra (one-time; may take several minutes)...")
    y = hp.read_map(str(COMP / "tsz" / "compton_y_nside4096.fits"), dtype=np.float64)
    ksz_b = hp.read_map(str(COMP / "ksz" / "doppler_b_nside4096.fits"), dtype=np.float64)
    kappa = hp.read_map(str(DATA / "kappa_rot.fits"), dtype=np.float64)
    lmax = min(3 * hp.get_nside(y) - 1, 6000)

    cls: dict[str, np.ndarray] = {}
    print("  tSZ auto...")
    cls["tsz_auto"] = compute_cl(y, lmax=lmax)
    print("  kSZ auto (Doppler b)...")
    cls["ksz_auto"] = compute_cl(ksz_b, lmax=lmax)
    print("  kappa auto...")
    cls["kappa_auto"] = compute_cl(kappa, lmax=lmax)

    cib_freqs = (217, 353, 545, 857)
    cib = {}
    for nu in cib_freqs:
        path = COMP / "cib" / f"CIB_I_{nu}GHz_nside4096.fits"
        if not path.exists():
            # may be symlink target only for some bands
            alt = DATA / f"lensed_CIB_rot_BANDPASS_F{nu}_three_params.fits"
            path = alt
        print(f"  CIB {nu} GHz...")
        cib[nu] = hp.read_map(str(path), dtype=np.float64)

    for nu in cib_freqs:
        print(f"  CIB auto {nu}...")
        cls[f"cib_auto_{nu}"] = compute_cl(cib[nu], lmax=lmax)
        print(f"  CIB x y {nu}...")
        cls[f"cib_x_tsz_{nu}"] = compute_cl(cib[nu], y, lmax=lmax)
        print(f"  CIB x kappa {nu}...")
        cls[f"cib_x_kappa_{nu}"] = compute_cl(cib[nu], kappa, lmax=lmax)

    for i, nu1 in enumerate(cib_freqs):
        for nu2 in cib_freqs[i + 1 :]:
            print(f"  CIB x CIB {nu1}x{nu2}...")
            cls[f"cib_x_{nu1}_{nu2}"] = compute_cl(cib[nu1], cib[nu2], lmax=lmax)

    np.savez_compressed(cache_path, **cls)
    print(f"Cached spectra -> {cache_path}")
    return cls


def plot_yang_spectra(cls: dict[str, np.ndarray]) -> None:
    apply_pub_style()
    cib_colors = {217: "#0072B2", 353: "#E69F00", 545: "#009E73", 857: "#D55E00"}
    cib_freqs = (217, 353, 545, 857)

    # Fig. 8 left — CIB autos
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for nu in cib_freqs:
        ell_b, cl_b = _bin_cl(cls[f"cib_auto_{nu}"])
        dl_b = _smooth(_dl(ell_b, cl_b))
        ax.loglog(ell_b, dl_b, color=cib_colors[nu], lw=1.8, label=rf"${nu}\,\mathrm{{GHz}}$")
    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(
        r"$\ell(\ell+1)C_\ell^{\mathrm{CIB}}/(2\pi)\;[\mathrm{Jy}\,\mathrm{sr}^{-1}]^{2}$"
    )
    ax.set_xlim(200, LMAX_PLOT)
    ax.legend(title=r"FLAMINGO L2p8\_m9 lightcone0", loc="lower left")
    ax.set_title(r"CIB auto-power spectra (native $N_{\mathrm{side}}=4096$)")
    no_grid(ax)
    fig.tight_layout()
    savefig(fig, "yang26_fig8left_cib_auto_spectra")
    plt.close(fig)

    # Table 3 — decorrelation matrix
    from flamingo_mock.powerspectra import decorrelation

    n = len(cib_freqs)
    R = np.eye(n)
    for i, nu1 in enumerate(cib_freqs):
        for j, nu2 in enumerate(cib_freqs):
            if j <= i:
                continue
            r_m, _ = decorrelation(
                cls[f"cib_x_{nu1}_{nu2}"],
                cls[f"cib_auto_{nu1}"],
                cls[f"cib_auto_{nu2}"],
            )
            R[i, j] = R[j, i] = r_m

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(R, vmin=0.8, vmax=1.0, cmap="viridis", origin="upper")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([rf"${nu}$" for nu in cib_freqs])
    ax.set_yticklabels([rf"${nu}$" for nu in cib_freqs])
    ax.set_xlabel(r"Frequency $\nu'\;[\mathrm{GHz}]$")
    ax.set_ylabel(r"Frequency $\nu\;[\mathrm{GHz}]$")
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{R[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if R[i, j] < 0.92 else "black",
                fontsize=11,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\langle r_{\nu\nu'}\rangle_{150<\ell<1000}$")
    ax.set_title(r"CIB frequency decorrelation (lightcone0)")
    no_grid(ax)
    fig.tight_layout()
    savefig(fig, "yang26_table3_cib_decorrelation")
    plt.close(fig)

    # Fig. 10 — cross spectra
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True)
    for nu in (353, 545, 857):
        ell_b, cl_by = _bin_cl(cls[f"cib_x_tsz_{nu}"])
        ell_k, cl_bk = _bin_cl(cls[f"cib_x_kappa_{nu}"])
        y_tsz = _smooth(np.abs(cl_by) * 1.0e6)
        y_kap = _smooth(np.abs(cl_bk))
        axes[0].loglog(ell_b, y_tsz, color=cib_colors[nu], lw=1.8, label=rf"${nu}\,\mathrm{{GHz}}$")
        axes[1].loglog(ell_k, ell_k * y_kap, color=cib_colors[nu], lw=1.8, label=rf"${nu}\,\mathrm{{GHz}}$")
    axes[0].set_xlabel(r"Multipole $\ell$")
    axes[1].set_xlabel(r"Multipole $\ell$")
    axes[0].set_ylabel(r"$|C_\ell^{\mathrm{CIB}\times y}|\;[10^{-6}\,\mathrm{Jy}\,\mathrm{sr}^{-1}]$")
    axes[1].set_ylabel(r"$\ell\,|C_\ell^{\mathrm{CIB}\times\kappa}|\;[\mathrm{Jy}\,\mathrm{sr}^{-1}]$")
    axes[0].set_xlim(10, LMAX_PLOT)
    axes[0].set_title(r"CIB--tSZ")
    axes[1].set_title(r"CIB--$\kappa$")
    for ax in axes:
        ax.legend(loc="upper right")
        no_grid(ax)
    fig.suptitle(r"FLAMINGO L2p8\_m9 / lightcone0 ($N_{\mathrm{side}}=4096$)", y=1.02)
    fig.tight_layout()
    savefig(fig, "yang26_fig10_cib_tsz_kappa_cross")
    plt.close(fig)

    # Figs 6/7 style — tSZ / kSZ / kappa autos (NO grids)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ell_b, cl_b = _bin_cl(cls["tsz_auto"])
    axes[0].loglog(ell_b, _smooth(_dl(ell_b, cl_b)), color="k", lw=1.8)
    axes[0].set_xlabel(r"Multipole $\ell$")
    axes[0].set_ylabel(r"$\ell(\ell+1)C_\ell^{yy}/(2\pi)$")
    axes[0].set_title(r"tSZ auto-spectrum (Compton $y$)")
    axes[0].set_xlim(10, LMAX_PLOT)

    ell_b, cl_b = _bin_cl(cls["ksz_auto"])
    dl_ksz = _dl(ell_b, cl_b) * (T_CMB * 1.0e6) ** 2
    axes[1].loglog(ell_b, _smooth(dl_ksz), color="k", lw=1.8)
    axes[1].set_xlabel(r"Multipole $\ell$")
    axes[1].set_ylabel(r"$\ell(\ell+1)C_\ell^{TT,\mathrm{kSZ}}/(2\pi)\;[\mu\mathrm{K}^2]$")
    axes[1].set_title(r"kSZ auto-spectrum (thermodynamic)")
    axes[1].set_xlim(10, LMAX_PLOT)

    ell_b, cl_b = _bin_cl(cls["kappa_auto"])
    axes[2].loglog(ell_b, _smooth(_dl(ell_b, cl_b)), color="k", lw=1.8)
    axes[2].set_xlabel(r"Multipole $\ell$")
    axes[2].set_ylabel(r"$\ell(\ell+1)C_\ell^{\kappa\kappa}/(2\pi)$")
    axes[2].set_title(r"CMB lensing $\kappa$ auto-spectrum")
    axes[2].set_xlim(10, LMAX_PLOT)

    for ax in axes:
        no_grid(ax)
    fig.tight_layout()
    savefig(fig, "yang26_figs67_tsz_ksz_kappa_autos")
    plt.close(fig)
    print("Yang spectra figures regenerated.")


def completeness_bar_counts(benchmark: dict) -> tuple[int, int]:
    """Numerator/denominator for the completeness bar label.

    Completeness is defined on the *truth* side: among detectable truth
    systems, how many were found.  The correct fraction is

        (n_truth_detectable - n_undetected) / n_truth_detectable

    which equals ``completeness_detectable``.  Using ``n_true_positives`` as
    the numerator is wrong when one-to-many matching can make
    ``n_true_positives > n_truth_detectable`` (detection-side TP count).
    """
    den = int(benchmark["n_truth_detectable"])
    num = den - int(benchmark["n_undetected"])
    return num, den


def plot_szifi_benchmarks() -> None:
    """Replot purity/completeness from cached benchmark JSON (usetex, no grids)."""
    apply_pub_style()
    pairs = [
        (
            "immf",
            SZIFI_CAT / "footprint_splitA_immf_q5_benchmark_szifi.json",
            SZIFI_CAT / "footprint_splitA_immf_q5_benchmark_szifi_snr_bins.json",
            "iMMF",
        ),
        (
            "scimmf",
            SZIFI_CAT / "footprint_splitA_scimmf_q5_benchmark_szifi.json",
            SZIFI_CAT / "footprint_splitA_scimmf_q5_benchmark_szifi_snr_bins.json",
            "sciMMF",
        ),
    ]
    purity_compare = {}
    for tag, bench_path, bins_path, label in pairs:
        if not bench_path.is_file():
            print(f"Skip SZiFi {tag}: missing {bench_path}")
            continue
        b = json.loads(bench_path.read_text())
        purity_compare[label] = b

        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
        n_tp_det, n_truth_det = completeness_bar_counts(b)
        metrics = [
            ("Purity", b["purity"], b["n_true_positives"], b["n_detected"], "#2ca02c"),
            (
                "Completeness",
                b["completeness_detectable"],
                n_tp_det,
                n_truth_det,
                "#1f77b4",
            ),
        ]
        for ax, (name, frac, num, den, color) in zip(axes, metrics):
            ax.bar([name], [frac * 100], color=color, width=0.5, edgecolor="0.2")
            ax.set_ylim(0, 105)
            ax.set_ylabel("Percent")
            ax.set_title(rf"{name}: ${frac * 100:.1f}\%$")
            ax.text(
                0,
                min(frac * 100 + 5, 95),
                rf"${num}/{den}$",
                ha="center",
                fontsize=12,
                fontweight="bold",
            )
            ax.axhline(50, color="0.7", ls="--", lw=0.8)
            no_grid(ax)
        fig.suptitle(
            rf"{label} footprint ($q_{{\mathrm{{opt}}}}\ge 5$, Zubeldia match $10'$)",
            y=1.02,
        )
        note = (
            rf"PR4 GAL$\times$PS footprint; truth SNR fixed-MMF; "
            rf"$z\le{b['z_max']}$; N$_\mathrm{{det}}={b['n_detected']}$"
        )
        fig.text(0.5, -0.02, note, ha="center", fontsize=11, color="0.25")
        fig.tight_layout()
        savefig(fig, f"szifi_footprint_{tag}_benchmark_szifi")
        plt.close(fig)

        if bins_path.is_file():
            sb = json.loads(bins_path.read_text())
            edges = sb["bin_edges"]
            # Completeness bins may be one longer than edges-1 if open-ended last bin
            comp = np.array(sb["completeness"], dtype=float) * 100
            pur = np.array(sb["purity"], dtype=float) * 100
            pur_th = sb.get("purity_thresholds", list(range(len(pur))))
            n_comp = len(comp)
            # labels for completeness: use bin centers if available
            if "bin_centers" in sb and len(sb["bin_centers"]) == n_comp:
                comp_labels = [f"{c:.0f}" for c in sb["bin_centers"]]
            else:
                comp_labels = [f"{i}" for i in range(n_comp)]

            fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
            x_c = np.arange(n_comp)
            e_c = np.array(sb.get("completeness_err", [0] * n_comp), dtype=float) * 100
            m = np.isfinite(comp)
            axes[0].errorbar(
                x_c[m],
                comp[m],
                yerr=e_c[m] if e_c.size == n_comp else None,
                fmt="o-",
                color="#1f77b4",
                capsize=3,
                lw=1.6,
                markersize=6,
                label="empirical",
            )
            if sb.get("completeness_erf") is not None:
                ye = np.array(sb["completeness_erf"], dtype=float) * 100
                me = np.isfinite(ye)
                axes[0].plot(
                    x_c[me],
                    ye[me],
                    "s--",
                    color="#d62728",
                    lw=1.3,
                    markersize=5,
                    label=r"ERF (blind)",
                )
                axes[0].legend(loc="lower right")
            axes[0].set_ylim(0, 105)
            axes[0].set_ylabel("Completeness (\\%)")
            axes[0].set_title("Completeness vs truth SNR")
            axes[0].set_xticks(x_c)
            axes[0].set_xticklabels(comp_labels, rotation=35, ha="right")
            axes[0].set_xlabel(r"Truth SNR bin centre ($q_{\mathrm{true}}$)")
            no_grid(axes[0])

            x_p = np.arange(len(pur))
            e_p = np.array(sb.get("purity_err", [0] * len(pur)), dtype=float) * 100
            m = np.isfinite(pur) & (np.array(sb.get("purity_n", [1] * len(pur))) > 0)
            axes[1].errorbar(
                x_p[m],
                pur[m],
                yerr=e_p[m] if e_p.size == len(pur) else None,
                fmt="o-",
                color="#2ca02c",
                capsize=3,
                lw=1.6,
                markersize=6,
            )
            axes[1].set_ylim(0, 105)
            axes[1].set_ylabel("Purity (\\%)")
            axes[1].set_title("Purity vs detection threshold")
            axes[1].set_xticks(x_p)
            axes[1].set_xticklabels([rf"$\ge{t:.0f}$" for t in pur_th], rotation=35, ha="right")
            axes[1].set_xlabel(r"Detection SNR threshold ($q_{\mathrm{opt}}$)")
            no_grid(axes[1])
            fig.suptitle(rf"{label} benchmark vs SNR", y=1.01)
            fig.tight_layout()
            savefig(fig, f"szifi_footprint_{tag}_benchmark_snr_bins_szifi")
            plt.close(fig)

    # Side-by-side purity comparison
    if len(purity_compare) == 2:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        labels = list(purity_compare.keys())
        purities = [purity_compare[k]["purity"] * 100 for k in labels]
        comps = [purity_compare[k]["completeness_detectable"] * 100 for k in labels]
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, purities, w, color="#2ca02c", edgecolor="0.2", label="Purity")
        ax.bar(x + w / 2, comps, w, color="#1f77b4", edgecolor="0.2", label="Completeness")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Percent")
        ax.set_title(r"iMMF vs sciMMF ($q\ge 5$, fixed-MMF truth)")
        ax.legend(loc="lower right")
        no_grid(ax)
        for i, (p, c) in enumerate(zip(purities, comps)):
            ax.text(i - w / 2, p + 2, f"{p:.1f}", ha="center", fontsize=11)
            ax.text(i + w / 2, c + 2, f"{c:.1f}", ha="center", fontsize=11)
        fig.tight_layout()
        savefig(fig, "szifi_footprint_purity_immf_scimmf")
        plt.close(fig)
    print("SZiFi benchmark figures regenerated.")


def plot_ilc_spectra() -> None:
    """Beam-deconvolved HILC/NILC vs truth y spectra (usetex, no grids)."""
    apply_pub_style()
    try:
        from flamingo_mock.ilc.beams import deconvolve_cl_beam, gaussian_beam_bl
    except Exception:
        # ILC package only on needlet_ilc; fall back to minimal local helpers
        def gaussian_beam_bl(lmax, fwhm_arcmin):
            ell = np.arange(lmax + 1)
            sigma = np.deg2rad(fwhm_arcmin / 60.0) / np.sqrt(8.0 * np.log(2.0))
            return np.exp(-0.5 * ell * (ell + 1.0) * sigma**2)

        def deconvolve_cl_beam(cl, bl, floor=1e-3):
            out = np.full_like(cl, np.nan)
            good = bl**2 > floor
            out[good] = cl[good] / bl[good] ** 2
            return out

    truth_path = ILC / "inputs_nside2048_npipe" / "compton_y_nside2048.fits"
    if not truth_path.is_file():
        # try components downgraded path
        alt = list(ILC.glob("**/compton_y_nside2048.fits"))
        truth_path = alt[0] if alt else None
    hilc_path = (
        ILC
        / "hilc_output_npipe_splitA"
        / "flamingo_needletILCmap_component_tSZ_hilc_y_npipe_splitA.fits"
    )
    hilc_b = (
        ILC
        / "hilc_output_npipe_splitB"
        / "flamingo_needletILCmap_component_tSZ_hilc_y_npipe_splitB.fits"
    )
    nilc_path = (
        ILC
        / "nilc_output_npipe_splitA"
        / "flamingo_needletILCmap_component_tSZ_nilc_y_npipe_splitA.fits"
    )
    mask_path = ILC / "gal_ps_mask_nside2048.fits"
    if truth_path is None or not Path(truth_path).is_file() or not hilc_path.is_file():
        print("Skip ILC spectra: missing y-maps")
        return

    lmax = 3000
    fwhm = 5.0
    bl = gaussian_beam_bl(lmax, fwhm)

    def masked_cl(m1, m2=None):
        mask = hp.read_map(str(mask_path), dtype=np.float64) if mask_path.is_file() else None
        a = hp.read_map(str(m1), dtype=np.float64)
        if mask is None:
            mask = np.ones_like(a)
        fsky = float(mask.mean())
        a0 = (a - np.sum(a * mask) / np.sum(mask)) * mask
        if m2 is None:
            return hp.anafast(a0, lmax=lmax) / fsky
        b = hp.read_map(str(m2), dtype=np.float64)
        b0 = (b - np.sum(b * mask) / np.sum(mask)) * mask
        return hp.anafast(a0, b0, lmax=lmax) / fsky

    print("ILC: computing spectra...")
    cl_truth = masked_cl(truth_path)
    cl_hilc = masked_cl(hilc_path)
    cl_hilc_x = masked_cl(hilc_path, hilc_b) if hilc_b.is_file() else cl_hilc
    cl_nilc = masked_cl(nilc_path) if nilc_path.is_file() else None

    cl_hilc_d = deconvolve_cl_beam(cl_hilc, bl)
    cl_hilc_x_d = deconvolve_cl_beam(cl_hilc_x, bl)
    cl_nilc_d = deconvolve_cl_beam(cl_nilc, bl) if cl_nilc is not None else None

    ell = np.arange(lmax + 1)
    # bin
    def bin_mean(cl, de=20):
        n = (len(cl) - 2) // de
        eb = np.array([ell[2 + i * de : 2 + (i + 1) * de].mean() for i in range(n)])
        cb = np.array(
            [
                np.nanmean(cl[2 + i * de : 2 + (i + 1) * de])
                for i in range(n)
            ]
        )
        return eb, cb

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    eb, cb = bin_mean(ell * (ell + 1) * cl_truth / (2 * np.pi))
    ax.loglog(eb, cb, "k-", lw=1.8, label=r"Truth $y$")
    eb, cb = bin_mean(ell * (ell + 1) * cl_hilc_x_d / (2 * np.pi))
    ax.loglog(eb, cb, color="#1f77b4", lw=1.8, label=r"HILC $A\times B$ (deconv.\ $5'$)")
    if cl_nilc_d is not None:
        eb, cb = bin_mean(ell * (ell + 1) * cl_nilc_d / (2 * np.pi))
        ax.loglog(eb, cb, color="#d62728", lw=1.6, ls="--", label=r"NILC (deconv.\ $5'$)")
    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{yy}/(2\pi)$")
    ax.set_title(r"ILC Compton-$y$ vs FLAMINGO truth")
    ax.legend(loc="lower left")
    ax.set_xlim(10, lmax)
    no_grid(ax)
    fig.tight_layout()
    savefig(fig, "ilc_y_vs_truth_spectra_pub")
    plt.close(fig)

    # Transfer T = C_ell^{ILC x truth} / C_ell^{truth}
    cl_xt = masked_cl(hilc_path, truth_path)
    cl_xt_d = deconvolve_cl_beam(cl_xt, bl)
    # truth is beam-free; compare to beamed truth for fair transfer of ILC maps
    T = np.full_like(cl_truth, np.nan)
    good = (cl_truth > 0) & np.isfinite(cl_xt_d)
    T[good] = cl_xt_d[good] / cl_truth[good]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    eb, tb = bin_mean(T)
    ax.semilogx(eb, tb, color="#1f77b4", lw=1.8, label="HILC")
    if cl_nilc is not None:
        cl_xn = masked_cl(nilc_path, truth_path)
        cl_xn_d = deconvolve_cl_beam(cl_xn, bl)
        Tn = np.full_like(cl_truth, np.nan)
        good = (cl_truth > 0) & np.isfinite(cl_xn_d)
        Tn[good] = cl_xn_d[good] / cl_truth[good]
        eb, tb = bin_mean(Tn)
        ax.semilogx(eb, tb, color="#d62728", lw=1.6, ls="--", label="NILC")
    ax.axhline(1.0, color="k", ls=":", lw=1.0)
    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(r"$T_\ell = C_\ell^{y_{\mathrm{ILC}}\times y_{\mathrm{true}}}/C_\ell^{yy}$")
    ax.set_title(r"ILC transfer function vs truth $y$")
    ax.set_ylim(0, 1.3)
    ax.set_xlim(10, lmax)
    ax.legend(loc="lower left")
    no_grid(ax)
    fig.tight_layout()
    savefig(fig, "ilc_transfer_vs_truth_pub")
    plt.close(fig)
    print("ILC figures regenerated.")


def plot_noise_gallery() -> None:
    """Planck NPIPE noise gallery with usetex (no grids)."""
    apply_pub_style()
    freqs = (100, 143, 217, 353)
    maps = []
    for nu in freqs:
        p = NOISE / f"{nu}GHz" / "A" / f"npipe6v20_noise_{nu}_A_mc_00200.fits"
        if not p.is_file():
            print(f"Skip noise gallery: missing {p}")
            return
        m = hp.read_map(str(p), field=0, dtype=np.float64)
        # 100-353 are K_CMB
        maps.append((nu, m * 1e6))  # uK

    fig = plt.figure(figsize=(11, 8.5))
    for i, (nu, m) in enumerate(maps):
        # percentile clip
        lo, hi = np.percentile(m, [1, 99])
        hp.mollview(
            m,
            min=lo,
            max=hi,
            title=rf"{nu:g}\,GHz NPIPE A  mc\_00200",
            unit=r"$\mu\mathrm{K}$",
            sub=(2, 2, i + 1),
            cmap="RdBu_r",
            fig=fig,
        )
    fig.suptitle(r"Planck NPIPE noise realisations (detector-set A)", y=1.01, fontsize=14)
    # dpi=300 + usetex via apply_pub_style(); PNG is the paper raster product
    out = FIG_DIR / "planck_noise_gallery_mc00200.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    try:
        fig.savefig(FIG_DIR / "planck_noise_gallery_mc00200.pdf", dpi=300, bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    apply_pub_style()
    print("=== Yang power spectra ===")
    cls = load_or_compute_cls()
    plot_yang_spectra(cls)
    print("=== SZiFi benchmarks ===")
    plot_szifi_benchmarks()
    print("=== ILC spectra ===")
    plot_ilc_spectra()
    print("=== Noise gallery ===")
    plot_noise_gallery()
    print("Done.")


if __name__ == "__main__":
    main()
