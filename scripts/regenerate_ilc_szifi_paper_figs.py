#!/usr/bin/env python3
"""Publication figures: ILC deprojection suite + SZiFi found-cluster maps.

All plots: text.usetex=True, savefig dpi=300, no gridlines.
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

ILC = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc")
SZIFI_CAT = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues"
)
YMAP = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/tsz/"
    "compton_y_nside4096.fits"
)
MASK = ILC / "gal_ps_mask_nside2048.fits"

# Documented HILC deprojection suite metrics (needlet_ilc branch)
DEPROJ_METRICS = [
    # label, std_y, corr_beamed, rho_50_500, T_mid
    ("None", 1.807e-6, 0.473, 0.654, 0.909),
    ("CMB", 1.817e-6, 0.470, 0.655, 0.910),
    ("CIB", 1.838e-6, 0.496, 0.671, 0.960),
    (r"CIB+CMB", 1.850e-6, 0.493, 0.672, 0.959),
    (r"CIB+$\delta\beta$", 1.948e-6, 0.463, 0.614, 0.949),
    (r"CIB+$\delta\beta$+CMB", 1.968e-6, 0.458, 0.615, 0.948),
    (r"CIB+$\delta\beta$+$\delta T$", 4.194e-6, 0.226, 0.279, 1.000),
    (r"CIB+$\delta\beta$+$\delta T$+CMB", 5.998e-6, 0.159, 0.271, 0.995),
]

DEPROJ_YMAPS = {
    "None": ILC
    / "hilc_output_npipe_splitA"
    / "flamingo_needletILCmap_component_tSZ_hilc_y_npipe_splitA.fits",
    "CIB": ILC
    / "hilc_output_npipe_splitA_deproj_CIB"
    / "flamingo_needletILCmap_component_tSZ_deproject_CIB_hilc_y_npipe_splitA_deproj_CIB.fits",
    r"CIB+$\delta\beta$": ILC
    / "hilc_output_npipe_splitA_deproj_CIB_CIB_dbeta"
    / "flamingo_needletILCmap_component_tSZ_deproject_CIB_CIB_dbeta_hilc_y_npipe_splitA_deproj_CIB_CIB_dbeta.fits",
    r"CIB+$\delta\beta$+$\delta T$": ILC
    / "hilc_output_npipe_splitA_deproj_CIB_CIB_dbeta_CIB_dT"
    / "flamingo_needletILCmap_component_tSZ_deproject_CIB_CIB_dbeta_CIB_dT_hilc_y_npipe_splitA_deproj_CIB_CIB_dbeta_CIB_dT.fits",
}


def gaussian_beam_bl(lmax: int, fwhm_arcmin: float) -> np.ndarray:
    ell = np.arange(lmax + 1)
    sigma = np.deg2rad(fwhm_arcmin / 60.0) / np.sqrt(8.0 * np.log(2.0))
    return np.exp(-0.5 * ell * (ell + 1.0) * sigma**2)


def masked_cl(m1: np.ndarray, m2: np.ndarray | None, mask: np.ndarray, lmax: int):
    fsky = float(mask.mean())
    a = (m1 - np.sum(m1 * mask) / np.sum(mask)) * mask
    if m2 is None:
        return hp.anafast(a, lmax=lmax) / fsky
    b = (m2 - np.sum(m2 * mask) / np.sum(mask)) * mask
    return hp.anafast(a, b, lmax=lmax) / fsky


def bin_dl(cl: np.ndarray, de: int = 30):
    ell = np.arange(cl.size)
    n = (len(cl) - 2) // de
    eb = np.array([ell[2 + i * de : 2 + (i + 1) * de].mean() for i in range(n)])
    cb = np.array(
        [np.nanmean(ell[2 + i * de : 2 + (i + 1) * de]
                    * (ell[2 + i * de : 2 + (i + 1) * de] + 1)
                    * cl[2 + i * de : 2 + (i + 1) * de]
                    / (2 * np.pi)) for i in range(n)]
    )
    return eb, cb


def plot_deproj_metrics() -> None:
    apply_pub_style()
    labels = [r[0] for r in DEPROJ_METRICS]
    stds = np.array([r[1] for r in DEPROJ_METRICS]) * 1e6
    rho = np.array([r[3] for r in DEPROJ_METRICS])
    tmid = np.array([r[4] for r in DEPROJ_METRICS])
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    axes[0].bar(x, stds, color="#4c72b0", edgecolor="0.2", width=0.7)
    axes[0].set_ylabel(r"$\sigma_y\times 10^{6}$")
    axes[0].set_title(r"Map RMS")
    axes[1].bar(x, rho, color="#55a868", edgecolor="0.2", width=0.7)
    axes[1].set_ylabel(r"$\rho$ ($\ell=50$--$500$)")
    axes[1].set_title(r"Corr.\ with truth $y$")
    axes[1].set_ylim(0, 1.05)
    axes[2].bar(x, tmid, color="#c44e52", edgecolor="0.2", width=0.7)
    axes[2].set_ylabel(r"$T_{\mathrm{mid}}$")
    axes[2].set_title(r"Mid-$\ell$ transfer")
    axes[2].set_ylim(0, 1.15)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=10)
        no_grid(ax)
    fig.suptitle(
        r"HILC constrained-ILC deprojection suite (NPIPE split A, vs truth $y$)",
        y=1.02,
    )
    fig.tight_layout()
    savefig(fig, "ilc_deproj_suite_metrics")
    plt.close(fig)
    print("Wrote deproj metrics")


def plot_deproj_spectra() -> None:
    """Auto-power for selected deprojections vs truth (beam-deconvolved 5')."""
    apply_pub_style()
    truth_path = ILC / "inputs_nside2048_npipe" / "compton_y_nside2048.fits"
    if not truth_path.is_file():
        # fall back to downgrade
        y = hp.ud_grade(hp.read_map(str(YMAP), dtype=np.float64), 2048)
    else:
        y = hp.read_map(str(truth_path), dtype=np.float64)
    mask = (
        hp.read_map(str(MASK), dtype=np.float64)
        if MASK.is_file()
        else np.ones_like(y)
    )
    lmax = 2500
    bl = gaussian_beam_bl(lmax, 5.0)
    cl_t = masked_cl(y, None, mask, lmax)

    colors = {
        "None": "k",
        "CIB": "#1f77b4",
        r"CIB+$\delta\beta$": "#ff7f0e",
        r"CIB+$\delta\beta$+$\delta T$": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    eb, dl = bin_dl(cl_t)
    ax.loglog(eb, dl, "k-", lw=2.0, label=r"Truth $y$")
    for lab, path in DEPROJ_YMAPS.items():
        if not path.is_file():
            print("missing", path)
            continue
        m = hp.read_map(str(path), dtype=np.float64)
        cl = masked_cl(m, None, mask, lmax)
        cl_d = np.full_like(cl, np.nan)
        good = bl**2 > 1e-3
        cl_d[good] = cl[good] / bl[good] ** 2
        eb, dl = bin_dl(cl_d)
        ax.loglog(
            eb,
            dl,
            color=colors.get(lab, "C0"),
            lw=1.7,
            label=lab if lab != "None" else r"HILC (no deproj.)",
        )
    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{yy}/(2\pi)$")
    ax.set_title(r"HILC deprojection suite: auto-power vs truth")
    ax.set_xlim(20, lmax)
    ax.legend(loc="lower left", fontsize=10)
    no_grid(ax)
    fig.tight_layout()
    savefig(fig, "ilc_deproj_suite_autopower")
    plt.close(fig)
    print("Wrote deproj autopower")


def _load_cat(path: Path) -> dict:
    d = np.load(path)
    return {k: np.asarray(d[k]) for k in d.files}


def plot_cluster_mollviews() -> None:
    """Truth y + detections, and TP/FP coloured maps for iMMF and sciMMF."""
    apply_pub_style()
    from flamingo_mock.szifi.paths import SZiFiPaths
    from flamingo_mock.szifi.tiles import load_pr4_gal_ps
    from flamingo_mock.szifi.validate import match_detection_flags, run_match

    nside = 512  # paper resolution — keeps PDF small
    paths = SZiFiPaths()
    gal, ps = load_pr4_gal_ps(paths.masks_fits, nside=nside)
    mask = gal * ps
    y = hp.ud_grade(hp.read_map(str(YMAP), dtype=np.float64), nside)
    masked = y.copy()
    masked[mask < 0.5] = hp.UNSEEN
    good = masked[masked != hp.UNSEEN]
    vmax = float(np.percentile(good[good > 0], 99.5))
    vmin = float(np.percentile(good[good > 0], 1.0))

    truth_path = SZIFI_CAT / "footprint_splitA_truth_qtrue_mmf_qap2.npz"
    true_snr = None
    if truth_path.is_file():
        true_snr = _load_cat(truth_path)

    pairs = [
        ("immf", "iMMF", SZIFI_CAT / "footprint_splitA_immf_q5.npz", "cyan"),
        ("scimmf", "sciMMF", SZIFI_CAT / "footprint_splitA_scimmf_q5.npz", "deepskyblue"),
    ]

    for tag, label, cat_path, color in pairs:
        if not cat_path.is_file():
            print("skip missing cat", cat_path)
            continue

        # Footprint-matched detection list + TP flags (same path as benchmarks)
        match, _truth, det, _ = run_match(
            cat_path,
            paths=paths,
            q_th_obs=5.0,
            q_th_truth=5.0,
            z_max=1.0,
            match_radius_arcmin=10.0,
            true_snr=true_snr,
            apply_footprint_to_detections=True,
        )
        lon = match.det_lon
        lat = match.det_lat
        th = match.det_theta
        det_hit = match.det_hit

        # --- all detections ---
        fig = plt.figure(figsize=(11.0, 6.4))
        hp.mollview(
            masked,
            min=vmin,
            max=vmax,
            title=rf"Truth $y$ + {label} detections ($N={lon.size}$, $q\ge 5$)",
            unit=r"$y$",
            cmap="hot",
            cbar=True,
            fig=fig,
        )
        hp.graticule(dmer=30, dpar=30, alpha=0.2, verbose=False)
        s = np.clip((th / 60.0) ** 2 * 140.0, 8.0, 400.0)
        hp.projscatter(
            lon,
            lat,
            lonlat=True,
            s=s,
            facecolors="none",
            edgecolors=color,
            linewidths=0.7,
            alpha=0.9,
            label=label,
        )
        plt.legend(loc="lower left", fontsize=13, framealpha=0.92)
        for ext in ("png", "pdf"):
            fig.savefig(
                FIG_DIR / f"szifi_footprint_{tag}_mollview_detections.{ext}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)
        print(f"Wrote detection mollview {tag} N={lon.size}")

        # --- TP / FP ---
        fig = plt.figure(figsize=(11.0, 6.4))
        n_tp = int(det_hit.sum())
        n_fp = int((~det_hit).sum())
        hp.mollview(
            masked,
            min=vmin,
            max=vmax,
            title=(
                rf"{label}: TP (lime) / FP (red) "
                rf"($N_{{\mathrm{{TP}}}}={n_tp}$, $N_{{\mathrm{{FP}}}}={n_fp}$)"
            ),
            unit=r"$y$",
            cmap="hot",
            cbar=True,
            fig=fig,
        )
        hp.graticule(dmer=30, dpar=30, alpha=0.2, verbose=False)
        for hit, col, lab in (
            (det_hit, "lime", rf"TP ({n_tp})"),
            (~det_hit, "red", rf"FP ({n_fp})"),
        ):
            if not np.any(hit):
                continue
            s = np.clip((th[hit] / 60.0) ** 2 * 140.0, 8.0, 400.0)
            hp.projscatter(
                lon[hit],
                lat[hit],
                lonlat=True,
                s=s,
                facecolors="none",
                edgecolors=col,
                linewidths=0.75,
                alpha=0.95,
                label=lab,
            )
        plt.legend(loc="lower left", fontsize=13, framealpha=0.92)
        for ext in ("png", "pdf"):
            fig.savefig(
                FIG_DIR / f"szifi_footprint_{tag}_mollview_tp_fp.{ext}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)
        print(f"Wrote TP/FP mollview {tag} TP={n_tp} FP={n_fp}")


def compress_mollviews_for_paper() -> None:
    """Downsample huge native mollview PDFs to paper-ready PNG@300dpi max width."""
    apply_pub_style()
    from PIL import Image

    # Prefer existing PNG; resize max edge 2400 px JPEG-quality PNG
    names = [
        "components_overview_353GHz_mollview",
        "components_cmb_lensed_mollview",
        "components_tsz_compton_y_mollview",
        "components_tsz_deltaT_allfreq_mollview",
        "components_cib_deltaT_allfreq_mollview",
    ]
    out_dir = FIG_DIR / "paper"
    out_dir.mkdir(exist_ok=True)
    for name in names:
        src = FIG_DIR / f"{name}.png"
        if not src.is_file():
            print("skip", name)
            continue
        im = Image.open(src).convert("RGB")
        max_w = 2400
        if im.width > max_w:
            h = int(im.height * max_w / im.width)
            im = im.resize((max_w, h), Image.Resampling.LANCZOS)
        out = out_dir / f"{name}.png"
        im.save(out, format="PNG", optimize=True)
        print(f"Compressed {out} ({out.stat().st_size/1e6:.2f} MB)")


def main() -> None:
    apply_pub_style()
    print("=== ILC deproj metrics ===")
    plot_deproj_metrics()
    print("=== ILC deproj spectra ===")
    plot_deproj_spectra()
    print("=== SZiFi cluster maps ===")
    plot_cluster_mollviews()
    print("=== Compress component mollviews ===")
    compress_mollviews_for_paper()
    print("Done.")


if __name__ == "__main__":
    main()
