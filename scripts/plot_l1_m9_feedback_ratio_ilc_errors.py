"""L1_m9 feedback / fiducial ratio figure with ILC r1×r2 Gaussian errors.

Datapoints are the paper 18-bin qfrommap ratios (unchanged).  The shaded
1σ band around unity is

    σ / D_ℓ^{fid} ,   σ² = σ_ILC,G² + σ_T²

where σ_T is the one-halo trispectrum piece already in the theory covariance
and σ_ILC,G is the Gaussian split-cross error of the homog HILC y r1×r2
maps (includes residual foregrounds and independent instrumental noise).

Deprojection prescriptions on disk: none, CIB, CIB+δβ+δT; full sky and q>5.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from flamingo_mock.powerspectra import sigma_dl_cross_binned

ROOT = Path("/scratch/scratch-lxu/flamingo_mock_analysis")
FLAMINGO_REPO = Path("/scratch/scratch-lxu/flamingo_repo")
ILC = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc")
COV = FLAMINGO_REPO / "data_paper" / "covariance"
FIG_DIR = ROOT / "figures"
FWHM_ILC = 10.0
LMAX = 4096
FSKY_Q5_THEORY = 0.9493476504305253

# Paper 18-bin edges (inclusive).  sigma_dl_cross_binned uses exclusive hi.
ELL_MIN_18 = np.array(
    [9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835],
    dtype=int,
)
ELL_MAX_18_INCL = np.array(
    [12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835, 1085],
    dtype=int,
)
ELL_MAX_18_EXCL = ELL_MAX_18_INCL + 1

ILC_NPZ = {
    ("none", "fullsky"): ILC / "hilc_output_homog_r2" / "hilc_homog_r1xr2_cl_unbinned.npz",
    ("cib", "fullsky"): ILC / "hilc_output_homog_r2_deproj_CIB" / "hilc_homog_deproj_cib_r1xr2_cl_unbinned.npz",
    ("moments", "fullsky"): (
        ILC / "hilc_output_homog_r2_deproj_CIB_CIB_dbeta_CIB_dT"
        / "hilc_homog_deproj_cib_moments_r1xr2_cl_unbinned.npz"
    ),
    ("none", "q5"): ILC / "hilc_output_homog_q5masked_r2" / "hilc_homog_q5masked_r1xr2_cl_unbinned.npz",
    ("cib", "q5"): (
        ILC / "hilc_output_homog_q5masked_r2_deproj_CIB"
        / "hilc_homog_q5masked_deproj_cib_r1xr2_cl_unbinned.npz"
    ),
    ("moments", "q5"): (
        ILC / "hilc_output_homog_q5masked_r2_deproj_CIB_CIB_dbeta_CIB_dT"
        / "hilc_homog_q5masked_deproj_cib_moments_r1xr2_cl_unbinned.npz"
    ),
}

MASK_APO = ILC / "szifi_immf_q5_cluster_mask_c2_025deg_nside2048.fits"


def _load_ratio_mod():
    script = FLAMINGO_REPO / "scripts" / "plot_l1_m9_feedback_ratio_vs_q.py"
    spec = importlib.util.spec_from_file_location("plot_feedback_ratio_vs_q", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.TAG = "qfrommap"
    mod.OUTPUT_TAG = "qfrommap"
    mod.SELECTION_DESCRIPTION = "empirical aperture q"
    mod.ERROR_BAND_KINDS = (("fullsky", "full sky"), (5.0, r"$q>5$"))
    return mod


def deconv_auto(cl: np.ndarray, bl: np.ndarray, good: np.ndarray) -> np.ndarray:
    out = np.full_like(cl, np.nan, dtype=np.float64)
    out[good] = cl[good] / bl[good] ** 2
    return out


def ilc_fsky(sky: str) -> float:
    if sky == "fullsky":
        return 1.0
    w = np.asarray(hp.read_map(str(MASK_APO), field=0, dtype=np.float64))
    return float(np.mean(w**2))


def ilc_sigma_18(npz: Path, fsky: float, bl: np.ndarray, good: np.ndarray) -> np.ndarray:
    z = np.load(npz)
    c11 = deconv_auto(z["cl_11"], bl, good)
    c22 = deconv_auto(z["cl_22"], bl, good)
    c12 = deconv_auto(z["cl_12"], bl, good)
    return sigma_dl_cross_binned(c11, c22, c12, ELL_MIN_18, ELL_MAX_18_EXCL, fsky)


def knox_sigma(dl: np.ndarray, fsky: float) -> np.ndarray:
    sig = np.empty(len(ELL_MIN_18), dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(ELL_MIN_18, ELL_MAX_18_INCL)):
        els = np.arange(lo, hi + 1, dtype=np.float64)
        nmodes = fsky * np.sum(2.0 * els + 1.0)
        sig[i] = abs(dl[i]) * np.sqrt(2.0 / nmodes)
    return sig


def trispectrum_sigma(kind: str, dl_theory: np.ndarray, fsky: float) -> np.ndarray:
    label = "fullsky" if kind == "fullsky" else "masked_qgt5"
    cov = np.load(COV / f"cov_full_L1_m9_customgnfw_bestfit_{label}_Dl_yy_binned_18.npy")
    sig_full = np.sqrt(np.clip(np.diag(np.asarray(cov, dtype=float)), 0.0, None))
    sig_g = knox_sigma(dl_theory, fsky)
    return np.sqrt(np.clip(sig_full**2 - sig_g**2, 0.0, None))


def theory_dl(kind: str) -> np.ndarray:
    path = COV / "Dl_yy_customgnfw_bestfit_theory_binned_18.txt"
    data = np.loadtxt(path)
    # columns: ell, 1h, 2h, total, 1h_q5, 2h_q5, total_q5, ...
    if kind == "fullsky":
        return data[:, 3]
    return data[:, 6]


def relative_ilc_plus_t(
    kind: str,
    deproj: str,
    dl_fid_1e12: np.ndarray,
    bl: np.ndarray,
    good: np.ndarray,
    fsky_ilc: dict[str, float],
) -> np.ndarray:
    sky = "fullsky" if kind == "fullsky" else "q5"
    sig_ilc = ilc_sigma_18(ILC_NPZ[(deproj, sky)], fsky_ilc[sky], bl, good)
    fsky_t = 1.0 if kind == "fullsky" else FSKY_Q5_THEORY
    sig_t = trispectrum_sigma(kind, theory_dl(kind), fsky_t)
    sig = np.sqrt(sig_ilc**2 + sig_t**2)
    return sig * 1.0e12 / dl_fid_1e12


# One colour per error treatment.  Old = paper signal-only G+T.
TREATMENTS = (
    ("old", r"$C_\ell^{yy}$ only", "#bdbdbd"),
    ("none", r"ILC, no deproj.", "#9ecae1"),
    ("cib", r"ILC, CIB deproj.", "#a1d99b"),
    ("moments", r"ILC, CIB+$\delta\beta$+$\delta T$", "#fc9272"),
)
FOCUS_VARIANTS = ("fgas-8sigma", "Mstar-1sigma")
SKY_COLS = (("fullsky", r"full sky"), ("q5", r"$q>5$"))
ELL_RANGE = (10.0, 959.5)
PANEL_W, PANEL_H, LEGEND_H = 3.35, 2.7, 0.9


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "text.usetex": True,
            "axes.grid": False,
            "mathtext.fontset": "cm",
        }
    )


def _ratio_setup(mod):
    curves = list(mod.DEFAULT_CURVES)
    all_ratios: dict[str, np.ndarray] = {}
    ell_ref = None
    for variant in FOCUS_VARIANTS:
        ell, ratios = mod.ratios_by_q(variant, curves, log=False)
        ell_ref = ell if ell_ref is None else ell_ref
        for key, arr in ratios.items():
            all_ratios[f"{variant}:{key}"] = arr
    assert ell_ref is not None
    inside = (ell_ref >= ELL_RANGE[0]) & (ell_ref <= ELL_RANGE[1])
    ylim = mod._ylim_from_ratios(all_ratios, inside)
    return curves, ell_ref, inside, ylim, mod._color_map_for_curves(curves)


def _draw_bands(ax, x, rel_bands, sky, inside):
    for key, _lab, color in reversed(TREATMENTS):
        s = rel_bands[(sky, key)][inside]
        ax.fill_between(
            x, 1.0 - s, 1.0 + s, facecolor=color,
            alpha=0.12 if key == "old" else 0.14, lw=0, zorder=1,
        )
        ax.plot(x, 1.0 + s, color=color, lw=1.5, zorder=2)
        ax.plot(x, 1.0 - s, color=color, lw=1.5, zorder=2)


def _draw_ratio_curves(ax, x, ratios, inside, curves, curve_colors):
    ax.axhline(1.0, color="k", lw=1.1, zorder=3)
    for kind in curves:
        ax.semilogx(
            x, ratios[kind][inside], lw=1.7, color=curve_colors[kind],
            marker="o", markersize=3.6, zorder=4,
        )
    ax.set_xlim(*ELL_RANGE)
    ax.set_xscale("log")
    ax.grid(False)


def _legend_handles(mod, curves, curve_colors):
    handles = [
        Line2D(
            [0], [0], color=curve_colors[kind], marker="o", lw=1.6, markersize=4,
            label=mod.curve_label(kind),
        )
        for kind in curves
    ]
    handles += [
        Line2D([0], [0], color=color, lw=2.2, label=lab)
        for _key, lab, color in TREATMENTS
    ]
    return handles


def _save(fig, stem: Path) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png)
    print("wrote", pdf)
    return png


def plot_focus(mod, rel_bands: dict, stem: Path) -> Path:
    curves, ell_ref, inside, ylim, curve_colors = _ratio_setup(mod)
    x = ell_ref[inside]
    fig, axes = plt.subplots(
        2, 2, figsize=(PANEL_W * 2, PANEL_H * 2 + LEGEND_H), sharex=True, sharey=True,
    )
    for i, variant in enumerate(FOCUS_VARIANTS):
        _ell, ratios = mod.ratios_by_q(variant, curves, log=False)
        for j, (sky, sky_lab) in enumerate(SKY_COLS):
            ax = axes[i, j]
            _draw_bands(ax, x, rel_bands, sky, inside)
            _draw_ratio_curves(ax, x, ratios, inside, curves, curve_colors)
            ax.set_ylim(*ylim)
            ax.set_title(
                rf"{mod.VARIANT_LABELS.get(variant, variant)}, {sky_lab} $1\sigma$",
                fontsize=10,
            )
            if i == 1:
                ax.set_xlabel(r"multipole $\ell$")
            if j == 0:
                ax.set_ylabel(r"$D_\ell^{\rm variant}/D_\ell^{\rm fiducial}$")
    fig.legend(
        handles=_legend_handles(mod, curves, curve_colors),
        loc="lower center", ncol=4, frameon=False, fontsize=8.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        r"FLAMINGO L1\_m9 feedback / fiducial ratio (18 Planck bins, qfrommap)"
        "\n"
        r"$1\sigma$ envelopes about 1: grey = $C_\ell^{yy}$ only; "
        r"colours = $\sigma^2_{\mathrm{ILC},r_1\times r_2}+\sigma^2_{\mathrm{T}}$",
        fontsize=11, y=1.01,
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    return _save(fig, stem)


def plot_q5_only(mod, rel_bands: dict, stem: Path) -> Path:
    q5_curves = [5.0]
    all_ratios: dict[str, np.ndarray] = {}
    ell_ref = None
    for variant in FOCUS_VARIANTS:
        ell, ratios = mod.ratios_by_q(variant, q5_curves, log=False)
        ell_ref = ell if ell_ref is None else ell_ref
        all_ratios[variant] = ratios[5.0]
    assert ell_ref is not None
    inside = (ell_ref >= ELL_RANGE[0]) & (ell_ref <= ELL_RANGE[1])
    ylim = mod._ylim_from_ratios(all_ratios, inside)
    curve_colors = mod._color_map_for_curves(list(mod.DEFAULT_CURVES))
    x = ell_ref[inside]
    fig, axes = plt.subplots(
        1, 2, figsize=(PANEL_W * 2, PANEL_H + LEGEND_H), sharex=True, sharey=True,
    )
    for ax, variant in zip(axes, FOCUS_VARIANTS):
        _ell, ratios = mod.ratios_by_q(variant, q5_curves, log=False)
        _draw_bands(ax, x, rel_bands, "q5", inside)
        _draw_ratio_curves(ax, x, ratios, inside, q5_curves, curve_colors)
        ax.set_ylim(*ylim)
        ax.set_title(
            rf"{mod.VARIANT_LABELS.get(variant, variant)}, $q>5$ $1\sigma$",
            fontsize=10,
        )
        ax.set_xlabel(r"multipole $\ell$")
    axes[0].set_ylabel(r"$D_\ell^{\rm variant}/D_\ell^{\rm fiducial}$")
    fig.legend(
        handles=_legend_handles(mod, q5_curves, curve_colors),
        loc="lower center", ncol=5, frameon=False, fontsize=8.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        r"FLAMINGO L1\_m9 feedback / fiducial ratio (18 Planck bins, qfrommap)"
        "\n"
        r"$q>5$ $1\sigma$ envelopes about 1: grey = $C_\ell^{yy}$ only; "
        r"colours = $\sigma^2_{\mathrm{ILC},r_1\times r_2}+\sigma^2_{\mathrm{T}}$",
        fontsize=11, y=1.01,
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    return _save(fig, stem)


def main() -> None:
    _apply_style()
    mod = _load_ratio_mod()
    bl = hp.gauss_beam(np.deg2rad(FWHM_ILC / 60.0), lmax=LMAX)
    good = bl >= 1e-3
    fsky_ilc = {"fullsky": 1.0, "q5": ilc_fsky("q5")}
    print(f"ILC fsky full={fsky_ilc['fullsky']:.4f}  q5={fsky_ilc['q5']:.4f}")

    ell_f, dl_full = mod.load_bandpowers(mod.bandpower_path(mod.FIDUCIAL, "fullsky", log=False))
    _, dl_q5 = mod.load_bandpowers(mod.bandpower_path(mod.FIDUCIAL, 5.0, log=False))
    i10 = int(np.argmin(np.abs(ell_f - 10.0)))
    i335 = int(np.argmin(np.abs(ell_f - 335.5)))
    i959 = int(np.argmin(np.abs(ell_f - 959.5)))
    print(f"{'sky':<8} {'deproj':<10} {'piece':<8} {'ℓ=10':>8} {'ℓ=335':>8} {'ℓ=959':>8}")
    rel_bands = {}
    for sky, kind, dl_fid, fsky_t in (
        ("fullsky", "fullsky", dl_full, 1.0),
        ("q5", 5.0, dl_q5, FSKY_Q5_THEORY),
    ):
        sig_t = trispectrum_sigma(kind, theory_dl(kind), fsky_t)
        rel_t = sig_t * 1.0e12 / dl_fid
        _, rel_paper = mod.relative_error_band(kind, log=False)
        rel_bands[(sky, "old")] = rel_paper
        print(
            f"{sky:<8} {'(paper)':<10} {'G+T':<8} "
            f"{rel_paper[i10]:8.3f} {rel_paper[i335]:8.3f} {rel_paper[i959]:8.3f}"
        )
        print(
            f"{sky:<8} {'(keep)':<10} {'T':<8} "
            f"{rel_t[i10]:8.3f} {rel_t[i335]:8.3f} {rel_t[i959]:8.3f}"
        )
        for deproj in ("none", "cib", "moments"):
            rel = relative_ilc_plus_t(kind, deproj, dl_fid, bl, good, fsky_ilc)
            rel_bands[(sky, deproj)] = rel
            ilc_sky = "fullsky" if sky == "fullsky" else "q5"
            sig_ilc = ilc_sigma_18(
                ILC_NPZ[(deproj, ilc_sky)], fsky_ilc[ilc_sky], bl, good
            )
            rel_g = sig_ilc * 1.0e12 / dl_fid
            print(
                f"{sky:<8} {deproj:<10} {'ILC G':<8} "
                f"{rel_g[i10]:8.3f} {rel_g[i335]:8.3f} {rel_g[i959]:8.3f}"
            )
            print(
                f"{sky:<8} {deproj:<10} {'ILC+T':<8} "
                f"{rel[i10]:8.3f} {rel[i335]:8.3f} {rel[i959]:8.3f}"
            )

    plot_focus(
        mod, rel_bands,
        FIG_DIR / "l1_m9_all_feedback_ratio_vs_q_binned_18_qfrommap_ilc_errors",
    )
    plot_q5_only(
        mod, rel_bands,
        FIG_DIR / "l1_m9_fgas8_mstar_ratio_q5_ilc_deproj_errors",
    )


if __name__ == "__main__":
    main()
