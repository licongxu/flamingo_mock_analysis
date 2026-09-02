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

from flamingo_mock.powerspectra import dl_from_cl, sigma_dl_cross_binned

ROOT = Path(__file__).resolve().parents[1]
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
    ("none", "q5"): ILC / "hilc_output_homog_q5masked_r2" / "hilc_homog_q5masked_r1xr2_cl_unbinned_l1_m9.npz",
    ("cib", "q5"): (
        ILC / "hilc_output_homog_q5masked_r2_deproj_CIB"
        / "hilc_homog_q5masked_deproj_cib_r1xr2_cl_unbinned_l1_m9.npz"
    ),
    ("moments", "q5"): (
        ILC / "hilc_output_homog_q5masked_r2_deproj_CIB_CIB_dbeta_CIB_dT"
        / "hilc_homog_q5masked_deproj_cib_moments_r1xr2_cl_unbinned_l1_m9.npz"
    ),
}

# CIB maps passed through the same q>5 HILC weights (r1×r2).
FIG9_CIB = {
    "none": ILC / "hilc_output_homog_q5masked_r2" / "hilc_homog_q5masked_r1xr2_fig9_cl_l1_m9.npz",
    "cib": (
        ILC / "hilc_output_homog_q5masked_r2_deproj_CIB"
        / "hilc_homog_q5masked_deproj_cib_fig9_cl_l1_m9.npz"
    ),
    "moments": (
        ILC / "hilc_output_homog_q5masked_r2_deproj_CIB_CIB_dbeta_CIB_dT"
        / "hilc_homog_q5masked_deproj_cib_moments_fig9_cl_l1_m9.npz"
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


def _ilc_cls(npz: Path, bl: np.ndarray, good: np.ndarray):
    z = np.load(npz)
    return (
        deconv_auto(z["cl_11"], bl, good),
        deconv_auto(z["cl_22"], bl, good),
        deconv_auto(z["cl_12"], bl, good),
    )


def ilc_sigma_18(npz: Path, fsky: float, bl: np.ndarray, good: np.ndarray) -> np.ndarray:
    c11, c22, c12 = _ilc_cls(npz, bl, good)
    return sigma_dl_cross_binned(c11, c22, c12, ELL_MIN_18, ELL_MAX_18_EXCL, fsky)


def ilc_sigma_exp_18(npz: Path, fsky: float, bl: np.ndarray, good: np.ndarray) -> np.ndarray:
    """Gaussian experimental piece: (C11 C22 − C12²) / N_modes, no tSZ cosmic variance."""
    c11, c22, c12 = _ilc_cls(npz, bl, good)
    sig_tot = sigma_dl_cross_binned(c11, c22, c12, ELL_MIN_18, ELL_MAX_18_EXCL, fsky)
    sig_cv = sigma_dl_cross_binned(c12, c12, c12, ELL_MIN_18, ELL_MAX_18_EXCL, fsky)
    return np.sqrt(np.clip(sig_tot**2 - sig_cv**2, 0.0, None))


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


def _bin_mean_dl(cl: np.ndarray) -> np.ndarray:
    out = np.empty(len(ELL_MIN_18), dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(ELL_MIN_18, ELL_MAX_18_EXCL)):
        ell = np.arange(lo, hi, dtype=np.float64)
        out[i] = float(np.nanmean(dl_from_cl(ell, cl[lo:hi])))
    return out


N_MC = 10_000

CIB_BAND_LABELS = (
    ("old", r"$C_\ell^{yy}$ only", "#bdbdbd"),
    ("none", r"no deproj., joint fit", "#9ecae1"),
    ("cib", r"CIB deproj., joint fit", "#a1d99b"),
    ("moments", r"CIB+$\delta\beta$+$\delta T$, joint fit", "#fc9272"),
)


def _sym_psd(m: np.ndarray) -> np.ndarray:
    a = 0.5 * (m + m.T)
    w, v = np.linalg.eigh(a)
    w = np.clip(w, 0.0, None)
    w = np.maximum(w, 1e-12 * max(float(w.max()), 1e-60))
    return (v * w) @ v.T


def _chol_factor(m: np.ndarray) -> np.ndarray:
    a = _sym_psd(m)
    w, v = np.linalg.eigh(a)
    return v * np.sqrt(np.clip(w, 0.0, None))


def _ratio_quantiles(m: np.ndarray, t: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    d_d = rng.standard_normal((N_MC, t.size)) @ _chol_factor(m).T
    ratio = 1.0 + d_d / t
    return np.percentile(ratio, 16.0, axis=0), np.percentile(ratio, 84.0, axis=0)


def m_stat_q5(deproj: str, fsky_ilc: float, bl, good) -> np.ndarray:
    """HILC r1×r2 Gaussian (diagonal) + trispectrum (full 18×18)."""
    sig_ilc = ilc_sigma_18(ILC_NPZ[(deproj, "q5")], fsky_ilc, bl, good)
    cov = np.asarray(
        np.load(COV / "cov_full_L1_m9_customgnfw_bestfit_masked_qgt5_Dl_yy_binned_18.npy"),
        dtype=np.float64,
    )
    sig_g = knox_sigma(theory_dl(5.0), FSKY_Q5_THEORY)
    m_t = cov.copy()
    np.fill_diagonal(m_t, np.clip(np.diag(m_t) - sig_g**2, 0.0, None))
    return np.diag(sig_ilc**2) + m_t


def _cib_vec(deproj: str, bl, good) -> np.ndarray:
    return _bin_mean_dl(deconv_auto(np.load(FIG9_CIB[deproj])["cl_cib_w"], bl, good))


def joint_fisher(t: np.ndarray, f: np.ndarray, m: np.ndarray) -> dict:
    """Linear Gaussian d = X A + n, n~N(0,M), X = (t | T) with T = CIB only.

    Σ_A = (X^T M^{-1} X)^{-1} is the joint posterior covariance.
    σ(A_tSZ) marginalized over FG is sqrt((Σ_A)_{tt}); σ(A_CIB) is sqrt((Σ_A)_{AA}).
    """
    m = _sym_psd(m)
    x = np.column_stack((np.asarray(t, dtype=np.float64), np.asarray(f, dtype=np.float64)))
    fisher = x.T @ np.linalg.solve(m, x)
    cov_a = np.linalg.inv(_sym_psd(fisher))
    tt, ff, tf = float(fisher[0, 0]), float(fisher[1, 1]), float(fisher[0, 1])
    cos = tf / np.sqrt(tt * ff)
    sin2 = max(1.0 - cos**2, 1e-12)
    sig_yy = float(np.sqrt(cov_a[0, 0]))
    sig_a = float(np.sqrt(cov_a[1, 1]))
    return {
        "M": m,
        "f": f,
        "cov_A": cov_a,
        "cos": cos,
        "sig_A": sig_a,
        "sig_yy": sig_yy,
        "inflate": 1.0 / np.sqrt(sin2),
        "rel_cib": sig_a * f / t,
    }


def _deproj_joint(deproj: str, t: np.ndarray, fsky_ilc: float, bl, good) -> dict:
    f = _cib_vec(deproj, bl, good)
    return joint_fisher(t, f, m_stat_q5(deproj, fsky_ilc, bl, good))


def _joint_model_lohi(j: dict, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-sigma posterior band for A_tSZ + A_CIB f/t, including covariance."""
    response = np.column_stack((np.ones_like(t), j["f"] / t))
    sigma = np.sqrt(np.clip(np.diag(response @ j["cov_A"] @ response.T), 0.0, None))
    return 1.0 - sigma, 1.0 + sigma


def _cib_marg_lohi(j: dict, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Marginal CIB-component scale; not independent of the tSZ amplitude."""
    shift = np.abs(j["sig_A"] * j["f"] / t)
    return 1.0 - shift, 1.0 + shift


def _q5_ratio_layout(mod):
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
    return q5_curves, ell_ref, inside, ylim, curve_colors, ell_ref[inside]


def _band_pm(ax, x, s: np.ndarray, color: str, inside) -> None:
    ax.fill_between(
        x, 1.0 - s[inside], 1.0 + s[inside], facecolor=color, alpha=0.22, lw=0, zorder=1,
    )
    ax.plot(x, 1.0 - s[inside], color=color, lw=1.6, zorder=2)
    ax.plot(x, 1.0 + s[inside], color=color, lw=1.6, zorder=2)


def _band_lohi(ax, x, lo, hi, color: str, inside) -> None:
    ax.fill_between(x, lo[inside], hi[inside], facecolor=color, alpha=0.22, lw=0, zorder=1)
    ax.plot(x, lo[inside], color=color, lw=1.6, zorder=2)
    ax.plot(x, hi[inside], color=color, lw=1.6, zorder=2)


def plot_q5_cib_amp(
    mod, rel_yy: np.ndarray, dl_fid_1e12: np.ndarray, fsky_ilc: float, bl, good, stem: Path
) -> Path:
    """Paper C_ell^yy vs joint tSZ+CIB posterior model band, per deproj."""
    q5_curves, _ell, inside, ylim, curve_colors, x = _q5_ratio_layout(mod)
    t = np.asarray(dl_fid_1e12, dtype=np.float64) * 1.0e-12
    lohi = {}
    for key in ("none", "cib", "moments"):
        j = _deproj_joint(key, t, fsky_ilc, bl, good)
        lo, hi = _joint_model_lohi(j, t)
        lohi[key] = (lo, hi)
        print(
            f"joint fit {key:<10} cos={j['cos']:.3f}  sig_A={j['sig_A']:.4f}  "
            f"A_yy x{j['inflate']:.2f}  68% |R-1| ℓ=10,959 "
            f"{0.5*(hi[0]-lo[0]):.3f} {0.5*(hi[-1]-lo[-1]):.3f}"
        )
    fig, axes = plt.subplots(
        1, 2, figsize=(PANEL_W * 2, PANEL_H + LEGEND_H), sharex=True, sharey=True,
    )
    for ax, variant in zip(axes, FOCUS_VARIANTS):
        _e, ratios = mod.ratios_by_q(variant, q5_curves, log=False)
        _band_pm(ax, x, rel_yy, "#bdbdbd", inside)
        for key, _lab, color in reversed(CIB_BAND_LABELS):
            if key == "old":
                continue
            lo, hi = lohi[key]
            _band_lohi(ax, x, lo, hi, color, inside)
        _draw_ratio_curves(ax, x, ratios, inside, q5_curves, curve_colors)
        ax.set_ylim(*ylim)
        ax.set_title(
            rf"{mod.VARIANT_LABELS.get(variant, variant)}, $q>5$ 68\%",
            fontsize=10,
        )
        ax.set_xlabel(r"multipole $\ell$")
    axes[0].set_ylabel(r"$D_\ell^{\rm variant}/D_\ell^{\rm fiducial}$")
    handles = [
        Line2D(
            [0], [0], color=curve_colors[5.0], marker="o", lw=1.6, markersize=4,
            label=r"$q>5$",
        )
    ]
    handles += [Line2D([0], [0], color=c, lw=2.2, label=lab) for _k, lab, c in CIB_BAND_LABELS]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        r"FLAMINGO L1\_m9 $q>5$ (18 Planck bins, qfrommap). "
        r"HILC homog $r_1\times r_2$, TopHatHarmonic $10'$. CIB only."
        "\n"
        r"Grey: $C_\ell^{yy}$ G+$T$. Colours: joint 68\% posterior model bands from "
        r"$X(\mathcal F^{-1})X^{T}$, including $A_{yy}$--$A_{\rm CIB}$ covariance.",
        fontsize=10, y=1.01,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.96))
    return _save(fig, stem)


def plot_q5_linear_fg_amp(
    mod, dl_fid_1e12: np.ndarray, fsky_ilc: float, bl, good, stem: Path
) -> Path:
    r"""Linear model: d = A_tSZ C^{tSZ} + A_CIB C^{CIB} (fixed templates).

    Data scatter, marginal A_CIB component, the joint fitted model, and the
    marginal A_tSZ amplitude. The two fitted amplitudes retain their covariance.
    """
    q5_curves, _ell, inside, ylim, curve_colors, x = _q5_ratio_layout(mod)
    t = np.asarray(dl_fid_1e12, dtype=np.float64) * 1.0e-12
    j = _deproj_joint("none", t, fsky_ilc, bl, good)
    rng = np.random.default_rng(1)
    eps = rng.standard_normal((N_MC, t.size)) @ _chol_factor(j["M"]).T
    ratio_cv = 1.0 + eps / t
    lo_cv = np.percentile(ratio_cv, 16.0, axis=0)
    hi_cv = np.percentile(ratio_cv, 84.0, axis=0)
    lo_tot, hi_tot = _joint_model_lohi(j, t)
    lo_cib, hi_cib = _cib_marg_lohi(j, t)
    ones = np.ones_like(t)
    sig_tsz = float(j["sig_yy"])
    lo_tsz, hi_tsz = 1.0 - sig_tsz * ones, 1.0 + sig_tsz * ones
    print(
        f"linear FG  data scatter 68% |R-1| ℓ=10,959 "
        f"{0.5*(hi_cv[0]-lo_cv[0]):.3f} {0.5*(hi_cv[-1]-lo_cv[-1]):.3f}"
    )
    print(
        f"linear FG  marg A   68% |R-1| ℓ=10,959 "
        f"{j['sig_A'] * j['f'][0] / t[0]:.3f} {j['sig_A'] * j['f'][-1] / t[-1]:.3f}  "
        f"cos={j['cos']:.3f}  sig_A={j['sig_A']:.4f}"
    )
    print(
        f"linear FG  joint fit 68% |R-1| ℓ=10,959 "
        f"{0.5*(hi_tot[0]-lo_tot[0]):.3f} {0.5*(hi_tot[-1]-lo_tot[-1]):.3f}"
    )
    print(f"linear FG  σ(A_tSZ) marg={sig_tsz:.4f}")
    fig, axes = plt.subplots(
        1, 2, figsize=(PANEL_W * 2, PANEL_H + LEGEND_H + 0.15),
        sharex=True, sharey=True,
    )
    for ax, variant in zip(axes, FOCUS_VARIANTS):
        _e, ratios = mod.ratios_by_q(variant, q5_curves, log=False)
        ax.fill_between(
            x, lo_tot[inside], hi_tot[inside],
            facecolor="#6a51a3", alpha=0.10, lw=0, zorder=1,
        )
        ax.plot(x, lo_tot[inside], color="#6a51a3", lw=1.4, zorder=3, alpha=0.70)
        ax.plot(x, hi_tot[inside], color="#6a51a3", lw=1.4, zorder=3, alpha=0.70)
        ax.fill_between(
            x, lo_cv[inside], hi_cv[inside],
            facecolor="#9ecae1", alpha=0.22, lw=0, zorder=2,
        )
        ax.plot(x, lo_cv[inside], color="#3182bd", lw=1.5, zorder=3)
        ax.plot(x, hi_cv[inside], color="#3182bd", lw=1.5, zorder=3)
        ax.fill_between(
            x, lo_cib[inside], hi_cib[inside],
            facecolor="#fd8d3c", alpha=0.18, lw=0, zorder=4,
        )
        ax.plot(x, lo_cib[inside], color="#e6550d", lw=1.7, zorder=5)
        ax.plot(x, hi_cib[inside], color="#e6550d", lw=1.7, zorder=5)
        ax.fill_between(
            x, lo_tsz[inside], hi_tsz[inside],
            facecolor="#74c476", alpha=0.20, lw=0, zorder=6,
        )
        ax.plot(x, lo_tsz[inside], color="#238b45", lw=1.7, zorder=7)
        ax.plot(x, hi_tsz[inside], color="#238b45", lw=1.7, zorder=7)
        _draw_ratio_curves(ax, x, ratios, inside, q5_curves, curve_colors)
        ax.set_ylim(*ylim)
        ax.set_title(
            rf"{mod.VARIANT_LABELS.get(variant, variant)}, $q>5$ 68\%",
            fontsize=10,
        )
        ax.set_xlabel(r"multipole $\ell$")
    axes[0].set_ylabel(r"$D_\ell^{\rm variant}/D_\ell^{\rm fiducial}$")
    handles = [
        Line2D(
            [0], [0], color=curve_colors[5.0], marker="o", lw=1.6, markersize=4,
            label=r"$q>5$",
        ),
        Line2D(
            [0], [0], color="#3182bd", lw=2.2,
            label=r"HILC bandpower scatter",
        ),
        Line2D(
            [0], [0], color="#e6550d", lw=2.2,
            label=r"marg.\ $A_{\mathrm{CIB}}$ component",
        ),
        Line2D(
            [0], [0], color="#238b45", lw=2.2,
            label=r"$\sigma(A_{\mathrm{tSZ}})$, marg.\ FG",
        ),
        Line2D(
            [0], [0], color="#6a51a3", lw=2.2,
            label=r"joint tSZ+CIB model",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        r"$d_\ell = A_{\mathrm{tSZ}}\,C_\ell^{\mathrm{tSZ}}"
        r" + A_{\mathrm{CIB}}\,C_\ell^{\mathrm{CIB}}$"
        r" \ (fixed templates, free amplitudes; CIB is the only FG)"
        "\n"
        r"FLAMINGO L1\_m9 $q>5$, no-deproj.\ HILC. "
        r"Orange shows the CIB component only; it is correlated with tSZ. "
        r"Purple uses the full $\Sigma_A$; green is $1\pm\sqrt{(\Sigma_A)_{tt}}$.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))
    return _save(fig, stem)


def plot_q5_nodeproj_vs_cib_fg_marg(
    mod, dl_fid_1e12: np.ndarray, fsky_ilc: float, bl, good, stem: Path
) -> Path:
    """Joint model and marginal CIB-component bands, no deproj vs CIB deproj."""
    q5_curves, _ell, inside, ylim, curve_colors, x = _q5_ratio_layout(mod)
    t = np.asarray(dl_fid_1e12, dtype=np.float64) * 1.0e-12
    jn = _deproj_joint("none", t, fsky_ilc, bl, good)
    jc = _deproj_joint("cib", t, fsky_ilc, bl, good)
    lo_n, hi_n = _joint_model_lohi(jn, t)
    lo_c, hi_c = _joint_model_lohi(jc, t)
    lo_cn, hi_cn = _cib_marg_lohi(jn, t)
    lo_cc, hi_cc = _cib_marg_lohi(jc, t)
    print(
        f"none joint fit 68% |R-1| ℓ=10,959 "
        f"{0.5*(hi_n[0]-lo_n[0]):.3f} {0.5*(hi_n[-1]-lo_n[-1]):.3f}  "
        f"sig_A={jn['sig_A']:.4f}  A_yy x{jn['inflate']:.2f}"
    )
    print(
        f"CIB  joint fit 68% |R-1| ℓ=10,959 "
        f"{0.5*(hi_c[0]-lo_c[0]):.3f} {0.5*(hi_c[-1]-lo_c[-1]):.3f}  "
        f"sig_A={jc['sig_A']:.4f}  A_yy x{jc['inflate']:.2f}"
    )
    print(
        f"none CIB marg σ_A F/D ℓ=10,959 "
        f"{jn['sig_A'] * jn['f'][0] / t[0]:.3f} {jn['sig_A'] * jn['f'][-1] / t[-1]:.3f}"
    )
    print(
        f"CIB  CIB marg σ_A F/D ℓ=10,959 "
        f"{jc['sig_A'] * jc['f'][0] / t[0]:.3f} {jc['sig_A'] * jc['f'][-1] / t[-1]:.3f}"
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(PANEL_W * 2, PANEL_H + LEGEND_H), sharex=True, sharey=True,
    )
    for ax, variant in zip(axes, FOCUS_VARIANTS):
        _e, ratios = mod.ratios_by_q(variant, q5_curves, log=False)
        ax.fill_between(
            x, lo_c[inside], hi_c[inside],
            facecolor="#31a354", alpha=0.08, lw=0, zorder=1,
        )
        ax.plot(x, lo_c[inside], color="#238b45", lw=1.2, zorder=3, alpha=0.50)
        ax.plot(x, hi_c[inside], color="#238b45", lw=1.2, zorder=3, alpha=0.50)
        ax.fill_between(
            x, lo_n[inside], hi_n[inside],
            facecolor="#3182bd", alpha=0.10, lw=0, zorder=2,
        )
        ax.plot(x, lo_n[inside], color="#08519c", lw=1.2, zorder=3, alpha=0.50)
        ax.plot(x, hi_n[inside], color="#08519c", lw=1.2, zorder=3, alpha=0.50)
        ax.fill_between(
            x, lo_cc[inside], hi_cc[inside],
            facecolor="#74c476", alpha=0.16, lw=0, zorder=4,
        )
        ax.plot(x, lo_cc[inside], color="#006d2c", lw=1.6, zorder=5)
        ax.plot(x, hi_cc[inside], color="#006d2c", lw=1.6, zorder=5)
        ax.fill_between(
            x, lo_cn[inside], hi_cn[inside],
            facecolor="#fd8d3c", alpha=0.18, lw=0, zorder=6,
        )
        ax.plot(x, lo_cn[inside], color="#e6550d", lw=1.6, zorder=7)
        ax.plot(x, hi_cn[inside], color="#e6550d", lw=1.6, zorder=7)
        _draw_ratio_curves(ax, x, ratios, inside, q5_curves, curve_colors)
        ax.set_ylim(*ylim)
        ax.set_title(
            rf"{mod.VARIANT_LABELS.get(variant, variant)}, $q>5$ 68\%",
            fontsize=10,
        )
        ax.set_xlabel(r"multipole $\ell$")
    axes[0].set_ylabel(r"$D_\ell^{\rm variant}/D_\ell^{\rm fiducial}$")
    handles = [
        Line2D(
            [0], [0], color=curve_colors[5.0], marker="o", lw=1.6, markersize=4,
            label=r"$q>5$",
        ),
        Line2D([0], [0], color="#08519c", lw=2.2, label=r"no deproj., joint model"),
        Line2D([0], [0], color="#238b45", lw=2.2, label=r"CIB deproj., joint model"),
        Line2D([0], [0], color="#e6550d", lw=2.2, label=r"no deproj., CIB component"),
        Line2D([0], [0], color="#006d2c", lw=2.2, label=r"CIB deproj., CIB component"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.0,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        r"FLAMINGO L1\_m9 $q>5$. Faint: joint tSZ+CIB posterior model. "
        r"Solid: marginal CIB component (correlated with tSZ), no deproj.\ vs CIB deproj.",
        fontsize=10, y=1.01,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.96))
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
            rel_exp = (
                ilc_sigma_exp_18(
                    ILC_NPZ[(deproj, ilc_sky)], fsky_ilc[ilc_sky], bl, good
                )
                * 1.0e12
                / dl_fid
            )
            print(
                f"{sky:<8} {deproj:<10} {'ILC exp':<8} "
                f"{rel_exp[i10]:8.3f} {rel_exp[i335]:8.3f} {rel_exp[i959]:8.3f}"
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
    plot_q5_cib_amp(
        mod, rel_bands[("q5", "old")], dl_q5, fsky_ilc["q5"], bl, good,
        FIG_DIR / "l1_m9_fgas8_mstar_ratio_q5_cib_amplitude_marg",
    )
    plot_q5_linear_fg_amp(
        mod, dl_q5, fsky_ilc["q5"], bl, good,
        FIG_DIR / "l1_m9_fgas8_mstar_ratio_q5_linear_cib_amp_marg",
    )
    plot_q5_nodeproj_vs_cib_fg_marg(
        mod, dl_q5, fsky_ilc["q5"], bl, good,
        FIG_DIR / "l1_m9_fgas8_mstar_ratio_q5_nodeproj_vs_cib_fg_marg",
    )


if __name__ == "__main__":
    main()
