#!/usr/bin/env python3
"""L1_m9 HILC r1×r2 recovered y with CIB/CMB residuals.

One combined panel per sky (truth + recovered + residuals), plus CIB
residual full sky vs q>5 masked. 10^12 D_ℓ, 18 Planck bins.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from pub_style import apply_pub_style, savefig

from flamingo_mock.powerspectra import sigma_dl_cross_binned
from hilc_prescriptions import (
    ALL_DEPROJ,
    ALL_RUNS,
    FREQS_ILC,
    ILC,
    LABELS,
    SYNTH,
    catalogue_path,
)

ROOT = Path(__file__).resolve().parents[1]
PS = ILC / "ps_namaster"
DATA = ROOT / "data"
FIG_DIR = ROOT / "figures" / "l1_m9"
PAPER_FIGS = Path("/home/lxu/scratch/tsz_cnc_paper_plots/masking_giant_paper/figs")
COV_FULLSKY = Path(
    "/scratch/scratch-lxu/flamingo_repo/chains/l1_m9_qfrommap_asz_covariance"
    "/final/covariance/cov_trispectrum_L1_m9_customgnfw_qfrommap_fullsky_Dl_yy_binned_18.npy"
)
COV_Q5 = DATA / "cov_trispectrum_L1_m9_ilc_q5_Dl_yy_binned_18.npy"
STEM_FULL = "l1_m9_hilc_recovered_residuals_fullsky"
STEM_MASK = "l1_m9_hilc_recovered_residuals_q5masked"
STEM_DEPROJ = "l1_m9_hilc_recovered_residuals_deproj"
STEM_CIB = "l1_m9_hilc_cib_residual_full_vs_masked"
STEM_CIB_PRESC = "l1_m9_hilc_cib_residual_nodeproj_prescriptions"
STEM_CIB_PRESC_ONE = "l1_m9_hilc_cib_residual_nodeproj_prescriptions_full_vs_masked"
STEM_CIB_FREQ = "l1_m9_cib_auto_by_freq_prescriptions"
STEM_CROSS_FULL = "l1_m9_hilc_cib_tsz_cross_fullsky"
STEM_CROSS_MASK = "l1_m9_hilc_cib_tsz_cross_q5masked"
STEM_CROSS_PAIR = "l1_m9_hilc_cib_tsz_cross_residuals"
STEM_TSZXCIB = "l1_m9_hilc_tsz_cib_residual_full_vs_masked"
FREQS_PLOT = (100, 143, 217, 353, 545, 857)
FREQ_IDX = {int(f): i for i, f in enumerate(FREQS_ILC)}
SCALE = 1.0e12
ELL_MIN = np.array(
    [9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835]
)
ELL_MAX_EXCL = np.array(
    [13, 17, 22, 28, 36, 47, 61, 79, 103, 134, 174, 225, 293, 381, 495, 643, 836, 1086]
)
COLORS = {
    "nodeproj": "#0072B2",
    "deproj_cib": "#D55E00",
    "deproj_cib_dbeta": "#009E73",
    "deproj_cib_dbeta_cmb": "#CC79A7",
    "deproj_moments": "#E69F00",
}
MARKERS = {
    "nodeproj": "o",
    "deproj_cib": "s",
    "deproj_cib_dbeta": "^",
    "deproj_cib_dbeta_cmb": "v",
    "deproj_moments": "D",
}
RUN_COLORS = {
    "L1_m9": "k",
    "fgas-8sigma": "#0072B2",
    "Mstar-1sigma": "#D55E00",
    "LS8": "#009E73",
}
RUN_MARKERS = {
    "L1_m9": "o",
    "fgas-8sigma": "^",
    "Mstar-1sigma": "s",
    "LS8": "D",
}


def _n_q5(run: str = "L1_m9") -> int:
    p = SYNTH / f"szifi_homog/{run}/catalogues/szifi_jax_scimmf_splitA_immf_q5.npz"
    if not p.is_file():
        p = catalogue_path(run)
    if not p.is_file():
        p = SYNTH / f"szifi_homog/{run}/catalogues/szifi_jax_splitA_immf_q5.npz"
    return int(np.load(p)["q_opt"].size)


def _p18(deproj: str, kind: str, run: str = "L1_m9"):
    z = np.load(PS / f"{run}_{deproj}_{kind}_p18.npz", allow_pickle=True)
    return np.asarray(z["ell"], dtype=np.float64), np.asarray(z["dl"], dtype=np.float64)


def _truth(kind: str, run: str = "L1_m9"):
    z = np.load(PS / f"{run}_truth_{kind}.npz")
    return np.asarray(z["ell"], dtype=np.float64), np.asarray(z["dl"], dtype=np.float64)


def _resid(run: str, deproj: str, kind: str):
    z = np.load(PS / f"{run}_{deproj}_{kind}_residuals.npz")
    return np.asarray(z["dl_cib"]), np.asarray(z["dl_cmb"])


def _tsz_cib(run: str, deproj: str, kind: str) -> np.ndarray:
    z = np.load(PS / f"{run}_{deproj}_{kind}_residuals.npz")
    return np.asarray(z["dl_tsz_cib"])


def sigma_g(deproj: str, kind: str, run: str = "L1_m9") -> np.ndarray:
    z = np.load(PS / f"{run}_{deproj}_{kind}_splitcross.npz")
    return sigma_dl_cross_binned(
        np.asarray(z["cl_11"]),
        np.asarray(z["cl_22"]),
        np.asarray(z["cl_12"]),
        ELL_MIN,
        ELL_MAX_EXCL,
        float(z["fsky_eff"]),
    )


def sigma_t(kind: str) -> np.ndarray:
    path = COV_FULLSKY if kind == "total" else COV_Q5
    return np.sqrt(np.clip(np.diag(np.asarray(np.load(path), dtype=np.float64)), 0.0, None))


def sigma_gt(deproj: str, kind: str, run: str = "L1_m9") -> np.ndarray:
    # sigma_t is the L1_m9 one-halo trispectrum used in the fiducial figure.
    return np.sqrt(sigma_g(deproj, kind, run) ** 2 + sigma_t(kind) ** 2)


def _print_table() -> None:
    i10, i335, i959 = 0, 13, 17
    print(
        f"{'sky':<8} {'deproj':<22} {'piece':<10} "
        f"{'ℓ=10':>8} {'ℓ=335':>8} {'ℓ=959':>8}"
    )
    for kind, sky in (("total", "fullsky"), ("masked", "q5")):
        _, t = _truth(kind)
        for d in ALL_DEPROJ:
            cib, cmb = _resid("L1_m9", d.key, kind)
            rec = _p18(d.key, kind)[1]
            x = _tsz_cib("L1_m9", d.key, kind)
            for lab, arr in (
                ("CIB/T", cib / t),
                ("tSZxCIB/T", x / t),
                ("CMB/T", cmb / t),
                ("rec/T", rec / t),
            ):
                print(
                    f"{sky:<8} {d.key:<22} {lab:<10} "
                    f"{arr[i10]:8.3f} {arr[i335]:8.3f} {arr[i959]:8.3f}"
                )


def _errbar(ax, x, y, sig, *, color, marker, zorder=5):
    y = np.asarray(y, dtype=np.float64)
    sig = np.asarray(sig, dtype=np.float64)
    lo = np.minimum(sig, np.maximum(np.abs(y), 1e-40) * 0.99)
    ax.errorbar(
        x, y, yerr=[lo, sig], fmt=marker, color=color, ms=4.0,
        elinewidth=0.9, capsize=2.0, zorder=zorder, lw=0,
    )


def _save(fig, stem: str) -> None:
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    for p in savefig(fig, stem, fig_dir=FIG_DIR):
        dest = PAPER_FIGS / p.name
        shutil.copy2(p, dest)
        print("copied", dest, flush=True)
    plt.close(fig)


def _scheme_line_handles():
    return [
        Line2D(
            [0], [0], color=COLORS[d.key], marker=MARKERS[d.key],
            ms=5, lw=1.6, label=d.label,
        )
        for d in ALL_DEPROJ
    ]


def _style_handles():
    return [
        Line2D([0], [0], color="0.25", ls="-", lw=1.8, label=r"CIB residual"),
        Line2D([0], [0], color="0.25", ls="--", lw=1.7, label=r"tSZ$\times$CIB"),
        Line2D(
            [0], [0], color="0.25", ls=":", lw=2.0,
            label=r"CIB + tSZ$\times$CIB",
        ),
    ]


def _row_major_handles(row1, row2):
    """Order handles so ``ncol=len(row1)`` displays row1 then row2."""
    pad = Line2D([0], [0], color="none", lw=0, label=" ")
    row2 = list(row2) + [pad] * (len(row1) - len(row2))
    out = []
    for a, b in zip(row1, row2):
        out.extend((a, b))
    return out


def _cross_legend_handles():
    return _row_major_handles(_scheme_line_handles(), _style_handles())


def _outside_legend(fig, handles, *, ncol: int, fontsize: float = 8) -> None:
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0.2,
    )


def _draw_cib_tsz_cross(ax, kind: str) -> None:
    ell, t = _truth(kind)
    ax.axhline(0.0, color="0.55", lw=0.8, zorder=1)
    for d in ALL_DEPROJ:
        cib, _ = _resid("L1_m9", d.key, kind)
        x = _tsz_cib("L1_m9", d.key, kind)
        color = COLORS[d.key]
        ax.semilogx(
            ell, cib / t, color=color, ls="-", marker=MARKERS[d.key],
            ms=4.0, lw=1.6, zorder=3,
        )
        ax.semilogx(ell, x / t, color=color, ls="--", lw=1.6, zorder=3)
        ax.semilogx(ell, (cib + x) / t, color=color, ls=":", lw=2.0, zorder=4)
    ax.set_xlim(10.0, 959.5)
    ax.set_ylim(-0.95, 1.22)
    ax.set_xlabel(r"$\ell$")
    ax.grid(False)


def build_cib_tsz_cross(kind: str, title: str) -> plt.Figure:
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(5.4, 5.0), layout="constrained")
    _draw_cib_tsz_cross(ax, kind)
    ax.set_ylabel(r"$D_\ell / D_\ell^{y,\mathrm{true}}$")
    ax.set_title(title, fontsize=12)
    _outside_legend(fig, _cross_legend_handles(), ncol=5, fontsize=8)
    return fig


def build_cib_tsz_cross_pair() -> plt.Figure:
    apply_pub_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(10.8, 5.2), sharey=True, layout="constrained"
    )
    titles = ("Total", r"$q>5$ masked")
    for ax, kind, title in zip(axes, ("total", "masked"), titles):
        _draw_cib_tsz_cross(ax, kind)
        ax.set_title(title, fontsize=12)
    axes[0].set_ylabel(r"$D_\ell / D_\ell^{y,\mathrm{true}}$")
    _outside_legend(fig, _cross_legend_handles(), ncol=5, fontsize=9)
    return fig


def build_tsz_cib_compare() -> plt.Figure:
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(5.4, 5.0), layout="constrained")
    ell, t_full = _truth("total")
    _, t_mask = _truth("masked")
    ax.axhline(0.0, color="0.55", lw=0.8, zorder=1)
    print(f"{'deproj':<22} {'ℓ=10':>8} {'ℓ=335':>8} {'ℓ=959':>8}  (tSZ×CIB / true)")
    for d in ALL_DEPROJ:
        xf = _tsz_cib("L1_m9", d.key, "total") / t_full
        xm = _tsz_cib("L1_m9", d.key, "masked") / t_mask
        color = COLORS[d.key]
        ax.semilogx(
            ell, xf, color=color, ls="-", marker=MARKERS[d.key],
            ms=4.0, lw=1.6, label=d.label,
        )
        ax.semilogx(ell, xm, color=color, ls="--", lw=1.5)
        print(f"{d.key:<22} {xf[0]:8.3f} {xf[13]:8.3f} {xf[17]:8.3f}  full")
        print(f"{'':<22} {xm[0]:8.3f} {xm[13]:8.3f} {xm[17]:8.3f}  q5")
    ax.set_xlim(10.0, 959.5)
    ax.set_ylim(-0.95, 0.22)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell^{\mathrm{tSZ}\times\mathrm{CIB}} / D_\ell^{y,\mathrm{true}}$")
    ax.set_title(r"tSZ$\times$CIB residual, full sky vs $q>5$ masked")
    h, _ = ax.get_legend_handles_labels()
    extra = [
        Line2D([0], [0], color="0.3", ls="-", lw=1.5, label="full sky"),
        Line2D([0], [0], color="0.3", ls="--", lw=1.4, label=r"$q>5$ masked"),
    ]
    _outside_legend(
        fig, _row_major_handles(h, extra), ncol=len(h), fontsize=8,
    )
    ax.grid(False)
    return fig


def _scheme_legend():
    return [
        Line2D([0], [0], color="k", lw=1.8, label=r"input truth $y$"),
    ] + [
        Line2D(
            [0], [0], color=COLORS[d.key], marker=MARKERS[d.key], lw=0, ms=5,
            label=d.label,
        )
        for d in ALL_DEPROJ
    ] + [
        Line2D([0], [0], color="0.3", ls="-", lw=1.5, label=r"CIB residual"),
        Line2D([0], [0], color="0.3", ls="--", lw=1.3, label=r"CMB residual"),
    ]


def _draw_combined(ax, kind: str, title: str, *, ylabel: bool, run: str = "L1_m9") -> None:
    ell, t = _truth(kind, run)
    n = len(ALL_DEPROJ)
    ax.loglog(ell, t * SCALE, color="k", lw=1.8, zorder=4)
    for i, d in enumerate(ALL_DEPROJ):
        color, mk = COLORS[d.key], MARKERS[d.key]
        x = ell * (1.05 ** (i - 0.5 * (n - 1)))
        _, rec = _p18(d.key, kind, run)
        _errbar(ax, x, rec * SCALE, sigma_gt(d.key, kind, run) * SCALE, color=color, marker=mk)
        cib, cmb = _resid(run, d.key, kind)
        ax.loglog(ell, np.abs(cib) * SCALE, color=color, ls="-", lw=1.4, zorder=3)
        ax.loglog(ell, np.abs(cmb) * SCALE, color=color, ls="--", lw=1.2, zorder=2)
    ax.set_xlim(10.0, 959.5)
    ax.set_ylim(3.0e-7, 40.0)
    ax.set_xlabel(r"$\ell$")
    if ylabel:
        ax.set_ylabel(r"$10^{12}D_\ell$")
    ax.set_title(title, fontsize=12)
    ax.legend(
        handles=_scheme_legend(), frameon=False, fontsize=7,
        loc="upper left", ncol=1, borderaxespad=0.3,
    )
    ax.grid(False)


def build_combined(kind: str, title: str, run: str = "L1_m9") -> plt.Figure:
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    _draw_combined(ax, kind, title, ylabel=True, run=run)
    fig.tight_layout()
    return fig


def save_feedback_recovered() -> None:
    """Same two panels as the fiducial figure. Analysis copies only."""
    for run in ("fgas-8sigma", "Mstar-1sigma"):
        nq = _n_q5(run)
        panels = (
            (build_combined("total", "Total", run=run), f"{run}_hilc_recovered_residuals_fullsky"),
            (
                build_combined("masked", rf"$q>5$ masked ($N={nq}$)", run=run),
                f"{run}_hilc_recovered_residuals_q5masked",
            ),
        )
        for fig, stem in panels:
            savefig(fig, stem, fig_dir=FIG_DIR)
            plt.close(fig)


def build_deproj_pair(nq: int) -> plt.Figure:
    apply_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)
    _draw_combined(axes[0], "total", "Total", ylabel=True)
    _draw_combined(axes[1], "masked", rf"$q>5$ masked ($N={nq}$)", ylabel=False)
    fig.tight_layout()
    return fig


def build_cib_compare() -> plt.Figure:
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    print(f"{'deproj':<22} {'ℓ=10':>8} {'ℓ=335':>8} {'ℓ=959':>8}  (masked/full CIB)")
    ell, _ = _p18("nodeproj", "total")
    for d in ALL_DEPROJ:
        cib_f = _resid("L1_m9", d.key, "total")[0]
        cib_m = _resid("L1_m9", d.key, "masked")[0]
        color = COLORS[d.key]
        ax.loglog(ell, np.abs(cib_f) * SCALE, color=color, ls="-", lw=1.6, label=d.label)
        ax.loglog(ell, np.abs(cib_m) * SCALE, color=color, ls="--", lw=1.4)
        r = np.abs(cib_m) / np.maximum(np.abs(cib_f), 1e-40)
        print(f"{d.key:<22} {r[0]:8.3f} {r[13]:8.3f} {r[17]:8.3f}")
    ax.set_xlim(10.0, 959.5)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$10^{12}|D_\ell^{\mathrm{CIB}}|$")
    ax.set_title(r"CIB residual, full sky vs $q>5$ masked")
    extra = [
        Line2D([0], [0], color="0.3", ls="-", lw=1.5, label="full sky"),
        Line2D([0], [0], color="0.3", ls="--", lw=1.4, label=r"$q>5$ masked"),
    ]
    h, lab = ax.get_legend_handles_labels()
    ax.legend(
        h + extra, lab + [e.get_label() for e in extra],
        frameon=False, fontsize=7.5, loc="upper left",
    )
    ax.grid(False)
    fig.tight_layout()
    return fig


def build_cib_feedback_overlay(deprojs=ALL_DEPROJ) -> plt.Figure:
    """Fiducial, fgas-8sigma, and Mstar-1sigma on one axes.

    Colour is the feedback run, marker the deprojection, solid the full sky,
    and dashed the q>5 masked sky.
    """
    apply_pub_style()
    runs = ("L1_m9", "fgas-8sigma", "Mstar-1sigma")
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ell, _ = _p18("nodeproj", "total")
    for run in runs:
        for d in deprojs:
            cib_f = np.abs(_resid(run, d.key, "total")[0])
            cib_m = np.abs(_resid(run, d.key, "masked")[0])
            kw = dict(color=RUN_COLORS[run], marker=MARKERS[d.key], ms=3.6, lw=1.25)
            ax.loglog(ell, cib_f * SCALE, ls="-", **kw)
            ax.loglog(ell, cib_m * SCALE, ls="--", **kw)
    ax.set_xlim(10.0, 959.5)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$10^{12}|D_\ell^{\mathrm{CIB}}|$")
    run_h = [
        Line2D([0], [0], color=RUN_COLORS[name], lw=1.6, label=LABELS[name])
        for name in runs
    ]
    deproj_h = [
        Line2D(
            [0], [0], color="0.25", marker=MARKERS[d.key], lw=0, ms=5, label=d.label,
        )
        for d in deprojs
    ]
    sky_h = [
        Line2D([0], [0], color="0.25", ls="-", lw=1.5, label="full sky"),
        Line2D([0], [0], color="0.25", ls="--", lw=1.4, label=r"$q>5$ masked"),
    ]
    ax.legend(
        handles=run_h + deproj_h + sky_h,
        frameon=False, fontsize=7, loc="upper left",
    )
    ax.grid(False)
    fig.tight_layout()
    return fig


def build_cib_mstar_fid_ratio() -> plt.Figure:
    """Masked q>5 CIB residual, M*-1σ divided by the L1_m9 fiducial."""
    apply_pub_style()
    deprojs = tuple(
        d for d in ALL_DEPROJ if d.key in ("deproj_cib_dbeta", "deproj_cib_dbeta_cmb")
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ell, _ = _p18("nodeproj", "masked")
    ax.axhline(1.0, color="0.45", lw=1.0, zorder=1)
    for d in deprojs:
        fid = _resid("L1_m9", d.key, "masked")[0]
        var = _resid("Mstar-1sigma", d.key, "masked")[0]
        ax.semilogx(
            ell, var / fid,
            color=COLORS[d.key], marker=MARKERS[d.key], ms=4.5, lw=1.6,
            label=d.label,
        )
    ax.set_xlim(10.0, 959.5)
    ax.set_ylim(0.70, 1.22)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell^{\mathrm{CIB}}(M_\ast-1\sigma)\,/\,D_\ell^{\mathrm{CIB}}(\mathrm{fiducial})$")
    ax.set_title(r"$q>5$ masked")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(False)
    fig.tight_layout()
    return fig


def build_cib_cross_mstar_ratios() -> plt.Figure:
    """q>5 masked M*-1σ / fiducial for CIB, tSZ×CIB, and their sum."""
    apply_pub_style()
    deprojs = tuple(
        d for d in ALL_DEPROJ if d.key in ("deproj_cib_dbeta", "deproj_cib_dbeta_cmb")
    )
    pieces = (
        ("CIB auto", "-", "#0072B2"),
        (r"tSZ$\times$CIB", "--", "#D55E00"),
        (r"CIB + tSZ$\times$CIB", ":", "#009E73"),
    )
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ell, truth = _truth("masked", "L1_m9")
    ax.axhline(1.0, color="0.45", lw=1.0, zorder=1)
    ax.axhline(0.0, color="0.75", lw=0.6, zorder=0)
    for d in deprojs:
        cib_f = _resid("L1_m9", d.key, "masked")[0]
        cib_m = _resid("Mstar-1sigma", d.key, "masked")[0]
        x_f = _tsz_cib("L1_m9", d.key, "masked")
        x_m = _tsz_cib("Mstar-1sigma", d.key, "masked")
        series = (
            (cib_m / cib_f, cib_f),
            (x_m / x_f, x_f),
            ((cib_m + x_m) / (cib_f + x_f), cib_f + x_f),
        )
        for (label, ls, color), (ratio, denom) in zip(pieces, series):
            # A near-zero fiducial denominator makes the ratio meaningless.
            shown = np.where(np.abs(denom) > 0.02 * np.abs(truth), ratio, np.nan)
            ax.semilogx(
                ell, shown,
                color=color, ls=ls, marker=MARKERS[d.key], ms=4.2, lw=1.6,
                label=label if d.key == "deproj_cib_dbeta" else None,
            )
    ax.set_xlim(10.0, 959.5)
    ax.set_ylim(-1.7, 2.3)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(
        r"$D_\ell(M_\ast-1\sigma)\,/\,D_\ell(\mathrm{fiducial})$"
    )
    ax.set_title(r"$q>5$ masked")
    extra = [
        Line2D([0], [0], color="0.25", marker="^", lw=0, ms=5, label=r"CIB + $\delta\beta$"),
        Line2D([0], [0], color="0.25", marker="v", lw=0, ms=5, label=r"CIB + $\delta\beta$ + CMB"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + extra, labels + [e.get_label() for e in extra], frameon=False, fontsize=8, loc="lower right")
    ax.grid(False)
    fig.tight_layout()
    return fig


def build_cib_prescriptions() -> plt.Figure:
    apply_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True, sharey=True)
    ell, _ = _p18("nodeproj", "total")
    titles = ("full sky", r"$q>5$ masked")
    kinds = ("total", "masked")
    names = ("L1_m9", "fgas-8sigma", "Mstar-1sigma")
    for ax, kind, title in zip(axes, kinds, titles):
        for name in names:
            cib = _resid(name, "nodeproj", kind)[0]
            ax.loglog(
                ell, cib * SCALE,
                color=RUN_COLORS[name], marker=RUN_MARKERS[name], ms=4.0,
                lw=1.5, label=LABELS[name],
            )
        ax.set_xlim(10.0, 959.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r"$\ell$")
        ax.grid(False)
    axes[0].set_ylabel(r"$10^{12}D_\ell^{\mathrm{CIB}}$")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def build_cib_prescriptions_overlay() -> plt.Figure:
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ell, _ = _p18("nodeproj", "total")
    print(f"{'run':<16} {'ℓ=10':>8} {'ℓ=335':>8} {'ℓ=959':>8}  (masked/full CIB)")
    for name in ALL_RUNS:
        cib_f = _resid(name, "nodeproj", "total")[0]
        cib_m = _resid(name, "nodeproj", "masked")[0]
        color, mk = RUN_COLORS[name], RUN_MARKERS[name]
        ax.loglog(ell, cib_f * SCALE, color=color, marker=mk, ms=4.0, lw=1.6, label=LABELS[name])
        ax.loglog(ell, cib_m * SCALE, color=color, ls="--", lw=1.4)
        r = cib_m / cib_f
        print(f"{name:<16} {r[0]:8.3f} {r[13]:8.3f} {r[17]:8.3f}")
    ax.set_xlim(10.0, 959.5)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$10^{12}D_\ell^{\mathrm{CIB}}$")
    extra = [
        Line2D([0], [0], color="0.3", ls="-", lw=1.5, label="full sky"),
        Line2D([0], [0], color="0.3", ls="--", lw=1.4, label=r"$q>5$ masked"),
    ]
    h, lab = ax.get_legend_handles_labels()
    ax.legend(
        h + extra, lab + [e.get_label() for e in extra],
        frameon=False, fontsize=8, loc="lower right",
    )
    ax.grid(False)
    fig.tight_layout()
    return fig


def _bin_dl_18(ell: np.ndarray, cl: np.ndarray) -> np.ndarray:
    dl = ell * (ell + 1.0) * cl / (2.0 * np.pi)
    out = np.empty(len(ELL_MIN))
    for i, (lo, hi) in enumerate(zip(ELL_MIN, ELL_MAX_EXCL)):
        out[i] = np.nanmean(dl[(ell >= lo) & (ell < hi)])
    return out


def _cib_freq_dl(name: str) -> dict[int, np.ndarray]:
    z = np.load(ILC / "plot_cache" / f"signal_alms_{name}_cib_cmb_lmax1085.npz")
    ell = np.arange(1086, dtype=np.float64)
    out = {}
    for nu in FREQS_PLOT:
        cl = np.asarray(hp.alm2cl(z[f"cib_alm_{FREQ_IDX[nu]}"]), dtype=np.float64)
        out[nu] = _bin_dl_18(ell, cl) * SCALE
    return out


def build_cib_freq_autos() -> plt.Figure:
    apply_pub_style()
    packs = {name: _cib_freq_dl(name) for name in ALL_RUNS}
    ell, _ = _p18("nodeproj", "total")
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.8), sharex=True)
    print(f"{'run':<16} " + " ".join(f"{nu:>8d}" for nu in FREQS_PLOT) + "  (D_959 [µK²])")
    for name in ALL_RUNS:
        vals = " ".join(f"{packs[name][nu][-1]:8.3g}" for nu in FREQS_PLOT)
        print(f"{name:<16} {vals}")
    for ax, nu in zip(axes.ravel(), FREQS_PLOT):
        for name in ALL_RUNS:
            ax.loglog(
                ell, packs[name][nu],
                color=RUN_COLORS[name], marker=RUN_MARKERS[name], ms=3.5,
                lw=1.4, label=LABELS[name],
            )
        ax.set_xlim(10.0, 959.5)
        ax.set_title(rf"${nu}\,\mathrm{{GHz}}$", fontsize=12)
        ax.grid(False)
    for ax in axes[1]:
        ax.set_xlabel(r"$\ell$")
    axes[0, 0].set_ylabel(r"$D_\ell^{\mathrm{CIB}}\,[\mu\mathrm{K}_{\mathrm{CMB}}^2]$")
    axes[1, 0].set_ylabel(r"$D_\ell^{\mathrm{CIB}}\,[\mu\mathrm{K}_{\mathrm{CMB}}^2]$")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def main() -> None:
    _print_table()
    nq = _n_q5()
    _save(build_combined("total", "Total"), STEM_FULL)
    _save(build_combined("masked", rf"$q>5$ masked ($N={nq}$)"), STEM_MASK)
    _save(build_deproj_pair(nq), STEM_DEPROJ)
    _save(build_cib_compare(), STEM_CIB)
    _save(build_cib_prescriptions(), STEM_CIB_PRESC)
    _save(build_cib_prescriptions_overlay(), STEM_CIB_PRESC_ONE)
    _save(build_cib_freq_autos(), STEM_CIB_FREQ)
    _save(build_cib_tsz_cross("total", "Total"), STEM_CROSS_FULL)
    _save(build_cib_tsz_cross("masked", r"$q>5$ masked"), STEM_CROSS_MASK)
    _save(build_cib_tsz_cross_pair(), STEM_CROSS_PAIR)
    _save(build_tsz_cib_compare(), STEM_TSZXCIB)


if __name__ == "__main__":
    main()
