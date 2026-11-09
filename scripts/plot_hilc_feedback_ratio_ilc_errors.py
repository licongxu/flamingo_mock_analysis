#!/usr/bin/env python3
"""Two-panel HILC r1 x r2 variant/fiducial ratio.

Ratio curves: D_var / D_fid of the same reconstruction.
1σ fills: 1 ± σ / D_fid for the same scheme (deproj + sky). σ² = σ_G² + σ_T².
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from plot_hilc_recovered_residuals_deproj import COLORS as REC_COLORS
from plot_hilc_recovered_residuals_deproj import MARKERS as REC_MARKERS
from plot_hilc_truth_yy_ratio import _truth as truth_spec
from pub_style import apply_pub_style, savefig

from flamingo_mock.powerspectra import sigma_dl_cross_binned
from hilc_prescriptions import ILC, LABELS

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
META_Q5 = DATA / "cov_trispectrum_L1_m9_ilc_q5_metadata.json"

VARIANTS = ("fgas-8sigma", "Mstar-1sigma")
ELL_MIN = np.array(
    [9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835]
)
ELL_MAX_EXCL = np.array(
    [13, 17, 22, 28, 36, 47, 61, 79, 103, 134, 174, 225, 293, 381, 495, 643, 836, 1086]
)
STEM = "l1_m9_hilc_feedback_ratio_ilc_errors"

# High-contrast pairs: dark line + light fill + band edge in line color.
(LINE_TOT, BAND_TOT) = "#1a1a1a", "#d4d4d4"
(LINE_Q5, BAND_Q5) = "#0055A4", "#9ecae1"
(LINE_CIB, BAND_CIB) = "#C44E00", "#fdd0a2"
(LINE_DBETA, BAND_DBETA) = "#6A1B9A", "#d4b9e8"
(LINE_CMB, BAND_CMB) = "#00796B", "#80cbc4"
(LINE_MOM, BAND_MOM) = "#2E7D32", "#a5d6a7"


def _p18(run: str, deproj: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(PS / f"{run}_{deproj}_{kind}_p18.npz", allow_pickle=True)
    return np.asarray(z["ell"], dtype=np.float64), np.asarray(z["dl"], dtype=np.float64)


def _splitcross(deproj: str, kind: str):
    z = np.load(PS / f"L1_m9_{deproj}_{kind}_splitcross.npz")
    return (
        np.asarray(z["cl_11"], dtype=np.float64),
        np.asarray(z["cl_22"], dtype=np.float64),
        np.asarray(z["cl_12"], dtype=np.float64),
        float(z["fsky_eff"]),
    )


def sigma_g(deproj: str, kind: str) -> np.ndarray:
    c11, c22, c12, fsky = _splitcross(deproj, kind)
    return sigma_dl_cross_binned(c11, c22, c12, ELL_MIN, ELL_MAX_EXCL, fsky)


def sigma_t(kind: str) -> np.ndarray:
    path = COV_FULLSKY if kind == "total" else COV_Q5
    cov = np.asarray(np.load(path), dtype=np.float64)
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


def rel_band(deproj: str, kind: str, dl_fid: np.ndarray) -> np.ndarray:
    return np.sqrt(sigma_g(deproj, kind) ** 2 + sigma_t(kind) ** 2) / dl_fid


def _fill(ax, ell, rel, line_color, band_color, zorder) -> None:
    ax.fill_between(
        ell,
        1.0 - rel,
        1.0 + rel,
        color=band_color,
        alpha=0.40,
        edgecolor=line_color,
        linewidth=0.9,
        zorder=zorder,
    )


def _handle(line, band, marker, ls="-"):
    return (
        Line2D([0], [0], color=line, marker=marker, lw=2.0, ms=5.5, ls=ls),
        Patch(facecolor=band, edgecolor=line, linewidth=0.9),
    )


def _print_table() -> None:
    i10, i335, i959 = 0, 13, 17
    print(f"{'sky':<8} {'deproj':<22} {'ℓ=10':>8} {'ℓ=335':>8} {'ℓ=959':>8}")
    for kind, deproj in (
        ("total", "nodeproj"),
        ("masked", "nodeproj"),
        ("masked", "deproj_cib"),
        ("masked", "deproj_cib_dbeta"),
        ("masked", "deproj_cib_dbeta_cmb"),
        ("masked", "deproj_moments"),
    ):
        sky = "fullsky" if kind == "total" else "q5"
        _, dl_ref = _p18("L1_m9", deproj, kind)
        stot = rel_band(deproj, kind, dl_ref)
        print(
            f"{sky:<8} {deproj:<22} "
            f"{stot[i10]:8.3f} {stot[i335]:8.3f} {stot[i959]:8.3f}"
        )
    if META_Q5.is_file():
        print("q5 T f_sky", json.loads(META_Q5.read_text())["f_sky"])


def build_figure() -> plt.Figure:
    apply_pub_style()
    ell, dl_fid_tot = _p18("L1_m9", "nodeproj", "total")
    _, dl_fid_q5 = _p18("L1_m9", "nodeproj", "masked")
    _, dl_fid_cib = _p18("L1_m9", "deproj_cib", "masked")
    _, dl_fid_dbeta = _p18("L1_m9", "deproj_cib_dbeta", "masked")
    _, dl_fid_cmb = _p18("L1_m9", "deproj_cib_dbeta_cmb", "masked")
    _, dl_fid_mom = _p18("L1_m9", "deproj_moments", "masked")
    rel_full = rel_band("nodeproj", "total", dl_fid_tot)
    rel_q5 = rel_band("nodeproj", "masked", dl_fid_q5)
    rel_cib = rel_band("deproj_cib", "masked", dl_fid_cib)
    rel_dbeta = rel_band("deproj_cib_dbeta", "masked", dl_fid_dbeta)
    rel_cmb = rel_band("deproj_cib_dbeta_cmb", "masked", dl_fid_cmb)
    rel_mom = rel_band("deproj_moments", "masked", dl_fid_mom)

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.4), sharex=True, sharey=True, layout="constrained",
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.02)
    for ax, run in zip(axes, VARIANTS):
        _, r_tot = _p18(run, "nodeproj", "total")
        _, r_q5 = _p18(run, "nodeproj", "masked")
        _, r_cib = _p18(run, "deproj_cib", "masked")
        _, r_dbeta = _p18(run, "deproj_cib_dbeta", "masked")
        _, r_cmb = _p18(run, "deproj_cib_dbeta_cmb", "masked")
        _, r_mom = _p18(run, "deproj_moments", "masked")
        _fill(ax, ell, rel_full, LINE_TOT, BAND_TOT, 1)
        _fill(ax, ell, rel_mom, LINE_MOM, BAND_MOM, 2)
        _fill(ax, ell, rel_cmb, LINE_CMB, BAND_CMB, 3)
        _fill(ax, ell, rel_dbeta, LINE_DBETA, BAND_DBETA, 4)
        _fill(ax, ell, rel_cib, LINE_CIB, BAND_CIB, 5)
        _fill(ax, ell, rel_q5, LINE_Q5, BAND_Q5, 6)
        ax.axhline(1.0, color="k", lw=1.0, zorder=7)
        ax.semilogx(
            ell, r_tot / dl_fid_tot, color=LINE_TOT, marker="o", ms=4.0, lw=2.0, zorder=8,
        )
        ax.semilogx(
            ell, r_q5 / dl_fid_q5, color=LINE_Q5, marker="o", ms=4.0, lw=2.0, zorder=9,
        )
        ax.semilogx(
            ell, r_cib / dl_fid_cib, color=LINE_CIB, marker="s", ms=4.0,
            lw=1.8, ls="--", zorder=10,
        )
        ax.semilogx(
            ell, r_dbeta / dl_fid_dbeta, color=LINE_DBETA, marker="^", ms=4.2,
            lw=1.8, ls="--", zorder=11,
        )
        ax.semilogx(
            ell, r_cmb / dl_fid_cmb, color=LINE_CMB, marker="v", ms=4.2,
            lw=1.8, ls="-.", zorder=12,
        )
        ax.semilogx(
            ell, r_mom / dl_fid_mom, color=LINE_MOM, marker="D", ms=4.0,
            lw=1.8, ls=":", zorder=13,
        )
        ax.set_xlim(10.0, 959.5)
        ax.set_ylim(0.78, 1.30)
        ax.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
        ax.set_title(LABELS.get(run, run), fontsize=12)
        ax.set_xlabel(r"$\ell$")
        ax.grid(False)
    axes[0].set_ylabel(r"$D_\ell^{\rm variant}/D_\ell^{\rm fiducial}$")
    handles = [
        _handle(LINE_TOT, BAND_TOT, "o"),
        _handle(LINE_Q5, BAND_Q5, "o"),
        _handle(LINE_CIB, BAND_CIB, "s", "--"),
        _handle(LINE_DBETA, BAND_DBETA, "^", "--"),
        _handle(LINE_CMB, BAND_CMB, "v", "-."),
        _handle(LINE_MOM, BAND_MOM, "D", ":"),
    ]
    labels = [
        r"Total, no deproj.",
        r"$q>5$, no deproj.",
        r"$q>5$, CIB",
        r"$q>5$, CIB+$\delta\beta$",
        r"$q>5$, CIB+$\delta\beta$+CMB",
        r"$q>5$, CIB+$\delta\beta$+$\delta T$",
    ]
    fig.legend(
        handles, labels, loc="outside upper center", ncol=3, frameon=False,
        fontsize=8.5, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.12)},
        columnspacing=1.2, handletextpad=0.4, handlelength=2.3, labelspacing=0.35,
    )
    return fig


RESIDUAL_SPECS = (
    ("total", "nodeproj", "-"),
    ("masked", "nodeproj", "--"),
    ("masked", "deproj_cib", "--"),
    ("masked", "deproj_cib_dbeta", "--"),
    ("masked", "deproj_cib_dbeta_cmb", "-."),
    ("masked", "deproj_moments", ":"),
)
FRAC_STEM = "l1_m9_hilc_feedback_fractional_fg_residual"


def _truth_dl(run: str, kind: str) -> np.ndarray:
    z = np.load(PS / f"{run}_truth_{kind}.npz")
    return np.asarray(z["dl"], dtype=np.float64)


def _band_colors(deproj: str, kind: str) -> tuple[str, str]:
    """Full-sky no-deproj is grey; q>5 no-deproj keeps the recovered-plot blue."""
    if deproj == "nodeproj" and kind == "total":
        return LINE_TOT, BAND_TOT
    if deproj == "nodeproj":
        return REC_COLORS["nodeproj"], "#9ecae1"
    color = REC_COLORS[deproj]
    return color, color


def _fill0(ax, ell, rel, line_color, band_color, zorder) -> None:
    ax.fill_between(
        ell, -rel, rel,
        color=band_color, alpha=0.40, edgecolor=line_color, linewidth=0.9, zorder=zorder,
    )


def build_fractional_figure(
    ylim: tuple[float, float] = (-0.8, 3.2),
    subtract: str = "variant",
) -> plt.Figure:
    """(D_yy^var / D_yy^fid - 1) plus (D_rec - D_yy^var) / D_yy^fid, centred on 0.

    Shaded bands are the fiducial HILC errors of that reconstruction, now
    ±σ/D_ℓ^{yy,fid} so they share the axis denominator. σ² = σ_G² + σ_T².
    """
    apply_pub_style()
    ell, dl_fid_tot, _, _ = truth_spec("L1_m9", "total")
    _, dl_fid_q5, _, _ = truth_spec("L1_m9", "masked")
    fid = {"total": dl_fid_tot, "masked": dl_fid_q5}
    bands = (
        ("nodeproj", "total", 1),
        ("deproj_moments", "masked", 2),
        ("deproj_cib_dbeta_cmb", "masked", 3),
        ("deproj_cib_dbeta", "masked", 4),
        ("deproj_cib", "masked", 5),
        ("nodeproj", "masked", 6),
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.6), sharex=True, sharey=True, layout="constrained",
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.02)
    for ax, run in zip(axes, VARIANTS):
        yy_tot = _truth_dl(run, "total")
        yy_q5 = _truth_dl(run, "masked")
        yy = {"total": yy_tot, "masked": yy_q5}
        for deproj, kind, zorder in bands:
            line, band = _band_colors(deproj, kind)
            _fill0(ax, ell, rel_band(deproj, kind, fid[kind]), line, band, zorder)
        ax.axhline(0.0, color="k", lw=1.0, zorder=7)
        ax.semilogx(
            ell, yy_tot / dl_fid_tot - 1.0,
            color="k", marker="o", ms=4.5, lw=2.2, zorder=11,
        )
        ax.semilogx(
            ell, yy_q5 / dl_fid_q5 - 1.0,
            color=LINE_Q5, marker="o", ms=4.5, lw=2.2, zorder=12,
        )
        for kind, deproj, ls in RESIDUAL_SPECS:
            _, dl_rec = _p18(run, deproj, kind)
            yy_sub = fid[kind] if subtract == "fiducial" else yy[kind]
            frac = (dl_rec - yy_sub) / fid[kind]
            ax.semilogx(
                ell, frac, color=_band_colors(deproj, kind)[0],
                marker=REC_MARKERS[deproj],
                ms=3.6, lw=1.3, ls=ls, zorder=8,
            )
        ax.set_xlim(10.0, 959.5)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(LABELS.get(run, run), fontsize=12)
        ax.set_xlabel(r"$\ell$")
        ax.grid(False)
    if subtract == "fiducial":
        axes[0].set_ylabel(r"$(D_\ell^{\mathrm{rec}}-D_\ell^{yy,\mathrm{fid}})/D_\ell^{yy,\mathrm{fid}}$")
    else:
        axes[0].set_ylabel(r"$\Delta D_\ell^{yy}/D_\ell^{yy,\mathrm{fid}}$")
    handles = [
        Line2D([0], [0], color="k", marker="o", lw=2.2, ms=5.0),
        Line2D([0], [0], color=LINE_Q5, marker="o", lw=2.2, ms=5.0),
        _handle(*_band_colors("nodeproj", "total"), "o", "-"),
        _handle(*_band_colors("nodeproj", "masked"), "o", "--"),
        _handle(REC_COLORS["deproj_cib"], REC_COLORS["deproj_cib"], "s", "--"),
        _handle(REC_COLORS["deproj_cib_dbeta"], REC_COLORS["deproj_cib_dbeta"], "^", "--"),
        _handle(REC_COLORS["deproj_cib_dbeta_cmb"], REC_COLORS["deproj_cib_dbeta_cmb"], "v", "-."),
        _handle(REC_COLORS["deproj_moments"], REC_COLORS["deproj_moments"], "D", ":"),
    ]
    labels = [
        r"Noiseless $y$, total",
        r"Noiseless $y$, $q>5$",
        r"FG residual, total, no deproj.",
        r"FG residual, $q>5$, no deproj.",
        r"FG residual, $q>5$, CIB",
        r"FG residual, $q>5$, CIB+$\delta\beta$",
        r"FG residual, $q>5$, CIB+$\delta\beta$+CMB",
        r"FG residual, $q>5$, CIB+$\delta\beta$+$\delta T$",
    ]
    fig.legend(
        handles, labels, loc="outside upper center", ncol=2, frameon=False,
        fontsize=8.0, columnspacing=1.0, handletextpad=0.4, handlelength=2.6,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.12)},
    )
    return fig


def main() -> None:
    _print_table()
    fig = build_figure()
    paths = savefig(fig, STEM, fig_dir=FIG_DIR)
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    for p in paths:
        dest = PAPER_FIGS / p.name
        shutil.copy2(p, dest)
        print("copied", dest, flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
