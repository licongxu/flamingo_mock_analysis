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
