#!/usr/bin/env python3
"""Two-panel noiseless Compton-y variant/fiducial ratio.

Same q>5 own-mask as the HILC ratio figure; no ILC, no deprojection.
Grey fills are 1 ± σ/D_fid with σ² = σ_G² + σ_T² from the truth C_ℓ^yy
auto and the same trispectrum covariances as the HILC figure.
"""
from __future__ import annotations

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

from flamingo_mock.powerspectra import sigma_dl_auto_binned
from hilc_prescriptions import ILC, LABELS

ROOT = Path(__file__).resolve().parents[1]
PS = ILC / "ps_namaster"
DATA = ROOT / "data"
FIG_DIR = ROOT / "figures" / "l1_m9"
PAPER_FIGS = (
    Path("/home/lxu/scratch/tsz_cnc_paper_plots/masking_giant_paper/figs"),
    Path("/scratch/scratch-lxu/tsz_cnc_paper_plots/masking_giant_paper_astra/figs"),
)
COV_FULLSKY = Path(
    "/scratch/scratch-lxu/flamingo_repo/chains/l1_m9_qfrommap_asz_covariance"
    "/final/covariance/cov_trispectrum_L1_m9_customgnfw_qfrommap_fullsky_Dl_yy_binned_18.npy"
)
COV_Q5 = DATA / "cov_trispectrum_L1_m9_ilc_q5_Dl_yy_binned_18.npy"

VARIANTS = ("fgas-8sigma", "Mstar-1sigma")
ELL_MIN = np.array(
    [9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835]
)
ELL_MAX_EXCL = np.array(
    [13, 17, 22, 28, 36, 47, 61, 79, 103, 134, 174, 225, 293, 381, 495, 643, 836, 1086]
)
STEM = "l1_m9_hilc_truth_yy_ratio"

(LINE_TOT, BAND_TOT) = "#1a1a1a", "#d4d4d4"
(LINE_Q5, BAND_Q5) = "#0055A4", "#9ecae1"


def _truth(name: str, kind: str):
    z = np.load(PS / f"{name}_truth_{kind}.npz")
    return (
        np.asarray(z["ell"], dtype=np.float64),
        np.asarray(z["dl"], dtype=np.float64),
        np.asarray(z["cl_ell"], dtype=np.float64),
        float(z["fsky_eff"]),
    )


def sigma_t(kind: str) -> np.ndarray:
    path = COV_FULLSKY if kind == "total" else COV_Q5
    cov = np.asarray(np.load(path), dtype=np.float64)
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


def rel_band(kind: str, cl: np.ndarray, fsky: float, dl_fid: np.ndarray) -> np.ndarray:
    sig_g = sigma_dl_auto_binned(cl, ELL_MIN, ELL_MAX_EXCL, fsky)
    return np.sqrt(sig_g**2 + sigma_t(kind) ** 2) / dl_fid


def _fill(ax, ell, rel, line_color, band_color, zorder) -> None:
    ax.fill_between(
        ell,
        -rel,
        rel,
        color=band_color,
        alpha=0.40,
        edgecolor=line_color,
        linewidth=0.9,
        zorder=zorder,
    )


def _handle(line, band, marker):
    return (
        Line2D([0], [0], color=line, marker=marker, lw=2.0, ms=5.5),
        Patch(facecolor=band, edgecolor=line, linewidth=0.9),
    )


def build_figure() -> plt.Figure:
    apply_pub_style()
    ell, dl_fid_tot, cl_fid_tot, fsky_tot = _truth("L1_m9", "total")
    _, dl_fid_q5, cl_fid_q5, fsky_q5 = _truth("L1_m9", "masked")
    rel_full = rel_band("total", cl_fid_tot, fsky_tot, dl_fid_tot)
    rel_q5 = rel_band("masked", cl_fid_q5, fsky_q5, dl_fid_q5)

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.4), sharex=True, sharey=True, layout="constrained",
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.02)
    for ax, run in zip(axes, VARIANTS):
        _, r_tot, _, _ = _truth(run, "total")
        _, r_q5, _, _ = _truth(run, "masked")
        d_tot = r_tot / dl_fid_tot - 1.0
        d_q5 = r_q5 / dl_fid_q5 - 1.0
        _fill(ax, ell, rel_full, LINE_TOT, BAND_TOT, 1)
        _fill(ax, ell, rel_q5, LINE_Q5, BAND_Q5, 2)
        ax.axhline(0.0, color="k", lw=1.0, zorder=3)
        ax.semilogx(
            ell, d_tot, color=LINE_TOT, marker="o", ms=4.0, lw=2.0, zorder=4,
        )
        ax.semilogx(
            ell, d_q5, color=LINE_Q5, marker="o", ms=4.0, lw=2.0, zorder=5,
        )
        ax.set_xlim(10.0, 959.5)
        ax.set_title(LABELS.get(run, run), fontsize=12)
        ax.set_xlabel(r"$\ell$")
        ax.grid(False)
    for ax in axes:
        ax.set_ylim(-0.3, 0.3)
    axes[0].set_ylabel(r"$\Delta D_\ell^{yy}/D_\ell^{yy,\mathrm{fid}}$")
    fig.legend(
        [_handle(LINE_TOT, BAND_TOT, "o"), _handle(LINE_Q5, BAND_Q5, "o")],
        [r"Total (noiseless $y$)", r"$q>5$ (noiseless $y$)"],
        loc="outside upper center",
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.12)},
        columnspacing=1.2,
        handletextpad=0.4,
        handlelength=2.3,
        labelspacing=0.35,
    )
    return fig


def main() -> None:
    fig = build_figure()
    paths = savefig(fig, STEM, fig_dir=FIG_DIR)
    for paper in PAPER_FIGS:
        paper.mkdir(parents=True, exist_ok=True)
        for p in paths:
            dest = paper / p.name
            shutil.copy2(p, dest)
            print("copied", dest, flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
