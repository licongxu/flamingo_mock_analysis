#!/usr/bin/env python3
"""Three-panel HILC r1×r2 variant/fiducial ratio with ILC Gaussian + theory T bands.

Datapoints: NaMaster 18-bin r1×r2 HILC D_ℓ (ps_namaster *_p18.npz).
Bands: 1 ± σ_b / D_b^{fid,HILC} with σ² = σ_{G,split-cross}² + σ_T².
Full sky: no deproj, light grey. q>5: no deproj and CIB deproj.
Knox is not used.
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

# full sky grey; q>5 no deproj blue; q>5 CIB deproj green
BAND_FULL = "#bdbdbd"
BAND_Q5 = "#9ecae1"
BAND_Q5_CIB = "#a1d99b"
CURVE_FULL = "#4d4d4d"
CURVE_Q5 = "#2171b5"
CURVE_Q5_CIB = "#238b45"


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


def _print_table() -> None:
    _, dl_tot = _p18("L1_m9", "nodeproj", "total")
    _, dl_msk = _p18("L1_m9", "nodeproj", "masked")
    i10, i335, i959 = 0, 13, 17
    print(f"{'sky':<8} {'deproj':<12} {'piece':<8} {'ℓ=10':>8} {'ℓ=335':>8} {'ℓ=959':>8}")
    for kind, deproj, dl in (
        ("total", "nodeproj", dl_tot),
        ("masked", "nodeproj", dl_msk),
        ("masked", "deproj_cib", dl_msk),
    ):
        sky = "fullsky" if kind == "total" else "q5"
        # CIB-deproj ratio uses CIB-deproj fiducial D_ℓ
        if deproj == "deproj_cib":
            _, dl = _p18("L1_m9", "deproj_cib", "masked")
        sg = sigma_g(deproj, kind) / dl
        st = sigma_t(kind) / dl
        stot = np.sqrt(sg**2 + st**2)
        for lab, arr in (("ILC G", sg), ("T", st), ("G+T", stot)):
            print(
                f"{sky:<8} {deproj:<12} {lab:<8} "
                f"{arr[i10]:8.3f} {arr[i335]:8.3f} {arr[i959]:8.3f}"
            )
    if META_Q5.is_file():
        print("q5 T f_sky", json.loads(META_Q5.read_text())["f_sky"])


def build_figure() -> plt.Figure:
    apply_pub_style()
    ell, dl_fid_tot = _p18("L1_m9", "nodeproj", "total")
    _, dl_fid_q5 = _p18("L1_m9", "nodeproj", "masked")
    _, dl_fid_q5_cib = _p18("L1_m9", "deproj_cib", "masked")
    rel_full = rel_band("nodeproj", "total", dl_fid_tot)
    rel_q5 = rel_band("nodeproj", "masked", dl_fid_q5)
    rel_q5_cib = rel_band("deproj_cib", "masked", dl_fid_q5_cib)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True, sharey=True)
    for ax, run in zip(axes, VARIANTS):
        _, r_full = _p18(run, "nodeproj", "total")
        _, r_q5 = _p18(run, "nodeproj", "masked")
        _, r_q5_cib = _p18(run, "deproj_cib", "masked")
        ax.fill_between(ell, 1.0 - rel_full, 1.0 + rel_full, color=BAND_FULL, alpha=0.45, lw=0, zorder=1)
        ax.fill_between(ell, 1.0 - rel_q5, 1.0 + rel_q5, color=BAND_Q5, alpha=0.50, lw=0, zorder=2)
        ax.fill_between(
            ell, 1.0 - rel_q5_cib, 1.0 + rel_q5_cib,
            facecolor="none", edgecolor=CURVE_Q5_CIB, hatch="///", lw=1.2, zorder=3,
        )
        ax.axhline(1.0, color="k", lw=1.0, zorder=4)
        ax.semilogx(ell, r_full / dl_fid_tot, color=CURVE_FULL, marker="o", ms=3.5, lw=1.6, zorder=5)
        ax.semilogx(ell, r_q5 / dl_fid_q5, color=CURVE_Q5, marker="o", ms=3.5, lw=1.6, zorder=6)
        ax.semilogx(
            ell, r_q5_cib / dl_fid_q5_cib, color=CURVE_Q5_CIB, marker="s", ms=3.2,
            lw=1.4, ls="--", zorder=7,
        )
        ax.set_xlim(10.0, 959.5)
        ax.set_ylim(0.78, 1.30)
        ax.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
        ax.set_title(LABELS.get(run, run), fontsize=12)
        ax.set_xlabel(r"$\ell$")
        ax.grid(False)
    axes[0].set_ylabel(r"$D_\ell^{\rm variant}/D_\ell^{\rm fiducial}$")
    handles = [
        Line2D([0], [0], color=CURVE_FULL, marker="o", lw=1.6, ms=4, label="full sky"),
        Line2D([0], [0], color=CURVE_Q5, marker="o", lw=1.6, ms=4, label=r"$q>5$, no deproj."),
        Line2D(
            [0], [0], color=CURVE_Q5_CIB, marker="s", lw=1.4, ms=4, ls="--",
            label=r"$q>5$, CIB deproj.",
        ),
        Patch(facecolor=BAND_FULL, edgecolor="none", label=r"full sky $1\sigma$ (no deproj.)"),
        Patch(facecolor=BAND_Q5, edgecolor="none", label=r"$q>5$ $1\sigma$ (no deproj.)"),
        Patch(
            facecolor="white", edgecolor=CURVE_Q5_CIB, hatch="///",
            label=r"$q>5$ $1\sigma$ (CIB deproj.)",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0.0, 0.20, 1.0, 1.0))
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
