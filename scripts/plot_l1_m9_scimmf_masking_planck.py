"""Single-panel L1_m9 truth-y PS with sciMMF masking vs Planck.

Run after compute_l1_m9_scimmf_masked_ps.py::

    python scripts/plot_l1_m9_scimmf_masking_planck.py
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

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from pub_style import apply_pub_style, savefig

PS_DIR = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi_homog/L1_m9"
    "/ps_masked_scimmf"
)
FULLSKY_18 = Path(
    "/scratch/scratch-lxu/flamingo_repo/data_paper/binned_bandpowers"
    "/Dl_yy_L1_m9_fullsky_binned_18.txt"
)
FULLSKY_LOG = Path(
    "/scratch/scratch-lxu/flamingo_repo/data_paper/binned_bandpowers"
    "/Dl_yy_L1_m9_fiducial_fullsky_logbins_dln0p4_lmax10000_pixwin_deconvolved.txt"
)
TOPOSZ = Path("/scratch/scratch-lxu/flamingo_repo/ref_package/TopoSZ/fig12 data")
PLANCK_B18 = TOPOSZ / "planck_sz_1712_00788v1.txt"
PLANCK_MASKED = TOPOSZ / (
    "szpowespectrum_measurement_urc_snr6_p18cmb_bf_fg_from_TSZ+P18_"
    "l_clyy_sigclyy_cib_ir_rs_cn.txt"
)
PAPER_FIGS = Path("/home/lxu/scratch/tsz_cnc_paper_plots/masking_giant_paper/figs")
FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "l1_m9"

CUTS = (
    (50.0, "qgt50", "#CC79A7", "-"),
    (20.0, "qgt20", "#56B4E9", "-"),
    (10.0, "qgt10", "#009E73", "-"),
    (6.0, "qgt6", "#D55E00", "--"),
)
N_PLANCK_BINS = 18
ELL_PLANCK_MAX = 959.5
STEM = "l1_m9_fiducial_masking_planck_scimmf"


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    return arr[:, 0], arr[:, 1]


def _hybrid(ell18: np.ndarray, dl18: np.ndarray, elllog: np.ndarray, dllog: np.ndarray):
    high = elllog > ELL_PLANCK_MAX
    return np.concatenate([ell18, elllog[high]]), np.concatenate([dl18, dllog[high]])


def build_figure() -> plt.Figure:
    apply_pub_style()
    meta = json.loads((PS_DIR / "L1_m9_masked_scimmf_metadata.json").read_text())["cuts"]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))

    ell_full, dl_full = _hybrid(*_load(FULLSKY_18), *_load(FULLSKY_LOG))
    (full_line,) = ax.loglog(
        ell_full, dl_full, color="#0072B2", lw=2.3, label="full sky", zorder=3
    )
    x_hi = float(ell_full[-1])

    cut_lines = []
    for cut, tag, color, ls in CUTS:
        ell, dl = _hybrid(
            *_load(PS_DIR / f"Dl_yy_L1_m9_masked_{tag}_scimmf_binned_18.txt"),
            *_load(PS_DIR / f"Dl_yy_L1_m9_masked_{tag}_scimmf_logbins_dln0p4_lmax10000.txt"),
        )
        n = meta[tag]["n_masked"]
        (line,) = ax.loglog(
            ell,
            dl,
            color=color,
            ls=ls,
            lw=1.9,
            label=rf"$q>{cut:g}$ ($N={n}$)",
            zorder=3,
        )
        cut_lines.append(line)
        x_hi = max(x_hi, float(ell[-1]))

    arr_b = np.loadtxt(PLANCK_B18)[:N_PLANCK_BINS]
    arr_r = np.loadtxt(PLANCK_MASKED)[:N_PLANCK_BINS]
    bolliet = ax.errorbar(
        arr_b[:, 0],
        arr_b[:, 1],
        yerr=arr_b[:, 2],
        fmt="o",
        mfc="white",
        mec="#0072B2",
        ecolor="#0072B2",
        capsize=2.5,
        elinewidth=1.1,
        markersize=4.5,
        label="Bolliet et al. (2018)",
        zorder=5,
    )
    rotti = ax.errorbar(
        arr_r[:, 0],
        arr_r[:, 1],
        yerr=arr_r[:, 2],
        fmt="s",
        mfc="white",
        mec="#D55E00",
        ecolor="#D55E00",
        capsize=2.5,
        elinewidth=1.1,
        markersize=4.5,
        label="Rotti et al. (2021)",
        zorder=5,
    )

    x_lo = min(float(ell_full[0]), float(arr_b[0, 0]), float(arr_r[0, 0]))
    ax.set_xlim(x_lo, x_hi)
    ax.margins(x=0)
    ax.set_ylim(5.0e-3, 3.0)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$10^{12}D_\ell^{yy}$")
    ax.legend(
        handles=(full_line, *cut_lines, bolliet, rotti),
        frameon=False,
        loc="lower right",
        handlelength=1.8,
        fontsize=8,
    )
    return fig


def main() -> None:
    fig = build_figure()
    paths = savefig(fig, STEM, fig_dir=FIG_DIR)
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    for p in paths:
        dest = PAPER_FIGS / p.name
        shutil.copy2(p, dest)
        print(f"copied {dest}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
