#!/usr/bin/env python3
"""Stacked HILC component budget vs ℓ (q>5), L1_m9 and M*-1σ, five schemes."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from pub_style import apply_pub_style, savefig
from hilc_prescriptions import ALL_DEPROJ, ILC, LABELS

ROOT = Path(__file__).resolve().parents[1]
PS = ILC / "ps_namaster"
FIG_DIR = ROOT / "figures" / "l1_m9"
STEM = "l1_m9_hilc_component_budget"
RUNS = ("L1_m9", "Mstar-1sigma")
SCALE = 1.0e12
COLORS = {
    "tsz": "#0055A4",
    "cib": "#C44E00",
    "cmb": "#00796B",
    "n": "#9e9e9e",
    "cross": "#6A1B9A",
}


def _load(name: str, deproj) -> dict:
    z = np.load(PS / f"{name}_{deproj.key}_masked_budget.npz")
    return {k: np.asarray(z[k]) for k in z.files}


def main() -> None:
    apply_pub_style()
    fig, axes = plt.subplots(
        2, 5, figsize=(12.4, 5.6), sharex=True, sharey=False, layout="constrained"
    )
    for i, name in enumerate(RUNS):
        for j, deproj in enumerate(ALL_DEPROJ):
            ax = axes[i, j]
            z = _load(name, deproj)
            ell = z["ell"]
            tsz = z["dl_tsz"] * SCALE
            cib = z["dl_cib"] * SCALE
            cmb = z["dl_cmb"] * SCALE
            n = z["dl_n"] * SCALE
            cross = (
                z["dl_tsz_cib"]
                + z["dl_tsz_cmb"]
                + z["dl_cib_cmb"]
                + z["dl_tsz_n"]
                + z["dl_cib_n"]
                + z["dl_cmb_n"]
            ) * SCALE
            ax.fill_between(ell, 0.0, tsz, color=COLORS["tsz"], alpha=0.85, lw=0)
            ax.fill_between(ell, tsz, tsz + cib, color=COLORS["cib"], alpha=0.85, lw=0)
            ax.fill_between(
                ell, tsz + cib, tsz + cib + cmb, color=COLORS["cmb"], alpha=0.85, lw=0
            )
            ax.fill_between(
                ell,
                tsz + cib + cmb,
                tsz + cib + cmb + n,
                color=COLORS["n"],
                alpha=0.85,
                lw=0,
            )
            ax.semilogx(ell, cross, color=COLORS["cross"], lw=1.4, ls="--")
            ax.semilogx(ell, z["dl_sum"] * SCALE, color="k", lw=1.1, ls=":")
            ax.semilogx(
                ell, z["dl_meas"] * SCALE, color="k", marker="o", ms=3.5, lw=0.0, zorder=5
            )
            ax.set_xlim(10.0, 959.5)
            ax.grid(False)
            if i == 0:
                ax.set_title(deproj.label, fontsize=11)
            if j == 0:
                ax.set_ylabel(rf"$10^{{12}}D_\ell$  {LABELS[name]}")
            if i == 1:
                ax.set_xlabel(r"$\ell$")
    handles = [
        Patch(facecolor=COLORS["tsz"], label="tSZ"),
        Patch(facecolor=COLORS["cib"], label="CIB"),
        Patch(facecolor=COLORS["cmb"], label="CMB"),
        Patch(facecolor=COLORS["n"], label="noise $r_1\\times r_2$"),
        Line2D([0], [0], color=COLORS["cross"], ls="--", lw=1.4, label="crosses"),
        Line2D([0], [0], color="k", ls=":", lw=1.1, label=r"$\sum$ terms"),
        Line2D([0], [0], color="k", marker="o", lw=0, ms=4, label="measured"),
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=7,
        frameon=False,
        fontsize=9,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    savefig(fig, STEM, fig_dir=FIG_DIR)
    plt.close(fig)


if __name__ == "__main__":
    main()
