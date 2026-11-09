#!/usr/bin/env python3
"""L1_m9 budget sum versus reconstructed HILC. Analysis figure only."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from pub_style import apply_pub_style, savefig
from hilc_prescriptions import ALL_DEPROJ, ILC
from plot_hilc_recovered_residuals_deproj import (
    COLORS,
    MARKERS,
    SCALE,
    _errbar,
    _truth,
    sigma_gt,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "l1_m9"
CLOSURE = Path("/tmp/l1_m9_budget_closure.npz")
STEM = "l1_m9_hilc_budget_sum_vs_recovered"


def _legend():
    return [
        Line2D([0], [0], color="k", lw=1.8, label=r"input truth $y$"),
    ] + [
        Line2D(
            [0], [0], color=COLORS[d.key], marker=MARKERS[d.key], lw=0, ms=5,
            label=d.label,
        )
        for d in ALL_DEPROJ
    ] + [
        Line2D([0], [0], color="0.3", ls="-", lw=1.5, label=r"sum of all terms"),
    ]


def _draw(ax, z, kind: str, title: str, *, ylabel: bool) -> None:
    ell, truth = _truth(kind)
    n = len(ALL_DEPROJ)
    ax.loglog(ell, truth * SCALE, color="k", lw=1.8, zorder=4)
    for i, d in enumerate(ALL_DEPROJ):
        color, mk = COLORS[d.key], MARKERS[d.key]
        x = ell * (1.05 ** (i - 0.5 * (n - 1)))
        rec = np.asarray(z[f"{kind}_{d.key}_p18"])
        total = np.asarray(z[f"{kind}_{d.key}_sum"])
        _errbar(ax, x, rec * SCALE, sigma_gt(d.key, kind) * SCALE, color=color, marker=mk)
        ax.loglog(ell, total * SCALE, color=color, ls="-", lw=1.4, zorder=3)
    ax.set_xlim(10.0, 959.5)
    ax.set_ylim(3.0e-7, 40.0)
    ax.set_xlabel(r"$\ell$")
    if ylabel:
        ax.set_ylabel(r"$10^{12}D_\ell$")
    ax.set_title(title, fontsize=12)
    ax.legend(handles=_legend(), frameon=False, fontsize=7, loc="upper left")
    ax.grid(False)


def main() -> None:
    apply_pub_style()
    z = np.load(CLOSURE)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)
    _draw(axes[0], z, "total", "Total", ylabel=True)
    _draw(axes[1], z, "masked", r"$q>5$ masked ($N=2868$)", ylabel=False)
    fig.tight_layout()
    savefig(fig, STEM, fig_dir=FIG_DIR)
    plt.close(fig)


if __name__ == "__main__":
    main()
