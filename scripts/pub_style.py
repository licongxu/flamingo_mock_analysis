"""Publication Matplotlib style for paper figures.

Rules for this project:
- text.usetex=True (LaTeX text rendering)
- readable axis / tick / legend fontsizes
- no gridlines (axes.grid=False; never call ax.grid(True))
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

PUB_RCPARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.linewidth": 1.0,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "legend.frameon": False,
}


def apply_pub_style() -> None:
    """Apply publication rcParams globally."""
    mpl.rcParams.update(PUB_RCPARAMS)


def no_grid(ax=None) -> None:
    """Ensure grid is off on an axes or all axes of current figure."""
    if ax is None:
        for a in plt.gcf().get_axes():
            a.grid(False)
    else:
        ax.grid(False)


def savefig(fig, name: str, fig_dir: Path | None = None) -> list[Path]:
    """Save PDF + PNG under figures/ (or fig_dir) at savefig.dpi=300.

    Requires ``apply_pub_style()`` (or equivalent) so that ``text.usetex`` is
    True and ``savefig.dpi`` is 300.  Explicit kwargs enforce dpi for PNG.
    """
    out = Path(fig_dir) if fig_dir is not None else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    # Enforce publication style at save time
    if not mpl.rcParams.get("text.usetex", False):
        apply_pub_style()
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
        print(f"Wrote {p}")
    return paths
