#!/usr/bin/env python3
"""N(q) marginal CNC for L1_m9 feedback SZiFi iMMF catalogues (q>=5).

Same q bins as tsz_cnc_paper_plots / plot_l1_m9_binned_cnc.py:
  Q_EDGES = geomspace(5, 40, 6)  → 5 bins.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi_homog")
FIG = Path(__file__).resolve().parents[1] / "figures"

Q_EDGES = np.geomspace(5.0, 40.0, 6)

PRESCRIPTIONS = ("L1_m9", "fgas-8sigma", "Mstar-1sigma", "LS8")
LABELS = {
    "L1_m9": r"fiducial (L1\_m9)",
    "fgas-8sigma": r"$f_{\rm gas}-8\sigma$",
    "Mstar-1sigma": r"$M_*-1\sigma$",
    "LS8": "LS8",
}
COLORS = {
    "L1_m9": "k",
    "fgas-8sigma": "#2ca02c",
    "Mstar-1sigma": "#17becf",
    "LS8": "#9467bd",
}

PAPER_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "mathtext.fontset": "cm",
    "text.latex.preamble": r"\usepackage{amsmath}",
}


def catalogue_path(prescription: str, cat_name: str = "fullsky_splitA_immf_q5.npz") -> Path:
    return ROOT / prescription / "catalogues" / cat_name


def bin_nq(path: Path) -> np.ndarray:
    q = np.load(path)["q_opt"].astype(np.float64)
    counts, _ = np.histogram(q, bins=Q_EDGES)
    return counts.astype(np.int64)


def legend_label(prescription: str, total: int) -> str:
    formatted = f"{total:,d}".replace(",", "{,}")
    return rf"{LABELS[prescription]} ($N={formatted}$)"


def grouped_bar_geometry(
    edges: np.ndarray, series_index: int, n_series: int
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.log(edges)
    bin_widths = np.diff(coordinates)
    bar_widths = 0.9 * bin_widths / n_series
    left = coordinates[:-1] + 0.05 * bin_widths + series_index * bar_widths
    right = left + bar_widths
    return np.exp(left), np.exp(right) - np.exp(left)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cat-name",
        default="fullsky_splitA_immf_q5.npz",
        help="Catalogue filename under each prescription (NumPy or szifi_jax_splitA_immf_q5.npz)",
    )
    p.add_argument("--stem", default="szifi_homog_cnc_binned_Nq_qgt5_immf")
    p.add_argument(
        "--cats",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit catalogue paths (overrides the four-prescription loop)",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Legend labels matching --cats (TeX ok). Default: path stems.",
    )
    args = p.parse_args()

    if args.cats is not None:
        keys = list(args.labels) if args.labels else [p.stem for p in args.cats]
        if len(keys) != len(args.cats):
            raise SystemExit("--labels must match --cats")
        series = [(k, Path(p)) for k, p in zip(keys, args.cats)]
        overlay_colors = ["k", "#d62728", "#2ca02c", "#17becf"]
        histograms = {k: bin_nq(p) for k, p in series}
        print("SZiFi iMMF N(q), q>=5, bins:", Q_EDGES.tolist())
        counts0 = None
        for i, (k, path) in enumerate(series):
            counts = histograms[k]
            n = int(counts.sum())
            ratio = ""
            if i == 0:
                counts0 = counts.astype(np.float64)
            else:
                r = counts.astype(np.float64) / np.maximum(counts0, 1.0)
                ratio = f"  ratio vs first={np.array2string(r, precision=3)}"
            print(f"{k}: N={n:,d}  N(q bins)={counts.tolist()}{ratio}  [{path}]")
        plt.rcParams.update(PAPER_RC)
        fig, ax = plt.subplots(figsize=(7.1, 4.4), layout="constrained")
        n_series = len(series)
        for index, (k, _) in enumerate(series):
            counts = histograms[k]
            left, widths = grouped_bar_geometry(Q_EDGES, index, n_series)
            formatted = f"{int(counts.sum()):,d}".replace(",", "{,}")
            ax.bar(
                left,
                counts,
                width=widths,
                align="edge",
                color=overlay_colors[index % len(overlay_colors)],
                alpha=1.0 if index == 0 else 0.72,
                edgecolor="white",
                linewidth=0.35,
                label=rf"{k} ($N={formatted}$)",
            )
    else:
        histograms = {p: bin_nq(catalogue_path(p, args.cat_name)) for p in PRESCRIPTIONS}
        print("SZiFi iMMF N(q), q>=5, bins:", Q_EDGES.tolist())
        for p in PRESCRIPTIONS:
            n = int(histograms[p].sum())
            print(f"{p}: N={n:,d}  N(q bins)={histograms[p].tolist()}")

        plt.rcParams.update(PAPER_RC)
        fig, ax = plt.subplots(figsize=(7.1, 4.4), layout="constrained")
        for index, prescription in enumerate(PRESCRIPTIONS):
            counts = histograms[prescription]
            left, widths = grouped_bar_geometry(Q_EDGES, index, len(PRESCRIPTIONS))
            ax.bar(
                left,
                counts,
                width=widths,
                align="edge",
                color=COLORS[prescription],
                alpha=1.0 if prescription == "L1_m9" else 0.68,
                edgecolor="white",
                linewidth=0.35,
                label=legend_label(prescription, int(counts.sum())),
            )

    ax.set_xscale("log")
    ax.set_xticks([5.0, 10.0, 20.0, 40.0], labels=[r"$5$", r"$10$", r"$20$", r"$40$"])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(4.7, 42.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$N$")
    ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        frameon=False,
        ncol=2,
        fontsize=12,
        columnspacing=0.8,
        handlelength=1.15,
        handletextpad=0.35,
    )

    FIG.mkdir(parents=True, exist_ok=True)
    stem = FIG / args.stem
    for suffix in ("png", "pdf"):
        out = stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
