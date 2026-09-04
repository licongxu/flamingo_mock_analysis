#!/usr/bin/env python3
"""Combined deprojection comparison plots (CIB overlay + 4-panel residuals).

Writes under figures/hilc/combined/deproj_compare/:
  r1xr2_fig9_{fullsky,q5masked}.png            — Fig. 9 stack (L1_m9, all schemes)
  cib_residual_{fullsky,q5masked}.png          — CIB curves, one axes (L1_m9)
  cib_residual_{fullsky,q5masked}_all.png      — same, one row per prescription
  residual_quad_{fullsky,q5masked}.png         — 2×2 signal / CIB / CMB / noise (L1_m9)
  residual_quad_{fullsky,q5masked}_all.png     — grid (prescription × component)

Includes no deprojection plus the four CIB deprojection schemes.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flamingo_mock.powerspectra import dl_from_cl

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from hilc_prescriptions import (  # noqa: E402
    ALL_RUNS,
    DEPROJ_CIB,
    DEPROJ_CIB_DBETA,
    DEPROJ_CIB_DBETA_CMB,
    DEPROJ_MOMENTS,
    DEPROJ_NONE,
    DeprojCase,
    LABELS,
    REPO,
    cluster_mask_apo,
)
from plot_hilc_homog_prescriptions import (  # noqa: E402
    ELL_EFF,
    ELL_PLOT_MAX,
    LMAX,
    YLIM,
    _curve_and_bins,
    _positive_dl,
    _residual_ylim,
    _tight_log_ylim,
    catalogue_path,
    diag,
    load_pack,
    signal_alms,
)

DEPROJ_SUITE = (
    DEPROJ_NONE,
    DEPROJ_CIB,
    DEPROJ_CIB_DBETA,
    DEPROJ_CIB_DBETA_CMB,
    DEPROJ_MOMENTS,
)
DEPROJ_STYLE = {
    DEPROJ_NONE.key: ("0.25", "s", ":"),
    DEPROJ_CIB.key: ("C2", "^", "-"),
    DEPROJ_CIB_DBETA.key: ("C4", "D", "-"),
    DEPROJ_CIB_DBETA_CMB.key: ("C5", "X", "--"),
    DEPROJ_MOMENTS.key: ("C1", "o", "-"),
}
SCHEME_TITLE = "no deprojection + four deprojections"
OUT = REPO / "figures" / "hilc" / "combined" / "deproj_compare"


def _load_deproj_packs(name: str, *, masked: bool, bl, good) -> list[tuple[DeprojCase, dict]]:
    sig = signal_alms(name)
    w_apo = diag.load_map(cluster_mask_apo(name)) if masked else None
    rows: list[tuple[DeprojCase, dict]] = []
    for deproj in DEPROJ_SUITE:
        pack = load_pack(
            name, masked=masked, deproj=deproj, sig=sig, bl=bl, good=good, w_apo=w_apo
        )
        if pack is not None:
            rows.append((deproj, pack))
    return rows


def _style_ax(ax, *, ylim: tuple[float, float] | None = None) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ELL_EFF[0], ELL_PLOT_MAX)
    ax.set_ylim(*(ylim if ylim is not None else YLIM))
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell$")


def _deproj_panel_ylim(
    rows: list[tuple[DeprojCase, dict]],
    ells: np.ndarray,
    sl: slice,
    clkey: str,
    dlkey: str,
    *,
    include_truth: bool = False,
) -> tuple[float, float]:
    vals = [
        np.abs(dl_from_cl(ells, pack[clkey])[sl])
        for _, pack in rows
    ] + [np.abs(pack[dlkey]) for _, pack in rows]
    if include_truth and rows:
        vals.append(np.abs(dl_from_cl(ells, rows[0][1]["cl_tt"])[sl]))
    return _tight_log_ylim(_positive_dl(*vals))


def plot_fig9_deproj_stack(
    rows: list[tuple[DeprojCase, dict]],
    *,
    name: str,
    masked: bool,
    out: Path,
) -> None:
    """Same single-panel Fig. 9 layout as nodeproj, stacked per deprojection."""
    fig9_rows = [(name, pack) for _, pack in rows]
    # plot_fig9 titles use deproj.label; stack one panel per deproj with correct metadata
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(8.6, 3.15 * n), sharex=True)
    if n == 1:
        axes = [axes]
    ells = np.arange(LMAX + 1, dtype=np.float64)
    sl = slice(2, ELL_PLOT_MAX + 1)
    for ax, (deproj, d) in zip(axes, rows):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ELL_EFF[0], ELL_PLOT_MAX)
        ax.set_ylim(*YLIM)
        ax.plot(ells[sl], np.abs(dl_from_cl(ells, d["cl_tt"]))[sl], color="k", lw=1.7, label=r"input truth $y$", zorder=3)
        ax.plot(ells[sl], d["d_bias"][sl], color="crimson", lw=1.6, ls="--", zorder=5, label=r"$|\Delta C_\ell^{y_1 y_2}|$")
        ax.plot(ells[sl], np.abs(dl_from_cl(ells, d["cl_12_d"]))[sl], color="C1", lw=1.8, alpha=0.45, zorder=2)
        y = np.abs(d["dl_cross"])
        sig = np.asarray(d["dl_cross_sigma"], dtype=np.float64)
        lo = np.maximum(y - sig, y * 1.0e-3)
        ax.fill_between(ELL_EFF, lo, y + sig, color="C1", alpha=0.28, zorder=1, lw=0)
        ax.errorbar(
            ELL_EFF, y, yerr=[y - lo, sig], fmt="o", color="C1", ms=6.5,
            elinewidth=2.2, capsize=4.5, capthick=1.8, zorder=6,
            label=r"HILC $y$ $r_1\times r_2$",
        )
        _curve_and_bins(ax, ells, sl, d["cl_cib_d"], d["dl_cib"], color="C2", marker="^", label=r"CIB residual")
        _curve_and_bins(ax, ells, sl, d["cl_cmb_d"], d["dl_cmb"], color="C4", marker="v", label=r"CMB residual")
        _curve_and_bins(ax, ells, sl, d["cl_n_d"], d["dl_n"], color="0.35", marker="+", label=r"noise residual", lw=1.0)
        ax.set_ylabel(r"$D_\ell$")
        nq = int(np.load(catalogue_path(name))["q_opt"].size)
        hole = rf"$q>5$ holes ($N={nq}$)" if masked else "full sky"
        ax.set_title(
            f"{LABELS[name]}  ({hole}, $N_\\mathrm{{deproj}}={deproj.n_deproj}$, {deproj.label})"
        )
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    axes[-1].set_xlabel(r"$\ell$")
    sky = r"$q>5$ cluster holes" if masked else "full sky"
    fig.suptitle(
        rf"HILC $y$ $r_1\times r_2$ ({sky}, {SCHEME_TITLE})"
        "\n"
        r"lines: unbinned $D_\ell$; points: Planck 2015 XXII bins",
        y=1.01, fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_cib_overlay(rows: list[tuple[DeprojCase, dict]], *, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ells = np.arange(LMAX + 1, dtype=np.float64)
    sl = slice(2, ELL_PLOT_MAX + 1)
    ylim = _deproj_panel_ylim(rows, ells, sl, "cl_cib_d", "dl_cib")
    _style_ax(ax, ylim=ylim)
    for deproj, pack in rows:
        color, marker, ls = DEPROJ_STYLE[deproj.key]
        _curve_and_bins(
            ax, ells, sl, pack["cl_cib_d"], pack["dl_cib"],
            color=color, marker=marker, label=deproj.label, lw=1.4,
        )
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_residual_quad(rows: list[tuple[DeprojCase, dict]], *, title: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
    ells = np.arange(LMAX + 1, dtype=np.float64)
    sl = slice(2, ELL_PLOT_MAX + 1)
    panels = (
        (axes[0, 0], "cl_12_d", "dl_cross", r"HILC $y$ $r_1\times r_2$", True),
        (axes[0, 1], "cl_cib_d", "dl_cib", r"CIB residual", False),
        (axes[1, 0], "cl_cmb_d", "dl_cmb", r"CMB residual", False),
        (axes[1, 1], "cl_n_d", "dl_n", r"noise residual", False),
    )
    for ax, clkey, dlkey, ylab, show_truth in panels:
        _style_ax(ax, ylim=_deproj_panel_ylim(rows, ells, sl, clkey, dlkey, include_truth=show_truth))
        if show_truth and rows:
            ax.plot(
                ells[sl],
                np.abs(dl_from_cl(ells, rows[0][1]["cl_tt"]))[sl],
                color="k",
                lw=1.5,
                ls=":",
                label=r"truth $y$",
                zorder=3,
            )
        for deproj, pack in rows:
            color, marker, _ = DEPROJ_STYLE[deproj.key]
            _curve_and_bins(
                ax, ells, sl, pack[clkey], pack[dlkey],
                color=color, marker=marker, label=deproj.label, lw=1.3,
            )
        ax.set_title(ylab)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_cib_all_prescriptions(
    by_name: dict[str, list[tuple[DeprojCase, dict]]], *, sky: str, out: Path
) -> None:
    n = len(by_name)
    fig, axes = plt.subplots(n, 1, figsize=(8.6, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    ells = np.arange(LMAX + 1, dtype=np.float64)
    sl = slice(2, ELL_PLOT_MAX + 1)
    for ax, (name, rows) in zip(axes, by_name.items()):
        ylim = _deproj_panel_ylim(rows, ells, sl, "cl_cib_d", "dl_cib")
        _style_ax(ax, ylim=ylim)
        for deproj, pack in rows:
            color, marker, _ = DEPROJ_STYLE[deproj.key]
            _curve_and_bins(
                ax, ells, sl, pack["cl_cib_d"], pack["dl_cib"],
                color=color, marker=marker, label=deproj.label, lw=1.3,
            )
        ax.set_title(f"{LABELS[name]}  ({sky})")
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    axes[-1].set_xlabel(r"$\ell$")
    fig.suptitle(rf"CIB residual — {SCHEME_TITLE} ({sky})", y=1.01, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_quad_all_prescriptions(
    by_name: dict[str, list[tuple[DeprojCase, dict]]], *, sky: str, out: Path
) -> None:
    names = list(by_name.keys())
    n = len(names)
    fig, axes = plt.subplots(n, 4, figsize=(16.0, 3.4 * n), sharex=True)
    if n == 1:
        axes = np.array([axes])
    ells = np.arange(LMAX + 1, dtype=np.float64)
    sl = slice(2, ELL_PLOT_MAX + 1)
    col_titles = (
        r"HILC $y$ $r_1\times r_2$",
        r"CIB residual",
        r"CMB residual",
        r"noise residual",
    )
    col_keys = (
        ("cl_12_d", "dl_cross"),
        ("cl_cib_d", "dl_cib"),
        ("cl_cmb_d", "dl_cmb"),
        ("cl_n_d", "dl_n"),
    )
    for i, name in enumerate(names):
        rows = by_name[name]
        for j, (col_title, (clkey, dlkey)) in enumerate(zip(col_titles, col_keys)):
            ax = axes[i, j]
            _style_ax(
                ax,
                ylim=_deproj_panel_ylim(
                    rows, ells, sl, clkey, dlkey, include_truth=(j == 0)
                ),
            )
            if j == 0 and rows:
                ax.plot(
                    ells[sl],
                    np.abs(dl_from_cl(ells, rows[0][1]["cl_tt"]))[sl],
                    color="k",
                    lw=1.2,
                    ls=":",
                    alpha=0.8,
                    zorder=3,
                    label=r"truth $y$",
                )
            for deproj, pack in rows:
                color, marker, _ = DEPROJ_STYLE[deproj.key]
                _curve_and_bins(
                    ax, ells, sl, pack[clkey], pack[dlkey],
                    color=color, marker=marker,
                    label=deproj.label,
                    lw=1.1,
                )
            if i == 0:
                ax.set_title(col_title, fontsize=10)
            if j == 0:
                ax.set_ylabel(LABELS[name], fontsize=10)
            ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\ell$")
    fig.suptitle(rf"Signal and weighted residuals — {SCHEME_TITLE} ({sky})", y=1.01, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    bl, good = diag.bl10(), diag.bl10() >= 1e-3
    for masked in (False, True):
        sky = r"$q>5$ masked" if masked else "full sky"
        tag = "q5masked" if masked else "fullsky"
        by_name: dict[str, list[tuple[DeprojCase, dict]]] = {}
        for name in ALL_RUNS:
            rows = _load_deproj_packs(name, masked=masked, bl=bl, good=good)
            if rows:
                by_name[name] = rows
        if not by_name:
            continue
        # Fig. 9-style residual stack (priority — same format as nodeproj)
        if "L1_m9" in by_name:
            plot_fig9_deproj_stack(
                by_name["L1_m9"],
                name="L1_m9",
                masked=masked,
                out=OUT / f"r1xr2_fig9_{tag}.png",
            )
            plot_cib_overlay(
                by_name["L1_m9"],
                title=rf"L1\_m9 CIB residual ({sky}, {SCHEME_TITLE})",
                out=OUT / f"cib_residual_{tag}.png",
            )
            plot_residual_quad(
                by_name["L1_m9"],
                title=rf"L1\_m9 weighted residuals ({sky}, {SCHEME_TITLE})",
                out=OUT / f"residual_quad_{tag}.png",
            )
        plot_cib_all_prescriptions(by_name, sky=sky, out=OUT / f"cib_residual_{tag}_all.png")
        plot_quad_all_prescriptions(by_name, sky=sky, out=OUT / f"residual_quad_{tag}_all.png")


if __name__ == "__main__":
    main()
