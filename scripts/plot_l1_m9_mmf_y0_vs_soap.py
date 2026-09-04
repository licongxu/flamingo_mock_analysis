#!/usr/bin/env python3
"""MMF ŷ0 at catalogue positions vs SOAP Y (L1_m9).

ŷ0 from spherical-harmonic convolution: y_lm → B_ℓ(θ_500) y_lm, with B_ℓ
the transform of the MMF kernel W (absolute, ∫ t_mmf W dΩ = 1).
θ_500 interpolation/extrapolation is linear in log θ (no clip).

Y_sph = ŷ0 k R_500^2 with FLAMINGO GNFW k.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("SZIFI_ARRAY_BACKEND", "numpy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/scratch/scratch-lxu/flamingo_repo/src")

from flamingo.inference.masked_ps import GNFW_SHAPE  # noqa: E402

from scripts.compute_mmf_W_yt_skyavg import (  # noqa: E402
    POLY_DEG,
    photometry_y0_harmonic,
)

CATALOGUE = Path(
    "/rds/rds-lxu/flamingo/L1_m9/catalogues/"
    "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommap.csv"
)
NOISE = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues/"
    "noise_curve_skyavg_flamingo_immf.npz"
)
FIG_BASE = REPO / "figures" / "l1_m9_mmf_y0_vs_soap"

PAPER_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "legend.fontsize": 9,
    "text.latex.preamble": r"\usepackage{amsmath}",
}


def gnfw_y_over_y0_r2(x_out: float) -> float:
    c500, gamma, alpha, beta = (
        GNFW_SHAPE["c500"],
        GNFW_SHAPE["gamma"],
        GNFW_SHAPE["alpha"],
        GNFW_SHAPE["beta"],
    )

    def p(x: float) -> float:
        cx = c500 * max(float(x), 1e-12)
        return cx ** (-gamma) * (1.0 + cx ** alpha) ** (-(beta - gamma) / alpha)

    i_sph = quad(lambda x: x**2 * p(x), 0.0, x_out, epsabs=1e-10)[0]
    i_los = quad(p, 0.0, np.inf, epsabs=1e-10, limit=200)[0]
    return float(2.0 * np.pi * i_sph / i_los)


K_R500 = gnfw_y_over_y0_r2(1.0)
K_5R500 = gnfw_y_over_y0_r2(5.0)


def load_frame(q_aperture_min: float | None, mmf_q_min: float | None) -> pd.DataFrame:
    cols = [
        "theta_rot_rad",
        "phi_rot_rad",
        "R_500c_Mpc",
        "theta_500_arcmin",
        "Y_500c_Mpc2",
        "Y_5R500c_Mpc2",
        "q_from_aperture",
    ]
    frame = pd.read_csv(CATALOGUE, comment="#", usecols=cols)
    for c in ("Y_500c_Mpc2", "Y_5R500c_Mpc2", "R_500c_Mpc"):
        v = frame[c].to_numpy(np.float64)
        if not np.all(np.isfinite(v)) or not np.all(v > 0):
            raise ValueError(f"{c} must be finite and positive")
    if q_aperture_min is not None:
        frame = frame.loc[frame["q_from_aperture"] > q_aperture_min].copy()
    return frame


def photometry_y0(frame: pd.DataFrame) -> np.ndarray:
    th500 = frame["theta_500_arcmin"].to_numpy(np.float64)
    th = frame["theta_rot_rad"].to_numpy(np.float64)
    ph = frame["phi_rot_rad"].to_numpy(np.float64)
    print(f"harmonic MMF photometry: N={len(frame):,}", flush=True)
    return photometry_y0_harmonic(th, ph, th500)


def mmf_q(y0: np.ndarray, theta500_arcmin: np.ndarray) -> np.ndarray:
    noise = np.load(NOISE)
    coeff = np.polyfit(
        np.log(noise["theta_500_arcmin"]),
        np.log(noise["sigma_y0_flamingo_mock"]),
        POLY_DEG,
    )
    sigma = np.exp(np.polyval(coeff, np.log(theta500_arcmin)))
    return y0 / sigma


def _series(ax, truth, inferred, color, label):
    ok = np.isfinite(truth) & np.isfinite(inferred) & (truth > 0) & (inferred > 0)
    x, y = truth[ok], inferred[ok]
    med = float(np.median(y / x)) if x.size else np.nan
    ax.plot(
        x,
        y,
        ".",
        ms=2.8,
        alpha=0.5,
        color=color,
        rasterized=True,
        zorder=2,
        label=rf"{label}, median ${med:.2f}$",
    )
    return x, y


def build_figure(truth_500, inf_500, truth_5r, inf_5r, *, title_suffix: str) -> plt.Figure:
    plt.rcParams.update(PAPER_RC)
    fig, ax = plt.subplots(figsize=(6.4, 6.0), layout="constrained")
    x500, y500 = _series(ax, truth_500, inf_500, "#1b9e77", r"$Y_{500c}$")
    x5, y5 = _series(ax, truth_5r, inf_5r, "#d95f02", r"$Y_{5R_{500}}$")
    lo = min(x500.min(), y500.min(), x5.min(), y5.min())
    hi = max(x500.max(), y500.max(), x5.max(), y5.max())
    ax.plot([lo, hi], [lo, hi], color="0.2", lw=1.0, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"SOAP $Y^{\rm sph}\,[{\rm Mpc}^{2}]$")
    ax.set_ylabel(
        r"inferred $Y^{\rm sph}= \hat y_0\,k\,R_{500}^{2}$"
        rf" ($k_{{500}}={K_R500:.3f}$, $k_{{5R}}={K_5R500:.3f}$)"
        r"$\,[{\rm Mpc}^{2}]$"
    )
    ax.set_title(rf"L1\_m9: aperture $\hat y_0$ vs SOAP{title_suffix}")
    ax.legend(loc="upper left", frameon=False, markerscale=4)
    return fig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--q-aperture-min", type=float, default=None)
    p.add_argument("--mmf-q-min", type=float, default=5.0)
    p.add_argument("--out-stem", type=Path, default=FIG_BASE)
    p.add_argument("--cache", type=Path, default=None)
    args = p.parse_args()

    frame = load_frame(args.q_aperture_min, args.mmf_q_min)
    th500 = frame["theta_500_arcmin"].to_numpy(np.float64)

    if args.cache and args.cache.is_file():
        y0 = np.load(args.cache)["y0_hat"]
        if y0.size != len(frame):
            raise ValueError("cache length mismatch")
    else:
        y0 = photometry_y0(frame)
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.cache, y0_hat=y0)
            print("wrote photometry cache", args.cache, flush=True)

    q = mmf_q(y0, th500)
    if args.mmf_q_min is not None:
        keep = np.isfinite(q) & (q > args.mmf_q_min)
        title_tag = rf", MMF $q>{args.mmf_q_min:g}$"
    elif args.q_aperture_min is not None:
        keep = np.ones(len(frame), dtype=bool)
        title_tag = rf", $q_{{\rm ap}}>{args.q_aperture_min:g}$"
    else:
        keep = np.ones(len(frame), dtype=bool)
        title_tag = ", all clusters"
    frame = frame.loc[keep].copy()
    y0 = y0[keep]
    q = q[keep]
    r500 = frame["R_500c_Mpc"].to_numpy(np.float64)
    print(f"N={len(frame):,}{title_tag}", flush=True)

    inf_500 = y0 * K_R500 * r500**2
    inf_5r = y0 * K_5R500 * r500**2
    truth_500 = frame["Y_500c_Mpc2"].to_numpy(np.float64)
    truth_5r = frame["Y_5R500c_Mpc2"].to_numpy(np.float64)

    suffix = rf"{title_tag}, $N={len(frame):,}$"
    fig = build_figure(truth_500, inf_500, truth_5r, inf_5r, title_suffix=suffix)
    out_stem = args.out_stem
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = out_stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out, flush=True)
    plt.close(fig)

    np.savez_compressed(
        out_stem.with_suffix(".npz"),
        y0_hat=y0,
        Y500_inferred=inf_500,
        Y5R500_inferred=inf_5r,
        Y500_soap=truth_500,
        Y5R500_soap=truth_5r,
        mmf_q=q,
    )


if __name__ == "__main__":
    main()
