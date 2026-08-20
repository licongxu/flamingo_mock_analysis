#!/usr/bin/env python3
"""L1_m9 q>5: Y_ap vs SOAP with sky-averaged W at each cluster's theta_500.

    W(x; theta_500) interpolated in log theta_500 from sky-avg profiles
    at 2', 5', 10' (x = theta / theta_500, tabulated to x=5).
    Y_ap = Omega_pix sum y_i W(|theta_i - theta_c| / theta_500)

Not a geometric stretch of a single 5' kernel. No mean subtraction.
sr -> Mpc^2 with D_A = R_500 / theta_500.
"""

from __future__ import annotations

import os
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("SZIFI_ARRAY_BACKEND", "numpy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CATALOGUE = Path(
    "/rds/rds-lxu/flamingo/.hbt_join_fix_staging/20260731/L1_m9/catalogues/"
    "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommap.csv"
)
YMAP = Path("/rds/rds-lxu/flamingo/L1_m9/maps/y_unlensed_L1_m9_lc0_nside4096.fits")
W_NPZ = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues/"
    "mmf_aperture/W_theta_skyavg.npz"
)
Q_MIN = 5.0
ARCMIN_PER_RAD = 180.0 * 60.0 / np.pi
X_MAX = 5.0
N_WORKERS = 8

_YMAP = None
_NSIDE = None
_OMEGA = None
_TE = None
_WGRID = None
_LOGTH = None


def w_profile(theta500_arcmin: float) -> np.ndarray:
    """Sky-avg W(x)/W(0) at this theta_500, linear in log theta_500."""
    return np.stack(
        [np.interp(np.log(theta500_arcmin), _LOGTH, _WGRID[:, i]) for i in range(_TE.size)],
        axis=0,
    )


def y_ap_at(lon_deg: float, lat_deg: float, theta500_arcmin: float) -> float:
    import healpy as hp

    r_max = (X_MAX * theta500_arcmin) / ARCMIN_PER_RAD
    vec = hp.ang2vec(lon_deg, lat_deg, lonlat=True)
    pix = hp.query_disc(_NSIDE, vec, r_max)
    if pix.size == 0:
        return np.nan
    x = hp.rotator.angdist(vec, hp.pix2vec(_NSIDE, pix)) * ARCMIN_PER_RAD / theta500_arcmin
    w = np.interp(x, _TE, w_profile(theta500_arcmin), left=1.0, right=0.0)
    return float(_OMEGA * np.sum(_YMAP[pix] * w, dtype=np.float64))


def _init(ymap, nside, omega, te, wgrid, logth):
    global _YMAP, _NSIDE, _OMEGA, _TE, _WGRID, _LOGTH
    _YMAP = ymap
    _NSIDE = nside
    _OMEGA = omega
    _TE = te
    _WGRID = wgrid
    _LOGTH = logth


def _chunk(payload):
    lon, lat, th500 = payload
    out = np.empty(lon.size, dtype=np.float64)
    for i, (lo, la, th) in enumerate(zip(lon, lat, th500)):
        out[i] = y_ap_at(float(lo), float(la), float(th))
    return out


def stats(name: str, truth: np.ndarray, inferred: np.ndarray) -> None:
    ratio = inferred / truth
    pos = np.isfinite(ratio) & (truth > 0) & (inferred > 0)
    print(
        f"{name}: N={pos.sum():,}  median={np.median(ratio[pos]):.4g}  "
        f"p16={np.percentile(ratio[pos], 16):.4g}  "
        f"p84={np.percentile(ratio[pos], 84):.4g}"
    )


def main() -> None:
    import healpy as hp

    wdata = np.load(W_NPZ)
    te = np.asarray(wdata["theta_eff"], dtype=np.float64)
    wgrid = np.asarray(wdata["W_skyavg"], dtype=np.float64)
    thetas = np.asarray(wdata["thetas"], dtype=np.float64)
    logth = np.log(thetas)
    print(
        f"sky-avg W at theta_500={thetas} arcmin, "
        f"N_tile={int(wdata['n_tiles'])}, x<= {te.max():g}"
    )

    cols = [
        "lon_rot_deg",
        "lat_rot_deg",
        "R_500c_Mpc",
        "theta_500_arcmin",
        "Y_500c_Mpc2",
        "Y_5R500c_Mpc2",
        "q_from_aperture",
    ]
    frame = pd.read_csv(CATALOGUE, comment="#", usecols=cols)
    frame = frame.loc[frame["q_from_aperture"].to_numpy(np.float64) > Q_MIN].copy()
    n = len(frame)
    th500 = frame["theta_500_arcmin"].to_numpy(np.float64)
    print(
        f"N(q_from_aperture>{Q_MIN:g})={n:,}  "
        f"theta_500 in [{th500.min():.2f}, {th500.max():.2f}]'  "
        f"n_outside_[{thetas[0]:g},{thetas[-1]:g}]'="
        f"{int(np.sum((th500 < thetas[0]) | (th500 > thetas[-1])))}"
    )

    print("reading y map (no mean subtraction)")
    ymap = np.asarray(hp.read_map(str(YMAP), dtype=np.float32), dtype=np.float32)
    nside = hp.npix2nside(ymap.size)
    omega = hp.nside2pixarea(nside)

    lon = frame["lon_rot_deg"].to_numpy(np.float64)
    lat = frame["lat_rot_deg"].to_numpy(np.float64)
    r500 = frame["R_500c_Mpc"].to_numpy(np.float64)
    truth_500 = frame["Y_500c_Mpc2"].to_numpy(np.float64)
    truth_5r = frame["Y_5R500c_Mpc2"].to_numpy(np.float64)
    q = frame["q_from_aperture"].to_numpy(np.float64)

    chunk = max(1, n // (N_WORKERS * 16))
    slices = [
        (lon[i : i + chunk], lat[i : i + chunk], th500[i : i + chunk])
        for i in range(0, n, chunk)
    ]
    print(f"Y_ap: {n:,} clusters, {N_WORKERS} workers")
    y_ap = np.empty(n, dtype=np.float64)
    done = 0
    with get_context("fork").Pool(
        N_WORKERS,
        initializer=_init,
        initargs=(ymap, nside, omega, te, wgrid, logth),
    ) as pool:
        for part in pool.imap(_chunk, slices, chunksize=1):
            y_ap[done : done + part.size] = part
            done += part.size
            if done % 500 == 0 or done == n:
                print(f"  {done}/{n}", flush=True)

    y_ap *= (r500 / (th500 / ARCMIN_PER_RAD)) ** 2

    print("\nY_ap (sky-avg W at each θ_500) vs SOAP:")
    stats("Y_500c", truth_500, y_ap)
    stats("Y_5R500c", truth_5r, y_ap)
    npos = int(np.sum(np.isfinite(y_ap) & (y_ap > 0)))
    print(f"Y_ap>0: {npos}/{n}")

    bins = [(0.0, 3.5), (3.5, 8.0), (8.0, np.inf)]
    print("median Y_ap/Y_500c by theta_500:")
    for lo, hi in bins:
        m = (th500 >= lo) & (th500 < hi)
        stats(f"  [{lo:g},{hi:g})'", truth_500[m], y_ap[m])

    fig_base = Path("figures/l1_m9_aperture_W_vs_soap")
    np.savez(
        fig_base.with_suffix(".npz"),
        Y500_soap=truth_500,
        Y5R500_soap=truth_5r,
        Y_ap=y_ap,
        q=q,
        theta_500_arcmin=th500,
        thetas_W=thetas,
    )

    ok = np.isfinite(y_ap) & (y_ap > 0) & (truth_500 > 0) & (truth_5r > 0)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0), layout="constrained")
    ax = axes[0]
    for truth, color, label in (
        (truth_500, "#1b9e77", r"$Y_{500c}$"),
        (truth_5r, "#d95f02", r"$Y_{5R_{500}}$"),
    ):
        m = ok & (truth > 0)
        ax.plot(
            truth[m], y_ap[m], ".", ms=2.4, alpha=0.45, color=color, rasterized=True,
            label=rf"{label}, median ${np.median(y_ap[m]/truth[m]):.2f}$",
        )
    lo = min(truth_500[ok].min(), truth_5r[ok].min(), y_ap[ok].min())
    hi = max(truth_500[ok].max(), truth_5r[ok].max(), y_ap[ok].max())
    ax.plot([lo, hi], [lo, hi], color="0.2", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"SOAP $Y\,[\mathrm{Mpc}^2]$")
    ax.set_ylabel(r"$Y_{\rm ap}\,[{\rm Mpc}^{2}]$")
    ax.legend(loc="upper left", frameon=False, markerscale=4)
    ax.set_title(rf"sky-avg $W(\theta;\theta_{{500}})$, $q>5$, $N={n:,}$")

    ax = axes[1]
    ax.plot(th500[ok], y_ap[ok] / truth_500[ok], ".", ms=2.6, alpha=0.45,
            color="#1b9e77", rasterized=True)
    ax.axhline(1.0, color="0.2", lw=1.0)
    ax.axvline(2.0, color="0.5", ls="--", lw=0.8)
    ax.axvline(10.0, color="0.5", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta_{500}\,[{\rm arcmin}]$")
    ax.set_ylabel(r"$Y_{\rm ap}/Y_{500c}$")
    ax.set_title(r"cluster by cluster")
    fig.savefig(fig_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(fig_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Wrote", fig_base.with_suffix(".png"))


if __name__ == "__main__":
    main()
