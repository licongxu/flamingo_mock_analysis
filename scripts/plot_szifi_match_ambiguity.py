#!/usr/bin/env python3
"""Gnomonic zooms of detections with ambiguous truth matches (small offsets)."""

from __future__ import annotations

from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.spatial import cKDTree

from flamingo_mock.szifi.paths import SZiFiPaths
from flamingo_mock.szifi.validate import (
    DEFAULT_TRUTH_CATALOGUE,
    _lonlat_to_vec,
    run_match,
)

YMAP = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/tsz/"
    "compton_y_nside4096.fits"
)
CAT = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues/"
    "footprint_splitA_immf_q5.npz"
)
OUT = Path("figures/szifi_match_ambiguity_zooms.png")
OUT_DIR = Path("figures/szifi_match_ambiguity")


def chord_to_arcmin(d: np.ndarray) -> np.ndarray:
    return 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0)) * 180.0 * 60.0 / np.pi


def pick_cases(match, truth, ang, jnn, *, r_match=10.0):
    """Select illustrative borderline cases around Zubeldia 2024 radius."""
    fp = ~match.det_hit
    # Should-be-TP under a tighter cut: matched at 10' but nearest was >5'
    should_tp = []
    for i in np.where(match.det_hit & (ang > 5) & (ang <= r_match))[0]:
        j = jnn[i]
        if truth["q_from_aperture"][j] >= 5 and truth["M_500c_Msun"][j] >= 1e14:
            should_tp.append(i)
    should_tp = sorted(should_tp, key=lambda i: -match.det_q[i])

    # Likely FP: unmatched at 10', weak nearest truth
    likely_fp = []
    for i in np.where(fp & (ang > r_match) & (ang < 20))[0]:
        j = jnn[i]
        if truth["q_from_aperture"][j] < 2 and match.det_q[i] >= 12:
            likely_fp.append(i)
    likely_fp = sorted(likely_fp, key=lambda i: -match.det_q[i])

    # Clean TP near the 10' edge
    edge_tp = np.where(match.det_hit & (ang > 7.0) & (ang <= r_match))[0]
    edge_tp = sorted(edge_tp, key=lambda i: -match.det_q[i])

    picks, labels = [], []
    for i in should_tp[:3]:
        picks.append(i)
        labels.append(f"TP@{r_match:.0f}' (would be FP@5'): offset {ang[i]:.1f}'")
    for i in likely_fp[:3]:
        if i not in picks:
            picks.append(i)
            labels.append(f"FP@{r_match:.0f}': no strong truth nearby")
    for i in edge_tp[:2]:
        if i not in picks:
            picks.append(i)
            labels.append(f"TP@{r_match:.0f}' near radius edge")
    return picks[:8], labels[:8]


def plot_panel(ax, ymap, nside, lon0, lat0, det_q, det_th, truth, tree, case_label):
    reso = 1.0  # arcmin/pix
    xsize = 80  # 80' field
    patch = hp.gnomview(
        ymap,
        rot=(lon0, lat0),
        reso=reso,
        xsize=xsize,
        return_projected_map=True,
        no_plot=True,
    )
    extent = [-xsize / 2, xsize / 2, -xsize / 2, xsize / 2]
    vmax = float(np.percentile(patch[patch > 0], 99.5)) if np.any(patch > 0) else 1e-5
    im = ax.imshow(
        patch,
        origin="lower",
        extent=extent,
        cmap="hot",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )

    # Detection at centre
    ax.plot(0, 0, "c+", ms=14, mew=2, zorder=5, label=f"detection q={det_q:.1f}")
    ax.add_patch(Circle((0, 0), 5, fill=False, ec="0.7", ls=":", lw=1.0, label="5'"))
    ax.add_patch(Circle((0, 0), 10, fill=False, ec="lime", ls="--", lw=1.4, label="10' (Zubeldia 2024)"))
    ax.add_patch(
        Circle((0, 0), float(det_th), fill=False, ec="white", ls=":", lw=1.0, alpha=0.8, label=r"det $\theta_{500}$")
    )

    # Nearby truth within 20'
    dvec = _lonlat_to_vec(np.array([lon0]), np.array([lat0]))[0]
    dists, idxs = tree.query(dvec, k=min(40, len(truth["z"])))
    angs = chord_to_arcmin(dists)
    for a, j in zip(np.atleast_1d(angs), np.atleast_1d(idxs)):
        if a > 20:
            continue
        lon_t = truth["lon_rot_deg"][j]
        lat_t = truth["lat_rot_deg"][j]
        dlon = np.deg2rad((lon_t - lon0 + 180) % 360 - 180)
        lat0r, lattr = np.deg2rad(lat0), np.deg2rad(lat_t)
        x = np.rad2deg(dlon * np.cos(0.5 * (lat0r + lattr))) * 60.0  # east
        y = (lat_t - lat0) * 60.0  # north
        qa = truth["q_from_aperture"][j]
        m14 = truth["M_500c_Msun"][j] / 1e14
        color = "lime" if qa >= 5 else "orange"
        ms = 8 + 4 * min(m14, 5)
        ax.plot(x, y, "o", mfc="none", mec=color, ms=ms, mew=1.5, zorder=4)
        ax.annotate(
            f"{a:.1f}'\nq={qa:.1f}\nM={m14:.1f}",
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6,
            color="white",
        )
        th500 = float(truth["theta_500_arcmin"][j])
        ax.add_patch(Circle((x, y), th500, fill=False, ec=color, lw=0.7, alpha=0.7))

    ax.set_xlim(-xsize / 2, xsize / 2)
    ax.set_ylim(-xsize / 2, xsize / 2)
    ax.set_aspect("equal")
    ax.set_xlabel("E (arcmin)")
    ax.set_ylabel("N (arcmin)")
    ax.set_title(f"{case_label}\nlon={lon0:.2f}, lat={lat0:.2f}", fontsize=8)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.8)
    return im


def main():
    paths = SZiFiPaths()
    match, truth, _det, _ = run_match(CAT, paths=paths, match_radius_arcmin=10.0)
    tvec = _lonlat_to_vec(truth["lon_rot_deg"], truth["lat_rot_deg"])
    tree = cKDTree(tvec)
    dist, jnn = tree.query(_lonlat_to_vec(match.det_lon, match.det_lat), k=1)
    ang = chord_to_arcmin(dist)

    picks, labels = pick_cases(match, truth, ang, jnn, r_match=10.0)
    print("Selected cases:")
    for i, lab in zip(picks, labels):
        print(f"  i={i} q={match.det_q[i]:.2f} d_nn={ang[i]:.2f}'  [{lab}]")

    ymap = np.asarray(hp.read_map(str(YMAP), dtype=np.float64))
    nside = hp.npix2nside(ymap.size)

    n = len(picks)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows))
    axes = np.atleast_2d(axes)
    for k, (i, lab) in enumerate(zip(picks, labels)):
        r, c = divmod(k, ncols)
        plot_panel(
            axes[r, c],
            ymap,
            nside,
            float(match.det_lon[i]),
            float(match.det_lat[i]),
            float(match.det_q[i]),
            float(match.det_theta[i]),
            truth,
            tree,
            lab,
        )
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    fig.suptitle(
        "Zubeldia 2024 matching: 10' radius (lime), 5' shown for comparison (grey)\n"
        "cyan=detection, green=truth q_ap>=5, orange=weaker truth",
        fontsize=11,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")

    # Also save individual high-res zooms for the clearest examples
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, lab in zip(picks[:4], labels[:4]):
        fig, ax = plt.subplots(figsize=(5.5, 5.2))
        plot_panel(
            ax,
            ymap,
            nside,
            float(match.det_lon[i]),
            float(match.det_lat[i]),
            float(match.det_q[i]),
            float(match.det_theta[i]),
            truth,
            tree,
            lab,
        )
        p = OUT_DIR / f"det_{i}_q{match.det_q[i]:.1f}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
