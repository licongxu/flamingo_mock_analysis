#!/usr/bin/env python3
"""Gnomonic zooms of in-mask true positives (matched truth within 10')."""

from __future__ import annotations

import argparse
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
    cross_match_greedy,
    run_match,
)

YMAP_DEFAULT = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/tsz/"
    "compton_y_nside4096.fits"
)
CAT_DIR = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues")


def chord_to_arcmin(d: np.ndarray) -> np.ndarray:
    return 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0)) * 180.0 * 60.0 / np.pi


def match_with_assignments(
    match,
    truth: dict[str, np.ndarray],
    *,
    r_match: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy match; return (det_hit, truth_index per det, -1 if unmatched)."""
    n_det = len(match.det_lon)
    n_truth = len(truth["z"])
    det_hit = np.zeros(n_det, dtype=bool)
    truth_j = np.full(n_det, -1, dtype=np.int64)
    truth_hit = np.zeros(n_truth, dtype=bool)
    if n_det == 0 or n_truth == 0:
        return det_hit, truth_j

    tree = cKDTree(_lonlat_to_vec(truth["lon_rot_deg"], truth["lat_rot_deg"]))
    order = np.argsort(-match.det_q)
    cos_r = np.cos(np.deg2rad(r_match / 60.0))

    for i in order:
        dist, j = tree.query(
            _lonlat_to_vec(match.det_lon[i : i + 1], match.det_lat[i : i + 1])[0], k=1
        )
        cos_ang = 1.0 - 0.5 * dist * dist
        if cos_ang >= cos_r and not truth_hit[j]:
            det_hit[i] = True
            truth_hit[j] = True
            truth_j[i] = j
    return det_hit, truth_j


def offset_arcmin(lon0: float, lat0: float, lon1: float, lat1: float) -> tuple[float, float]:
    dlon = np.deg2rad((lon1 - lon0 + 180) % 360 - 180)
    lat0r, lat1r = np.deg2rad(lat0), np.deg2rad(lat1)
    x = np.rad2deg(dlon * np.cos(0.5 * (lat0r + lat1r))) * 60.0
    y = (lat1 - lat0) * 60.0
    return float(x), float(y)


def pick_tp_cases(
    match,
    truth,
    truth_j: np.ndarray,
    ang_nn: np.ndarray,
    *,
    r_match: float = 10.0,
    n_pick: int = 8,
) -> tuple[list[int], list[str]]:
    tp_idx = np.where(match.det_hit)[0]
    if len(tp_idx) == 0:
        return [], []

    picks: list[int] = []
    labels: list[str] = []
    used: set[int] = set()

    def add(i: int, label: str) -> None:
        if i in used or len(picks) >= n_pick:
            return
        used.add(i)
        picks.append(i)
        labels.append(label)

    # Bright, well-centred matches
    for i in sorted(tp_idx, key=lambda k: -match.det_q[k])[:3]:
        j = int(truth_j[i])
        add(
            i,
            f"high-q TP: q={match.det_q[i]:.1f}, offset={ang_nn[i]:.1f}' "
            f"(truth q_ap={truth['q_from_aperture'][j]:.1f})",
        )

    # Clean centroid (small offset)
    tight = tp_idx[ang_nn[tp_idx] < 2.0]
    for i in sorted(tight, key=lambda k: -match.det_q[k])[:2]:
        j = int(truth_j[i])
        add(
            i,
            f"tight match: offset={ang_nn[i]:.2f}', q={match.det_q[i]:.1f}, "
            f"M={truth['M_500c_Msun'][j]/1e14:.1f}×10¹⁴",
        )

    # Near match-radius edge
    edge = tp_idx[(ang_nn[tp_idx] > 0.7 * r_match) & (ang_nn[tp_idx] <= r_match)]
    for i in sorted(edge, key=lambda k: -match.det_q[k])[:2]:
        j = int(truth_j[i])
        add(
            i,
            f"edge TP: offset={ang_nn[i]:.1f}' (≤{r_match:.0f}'), "
            f"truth q_ap={truth['q_from_aperture'][j]:.1f}",
        )

    # Threshold detections that still match
    low = tp_idx[(match.det_q[tp_idx] >= 5.0) & (match.det_q[tp_idx] < 6.5)]
    for i in sorted(low, key=lambda k: match.det_q[k])[:2]:
        j = int(truth_j[i])
        add(
            i,
            f"threshold TP: q={match.det_q[i]:.2f}, offset={ang_nn[i]:.1f}', "
            f"truth q_ap={truth['q_from_aperture'][j]:.1f}",
        )

    rest = [i for i in tp_idx if i not in used]
    for i in sorted(rest, key=lambda k: -match.det_q[k]):
        if len(picks) >= n_pick:
            break
        j = int(truth_j[i])
        add(
            i,
            f"TP q={match.det_q[i]:.1f}, offset={ang_nn[i]:.1f}', "
            f"truth q_ap={truth['q_from_aperture'][j]:.1f}",
        )

    return picks[:n_pick], labels[:n_pick]


def plot_panel(
    ax,
    ymap: np.ndarray,
    lon0: float,
    lat0: float,
    det_q: float,
    det_th: float,
    truth: dict[str, np.ndarray],
    tree: cKDTree,
    truth_j: int,
    case_label: str,
    *,
    r_match: float = 10.0,
) -> None:
    reso = 1.0
    xsize = 80
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
    ax.imshow(
        patch,
        origin="lower",
        extent=extent,
        cmap="hot",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.plot(0, 0, "c+", ms=14, mew=2, zorder=5, label=f"TP det q={det_q:.1f}")
    ax.add_patch(Circle((0, 0), 5, fill=False, ec="0.7", ls=":", lw=1.0, label="5'"))
    ax.add_patch(
        Circle((0, 0), r_match, fill=False, ec="lime", ls="--", lw=1.4, label=f"{r_match:.0f}' match")
    )
    ax.add_patch(
        Circle(
            (0, 0),
            float(det_th),
            fill=False,
            ec="white",
            ls=":",
            lw=1.0,
            alpha=0.8,
            label=r"det $\theta_{500}$",
        )
    )

    # Matched truth (star)
    if truth_j >= 0:
        lon_m = float(truth["lon_rot_deg"][truth_j])
        lat_m = float(truth["lat_rot_deg"][truth_j])
        xm, ym = offset_arcmin(lon0, lat0, lon_m, lat_m)
        qa = float(truth["q_from_aperture"][truth_j])
        m14 = float(truth["M_500c_Msun"][truth_j]) / 1e14
        th500 = float(truth["theta_500_arcmin"][truth_j])
        ax.plot(xm, ym, "*", mfc="lime", mec="white", ms=16, mew=0.8, zorder=6, label="matched truth")
        ax.annotate(
            f"matched\n{np.hypot(xm, ym):.1f}'\nq_ap={qa:.1f}\nM={m14:.1f}",
            (xm, ym),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
            color="lime",
            fontweight="bold",
        )
        ax.add_patch(Circle((xm, ym), th500, fill=False, ec="lime", lw=1.2, alpha=0.95))

    # Other nearby truth (fainter)
    dvec = _lonlat_to_vec(np.array([lon0]), np.array([lat0]))[0]
    dists, idxs = tree.query(dvec, k=min(40, len(truth["z"])))
    angs = chord_to_arcmin(np.atleast_1d(dists))
    for a, j in zip(angs, np.atleast_1d(idxs)):
        if a > 25 or j == truth_j:
            continue
        lon_t = float(truth["lon_rot_deg"][j])
        lat_t = float(truth["lat_rot_deg"][j])
        x, y = offset_arcmin(lon0, lat0, lon_t, lat_t)
        qa = float(truth["q_from_aperture"][j])
        color = "orange" if qa >= 2 else "0.55"
        ax.plot(x, y, "o", mfc="none", mec=color, ms=7, mew=1.0, zorder=3, alpha=0.8)

    ax.set_xlim(-xsize / 2, xsize / 2)
    ax.set_ylim(-xsize / 2, xsize / 2)
    ax.set_aspect("equal")
    ax.set_xlabel("E (arcmin)")
    ax.set_ylabel("N (arcmin)")
    ax.set_title(f"{case_label}\nlon={lon0:.2f}, lat={lat0:.2f}", fontsize=8)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.85)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--catalogue",
        type=Path,
        default=CAT_DIR / "footprint_splitA_immf_q5.npz",
    )
    p.add_argument("--truth", type=Path, default=DEFAULT_TRUTH_CATALOGUE)
    p.add_argument("--ymap", type=Path, default=YMAP_DEFAULT)
    p.add_argument("--match-radius-arcmin", type=float, default=10.0)
    p.add_argument("--n-panels", type=int, default=8)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("figures/szifi_true_positives_zooms.png"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/szifi_true_positives"),
    )
    args = p.parse_args()

    paths = SZiFiPaths()
    match, truth, _det, _ = run_match(
        args.catalogue,
        truth_csv=args.truth,
        paths=paths,
        match_radius_arcmin=args.match_radius_arcmin,
    )
    n_tp = int(match.det_hit.sum())
    print(f"in-mask detections={len(match.det_q)} TPs={n_tp}")

    tvec = _lonlat_to_vec(truth["lon_rot_deg"], truth["lat_rot_deg"])
    tree = cKDTree(tvec)
    dist, j_nn = tree.query(_lonlat_to_vec(match.det_lon, match.det_lat), k=1)
    ang_nn = chord_to_arcmin(dist)
    _, truth_j = match_with_assignments(match, truth, r_match=args.match_radius_arcmin)

    picks, labels = pick_tp_cases(
        match, truth, truth_j, ang_nn, r_match=args.match_radius_arcmin, n_pick=args.n_panels
    )
    if not picks:
        print("No true positives to plot.")
        return

    print("Selected TPs:")
    for i, lab in zip(picks, labels):
        j = int(truth_j[i])
        print(
            f"  i={i} q={match.det_q[i]:.2f} offset={ang_nn[i]:.2f}' "
            f"truth_j={j} q_ap={truth['q_from_aperture'][j]:.1f}  [{lab}]"
        )

    ymap = np.asarray(hp.read_map(str(args.ymap), dtype=np.float64))

    ncols = 4
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.3 * nrows))
    axes = np.atleast_2d(axes)
    for k, (i, lab) in enumerate(zip(picks, labels)):
        r, c = divmod(k, ncols)
        plot_panel(
            axes[r, c],
            ymap,
            float(match.det_lon[i]),
            float(match.det_lat[i]),
            float(match.det_q[i]),
            float(match.det_theta[i]),
            truth,
            tree,
            int(truth_j[i]),
            lab,
            r_match=args.match_radius_arcmin,
        )
    for k in range(len(picks), nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    method = "iMMF" if "immf" in args.catalogue.name.lower() else "sciMMF"
    fig.suptitle(
        f"{method} true positives (in-mask, q≥5): cyan=detection, lime star=matched truth\n"
        f"green dashed = {args.match_radius_arcmin:.0f}' match radius",
        fontsize=11,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=165, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, lab in zip(picks, labels):
        fig, ax = plt.subplots(figsize=(5.8, 5.4))
        plot_panel(
            ax,
            ymap,
            float(match.det_lon[i]),
            float(match.det_lat[i]),
            float(match.det_q[i]),
            float(match.det_theta[i]),
            truth,
            tree,
            int(truth_j[i]),
            lab,
            r_match=args.match_radius_arcmin,
        )
        safe = lab.split(":")[0].replace(" ", "_")[:24]
        out = args.out_dir / f"tp_{i}_q{match.det_q[i]:.1f}_{safe}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=175, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
