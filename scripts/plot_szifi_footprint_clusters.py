#!/usr/bin/env python3
"""Mollweide Compton-y map with SZiFi detections circled (theta_500 radius)."""

from __future__ import annotations

import argparse
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from flamingo_mock.szifi.paths import SZiFiPaths
from flamingo_mock.szifi.tiles import load_pr4_gal_ps
from flamingo_mock.szifi.validate import detection_q_mask


def load_catalogue(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {k: np.asarray(d[k]) for k in d.files}


def plot_detection_circles(lon, lat, theta_500, color: str):
    """Hollow markers scaled to theta_500 (arcmin) — fast for ~1k detections."""
    # Empirical scale: marker area ~ angular diameter on mollweide plot.
    s = np.clip((theta_500 / 60.0) ** 2 * 120.0, 4.0, 400.0)
    hp.projscatter(
        lon,
        lat,
        lonlat=True,
        s=s,
        facecolors="none",
        edgecolors=color,
        linewidths=0.45,
        alpha=0.85,
        label=f"iMMF ($n={len(lon)}$)",
    )


def plot_mollview(
    ymap: np.ndarray,
    mask: np.ndarray,
    catalogues: list[tuple[str, dict[str, np.ndarray], str]],
    out_path: Path,
    title: str,
    nside: int,
) -> None:
    masked = ymap.copy()
    masked[mask < 0.5] = hp.UNSEEN
    good = masked[masked != hp.UNSEEN]
    vmax = float(np.percentile(good[good > 0], 99.5))
    vmin = float(np.percentile(good[good > 0], 1.0))

    hp.mollview(
        masked,
        min=vmin,
        max=vmax,
        title=title,
        unit=r"$y$",
        cmap="hot",
        cbar=True,
        hold=True,
    )
    hp.graticule(dmer=30, dpar=30, alpha=0.25, verbose=False)

    for _label, cat, color in catalogues:
        plot_detection_circles(cat["lon"], cat["lat"], cat["theta_500"], color)

    n_det = sum(len(cat["lon"]) for _, cat, _ in catalogues)
    plt.legend(loc="lower right", fontsize=10, framealpha=0.9)
    plt.gcf().text(
        0.02,
        0.02,
        f"{n_det} clusters detected (q >= 5)",
        transform=plt.gcf().transFigure,
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ymap",
        type=Path,
        default=Path(
            "/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/tsz/"
            "compton_y_nside4096.fits"
        ),
    )
    parser.add_argument("--nside", type=int, default=2048)
    parser.add_argument(
        "--immf",
        type=Path,
        default=Path(
            "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues/"
            "footprint_splitA_immf_q5.npz"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/szifi_footprint_immf_mollview.png"),
    )
    args = parser.parse_args()

    paths = SZiFiPaths()
    gal, ps = load_pr4_gal_ps(paths.masks_fits, nside=args.nside)
    mask = gal * ps

    y4096 = np.asarray(hp.read_map(str(args.ymap), dtype=np.float64))
    if hp.npix2nside(y4096.size) != args.nside:
        ymap = hp.ud_grade(y4096, args.nside)
    else:
        ymap = y4096

    if not args.immf.is_file():
        raise FileNotFoundError(f"Missing iMMF catalogue: {args.immf}")

    cat = load_catalogue(args.immf)
    n_det = int(detection_q_mask(cat, 5.0).sum())
    catalogues: list[tuple[str, dict[str, np.ndarray], str]] = [
        ("iMMF", cat, "cyan"),
    ]

    title = (
        f"Truth Compton-y + iMMF detections "
        f"(N={n_det}, q>=5, GAL x PS mask)"
    )
    print(f"iMMF detections: N={n_det}")
    plot_mollview(ymap, mask, catalogues, args.out, title, args.nside)


if __name__ == "__main__":
    main()
