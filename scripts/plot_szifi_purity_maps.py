#!/usr/bin/env python3
"""Purity (iMMF vs sciMMF) and Compton-y maps with detection circles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from flamingo_mock.szifi.paths import SZiFiPaths
from flamingo_mock.szifi.tiles import load_pr4_gal_ps
from flamingo_mock.szifi.validate import (
    DEFAULT_TRUTH_CATALOGUE,
    benchmark_catalogue,
    benchmark_snr_bins,
    run_match,
    write_benchmark_json,
)

CAT_DIR = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues")
YMAP_DEFAULT = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/tsz/"
    "compton_y_nside4096.fits"
)


def plot_purity_comparison(
    series: list[tuple[str, list[float], list[float], list[int], list[float]]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"iMMF": "#1f77b4", "sciMMF": "#d62728"}
    markers = {"iMMF": "o", "sciMMF": "s"}
    for name, th, pur, n, err in series:
        x = np.asarray(th, dtype=np.float64)
        y = np.asarray(pur, dtype=np.float64) * 100.0
        e = np.asarray(err, dtype=np.float64) * 100.0
        m = np.isfinite(y)
        ax.errorbar(
            x[m],
            y[m],
            yerr=e[m],
            fmt=f"{markers[name]}-",
            color=colors[name],
            capsize=3,
            lw=1.6,
            markersize=6,
            label=name,
        )
        for xi, yi, ni in zip(x[m], y[m], np.asarray(n)[m]):
            ax.annotate(
                f"n={int(ni)}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
                color=colors[name],
            )
    ax.set_ylim(0, 105)
    ax.set_xlabel(r"Detection SNR threshold ($q_{\rm opt}\geq q_{\rm th}$)")
    ax.set_ylabel("Purity (%)")
    ax.set_title("Purity vs SNR threshold (Zubeldia 2024, 10′ match)")
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.95)
    note = (
        "PR4 GAL×PS footprint on detections and truth (Zubeldia in-mask purity); "
        "10′ match"
    )
    fig.text(0.5, -0.02, note, ha="center", fontsize=8.5, color="0.35")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_detection_map(
    ymap: np.ndarray,
    mask: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    theta_500: np.ndarray,
    det_hit: np.ndarray,
    *,
    title: str,
    out_path: Path,
    purity: float | None = None,
    n_tp: int | None = None,
    n_det: int | None = None,
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

    for hit, color, label in (
        (det_hit, "lime", f"true positive ({int(det_hit.sum())})"),
        (~det_hit, "red", f"false positive ({int((~det_hit).sum())})"),
    ):
        if not hit.any():
            continue
        s = np.clip((theta_500[hit] / 60.0) ** 2 * 120.0, 4.0, 400.0)
        hp.projscatter(
            lon[hit],
            lat[hit],
            lonlat=True,
            s=s,
            facecolors="none",
            edgecolors=color,
            linewidths=0.55,
            alpha=0.9,
            label=label,
        )
    plt.legend(loc="lower right", fontsize=9, framealpha=0.9)
    if purity is not None and n_tp is not None and n_det is not None:
        stats = f"Purity: {purity * 100:.1f}% ({n_tp}/{n_det})\nMatch: 10 arcmin"
        plt.gcf().text(
            0.02,
            0.02,
            stats,
            transform=plt.gcf().transFigure,
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="0.7"),
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--immf", type=Path, default=CAT_DIR / "footprint_splitA_immf_q5.npz")
    p.add_argument(
        "--scimmf", type=Path, default=CAT_DIR / "footprint_splitA_scimmf_q5.npz"
    )
    p.add_argument("--truth", type=Path, default=DEFAULT_TRUTH_CATALOGUE)
    p.add_argument("--ymap", type=Path, default=YMAP_DEFAULT)
    p.add_argument(
        "--out-purity",
        type=Path,
        default=Path("figures/szifi_footprint_purity_immf_scimmf.png"),
    )
    p.add_argument(
        "--out-immf-map",
        type=Path,
        default=Path("figures/szifi_footprint_immf_mollview_detections.png"),
    )
    p.add_argument(
        "--out-scimmf-map",
        type=Path,
        default=Path("figures/szifi_footprint_scimmf_mollview_detections.png"),
    )
    args = p.parse_args()

    paths = SZiFiPaths()
    configs = [
        ("iMMF", args.immf, "#00bcd4"),
        ("sciMMF", args.scimmf, "#ff9800"),
    ]

    purity_series = []
    results = {}
    for name, cat_path, _color in configs:
        if not cat_path.is_file():
            raise FileNotFoundError(cat_path)
        result = benchmark_catalogue(cat_path, truth_csv=args.truth, paths=paths)
        write_benchmark_json(
            result, CAT_DIR / f"footprint_splitA_{name.lower()}_q5_benchmark.json"
        )
        match, _, _, _ = run_match(cat_path, truth_csv=args.truth, paths=paths)
        binned = benchmark_snr_bins(match, q_th_truth=result.q_th_truth)
        purity_series.append(
            (
                name,
                binned.purity_thresholds,
                binned.purity,
                binned.purity_n,
                binned.purity_err,
            )
        )
        results[name] = result
        print(
            f"{name}: catalogue={result.n_detected_catalogue} in-mask={result.n_detected} "
            f"(excl {result.n_detected_excluded_mask} masked) "
            f"TP={result.n_true_positives} FP={result.n_false_positives} "
            f"purity={result.purity:.4f}"
        )

    plot_purity_comparison(purity_series, args.out_purity)

    gal, ps = load_pr4_gal_ps(paths.masks_fits, nside=paths.nside)
    mask = gal * ps
    y4096 = np.asarray(hp.read_map(str(args.ymap), dtype=np.float64))
    ymap = (
        hp.ud_grade(y4096, paths.nside)
        if hp.npix2nside(y4096.size) != paths.nside
        else y4096
    )

    out_maps = {"iMMF": args.out_immf_map, "sciMMF": args.out_scimmf_map}
    matches = {}
    for name, cat_path, _ in configs:
        match, _, _, _ = run_match(cat_path, truth_csv=args.truth, paths=paths)
        matches[name] = match
    for name, cat_path, _ in configs:
        match = matches[name]
        r = results[name]
        plot_detection_map(
            ymap,
            mask,
            match.det_lon,
            match.det_lat,
            match.det_theta,
            match.det_hit,
            title=(
                f"Truth Compton-y + {name} in-mask (N={r.n_detected}, q≥5)"
            ),
            out_path=out_maps[name],
            purity=r.purity,
            n_tp=r.n_true_positives,
            n_det=r.n_detected,
        )

    summary = {
        name: {
            "n_detected": r.n_detected,
            "n_true_positives": r.n_true_positives,
            "n_false_positives": r.n_false_positives,
            "purity": r.purity,
        }
        for name, r in results.items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
