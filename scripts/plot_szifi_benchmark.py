#!/usr/bin/env python3
"""Plot purity/completeness benchmark + TP/FP sky map for iMMF footprint."""

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
    load_benchmark_json,
    match_detection_flags,
    plot_benchmark_snr_bins,
    plot_benchmark_summary,
    run_match,
    write_benchmark_json,
    write_snr_bins_json,
)


def plot_mollview_tp_fp(
    ymap: np.ndarray,
    mask: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    theta_500: np.ndarray,
    det_hit: np.ndarray,
    result,
    out_path: Path,
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
        title=(
            f"Truth Compton-y + iMMF (N={result.n_detected}, q>={result.q_th_obs})"
        ),
        unit=r"$y$",
        cmap="hot",
        cbar=True,
        hold=True,
    )
    hp.graticule(dmer=30, dpar=30, alpha=0.25, verbose=False)

    for hit, color, label in (
        (det_hit, "lime", f"true positive ({det_hit.sum()})"),
        (~det_hit, "red", f"false positive ({(~det_hit).sum()})"),
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
            linewidths=0.5,
            alpha=0.9,
            label=label,
        )

    plt.legend(loc="lower right", fontsize=9, framealpha=0.9)
    stats = (
        f"Match: Zubeldia 2024, 10 arcmin\n"
        f"Purity: {result.purity * 100:.1f}% ({result.n_true_positives}/{result.n_detected})\n"
        f"Completeness: {result.completeness_detectable * 100:.1f}% "
        f"({result.n_true_positives}/{result.n_truth_detectable} detectable)"
    )
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
    p.add_argument(
        "--catalogue",
        type=Path,
        default=Path(
            "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues/"
            "footprint_splitA_immf_q5.npz"
        ),
    )
    p.add_argument("--truth", type=Path, default=DEFAULT_TRUTH_CATALOGUE)
    p.add_argument(
        "--benchmark-json",
        type=Path,
        default=Path(
            "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues/"
            "footprint_splitA_immf_q5_benchmark.json"
        ),
    )
    p.add_argument(
        "--ymap",
        type=Path,
        default=Path(
            "/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/tsz/"
            "compton_y_nside4096.fits"
        ),
    )
    p.add_argument("--out-bar", type=Path, default=Path("figures/szifi_footprint_immf_benchmark.png"))
    p.add_argument(
        "--out-snr",
        type=Path,
        default=Path("figures/szifi_footprint_immf_benchmark_snr_bins.png"),
    )
    p.add_argument(
        "--out-sky",
        type=Path,
        default=Path("figures/szifi_footprint_immf_mollview_benchmark.png"),
    )
    p.add_argument("--recompute", action="store_true")
    args = p.parse_args()

    paths = SZiFiPaths()
    if args.recompute or not args.benchmark_json.is_file():
        result = benchmark_catalogue(args.catalogue, truth_csv=args.truth, paths=paths)
        write_benchmark_json(result, args.benchmark_json)
    else:
        result = load_benchmark_json(args.benchmark_json)
        if result.footprint != "planck_gal_x_ps_unmasked":
            result = benchmark_catalogue(args.catalogue, truth_csv=args.truth, paths=paths)
            write_benchmark_json(result, args.benchmark_json)

    match, _, _, _ = run_match(args.catalogue, truth_csv=args.truth, paths=paths)
    binned = benchmark_snr_bins(match, q_th_truth=result.q_th_truth)
    snr_json = args.benchmark_json.with_name(
        args.benchmark_json.stem + "_snr_bins.json"
    )
    write_snr_bins_json(binned, snr_json)

    plot_benchmark_summary(result, args.out_bar)
    plot_benchmark_snr_bins(binned, result, args.out_snr)

    det, det_ok, det_hit = match_detection_flags(
        args.catalogue, truth_csv=args.truth, paths=paths
    )
    gal, ps = load_pr4_gal_ps(paths.masks_fits, nside=paths.nside)
    y4096 = np.asarray(hp.read_map(str(args.ymap), dtype=np.float64))
    ymap = hp.ud_grade(y4096, paths.nside) if hp.npix2nside(y4096.size) != paths.nside else y4096

    plot_mollview_tp_fp(
        ymap,
        gal * ps,
        det["lon"][det_ok],
        det["lat"][det_ok],
        det["theta_500"][det_ok],
        det_hit,
        result,
        args.out_sky,
    )

    print(json.dumps({"purity": result.purity, "completeness_detectable": result.completeness_detectable}, indent=2))


if __name__ == "__main__":
    main()
