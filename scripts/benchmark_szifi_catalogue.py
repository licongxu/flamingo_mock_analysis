#!/usr/bin/env python3
"""Benchmark SZiFi catalogue purity/completeness vs L2p8_m9 qfrommap truth."""

from __future__ import annotations

import argparse
from pathlib import Path

from flamingo_mock.szifi.paths import SZiFiPaths
from flamingo_mock.szifi.validate import (
    DEFAULT_TRUTH_CATALOGUE,
    benchmark_catalogue,
    write_benchmark_json,
)


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
    p.add_argument("--q-th-obs", type=float, default=5.0)
    p.add_argument("--q-th-truth", type=float, default=5.0)
    p.add_argument("--z-max", type=float, default=1.0)
    p.add_argument("--match-radius-arcmin", type=float, default=10.0)
    p.add_argument("--use-theta-500", action="store_true")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    result = benchmark_catalogue(
        args.catalogue,
        truth_csv=args.truth,
        paths=SZiFiPaths(),
        q_th_obs=args.q_th_obs,
        q_th_truth=args.q_th_truth,
        z_max=args.z_max,
        match_radius_arcmin=args.match_radius_arcmin,
        use_theta_500=args.use_theta_500,
    )

    print(f"Catalogue: {args.catalogue.name}")
    print(f"Truth:     {args.truth.name} (q_from_aperture)")
    print(f"Footprint: PR4 GAL×PS unmasked (detections + truth for purity)")
    print(
        f"On disk (q>={args.q_th_obs}):     {result.n_detected_catalogue} detections"
    )
    print(
        f"In-mask for purity:              {result.n_detected} detections "
        f"({result.n_detected_excluded_mask} excluded on GAL×PS mask)"
    )
    print(f"Truth in footprint (z<={args.z_max}): {result.n_truth_all}")
    print(
        f"Truth detectable (q_from_aperture>={args.q_th_truth}): "
        f"{result.n_truth_detectable}"
    )
    print(f"True positives:  {result.n_true_positives}")
    print(f"False positives: {result.n_false_positives}")
    print(f"Undetected truth: {result.n_undetected}")
    print()
    print(
        f"Purity (in-mask):          {result.purity:.4f}  "
        f"({result.n_true_positives}/{result.n_detected})"
    )
    print(
        f"Completeness (all truth):  {result.completeness_all:.4f}  "
        f"({result.n_true_positives}/{result.n_truth_all})"
    )
    print(
        f"Completeness (detectable): {result.completeness_detectable:.4f}  "
        f"({result.n_true_positives}/{result.n_truth_detectable})"
    )
    print(f"Match radius: {result.match_radius_arcmin} arcmin")

    if args.json_out:
        write_benchmark_json(result, args.json_out)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
