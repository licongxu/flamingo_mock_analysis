#!/usr/bin/env python3
"""SZiFi-style completeness: bin by fixed-mode q_true_mmf + ERF overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flamingo_mock.szifi.paths import SZiFiPaths
from flamingo_mock.szifi.true_snr import load_true_snr
from flamingo_mock.szifi.validate import (
    DEFAULT_QTRUE_BIN_EDGES,
    DEFAULT_TRUTH_CATALOGUE,
    benchmark_catalogue,
    benchmark_snr_bins,
    plot_benchmark_snr_bins,
    plot_benchmark_summary,
    run_match,
    write_benchmark_json,
    write_snr_bins_json,
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
    p.add_argument(
        "--true-snr",
        type=Path,
        default=Path(
            "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues/"
            "footprint_splitA_truth_qtrue_mmf_qap2.npz"
        ),
    )
    p.add_argument("--truth", type=Path, default=DEFAULT_TRUTH_CATALOGUE)
    p.add_argument("--q-th-obs", type=float, default=5.0)
    p.add_argument("--q-th-truth", type=float, default=5.0)
    p.add_argument("--match-radius-arcmin", type=float, default=10.0)
    p.add_argument(
        "--json-out",
        type=Path,
        default=Path(
            "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues/"
            "footprint_splitA_immf_q5_benchmark_szifi.json"
        ),
    )
    p.add_argument(
        "--out-bar",
        type=Path,
        default=Path("figures/szifi_footprint_immf_benchmark_szifi.png"),
    )
    p.add_argument(
        "--out-snr",
        type=Path,
        default=Path("figures/szifi_footprint_immf_benchmark_snr_bins_szifi.png"),
    )
    args = p.parse_args()

    if not args.true_snr.is_file():
        raise SystemExit(
            f"Missing true-SNR catalogue: {args.true_snr}\n"
            "Run: flamingo-szifi true-snr"
        )

    paths = SZiFiPaths()
    true_snr = load_true_snr(args.true_snr)
    print(
        f"true_snr: n={len(true_snr['lon'])} "
        f"q_true>=5: {(true_snr['q_true_mmf'] >= args.q_th_truth).sum()} "
        f"median q_true={float(true_snr['q_true_mmf'].mean()):.2f}"
    )

    result = benchmark_catalogue(
        args.catalogue,
        truth_csv=args.truth,
        paths=paths,
        q_th_obs=args.q_th_obs,
        q_th_truth=args.q_th_truth,
        match_radius_arcmin=args.match_radius_arcmin,
        true_snr=true_snr,
    )
    write_benchmark_json(result, args.json_out)

    match, _, _, _ = run_match(
        args.catalogue,
        truth_csv=args.truth,
        paths=paths,
        q_th_obs=args.q_th_obs,
        q_th_truth=args.q_th_truth,
        match_radius_arcmin=args.match_radius_arcmin,
        true_snr=true_snr,
    )
    binned = benchmark_snr_bins(
        match,
        bin_edges=DEFAULT_QTRUE_BIN_EDGES,
        q_th_truth=args.q_th_truth,
        q_th_obs=args.q_th_obs,
        restrict_truth_to_qth=False,
        erf_opt_bias=True,
    )
    snr_json = args.json_out.with_name(args.json_out.stem + "_snr_bins.json")
    write_snr_bins_json(binned, snr_json)

    plot_benchmark_summary(
        result,
        args.out_bar,
        title="iMMF SZiFi-style completeness (fixed-mode $\\bar{q}_t$, 10' match)",
        truth_snr_name="q_true_mmf",
    )
    plot_benchmark_snr_bins(
        binned,
        result,
        args.out_snr,
        title="iMMF vs fixed-mode $\\bar{q}_t$ (SZiFi-style + ERF)",
    )

    print(
        json.dumps(
            {
                "purity": result.purity,
                "completeness_qtrue_ge_5": result.completeness_detectable,
                "n_truth_qtrue_ge_5": result.n_truth_detectable,
                "n_tp": result.n_true_positives,
                "n_fp": result.n_false_positives,
                "completeness_bins": {
                    "centers": binned.bin_centers,
                    "C": binned.completeness,
                    "C_erf": binned.completeness_erf,
                    "n": binned.completeness_n,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
