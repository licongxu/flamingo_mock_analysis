#!/usr/bin/env python3
"""Extract SZiFi fixed-mode true SNR (q-bar_t) for footprint truth halos."""

from __future__ import annotations

import argparse
from pathlib import Path

from flamingo_mock.szifi.paths import SZiFiPaths
from flamingo_mock.szifi.true_snr import extract_true_snr
from flamingo_mock.szifi.validate import DEFAULT_TRUTH_CATALOGUE


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth", type=Path, default=DEFAULT_TRUTH_CATALOGUE)
    p.add_argument("--split", choices=("A", "B"), default="A")
    p.add_argument("--z-max", type=float, default=1.0)
    p.add_argument(
        "--q-ap-min",
        type=float,
        default=2.0,
        help="Parent pre-cut on aperture SNR (tractability only)",
    )
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument(
        "--threads-per-worker",
        type=int,
        default=None,
        help="OMP/BLAS threads per worker; with --n-workers, capped at half machine",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues/"
            "footprint_splitA_truth_qtrue_mmf_qap2.npz"
        ),
    )
    args = p.parse_args()

    extract_true_snr(
        SZiFiPaths(),
        truth_csv=args.truth,
        split=args.split,
        z_max=args.z_max,
        q_ap_min=args.q_ap_min,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
