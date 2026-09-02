"""CLI for FLAMINGO × SZiFi cluster finding.

Examples
--------
Prepare Planck-footprint tiles (GAL×PS, split A)::

    flamingo-szifi prepare --footprint --split A

Run footprint catalogues with CPU parallelism (≤ half machine)::

    flamingo-szifi run --footprint --split A --method immf --n-workers 8

Purity/completeness vs qfrommap truth::

    flamingo-szifi benchmark --catalogue PATH.npz

Fixed-mode true SNR at truth positions::

    flamingo-szifi true-snr --split A

Full-sky unmasked (all 768 tiles, GAL=PS=1)::

    flamingo-szifi run --kind homog --full-sky --method immf --n-workers 6

Pilot (few high-|b| tiles)::

    flamingo-szifi run --pilot --split A
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must run before numpy/jax/healpy import so fork workers inherit a 1-thread pool.
for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NPROC",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_NUM_CPU_DEVICES", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

from .paths import (
    DEFAULT_OUT_ROOT_HOMOG,
    DEFAULT_TOTAL_MAPS_HOMOG,
    SZiFiPaths,
)
from .run import run_imf_and_scimmf, run_mmf_batched
from .tiles import (
    prepare_tiles,
    select_all_tile_ids,
    select_footprint_tile_ids,
    select_pilot_tile_ids,
)
from .true_snr import extract_true_snr
from .validate import (
    DEFAULT_TRUTH_CATALOGUE,
    benchmark_catalogue,
    write_benchmark_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root (default: .../szifi or .../szifi_homog)",
    )
    common.add_argument(
        "--kind",
        choices=("npipe", "homog"),
        default="npipe",
        help="Total-map set: NPIPE ILC coadds or homogeneous-noise CMB+tSZ+CIB",
    )
    common.add_argument(
        "--total-maps-dir",
        type=Path,
        default=None,
        help="Directory of full-sky total maps",
    )

    p = argparse.ArgumentParser(
        prog="flamingo-szifi",
        description="SZiFi iMMF/sciMMF cluster finding on FLAMINGO mock Planck skies",
    )
    sub = p.add_subparsers(dest="command", required=True)

    prep = sub.add_parser(
        "prepare",
        parents=[common],
        help="Cut tiles + project PR4 masks",
    )
    prep.add_argument("--split", choices=("A", "B"), default="A")
    prep.add_argument("--pilot", action="store_true", help="Only ~4 high-|b| tiles")
    prep.add_argument(
        "--footprint",
        action="store_true",
        help="All nside=8 tiles with GAL×PS fraction >= min-ftile",
    )
    prep.add_argument(
        "--full-sky",
        action="store_true",
        help="All 768 nside=8 tiles; GAL and PS masks set to 1 (no footprint cut)",
    )
    prep.add_argument("--n-pilot", type=int, default=4)
    prep.add_argument("--b-min", type=float, default=40.0)
    prep.add_argument("--min-ftile", type=float, default=0.3)
    prep.add_argument("--field-ids", type=int, nargs="+", default=None)
    prep.add_argument("--overwrite", action="store_true")
    prep.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Parallel CPU workers for cutouts (default ~8, ≤ half machine)",
    )

    run = sub.add_parser(
        "run",
        parents=[common],
        help="Run iMMF / sciMMF; write q>5 catalogues",
    )
    run.add_argument("--split", choices=("A", "B"), default="A")
    run.add_argument("--pilot", action="store_true")
    run.add_argument("--footprint", action="store_true")
    run.add_argument(
        "--full-sky",
        action="store_true",
        help="All 768 nside=8 tiles; GAL and PS masks set to 1 (no footprint cut)",
    )
    run.add_argument("--n-pilot", type=int, default=4)
    run.add_argument("--b-min", type=float, default=40.0)
    run.add_argument("--min-ftile", type=float, default=0.3)
    run.add_argument("--field-ids", type=int, nargs="+", default=None)
    run.add_argument("--q-th", type=float, default=5.0)
    run.add_argument("--tag", default=None, help="Output name tag (default pilot|footprint|fullsky)")
    run.add_argument(
        "--method",
        choices=("immf", "scimmf", "both"),
        default="both",
        help="Which MMF product(s) to run",
    )
    run.add_argument("--batch-size", type=int, default=4, help="Tiles per SZiFi call")
    run.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Parallel CPU workers (default ~8, capped at half machine)",
    )
    run.add_argument(
        "--threads-per-worker",
        type=int,
        default=None,
        help="OMP/BLAS threads per worker (with --n-workers, uses half-machine budget)",
    )
    run.add_argument(
        "--backend",
        choices=("jax", "numpy"),
        default="jax",
        help="SZiFi array backend in workers (jax runs on CPU to allow many processes)",
    )

    bench = sub.add_parser(
        "benchmark",
        help="Purity/completeness vs qfrommap truth",
    )
    bench.add_argument(
        "--catalogue",
        type=Path,
        default=Path(
            "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues/"
            "footprint_splitA_immf_q5.npz"
        ),
    )
    bench.add_argument("--truth", type=Path, default=DEFAULT_TRUTH_CATALOGUE)
    bench.add_argument("--q-th-obs", type=float, default=5.0)
    bench.add_argument("--q-th-truth", type=float, default=5.0)
    bench.add_argument("--z-max", type=float, default=1.0)
    bench.add_argument("--match-radius-arcmin", type=float, default=10.0)
    bench.add_argument("--use-theta-500", action="store_true")
    bench.add_argument("--json-out", type=Path, default=None)

    snr = sub.add_parser(
        "true-snr",
        parents=[common],
        help="Fixed-mode true SNR at truth halo positions",
    )
    snr.add_argument("--split", choices=("A", "B"), default="A")
    snr.add_argument("--truth", type=Path, default=DEFAULT_TRUTH_CATALOGUE)
    snr.add_argument("--z-max", type=float, default=1.0)
    snr.add_argument(
        "--q-ap-min",
        type=float,
        default=2.0,
        help="Parent pre-cut on aperture SNR (tractability only)",
    )
    snr.add_argument("--n-workers", type=int, default=None)
    snr.add_argument(
        "--threads-per-worker",
        type=int,
        default=None,
        help="OMP/BLAS threads per worker; with --n-workers, capped at half machine",
    )
    snr.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues/"
            "footprint_splitA_truth_qtrue_mmf_qap2.npz"
        ),
    )

    return p.parse_args(argv)


def _paths_from_args(args: argparse.Namespace) -> SZiFiPaths:
    kind = args.kind
    out_root = args.out_root
    if out_root is None:
        out_root = DEFAULT_OUT_ROOT_HOMOG if kind == "homog" else SZiFiPaths().out_root
    total_maps_dir = args.total_maps_dir
    if total_maps_dir is None:
        total_maps_dir = (
            DEFAULT_TOTAL_MAPS_HOMOG if kind == "homog" else SZiFiPaths().total_maps_dir
        )
    return SZiFiPaths(
        out_root=out_root,
        total_maps_dir=total_maps_dir,
        kind=kind,
    )


def _cmd_benchmark(args: argparse.Namespace) -> None:
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
    print("Footprint: PR4 GAL×PS unmasked (detections + truth for purity)")
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


def _field_ids(args: argparse.Namespace, paths: SZiFiPaths) -> list[int]:
    if args.field_ids is not None:
        return list(args.field_ids)
    if getattr(args, "full_sky", False):
        return select_all_tile_ids()
    if getattr(args, "footprint", False):
        return select_footprint_tile_ids(paths.masks_fits, min_ftile=args.min_ftile)
    if getattr(args, "pilot", False):
        return select_pilot_tile_ids(n=args.n_pilot, b_min_deg=args.b_min)
    raise SystemExit("Provide --full-sky, --footprint, --pilot, or --field-ids")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "benchmark":
        _cmd_benchmark(args)
        return
    if args.command == "true-snr":
        extract_true_snr(
            _paths_from_args(args),
            truth_csv=args.truth,
            split=args.split,
            z_max=args.z_max,
            q_ap_min=args.q_ap_min,
            n_workers=args.n_workers,
            threads_per_worker=args.threads_per_worker,
            out_path=args.out,
        )
        return

    paths = _paths_from_args(args)
    paths.make_dirs(args.split)
    fids = _field_ids(args, paths)
    print(f"kind={paths.kind}  tiles: n={len(fids)}  first={fids[:5]} ...")
    print(f"out_root={paths.out_root}")
    print(f"total_maps={paths.total_maps_dir}")

    if args.command == "prepare":
        print(f"prepare split={args.split}")
        prepare_tiles(
            paths,
            fids,
            split=args.split,
            overwrite=args.overwrite,
            n_workers=args.n_workers,
            threads_per_worker=getattr(args, "threads_per_worker", None),
            unmasked=bool(getattr(args, "full_sky", False)),
        )
        print("prepare done")
        return

    if args.command == "run":
        prepare_tiles(
            paths,
            fids,
            split=args.split,
            overwrite=False,
            n_workers=args.n_workers,
            threads_per_worker=getattr(args, "threads_per_worker", None),
            unmasked=bool(getattr(args, "full_sky", False)),
        )
        if args.pilot:
            tag = args.tag or "pilot"
        elif getattr(args, "full_sky", False):
            tag = args.tag or "fullsky"
        else:
            tag = args.tag or "footprint"
        out_dir = paths.pilot_dir() if args.pilot else paths.catalogues_dir()
        methods = ("immf", "scimmf") if args.method == "both" else (args.method,)

        if args.footprint or getattr(args, "full_sky", False) or len(fids) > 16:
            written = {}
            for m in methods:
                written[m] = run_mmf_batched(
                    paths,
                    fids,
                    split=args.split,
                    method=m,
                    q_th_final=args.q_th,
                    batch_size=args.batch_size,
                    out_dir=out_dir,
                    tag=tag,
                    n_workers=args.n_workers,
                    threads_per_worker=getattr(args, "threads_per_worker", None),
                    array_backend=getattr(args, "backend", "jax"),
                )
            print("wrote", written)
            return

        written = run_imf_and_scimmf(
            paths,
            fids,
            split=args.split,
            q_th_final=args.q_th,
            out_dir=out_dir,
            tag=tag,
            methods=methods,
        )
        print("wrote", written)
        return


if __name__ == "__main__":
    main()
