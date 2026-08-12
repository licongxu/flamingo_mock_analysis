"""CLI for FLAMINGO × SZiFi cluster finding.

Examples
--------
Prepare Planck-footprint tiles (GAL×PS, split A)::

    flamingo-szifi prepare --footprint --split A

Run footprint catalogues with CPU parallelism (≤ half machine)::

    flamingo-szifi run --footprint --split A --method both --n-workers 8

Pilot (few high-|b| tiles)::

    flamingo-szifi run --pilot --split A
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .paths import SZiFiPaths
from .run import run_imf_and_scimmf, run_mmf_batched
from .tiles import prepare_tiles, select_footprint_tile_ids, select_pilot_tile_ids
from .true_snr import extract_true_snr
from .validate import DEFAULT_TRUTH_CATALOGUE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="flamingo-szifi",
        description="SZiFi iMMF/sciMMF cluster finding on FLAMINGO mock Planck skies",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=SZiFiPaths().out_root,
        help="Output root under /rds/.../szifi",
    )
    sub = p.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Cut tiles + project PR4 masks")
    prep.add_argument("--split", choices=("A", "B"), default="A")
    prep.add_argument("--pilot", action="store_true", help="Only ~4 high-|b| tiles")
    prep.add_argument(
        "--footprint",
        action="store_true",
        help="All nside=8 tiles with GAL×PS fraction >= min-ftile",
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

    run = sub.add_parser("run", help="Run iMMF / sciMMF; write q>5 catalogues")
    run.add_argument("--split", choices=("A", "B"), default="A")
    run.add_argument("--pilot", action="store_true")
    run.add_argument("--footprint", action="store_true")
    run.add_argument("--n-pilot", type=int, default=4)
    run.add_argument("--b-min", type=float, default=40.0)
    run.add_argument("--min-ftile", type=float, default=0.3)
    run.add_argument("--field-ids", type=int, nargs="+", default=None)
    run.add_argument("--q-th", type=float, default=5.0)
    run.add_argument("--tag", default=None, help="Output name tag (default pilot|footprint)")
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

    ts = sub.add_parser("true-snr", help="Fixed-mode true SNR for footprint truth halos")
    ts.add_argument("--truth", type=Path, default=DEFAULT_TRUTH_CATALOGUE)
    ts.add_argument("--split", choices=("A", "B"), default="A")
    ts.add_argument("--z-max", type=float, default=1.0)
    ts.add_argument("--q-ap-min", type=float, default=2.0)
    ts.add_argument("--n-workers", type=int, default=None)
    ts.add_argument("--threads-per-worker", type=int, default=None)
    ts.add_argument("--out", type=Path, default=None)

    return p.parse_args(argv)


def _field_ids(args: argparse.Namespace, paths: SZiFiPaths) -> list[int]:
    if args.field_ids is not None:
        return list(args.field_ids)
    if getattr(args, "footprint", False):
        return select_footprint_tile_ids(paths.masks_fits, min_ftile=args.min_ftile)
    if getattr(args, "pilot", False):
        return select_pilot_tile_ids(n=args.n_pilot, b_min_deg=args.b_min)
    raise SystemExit("Provide --footprint, --pilot, or --field-ids")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = SZiFiPaths(out_root=args.out_root)

    if args.command == "true-snr":
        out = args.out or (
            paths.catalogues_dir()
            / f"footprint_split{args.split}_truth_qtrue_mmf_qap{args.q_ap_min:g}.npz"
        )
        extract_true_snr(
            paths,
            truth_csv=args.truth,
            split=args.split,
            z_max=args.z_max,
            q_ap_min=args.q_ap_min,
            n_workers=args.n_workers,
            threads_per_worker=args.threads_per_worker,
            out_path=out,
        )
        return

    paths.make_dirs(args.split)
    fids = _field_ids(args, paths)
    print(f"tiles: n={len(fids)}  first={fids[:5]} ...")

    if args.command == "prepare":
        print(f"prepare split={args.split}")
        prepare_tiles(
            paths,
            fids,
            split=args.split,
            overwrite=args.overwrite,
            n_workers=args.n_workers,
        )
        print("prepare done")
        return

    if args.command == "run":
        prepare_tiles(
            paths, fids, split=args.split, overwrite=False, n_workers=args.n_workers
        )
        tag = args.tag or ("pilot" if args.pilot else "footprint")
        out_dir = paths.pilot_dir() if args.pilot else paths.catalogues_dir()
        methods = ("immf", "scimmf") if args.method == "both" else (args.method,)

        if args.footprint or len(fids) > 16:
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
