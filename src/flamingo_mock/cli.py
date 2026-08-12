"""Command-line entry point for synthetic map making.

Examples
--------
Build per-component maps only (default — no coaddition):

    flamingo-mock-maps build

Skip the CMB step:

    flamingo-mock-maps build --steps tsz ksz cib

Quick low-resolution test:

    flamingo-mock-maps build --nside 256 --frequencies 100 217 857
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_FREQUENCIES_GHZ, MockConfig

ALL_STEPS = ("cmb", "tsz", "ksz", "cib")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="flamingo-mock-maps",
        description="Synthetic multi-frequency skies from FLAMINGO integrated maps",
    )
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="Build per-component maps (no coaddition)")
    b.add_argument(
        "--data-dir",
        type=Path,
        default=MockConfig.data_dir,
        help="FLAMINGO lightcone directory with the integrated maps",
    )
    b.add_argument(
        "--out-dir",
        type=Path,
        default=MockConfig.out_dir,
        help="Output root for synthetic maps",
    )
    b.add_argument(
        "--frequencies",
        type=float,
        nargs="+",
        default=list(DEFAULT_FREQUENCIES_GHZ),
        metavar="GHZ",
        help="Observing frequencies in GHz",
    )
    b.add_argument(
        "--nside",
        type=int,
        default=4096,
        help="Working Nside (use < 4096 for quick tests; inputs are downgraded)",
    )
    b.add_argument("--seed", type=int, default=42, help="CMB realization seed")
    b.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=list(ALL_STEPS),
        help="Pipeline steps (default: cmb tsz ksz cib)",
    )
    return p.parse_args(argv)


def build(args: argparse.Namespace) -> None:
    # Imports deferred so `--help` stays instant and missing optional
    # dependencies (camb/pixell) only matter for the cmb step.
    from . import cib as cib_mod
    from . import ksz as ksz_mod
    from . import tsz as tsz_mod

    cfg = MockConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        frequencies=tuple(args.frequencies),
        nside=args.nside,
        seed=args.seed,
    )
    cfg.make_dirs()
    print(f"data: {cfg.data_dir}")
    print(f"out:  {cfg.out_dir}")
    print(f"nside={cfg.nside}, freqs={list(cfg.frequencies)}, steps={args.steps}")

    steps = set(args.steps)

    if "cmb" in steps:
        from .cmb import make_lensed_cmb

        make_lensed_cmb(cfg)
    if "tsz" in steps:
        tsz_mod.make_tsz_maps(cfg)
    if "ksz" in steps:
        ksz_mod.make_ksz_map(cfg)
    if "cib" in steps:
        cib_mod.make_cib_maps(cfg)

    print("Done.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "build":
        build(args)
