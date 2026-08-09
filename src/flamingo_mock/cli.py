"""Command-line entry point for synthetic map making.

Examples
--------
Full pipeline at native resolution (needs camb + pixell for the CMB step):

    flamingo-mock-maps build

Everything except the CMB (reuses a cached lensed CMB if present):

    flamingo-mock-maps build --steps tsz ksz cib coadd

Quick low-resolution test:

    flamingo-mock-maps build --nside 256 --frequencies 90 217 857
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import DEFAULT_FREQUENCIES_GHZ, MockConfig

ALL_STEPS = ("cmb", "tsz", "ksz", "cib", "coadd")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="flamingo-mock-maps",
        description="Synthetic multi-frequency skies from FLAMINGO integrated maps",
    )
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="Build component and coadded maps")
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
        help="Pipeline steps to run (default: all)",
    )
    b.add_argument(
        "--smooth",
        action="store_true",
        help="Smooth coadded skies with per-frequency Gaussian beams",
    )
    return p.parse_args(argv)


def build(args: argparse.Namespace) -> None:
    # Imports deferred so `--help` stays instant and missing optional
    # dependencies (camb/pixell) only matter for the cmb step.
    from . import cib as cib_mod
    from . import ksz as ksz_mod
    from . import tsz as tsz_mod
    from .sky import make_coadd_maps

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

    cmb_uK = None
    if "cmb" in steps:
        from .cmb import make_lensed_cmb

        cmb_uK = make_lensed_cmb(cfg)
    elif "coadd" in steps:
        import healpy as hp

        cached = cfg.raw_dir / f"primary_CMB_T_lensed_nside{cfg.nside}_seed{cfg.seed}.fits"
        if not cached.is_file():
            raise SystemExit(
                f"coadd needs the lensed CMB; run the 'cmb' step first "
                f"(missing {cached})"
            )
        print(f"CMB: loading cached {cached.name}")
        cmb_uK = hp.read_map(str(cached), dtype=np.float64)

    tsz_uK = tsz_mod.make_tsz_maps(cfg) if "tsz" in steps else None
    ksz_uK = ksz_mod.make_ksz_map(cfg) if "ksz" in steps else None
    cib_uK = cib_mod.make_cib_maps(cfg) if "cib" in steps else None

    if "coadd" in steps:
        import healpy as hp  # noqa: F401  (ensures healpy present for coadd)

        def load_component(pattern: str):
            matches = sorted(cfg.raw_dir.glob(pattern))
            if not matches:
                raise SystemExit(
                    f"coadd needs {pattern}; run the corresponding step first"
                )
            return {
                float(m.name.split("_")[-2].replace("GHz", "")): hp.read_map(
                    str(m), dtype=np.float64
                )
                for m in matches
            }

        if tsz_uK is None:
            tsz_uK = load_component(f"tSZ_deltaT_*GHz_nside{cfg.nside}.fits")
        if ksz_uK is None:
            ksz_uK = hp.read_map(
                str(cfg.raw_dir / f"kSZ_deltaT_nside{cfg.nside}.fits"),
                dtype=np.float64,
            )
        if cib_uK is None:
            cib_uK = load_component(f"CIB_deltaT_*GHz_nside{cfg.nside}.fits")

        print("Coadding CMB + tSZ + kSZ + CIB...")
        make_coadd_maps(cfg, cmb_uK, tsz_uK, ksz_uK, cib_uK, smooth=args.smooth)

    print("Done.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "build":
        build(args)
