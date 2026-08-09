"""Command-line entry point for the FLAMINGO pyILC Compton-y pipeline.

Examples
--------
Prepare noise-added split maps (Nside=2048, NPIPE splits A/B):

    flamingo-ilc prepare

Write pyILC YAML configs (HILC both splits + NILC split A):

    flamingo-ilc config --out-dir configs

Run the ILC (JAX backend on GPU by default):

    flamingo-ilc run configs/hilc_y_flamingo_npipe_splitA.yml
    flamingo-ilc run configs/hilc_y_flamingo_npipe_splitB.yml

Validate against the truth y-map (beam-deconvolved spectra):

    flamingo-ilc validate --ymap <splitA y-map> --ymap-split <splitB y-map>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .paths import (
    ILC_BEAM_FWHM_ARCMIN,
    ILC_FREQUENCIES_GHZ,
    NPIPE_MC_DEFAULT,
    NPIPE_SPLITS,
    ILCPaths,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="flamingo-ilc",
        description="pyILC Compton-y on FLAMINGO mock skies + Planck NPIPE noise",
    )
    sub = p.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Build coadd+noise split maps and truth y")
    prep.add_argument("--nside", type=int, default=2048, help="Working Nside")
    prep.add_argument("--mc", type=int, default=NPIPE_MC_DEFAULT, help="NPIPE MC index")
    prep.add_argument(
        "--ellmax-smooth",
        type=int,
        default=6000,
        help="lmax for the channel-beam smoothing at native resolution",
    )
    prep.add_argument("--overwrite", action="store_true")

    cfg = sub.add_parser("config", help="Write pyILC YAML configs")
    cfg.add_argument("--out-dir", type=Path, default=Path("configs"))
    cfg.add_argument(
        "--methods", nargs="+", choices=("hilc", "nilc"), default=["hilc", "nilc"]
    )
    cfg.add_argument("--splits", nargs="+", choices=NPIPE_SPLITS, default=list(NPIPE_SPLITS))
    cfg.add_argument(
        "--backend",
        default="jax",
        choices=("auto", "numpy", "numba", "jax", "cupy"),
        help="ilc_backend written into the YAMLs (default: jax)",
    )
    cfg.add_argument("--nside", type=int, default=2048)

    run = sub.add_parser("run", help="Run pyILC from a YAML config")
    run.add_argument("yaml_config", type=Path)
    run.add_argument(
        "--backend",
        default=None,
        choices=("auto", "numpy", "numba", "jax", "cupy"),
        help="Override ilc_backend (sets PYILC_BACKEND for the run)",
    )

    val = sub.add_parser("validate", help="Validate an ILC y-map against truth y")
    val.add_argument("--ymap", type=Path, default=None, help="ILC y-map FITS path")
    val.add_argument("--truth", type=Path, default=None, help="Truth Compton-y FITS")
    val.add_argument(
        "--ymap-split",
        type=Path,
        default=None,
        help="Second-split ILC y-map for the noise-decoupled cross spectrum",
    )
    val.add_argument("--lmax", type=int, default=3000)
    val.add_argument(
        "--ilc-beam-fwhm-arcmin",
        type=float,
        default=ILC_BEAM_FWHM_ARCMIN,
        help="Common ILC beam deconvolved from the spectra",
    )
    val.add_argument("--bl-floor", type=float, default=1e-3)
    val.add_argument("--figures-dir", type=Path, default=Path("figures"))
    val.add_argument("--log", type=Path, default=None, help="Write the JSON summary here")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "prepare":
        from .prepare import build_split_maps

        paths = ILCPaths(nside=args.nside)
        print(f"inputs: {paths.inputs_dir}")
        build_split_maps(
            paths, mc=args.mc, ellmax_smooth=args.ellmax_smooth, overwrite=args.overwrite
        )
        return 0

    if args.command == "config":
        from .config import write_config_set

        paths = ILCPaths(nside=args.nside)
        written = write_config_set(
            args.out_dir,
            methods=tuple(args.methods),
            splits=tuple(args.splits),
            paths=paths,
            ilc_backend=args.backend,
        )
        print(f"freqs: {list(ILC_FREQUENCIES_GHZ)} GHz; inputs: {paths.inputs_dir}")
        return 0 if written else 1

    if args.command == "run":
        from .run import run_ilc

        run_ilc(args.yaml_config, backend=args.backend)
        return 0

    if args.command == "validate":
        from .validate import print_summary, summary_ok, validate_ymap

        summary = validate_ymap(
            args.ymap,
            truth=args.truth,
            ymap_split=args.ymap_split,
            lmax=args.lmax,
            ilc_beam_fwhm_arcmin=args.ilc_beam_fwhm_arcmin,
            bl_floor=args.bl_floor,
            figures_dir=args.figures_dir,
        )
        text = print_summary(summary)
        print(text)
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            args.log.write_text(text + "\n")
            print(f"wrote log {args.log}")
        if not summary_ok(summary):
            print("FAIL: y-map validation checks did not pass")
            return 1
        print("PASS: y-map amplitude, correlation, and beam-deconvolved Cl OK")
        return 0

    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
