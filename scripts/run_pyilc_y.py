#!/usr/bin/env python3
"""Run pyILC (NILC or HILC) from a YAML config — McCarthy & Hill (2024) entry path.

Usage (with cosmo_env activated)::

    python scripts/run_pyilc_y.py configs/hilc_y_flamingo_noise_split0.yml
    python scripts/run_pyilc_y.py configs/nilc_y_flamingo_noise_split0.yml

This is a thin wrapper around pyilc.main that records the resolved y-map path.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pyILC y-map reconstruction from YAML")
    parser.add_argument("yaml_config", type=str, help="Path to pyILC YAML config")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Optional PYILC_BACKEND override (jax|numba|numpy|auto)",
    )
    args = parser.parse_args(argv)

    cfg = Path(args.yaml_config).expanduser().resolve()
    if not cfg.is_file():
        print(f"ERROR: config not found: {cfg}", file=sys.stderr)
        return 2

    if args.backend:
        os.environ["PYILC_BACKEND"] = args.backend

    # Default: numba on host RAM. JAX/TPU can OOM on large needlet scales.
    if "PYILC_BACKEND" not in os.environ:
        os.environ.setdefault("PYILC_BACKEND", "numba")

    from pyilc.input import ILCInfo
    from pyilc.wavelets import Wavelets, wavelet_ILC, harmonic_ILC, _ILC_map_filename

    t0 = time.time()
    print(f"[run_pyilc_y] config={cfg}")
    print(f"[run_pyilc_y] PYILC_BACKEND={os.environ.get('PYILC_BACKEND')}")

    info = ILCInfo(str(cfg))
    print(
        f"[run_pyilc_y] wavelet_type={info.wavelet_type} "
        f"N_freqs={info.N_freqs} preserved={info.ILC_preserved_comp} "
        f"N_deproj={info.N_deproj} N_side={info.N_side} ELLMAX={info.ELLMAX}"
    )
    print(f"[run_pyilc_y] output_dir={info.output_dir}")
    os.makedirs(info.output_dir, exist_ok=True)

    info.read_bandpasses()
    info.read_beams()
    if getattr(info, "work_in_car", False):
        info.read_geometries()

    wv = Wavelets(
        N_scales=info.N_scales,
        ELLMAX=info.ELLMAX,
        tol=1.0e-6,
        taper_width=info.taper_width,
    )
    if info.wavelet_type == "GaussianNeedlets":
        wv.GaussianNeedlets(FWHM_arcmin=info.GN_FWHM_arcmin)
    elif info.wavelet_type == "CosineNeedlets":
        wv.CosineNeedlets(ellmin=info.ellmin, ellpeaks=info.ellpeaks)
    elif info.wavelet_type == "TopHatHarmonic":
        wv.TopHatHarmonic(info.ellbins)
    elif info.wavelet_type == "TaperedTopHats":
        wv.TaperedTopHats(ellboundaries=info.ellboundaries, taperwidths=info.taperwidths)
    else:
        raise TypeError(f"unsupported wavelet type: {info.wavelet_type}")

    if info.wavelet_type == "TopHatHarmonic":
        info.read_maps()
        # healpix path: pix_size (arcmin) is only set by read_geometries for CAR.
        if not hasattr(info, "pix_size") or info.pix_size is None:
            import healpy as hp

            info.pix_size = float(hp.nside2resol(info.N_side, arcmin=True))
        if not info.weights_exist:
            info.maps2alms()
            info.alms2cls()
        # pyILC main.py checks info.maps_to_apply_weights, but ILCInfo only
        # sets freq_map_files_for_weights when YAML has maps_to_apply_weights.
        if getattr(info, "freq_map_files_for_weights", None) is not None:
            info.maps_to_apply_weights2alms()
        harmonic_ILC(wv, info, resp_tol=info.resp_tol, map_images=False)
    else:
        wavelet_ILC(wv, info, resp_tol=info.resp_tol, map_images=False)

    ypath = _ILC_map_filename(info)
    dt = time.time() - t0
    print(f"[run_pyilc_y] finished in {dt:.1f}s")
    print(f"[run_pyilc_y] y_map={ypath}")
    if not Path(ypath).is_file():
        print(f"ERROR: expected y-map missing: {ypath}", file=sys.stderr)
        return 1
    print(f"[run_pyilc_y] y_map_bytes={Path(ypath).stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
