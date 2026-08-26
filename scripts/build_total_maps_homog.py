#!/usr/bin/env python3
"""L1_m9 fiducial Planck HFI mocks: Beam(CMB + tSZ + CIB) + homogeneous white noise.

Writes
  .../total_maps/sky_CMB_tSZ_CIB_homog_{nu}GHz_nside2048_uK.fits

Same recipe as notebooks/build_total_maps_homog_noise.ipynb (the L2p8 demo
in total_maps/test/). No kSZ. Beam is Table I FWHM on the signal only.
Noise is the existing homogeneous pixel-white realisation (not beamed).
Pixel window is not deconvolved.
"""
from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np

from flamingo_mock.config import BEAM_FWHM_ARCMIN, PLANCK_FREQUENCIES_GHZ
from flamingo_mock.io import write_map

COMP = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/components")
NOISE = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/planck_noise/homogeneous")
OUT = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/total_maps")

NSIDE = 2048
NSIDE_IN = 4096
PRESCRIP = "L1_m9"  # fiducial hydro, D3A CMB
FREQS = tuple(int(f) for f in PLANCK_FREQUENCIES_GHZ)

CMB = COMP / "cmb" / "primary_CMB_T_lensed_nside4096_seed42.fits"
CIB_DIR = COMP / "cib" / PRESCRIP
TSZ_DIR = COMP / "tsz"  # L1_m9 fiducial products (unlabeled)


def load_uK(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(hp.read_map(str(path), field=0, dtype=np.float64))


def to_nside(m: np.ndarray, nside: int = NSIDE) -> np.ndarray:
    if hp.get_nside(m) == nside:
        return m
    return hp.ud_grade(m, nside)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("prescription:", PRESCRIP, "(fiducial hydro)")
    print("CMB:", CMB)
    print("CIB:", CIB_DIR)
    print("tSZ:", TSZ_DIR)
    print("out:", OUT)

    print("CMB: load + ud_grade ...", flush=True)
    cmb = to_nside(load_uK(CMB))
    print(f"  nside={hp.get_nside(cmb)}  rms={cmb.std():.2f} uK")

    print(f"{'nu':>5}  {'CMB':>8}  {'tSZ':>8}  {'CIB':>8}  {'beamed':>8}  {'noise':>8}  {'total':>8}")
    for nu in FREQS:
        tsz = to_nside(load_uK(TSZ_DIR / f"tSZ_deltaT_{nu}GHz_nside{NSIDE_IN}.fits"))
        cib = to_nside(load_uK(CIB_DIR / f"CIB_deltaT_{nu}GHz_nside{NSIDE_IN}.fits"))
        noise = load_uK(NOISE / f"{nu}GHz" / f"white_noise_{nu}GHz_nside{NSIDE}_uK.fits")
        if hp.get_nside(noise) != NSIDE:
            raise ValueError(f"noise nside {hp.get_nside(noise)} != {NSIDE}")

        signal = cmb + tsz + cib
        fwhm = float(BEAM_FWHM_ARCMIN[nu])
        print(f"\n=== {nu} GHz  FWHM={fwhm:.2f}' ===", flush=True)
        signal_b = hp.smoothing(signal, fwhm=np.radians(fwhm / 60.0))
        total = signal_b + noise
        print(
            f"{nu:5d}  {cmb.std():8.2f}  {tsz.std():8.2f}  {cib.std():8.2f}  "
            f"{signal_b.std():8.2f}  {noise.std():8.2f}  {total.std():8.2f}"
        )
        dest = OUT / f"sky_CMB_tSZ_CIB_homog_{nu}GHz_nside{NSIDE}_uK.fits"
        write_map(
            dest,
            total,
            unit="uK_CMB",
            freq=float(nu),
            extra=[
                ("COMPS", "CMB+tSZ+CIB+homog_noise"),
                ("PRESCRIP", PRESCRIP, "L1_m9 fiducial hydro"),
                ("FWHM", fwhm, "Gaussian beam [arcmin] on signal only"),
                ("BEAMON", 1),
                ("NOISE", "homogeneous white, not beamed"),
                ("NKSZ", 1, "kSZ not included"),
                ("PIXWIN", 0, "0=do NOT deconvolve HEALPix pixwin"),
                ("COMMENT", "Planck-like mock: beam on signal only; no pixwin inverse"),
            ],
            dtype=np.float32,
        )
        del tsz, cib, noise, signal, signal_b, total
    print("Done.")


if __name__ == "__main__":
    main()
