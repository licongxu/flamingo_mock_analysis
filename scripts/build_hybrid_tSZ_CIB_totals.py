#!/usr/bin/env python3
"""Hybrid homog totals: fid tSZ + M* CIB, and M* tSZ + fid CIB.

Shared CMB (seed 42) and r1 white noise. Does not touch L1_m9 or Mstar totals.
"""
from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np

from flamingo_mock.config import BEAM_FWHM_ARCMIN, PLANCK_FREQUENCIES_GHZ
from flamingo_mock.io import write_map

_SCRIPTS = Path(__file__).resolve().parent
import sys

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from hilc_prescriptions import (
    HYBRID_FID_TSZ_MSTAR_CIB,
    HYBRID_MSTAR_TSZ_FID_CIB,
    cmb_path,
    cib_dir,
    tsz_dir,
    total_maps_dir,
)

NOISE = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/planck_noise/homogeneous")
NSIDE = 2048
NSIDE_IN = 4096
FREQS = tuple(int(f) for f in PLANCK_FREQUENCIES_GHZ)
HYBRIDS = (
    (HYBRID_FID_TSZ_MSTAR_CIB, "L1_m9", "Mstar-1sigma"),
    (HYBRID_MSTAR_TSZ_FID_CIB, "Mstar-1sigma", "L1_m9"),
)


def load_uK(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(hp.read_map(str(path), field=0, dtype=np.float64))


def to_nside(m: np.ndarray, nside: int = NSIDE) -> np.ndarray:
    if hp.get_nside(m) == nside:
        return m
    return hp.ud_grade(m, nside)


def _complete(name: str) -> bool:
    d = total_maps_dir(name)
    return all(
        (d / f"sky_CMB_tSZ_CIB_homog_{nu}GHz_nside{NSIDE}_uK.fits").is_file()
        for nu in FREQS
    )


def build(name: str, tsz_name: str, cib_name: str) -> None:
    out = total_maps_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    cmb = to_nside(load_uK(cmb_path("L1_m9")))
    print(f"\n=== totals {name}  tSZ={tsz_name}  CIB={cib_name} ===")
    for nu in FREQS:
        tsz_m = to_nside(load_uK(tsz_dir(tsz_name) / f"tSZ_deltaT_{nu}GHz_nside{NSIDE_IN}.fits"))
        cib = to_nside(load_uK(cib_dir(cib_name) / f"CIB_deltaT_{nu}GHz_nside{NSIDE_IN}.fits"))
        noise = load_uK(NOISE / f"{nu}GHz" / f"white_noise_{nu}GHz_nside{NSIDE}_uK.fits")
        fwhm = float(BEAM_FWHM_ARCMIN[nu])
        print(f"  {nu} GHz FWHM={fwhm:.2f}' ...", flush=True)
        signal_b = hp.smoothing(cmb + tsz_m + cib, fwhm=np.radians(fwhm / 60.0))
        total = signal_b + noise
        write_map(
            out / f"sky_CMB_tSZ_CIB_homog_{nu}GHz_nside{NSIDE}_uK.fits",
            total,
            unit="uK_CMB",
            freq=float(nu),
            extra=[
                ("COMPS", "CMB+tSZ+CIB+homog_noise"),
                ("PRESCRIP", name),
                ("TSZFROM", tsz_name),
                ("CIBFROM", cib_name),
                ("FWHM", fwhm, "Gaussian beam [arcmin] on signal only"),
                ("BEAMON", 1),
                ("NOISE", "homogeneous white, not beamed"),
                ("NKSZ", 1, "kSZ not included"),
                ("PIXWIN", 0, "0=do NOT deconvolve HEALPix pixwin"),
            ],
            dtype=np.float32,
        )
        del tsz_m, cib, noise, signal_b, total


def main() -> None:
    for name, tsz_name, cib_name in HYBRIDS:
        if _complete(name):
            print(f"{name}: totals already present")
            continue
        build(name, tsz_name, cib_name)
    print("Done.")


if __name__ == "__main__":
    main()
