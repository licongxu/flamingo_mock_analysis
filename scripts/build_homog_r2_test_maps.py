#!/usr/bin/env python3
"""Independent homogeneous white-noise realisation (r2) on beamed totals.

Same pixel-white recipe as notebooks/homogeneous_planck_white_noise.ipynb
(SEED=1042 instead of 42). Observation maps keep the r1 beamed signal:

  total_r2 = total_r1 - noise_r1 + noise_r2

Default prescription is L1_m9. Other hydro/cosmology variants share the same
r1/r2 noise maps and only swap the beamed CMB+tSZ+CIB. Does not touch
total_maps/test/ (L2p8 demo).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import healpy as hp
import numpy as np

from flamingo_mock.config import BEAM_FWHM_ARCMIN, PLANCK_FREQUENCIES_GHZ
from flamingo_mock.io import write_map

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from hilc_prescriptions import ALL_RUNS, ilc_input_dir, total_maps_dir

NSIDE = 2048
SEED_R2 = 1042  # r1 used 42; independent streams
FREQS = tuple(int(f) for f in PLANCK_FREQUENCIES_GHZ)
UK_ARCMIN = {
    100: 77.4,
    143: 33.0,
    217: 46.8,
    353: 154.0,
    545: 806.7,
    857: 19115.0,
}
NELL = {
    100: 5.07e-4,
    143: 9.21e-5,
    217: 1.85e-4,
    353: 2.00e-3,
    545: 5.51e-2,
    857: 30.9,
}

NOISE = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/planck_noise/homogeneous")

OMEGA_ARCMIN2 = hp.nside2pixarea(NSIDE, degrees=True) * 3600.0


def load_uK(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(hp.read_map(str(path), field=0, dtype=np.float64))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prescription",
        choices=ALL_RUNS,
        default="L1_m9",
        help="Beamed totals to pair with the shared r1/r2 white noise",
    )
    args = p.parse_args()
    name = args.prescription
    total_r1_dir = total_maps_dir(name)
    ilc_k_r1 = ilc_input_dir(name, 1)
    ilc_k_r2 = ilc_input_dir(name, 2)
    ilc_k_r1.mkdir(parents=True, exist_ok=True)
    ilc_k_r2.mkdir(parents=True, exist_ok=True)
    print(f"prescription={name}  seed_r2={SEED_R2}  nside={NSIDE}  Omega={OMEGA_ARCMIN2:.4f} arcmin^2")
    print(f"r1 K maps <- {total_r1_dir}")

    for nu in FREQS:
        sigma_pix = UK_ARCMIN[nu] / np.sqrt(OMEGA_ARCMIN2)
        dest_n = NOISE / f"{nu}GHz" / f"white_noise_{nu}GHz_nside{NSIDE}_uK_r2.fits"
        if dest_n.is_file():
            noise_r2 = load_uK(dest_n)
        else:
            rng = np.random.default_rng(SEED_R2 + int(nu))
            noise_r2 = rng.normal(0.0, sigma_pix, hp.nside2npix(NSIDE))
            write_map(
                dest_n,
                noise_r2,
                unit="uK_CMB",
                freq=float(nu),
                extra=[
                    ("FWHM", float(BEAM_FWHM_ARCMIN[nu]), "beam FWHM [arcmin], NOT applied"),
                    ("UKARCMIN", UK_ARCMIN[nu], "white noise w^{-1/2} [uK arcmin]"),
                    ("NELL", NELL[nu], "target white N_ell [uK^2]"),
                    ("SEED", SEED_R2 + int(nu), "np.random.default_rng seed"),
                    ("REAL", 2, "independent realisation (r1 used SEED=42+nu)"),
                    ("COMMENT", "homogeneous pixel white noise; no beam smoothing"),
                ],
                dtype=np.float32,
            )

        noise_r1 = load_uK(NOISE / f"{nu}GHz" / f"white_noise_{nu}GHz_nside{NSIDE}_uK.fits")
        total_r1 = load_uK(
            total_r1_dir / f"sky_CMB_tSZ_CIB_homog_{nu}GHz_nside{NSIDE}_uK.fits"
        )
        if hp.get_nside(noise_r1) != NSIDE or hp.get_nside(total_r1) != NSIDE:
            raise ValueError(f"{nu} GHz nside mismatch")
        total_r2 = total_r1 - noise_r1 + noise_r2
        corr = float(np.corrcoef(noise_r1, noise_r2)[0, 1])
        print(
            f"{nu:5d}  sig_pix={sigma_pix:8.3f}  rms_n1={noise_r1.std():8.3f}  "
            f"rms_n2={noise_r2.std():8.3f}  corr(n1,n2)={corr:+.4f}  "
            f"rms_tot2={total_r2.std():8.3f}"
        )
        dest_k1 = ilc_k_r1 / f"sky_CMB_tSZ_CIB_homog_{nu}GHz_nside{NSIDE}_K.fits"
        write_map(
            dest_k1,
            total_r1 * 1e-6,
            unit="K_CMB",
            freq=float(nu),
            extra=[
                ("COMPS", "CMB+tSZ+CIB+homog_noise"),
                ("PRESCRIP", name),
                ("FROMUK", str(total_r1_dir / f"sky_CMB_tSZ_CIB_homog_{nu}GHz_nside{NSIDE}_uK.fits")),
                ("REAL", 1),
            ],
            dtype=np.float32,
        )
        dest_t = total_r1_dir / f"sky_CMB_tSZ_CIB_homog_{nu}GHz_nside{NSIDE}_uK_r2.fits"
        write_map(
            dest_t,
            total_r2,
            unit="uK_CMB",
            freq=float(nu),
            extra=[
                ("COMPS", "CMB+tSZ+CIB+homog_noise"),
                ("FWHM", float(BEAM_FWHM_ARCMIN[nu]), "Gaussian beam [arcmin] on signal only"),
                ("BEAMON", 1),
                ("NOISE", "homogeneous white r2, not beamed"),
                ("REAL", 2),
                ("NKSZ", 1, "kSZ not included"),
                ("PIXWIN", 0, "0=do NOT deconvolve HEALPix pixwin"),
                ("PRESCRIP", name),
                ("COMMENT", "same beamed signal as r1; independent white noise"),
            ],
            dtype=np.float32,
        )
        dest_k = ilc_k_r2 / f"sky_CMB_tSZ_CIB_homog_{nu}GHz_nside{NSIDE}_K.fits"
        write_map(
            dest_k,
            total_r2 * 1e-6,
            unit="K_CMB",
            freq=float(nu),
            extra=[
                ("COMPS", "CMB+tSZ+CIB+homog_noise"),
                ("PRESCRIP", name),
                ("FROMUK", str(dest_t)),
                ("REAL", 2),
            ],
            dtype=np.float32,
        )
        del noise_r1, noise_r2, total_r1, total_r2
    print("Done.")


if __name__ == "__main__":
    main()
