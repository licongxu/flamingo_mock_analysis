#!/usr/bin/env python3
"""Masked-HILC input maps: common 10' beam, mask-weighted monopole removed.

run_ilc multiplies the inputs by the q>5 cluster mask and pyILC then rebeams
B_10/B_nu. Masking at channel beams breaks sum_nu w_nu a_nu = 0 off the
diagonal (mask and beam do not commute), and the CIB monopole becomes a
hole pattern. So each full-sky channel is brought to 10' and its
mask-weighted mean removed here; the masked YAMLs declare 10' beams.

  python scripts/build_hilc_q5prep_inputs.py --prescription L1_m9
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import healpy as hp
import numpy as np
from pyilc.sht import alm2map, map2alm

from flamingo_mock.config import BEAM_FWHM_ARCMIN
from flamingo_mock.io import write_map

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from hilc_prescriptions import (  # noqa: E402
    ALL_RUNS,
    FREQS_ILC,
    cluster_mask_apo,
    freq_map_files,
    q5prep_map_files,
)

LMAX = 4096
FWHM_COMMON = 10.0


def to_common_beam_monosub(m: np.ndarray, fwhm_in: float, mask: np.ndarray) -> np.ndarray:
    """Full-sky map at ``fwhm_in`` -> 10' (lmax 4096), minus its mask-weighted mean."""
    alm, _ = map2alm(m, lmax=LMAX, niter=3, backend="ducc0")
    b_in = hp.gauss_beam(np.radians(fwhm_in / 60.0), lmax=LMAX)
    b10 = hp.gauss_beam(np.radians(FWHM_COMMON / 60.0), lmax=LMAX)
    out, _ = alm2map(hp.almxfl(alm, b10 / b_in), nside=hp.get_nside(m), lmax=LMAX, backend="ducc0")
    return out - np.sum(mask * out) / np.sum(mask)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prescription", required=True)
    p.add_argument("--mask-name", choices=ALL_RUNS, default=None, help="cluster mask (default: own)")
    args = p.parse_args()
    name = args.prescription
    mask_name = args.mask_name or name
    mask = np.asarray(hp.read_map(str(cluster_mask_apo(mask_name)), dtype=np.float64))
    for real in (1, 2):
        dests = q5prep_map_files(name, real, mask_name)
        for nu, src, dest in zip(FREQS_ILC, freq_map_files(name, real), dests):
            m = np.asarray(hp.read_map(src, dtype=np.float64))
            out = to_common_beam_monosub(m, float(BEAM_FWHM_ARCMIN[nu]), mask)
            print(f"{name} r{real} {nu:4d} GHz  mean {m.mean():+.3e} -> {out.mean():+.3e}  "
                  f"masked mean {np.sum(mask * out) / np.sum(mask):+.1e}", flush=True)
            write_map(
                Path(dest),
                out,
                unit="K_CMB",
                freq=float(nu),
                extra=[
                    ("FROMMAP", Path(src).name),
                    ("FWHM", FWHM_COMMON, "common Gaussian beam [arcmin]"),
                    ("MASK", cluster_mask_apo(mask_name).name, "mask-weighted mean removed"),
                    ("REAL", real),
                ],
                dtype=np.float64,
            )


if __name__ == "__main__":
    main()
