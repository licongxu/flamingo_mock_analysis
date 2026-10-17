#!/usr/bin/env python3
"""NaMaster r1×r2 autos and cross of L1_m9 HILC y for split-cross Gaussian errors.

Same MASTER nlb=1 → per-ℓ beam×pixwin deconvolution as compute_hilc_y_namaster_ps.py.
lmax=1085 covers the 18 Planck bins. Writes cl_11, cl_22, cl_12, fsky_eff.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import healpy as hp
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "compute_hilc_y_namaster_ps", _SCRIPTS / "compute_hilc_y_namaster_ps.py"
)
assert _SPEC is not None and _SPEC.loader is not None
ps = importlib.util.module_from_spec(_SPEC)
sys.modules["compute_hilc_y_namaster_ps"] = ps
_SPEC.loader.exec_module(ps)

from hilc_prescriptions import ALL_DEPROJ, cluster_mask_apo  # noqa: E402

ps.LMAX = 1085
OUT = ps.OUT
CASES = ALL_DEPROJ


def main() -> None:
    trans = ps.transfer(ps.NSIDE, ps.LMAX)
    ones = np.ones(hp.nside2npix(ps.NSIDE), dtype=np.float64)
    mask = np.asarray(hp.read_map(str(cluster_mask_apo("L1_m9")), dtype=np.float64))
    fsky_eff = float(np.mean(mask**2))
    fsky_raw = float((mask > 0.5).mean())
    print("coupling full-sky ...", flush=True)
    ws_full, bins_full = ps.coupling(ones)
    print("coupling masked ...", flush=True)
    ws_mask, bins_mask = ps.coupling(mask)

    for deproj in CASES:
        for kind, masked, mask_use, ws, bins, fsky in (
            ("total", False, ones, ws_full, bins_full, 1.0),
            ("masked", True, mask, ws_mask, bins_mask, fsky_eff),
        ):
            path = OUT / f"L1_m9_{deproj.key}_{kind}_splitcross.npz"
            if path.is_file():
                print("skip", path, flush=True)
                continue
            y1 = ps._load_y("L1_m9", masked=masked, real=1, deproj=deproj)
            y2 = ps._load_y("L1_m9", masked=masked, real=2, deproj=deproj)
            cl11 = ps.deconv_per_ell(ps.decoupled_cross(y1, y1, mask_use, ws, bins), trans)
            cl22 = ps.deconv_per_ell(ps.decoupled_cross(y2, y2, mask_use, ws, bins), trans)
            cl12 = ps.deconv_per_ell(ps.decoupled_cross(y1, y2, mask_use, ws, bins), trans)
            np.savez_compressed(
                path,
                cl_11=cl11,
                cl_22=cl22,
                cl_12=cl12,
                fsky_eff=fsky,
                fsky_raw=fsky_raw if masked else 1.0,
                deproj=deproj.key,
                kind=kind,
                lmax=ps.LMAX,
            )
            print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
