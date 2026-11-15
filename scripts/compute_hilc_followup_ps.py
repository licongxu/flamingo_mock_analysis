#!/usr/bin/env python3
"""NaMaster 18-bin r1×r2 of follow-up HILC y-maps (fiducial q>5 mask)."""
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

from hilc_prescriptions import (  # noqa: E402
    CIBSHUFFLE,
    DEPROJ_CIB_DBETA,
    DEPROJ_FOLLOWUP,
    DEPROJ_NONE,
    FIDMASK_TAG,
    HYBRID_FID_TSZ_MSTAR_CIB,
    HYBRID_MSTAR_TSZ_FID_CIB,
    cluster_mask_apo,
    hilc_ymap,
)

ps.LMAX = 1085
OUT = ps.OUT
CASES = (
    *[(HYBRID_FID_TSZ_MSTAR_CIB, d, "") for d in DEPROJ_FOLLOWUP],
    *[(HYBRID_MSTAR_TSZ_FID_CIB, d, "") for d in DEPROJ_FOLLOWUP],
    *[(n, d, FIDMASK_TAG) for n in ("Mstar-1sigma", "fgas-8sigma") for d in (DEPROJ_NONE, DEPROJ_CIB_DBETA)],
    *[(CIBSHUFFLE, d, "") for d in (DEPROJ_NONE, DEPROJ_CIB_DBETA)],
)


def _stem(name: str, deproj, extra_tag: str) -> str:
    tag = f"_{extra_tag}" if extra_tag else ""
    return f"{name}_{deproj.key}_masked{tag}"


def main() -> None:
    trans = ps.transfer(ps.NSIDE, ps.LMAX)
    ell = np.arange(ps.LMAX + 1, dtype=np.float64)
    mask = np.asarray(hp.read_map(str(cluster_mask_apo("L1_m9")), dtype=np.float64))
    fsky_eff = float(np.mean(mask**2))
    fsky_raw = float((mask > 0.5).mean())
    print("coupling masked L1_m9 ...", flush=True)
    ws, bins = ps.coupling(mask)
    for name, deproj, extra_tag in CASES:
        stem = OUT / _stem(name, deproj, extra_tag)
        path = Path(str(stem) + "_p18.npz")
        if path.is_file():
            print("skip", path, flush=True)
            continue
        y1 = np.asarray(
            hp.read_map(
                str(hilc_ymap(name, masked=True, real=1, deproj=deproj, extra_tag=extra_tag)),
                dtype=np.float64,
            )
        )
        y2 = np.asarray(
            hp.read_map(
                str(hilc_ymap(name, masked=True, real=2, deproj=deproj, extra_tag=extra_tag)),
                dtype=np.float64,
            )
        )
        cl = ps.deconv_per_ell(ps.decoupled_cross(y1, y2, mask, ws, bins), trans)
        extra = {
            "prescription": name,
            "deproj": deproj.key,
            "extra_tag": extra_tag,
            "fsky_raw": fsky_raw,
            "fsky_eff": fsky_eff,
            "lmax": ps.LMAX,
        }
        ps._write(stem.with_name(stem.name + "_p18"), ell_b=ps.ELL_EFF, dl=ps.bin_dl_18(ell, cl), cl_ell=cl, extra=extra)
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
