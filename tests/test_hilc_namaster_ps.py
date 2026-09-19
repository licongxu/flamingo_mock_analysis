"""Deconv-then-bin for HILC r1×r2 NaMaster (beam and binning do not commute)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_SPEC = importlib.util.spec_from_file_location(
    "compute_hilc_y_namaster_ps", _SCRIPTS / "compute_hilc_y_namaster_ps.py"
)
assert _SPEC is not None and _SPEC.loader is not None
ps = importlib.util.module_from_spec(_SPEC)
sys.modules["compute_hilc_y_namaster_ps"] = ps
_SPEC.loader.exec_module(ps)


def test_planck18_and_log_edges():
    assert ps.ELL_MIN[0] == 9 and ps.ELL_MAX[-1] == 1085
    assert ps.ELL_MIN.size == 18
    assert ps.LOG_EDGES.size == 13
    assert abs(ps.LOG_EDGES[-1] - ps.LMAX) < 1e-6


def test_deconv_then_bin_differs_from_bin_then_deconv():
    ell = np.arange(ps.LMAX + 1, dtype=np.float64)
    cl = 1.0 / np.maximum(ell * (ell + 1.0), 1.0)
    trans = ps.transfer(ps.NSIDE, ps.LMAX)
    cl_de = ps.deconv_per_ell(cl, trans)
    dl_ok = ps.bin_dl_18(ell, cl_de)
    # Wrong order: bin C_ell, then divide by transfer at ell_eff.
    dl_wrong = np.empty_like(dl_ok)
    for i, (lo, hi, leff) in enumerate(zip(ps.ELL_MIN, ps.ELL_MAX, ps.ELL_EFF)):
        inside = (ell >= lo) & (ell <= hi)
        dl_binned = np.nanmean(ell[inside] * (ell[inside] + 1.0) * cl[inside] / (2.0 * np.pi))
        t = trans[int(round(leff))]
        dl_wrong[i] = dl_binned / t
    assert np.nanmax(np.abs(dl_ok / dl_wrong - 1.0)) > 1e-4
