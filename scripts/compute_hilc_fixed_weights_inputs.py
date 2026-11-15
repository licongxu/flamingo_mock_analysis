#!/usr/bin/env python3
"""Apply saved HILC weights to ILC input frequency maps (Exp 2).

pyILC sees already-beamed totals + unbeamed noise, so the filter is
w × B_10 / B_ν × taper. q>5 fiducial mask. Writes
ps_namaster/fixedw_inputs_deproj_cib_dbeta_masked.npz and checks that
fiducial weights on fiducial inputs recover the measured y-map.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import healpy as hp
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location(
    "compute_hilc_y_namaster_ps", _SCRIPTS / "compute_hilc_y_namaster_ps.py"
)
assert _SPEC is not None and _SPEC.loader is not None
ps = importlib.util.module_from_spec(_SPEC)
sys.modules["compute_hilc_y_namaster_ps"] = ps
_SPEC.loader.exec_module(ps)

_DSPEC = importlib.util.spec_from_file_location(
    "hilc_r1xr2_diag", _SCRIPTS / "plot_hilc_homog_r1xr2_split_diagnostics.py"
)
assert _DSPEC is not None and _DSPEC.loader is not None
diag = importlib.util.module_from_spec(_DSPEC)
sys.modules["hilc_r1xr2_diag"] = diag
_DSPEC.loader.exec_module(diag)

from hilc_prescriptions import (
    DEPROJ_CIB_DBETA,
    ILC,
    cluster_mask_apo,
    hilc_output_dir,
    ilc_input_dir,
)

ps.LMAX = 1085
LMAX = 1085
OUT = ps.OUT
FREQS = diag.FREQS
DEPROJ = DEPROJ_CIB_DBETA


def _slice(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr[: LMAX + 1], dtype=np.float64)


def _cache(name: str, real: int) -> Path:
    return ILC / "plot_cache" / f"ilc_input_alms_{name}_r{real}_lmax{LMAX}.npz"


def _input_alms(name: str, real: int) -> tuple[np.ndarray, ...]:
    path = _cache(name, real)
    if path.is_file():
        z = np.load(path)
        print("loaded", path, flush=True)
        return tuple(z[f"a{i}"] for i in range(len(FREQS)))
    print(f"map2alm ILC inputs {name} r{real} ...", flush=True)
    d = ilc_input_dir(name, real)
    alms = []
    for f in FREQS:
        m = np.asarray(
            hp.read_map(str(d / f"sky_CMB_tSZ_CIB_homog_{f}GHz_nside2048_K.fits"), dtype=np.float64)
        )
        m = m - np.mean(m)
        alms.append(hp.map2alm(m, lmax=LMAX, iter=0))
        print(f"  {f} GHz", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"a{i}": a for i, a in enumerate(alms)})
    return tuple(alms)


def _weights(name: str, real: int) -> np.ndarray:
    w = diag.hilc_weights(
        hilc_output_dir(name, masked=True, real=real, deproj=DEPROJ),
        DEPROJ.wtag,
        diag.LMAX,
    )
    return np.asarray(w[:, : LMAX + 1], dtype=np.float64)


def _filt(w_nu: np.ndarray, a: int) -> np.ndarray:
    bnu = np.maximum(_slice(diag.bnu()[a]), 1e-30)
    return w_nu * _slice(diag.bl10()) / bnu * _slice(diag.taper())


def _y(w, alms):
    y = None
    for a in range(len(FREQS)):
        c = hp.almxfl(alms[a], _filt(w[a], a))
        y = c if y is None else y + c
    return hp.alm2map(y, nside=ps.NSIDE, lmax=LMAX)


def _dl(m1, m2, mask, ws, bins, trans, ell):
    cl = ps.deconv_per_ell(ps.decoupled_cross(m1, m2, mask, ws, bins), trans)
    return ps.bin_dl_18(ell, cl)


def main() -> None:
    out = OUT / "fixedw_inputs_deproj_cib_dbeta_masked.npz"
    trans = ps.transfer(ps.NSIDE, LMAX)
    ell = np.arange(LMAX + 1, dtype=np.float64)
    mask = np.asarray(hp.read_map(str(cluster_mask_apo("L1_m9")), dtype=np.float64))
    print("coupling masked L1_m9 ...", flush=True)
    ws, bins = ps.coupling(mask)
    alms = {
        (name, real): _input_alms(name, real)
        for name in ("L1_m9", "Mstar-1sigma")
        for real in (1, 2)
    }
    w = {
        (name, real): _weights(name, real)
        for name in ("L1_m9", "Mstar-1sigma")
        for real in (1, 2)
    }
    cases = {
        "fid_fidw": ("L1_m9", "L1_m9"),
        "mstar_fidw": ("Mstar-1sigma", "L1_m9"),
        "fid_mstarw": ("L1_m9", "Mstar-1sigma"),
        "mstar_mstarw": ("Mstar-1sigma", "Mstar-1sigma"),
    }
    payload = {"ell": ps.ELL_EFF}
    for key, (sky, wname) in cases.items():
        print(f"{key} ...", flush=True)
        y1 = _y(w[(wname, 1)], alms[(sky, 1)])
        y2 = _y(w[(wname, 2)], alms[(sky, 2)])
        payload[f"{key}_dl"] = _dl(y1, y2, mask, ws, bins, trans, ell)
        del y1, y2
    meas = np.asarray(np.load(OUT / "L1_m9_deproj_cib_dbeta_masked_p18.npz")["dl"])
    rel = np.abs(payload["fid_fidw_dl"] - meas) / np.maximum(np.abs(meas), 1e-30)
    print(
        f"fid inputs+weights vs p18  ell10={rel[0]:.3f} ell40={rel[5]:.3f}",
        flush=True,
    )
    r_own = payload["mstar_mstarw_dl"] / np.maximum(payload["fid_fidw_dl"], 1e-30)
    r_fix = payload["mstar_fidw_dl"] / np.maximum(payload["fid_fidw_dl"], 1e-30)
    print(f"own-w ell10={r_own[0]:.3f}  fid-w ell10={r_fix[0]:.3f}", flush=True)
    np.savez_compressed(out, **payload, dl_meas_fid=meas)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
