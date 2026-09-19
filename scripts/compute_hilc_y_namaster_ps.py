#!/usr/bin/env python3
"""Beam-deconvolved NaMaster r1×r2 cross of homog HILC y-maps.

Order: MASTER nlb=1 → C_ell / (b_ell^2 w_ell^2) per ell → then bin.
Planck-18 (mean D_ell) and log bins (mean C_ell → D_ell at geometric centre).
"""
from __future__ import annotations

import sys
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from hilc_prescriptions import (  # noqa: E402
    ALL_DEPROJ,
    ALL_RUNS,
    ILC,
    cluster_mask_apo,
    hilc_ymap,
)

NSIDE = 2048
LMAX = 4096
FWHM_ARCMIN = 10.0
OUT = ILC / "ps_namaster"

ELL_MIN = np.array(
    [9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835]
)
ELL_MAX = np.array(
    [12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835, 1085]
)
ELL_EFF = np.array(
    [
        10.0, 13.5, 18.0, 23.5, 30.5, 40.0, 52.5, 68.5, 89.5, 117.0, 152.5,
        198.0, 257.5, 335.5, 436.5, 567.5, 738.0, 959.5,
    ]
)
N_LOG_BINS = 12
DLN_ELL = 0.4
LOG_EDGES = LMAX * np.exp(-N_LOG_BINS * DLN_ELL) * np.exp(DLN_ELL * np.arange(N_LOG_BINS + 1))


def transfer(nside: int, lmax: int) -> np.ndarray:
    """b_ell^2 w_ell^2; NaN where b_ell w_ell is tiny."""
    b = np.asarray(hp.gauss_beam(np.radians(FWHM_ARCMIN / 60.0), lmax=lmax), dtype=np.float64)
    w = np.asarray(hp.pixwin(nside, lmax=lmax, pol=False), dtype=np.float64)
    bw = b * w
    t = np.full(lmax + 1, np.nan)
    good = bw > 1e-6
    t[good] = bw[good] ** 2
    return t


def deconv_per_ell(cl: np.ndarray, trans: np.ndarray) -> np.ndarray:
    out = np.full_like(cl, np.nan)
    good = np.isfinite(cl) & np.isfinite(trans) & (trans > 0.0)
    out[good] = cl[good] / trans[good]
    return out


def bin_dl_18(ell: np.ndarray, cl: np.ndarray) -> np.ndarray:
    dl = ell * (ell + 1.0) * cl / (2.0 * np.pi)
    out = np.full(len(ELL_MIN), np.nan)
    for i, (lo, hi) in enumerate(zip(ELL_MIN, ELL_MAX)):
        inside = (ell >= lo) & (ell <= hi)
        out[i] = np.nanmean(dl[inside])
    return out


def bin_cl_log(ell: np.ndarray, cl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centres = np.sqrt(LOG_EDGES[:-1] * LOG_EDGES[1:])
    out = np.empty(N_LOG_BINS)
    for i, (lo, hi) in enumerate(zip(LOG_EDGES[:-1], LOG_EDGES[1:])):
        inside = (ell >= lo) & (ell <= hi) if i == N_LOG_BINS - 1 else (ell >= lo) & (ell < hi)
        if not np.any(inside):
            raise ValueError(f"log bin [{lo}, {hi}] is empty")
        out[i] = np.nanmean(cl[inside])
    return centres, centres * (centres + 1.0) * out / (2.0 * np.pi)


def _subtract_monopole(m: np.ndarray, w: np.ndarray) -> np.ndarray:
    return m - float(np.sum(w * m) / np.sum(w))


def coupling(mask: np.ndarray) -> tuple[nmt.NmtWorkspace, nmt.NmtBin]:
    bins = nmt.NmtBin.from_lmax_linear(LMAX, nlb=1)
    tmpl = nmt.NmtField(mask, [np.zeros_like(mask)], lmax=LMAX)
    ws = nmt.NmtWorkspace()
    ws.compute_coupling_matrix(tmpl, tmpl, bins)
    return ws, bins


def decoupled_cross(
    m1: np.ndarray, m2: np.ndarray, mask: np.ndarray, ws: nmt.NmtWorkspace, bins: nmt.NmtBin
) -> np.ndarray:
    a = _subtract_monopole(m1, mask)
    b = _subtract_monopole(m2, mask)
    f1 = nmt.NmtField(mask, [a], lmax=LMAX)
    f2 = nmt.NmtField(mask, [b], lmax=LMAX)
    cl = ws.decouple_cell(nmt.compute_coupled_cell(f1, f2))[0]
    ell_eff = bins.get_effective_ells().astype(int)
    cl_full = np.full(LMAX + 1, np.nan)
    cl_full[ell_eff] = cl
    return cl_full


def _load_y(name: str, *, masked: bool, real: int, deproj) -> np.ndarray:
    path = hilc_ymap(name, masked=masked, real=real, deproj=deproj)
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(hp.read_map(str(path), dtype=np.float64))


def _write(stem: Path, *, ell_b: np.ndarray, dl: np.ndarray, cl_ell: np.ndarray, extra: dict) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stem.with_suffix(".npz"),
        ell=ell_b,
        dl=dl,
        cl_ell=cl_ell,
        ell_full=np.arange(LMAX + 1, dtype=np.float64),
        **extra,
    )
    print("wrote", stem.with_suffix(".npz"), flush=True)


def main() -> None:
    trans = transfer(NSIDE, LMAX)
    ell = np.arange(LMAX + 1, dtype=np.float64)
    ones = np.ones(hp.nside2npix(NSIDE), dtype=np.float64)
    print("coupling full-sky ...", flush=True)
    ws_full, bins = coupling(ones)
    ws_mask: dict[str, tuple] = {}

    for name in ALL_RUNS:
        mask_apo = np.asarray(hp.read_map(str(cluster_mask_apo(name)), dtype=np.float64))
        print(f"coupling masked {name} ...", flush=True)
        ws_mask[name] = coupling(mask_apo)
        fsky_raw = float((mask_apo > 0.5).mean())
        fsky_eff = float(np.mean(mask_apo**2))

        for deproj in ALL_DEPROJ:
            y1 = _load_y(name, masked=False, real=1, deproj=deproj)
            y2 = _load_y(name, masked=False, real=2, deproj=deproj)
            cl_tot = deconv_per_ell(decoupled_cross(y1, y2, ones, ws_full, bins), trans)
            y1m = _load_y(name, masked=True, real=1, deproj=deproj)
            y2m = _load_y(name, masked=True, real=2, deproj=deproj)
            ws_m, bins_m = ws_mask[name]
            cl_msk = deconv_per_ell(decoupled_cross(y1m, y2m, mask_apo, ws_m, bins_m), trans)
            extra = {
                "prescription": name,
                "deproj": deproj.key,
                "fsky_raw": fsky_raw,
                "fsky_eff": fsky_eff,
                "lmax": LMAX,
                "fwhm_arcmin": FWHM_ARCMIN,
            }
            tag = f"{name}_{deproj.key}"
            for kind, cl in (("total", cl_tot), ("masked", cl_msk)):
                _write(
                    OUT / f"{tag}_{kind}_p18",
                    ell_b=ELL_EFF,
                    dl=bin_dl_18(ell, cl),
                    cl_ell=cl,
                    extra=extra,
                )
                ell_log, dl_log = bin_cl_log(ell, cl)
                _write(
                    OUT / f"{tag}_{kind}_log",
                    ell_b=ell_log,
                    dl=dl_log,
                    cl_ell=cl,
                    extra=extra,
                )


if __name__ == "__main__":
    main()
