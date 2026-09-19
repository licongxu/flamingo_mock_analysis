#!/usr/bin/env python3
"""NaMaster truth y and HILC-weighted CIB/CMB r1×r2 residuals.

Truth: ud-grade Compton-y to Nside=2048, MASTER nlb=1, pixwin-only deconv.
Written for L1_m9, fgas-8sigma, and Mstar-1sigma (own q>5 mask).
Residuals: pyILC weights × (B_10 taper) on CIB/CMB alms → alm2map → same
workspace, deconv B_10^2 only. lmax=1085 covers the 18 Planck bins.
q>5 residuals multiply maps by the C2 0.25° apodized cluster mask before
map2alm (same mask as the synthetic MASTER spectra), then apply HILC
weights. L1_m9 also stores tSZ×CIB.
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

_DSPEC = importlib.util.spec_from_file_location(
    "hilc_r1xr2_diag", _SCRIPTS / "plot_hilc_homog_r1xr2_split_diagnostics.py"
)
assert _DSPEC is not None and _DSPEC.loader is not None
diag = importlib.util.module_from_spec(_DSPEC)
sys.modules["hilc_r1xr2_diag"] = diag
_DSPEC.loader.exec_module(diag)

from hilc_prescriptions import (  # noqa: E402
    ALL_DEPROJ,
    ALL_RUNS,
    DEPROJ_NONE,
    ILC,
    cib_dir,
    cluster_mask_apo,
    cmb_path,
    hilc_output_dir,
    tsz_dir,
)

ps.LMAX = 1085
LMAX = 1085
OUT = ps.OUT
TRUTH_RUNS = ("L1_m9", "fgas-8sigma", "Mstar-1sigma")


def _pixwin2() -> np.ndarray:
    w = np.asarray(hp.pixwin(ps.NSIDE, lmax=LMAX, pol=False), dtype=np.float64)
    t = np.full(LMAX + 1, np.nan)
    good = w > 1e-6
    t[good] = w[good] ** 2
    return t


def _beam2() -> np.ndarray:
    b = np.asarray(diag.bl10()[: LMAX + 1], dtype=np.float64)
    t = np.full(LMAX + 1, np.nan)
    good = b > 1e-6
    t[good] = b[good] ** 2
    return t


def _cache_path(name: str, *, masked: bool) -> Path:
    tag = "_q5apo" if masked else ""
    return ILC / "plot_cache" / f"signal_alms_{name}_cib_cmb_lmax1085{tag}.npz"


def _premask(m: np.ndarray, wapo: np.ndarray | None) -> np.ndarray:
    if wapo is None:
        return m
    if hp.get_nside(m) != ps.NSIDE:
        m = hp.ud_grade(m, ps.NSIDE)
    return m * wapo


def _apo_mask(name: str) -> np.ndarray:
    return np.asarray(hp.read_map(str(cluster_mask_apo(name)), dtype=np.float64))


def _tsz_alms(name: str, *, masked: bool) -> tuple[np.ndarray, ...]:
    wapo = _apo_mask(name) if masked else None
    if not masked:
        budget = ILC / "plot_cache" / f"budget_alms_{name}_lmax{LMAX}.npz"
        if budget.is_file():
            z = np.load(budget)
            print("loaded tSZ", budget, flush=True)
            return tuple(z[f"tsz_{i}"] for i in range(len(diag.FREQS)))
    cache = ILC / "plot_cache" / f"tsz_alms_{name}_lmax{LMAX}_q5apo.npz"
    if masked and cache.is_file():
        z = np.load(cache)
        print("loaded tSZ", cache, flush=True)
        return tuple(z[f"tsz_{i}"] for i in range(len(diag.FREQS)))
    print(f"map2alm tSZ lmax=1085 ({name}{' q5apo' if masked else ''}) ...", flush=True)
    out = []
    for f in diag.FREQS:
        m = _premask(
            diag.load_uk_to_k(tsz_dir(name) / f"tSZ_deltaT_{f}GHz_nside4096.fits"),
            wapo,
        )
        out.append(hp.map2alm(m, lmax=LMAX, iter=0))
        print(f"  tSZ {f} GHz", flush=True)
        del m
    if masked:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **{f"tsz_{i}": a for i, a in enumerate(out)})
        print("cached", cache, flush=True)
    return tuple(out)


def _cmb_cib_alms(
    name: str, *, masked: bool
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    cache = _cache_path(name, masked=masked)
    if cache.is_file():
        z = np.load(cache)
        cib = tuple(z[f"cib_alm_{i}"] for i in range(len(diag.FREQS)))
        print("loaded", cache, flush=True)
        return z["cmb_alm"], cib
    wapo = _apo_mask(name) if masked else None
    print(
        f"map2alm CMB / CIB lmax=1085 ({name}{' q5apo' if masked else ''}) ...",
        flush=True,
    )
    cmb = hp.map2alm(_premask(diag.load_uk_to_k(cmb_path(name)), wapo), lmax=LMAX, iter=0)
    cib = []
    for f in diag.FREQS:
        m = _premask(
            diag.load_uk_to_k(cib_dir(name) / f"CIB_deltaT_{f}GHz_nside4096.fits"),
            wapo,
        )
        cib.append(hp.map2alm(m, lmax=LMAX, iter=0))
        print(f"  CIB {f} GHz", flush=True)
        del m
    cache.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cmb_alm": cmb}
    for i, a in enumerate(cib):
        payload[f"cib_alm_{i}"] = a
    np.savez(cache, **payload)
    print("cached", cache, flush=True)
    return cmb, tuple(cib)


def _y_alms(w1: np.ndarray, w2: np.ndarray, alms: tuple[np.ndarray, ...], same_all_freq: bool):
    bl = diag.bl10()[: w1.shape[1]] * diag.taper()[: w1.shape[1]]
    y1 = y2 = None
    for a in range(len(diag.FREQS)):
        alm = alms[0] if same_all_freq else alms[a]
        c1 = hp.almxfl(alm, w1[a] * bl)
        c2 = hp.almxfl(alm, w2[a] * bl)
        y1 = c1 if y1 is None else y1 + c1
        y2 = c2 if y2 is None else y2 + c2
    return y1, y2


def _to_map(alm: np.ndarray) -> np.ndarray:
    return hp.alm2map(alm, nside=ps.NSIDE, lmax=LMAX)


def _merge_npz(path: Path, **extra) -> None:
    z = np.load(path, allow_pickle=True)
    payload = {k: z[k] for k in z.files}
    payload.update(extra)
    np.savez_compressed(path, **payload)


def _write_truth(
    name: str,
    *,
    ones: np.ndarray,
    ws_full,
    bins_full,
    trans_pix: np.ndarray,
    ell: np.ndarray,
) -> None:
    pending = [
        kind
        for kind in ("total", "masked")
        if not (OUT / f"{name}_truth_{kind}.npz").is_file()
    ]
    for kind in ("total", "masked"):
        if kind not in pending:
            print("skip", OUT / f"{name}_truth_{kind}.npz", flush=True)
    if not pending:
        return
    mask = np.asarray(hp.read_map(str(cluster_mask_apo(name)), dtype=np.float64))
    fsky_eff = float(np.mean(mask**2))
    print(f"coupling masked {name} (truth) ...", flush=True)
    ws_mask, bins_mask = ps.coupling(mask)
    yt = diag.load_map(tsz_dir(name) / "compton_y_nside4096.fits")
    skies = {
        "total": (ones, ws_full, bins_full, 1.0),
        "masked": (mask, ws_mask, bins_mask, fsky_eff),
    }
    for kind in pending:
        mask_use, ws, bins, fsky = skies[kind]
        path = OUT / f"{name}_truth_{kind}.npz"
        print(f"truth {name} {kind} ...", flush=True)
        cl = ps.deconv_per_ell(ps.decoupled_cross(yt, yt, mask_use, ws, bins), trans_pix)
        np.savez_compressed(
            path,
            ell=ps.ELL_EFF,
            dl=ps.bin_dl_18(ell, cl),
            cl_ell=cl,
            fsky_eff=fsky,
            kind=kind,
            prescription=name,
            lmax=LMAX,
        )
        print("wrote", path, flush=True)


def _write_residuals(
    name: str,
    deprojs,
    *,
    ones,
    ws_full,
    bins_full,
    trans_b,
    ell,
    with_tsz_cib: bool = False,
) -> None:
    mask = np.asarray(hp.read_map(str(cluster_mask_apo(name)), dtype=np.float64))
    fsky_eff = float(np.mean(mask**2))
    print(f"coupling masked {name} ...", flush=True)
    ws_mask, bins_mask = ps.coupling(mask)
    cmb_full, cib_full = _cmb_cib_alms(name, masked=False)
    cmb_q5, cib_q5 = _cmb_cib_alms(name, masked=True)
    tsz_full = _tsz_alms(name, masked=False) if with_tsz_cib else None
    tsz_q5 = _tsz_alms(name, masked=True) if with_tsz_cib else None
    skies = (
        ("total", False, ones, ws_full, bins_full, 1.0),
        ("masked", True, mask, ws_mask, bins_mask, fsky_eff),
    )
    for deproj in deprojs:
        for kind, masked, mask_use, ws, bins, fsky in skies:
            path = OUT / f"{name}_{deproj.key}_{kind}_residuals.npz"
            have = path.is_file()
            have_cross = False
            apo_sht = False
            if have:
                with np.load(path) as z:
                    have_cross = "dl_tsz_cib" in z
                    apo_sht = bool(z["apodized_sht"]) if "apodized_sht" in z.files else False
            complete = have and (have_cross or not with_tsz_cib)
            if complete and ((not masked) or apo_sht):
                print("skip", path, flush=True)
                continue
            print(f"residuals {name} {deproj.key} {kind} ...", flush=True)
            cmb_alm, cib_alms = (cmb_q5, cib_q5) if masked else (cmb_full, cib_full)
            tsz_alms = tsz_q5 if masked else tsz_full
            w1 = diag.hilc_weights(
                hilc_output_dir(name, masked=masked, real=1, deproj=deproj),
                deproj.wtag,
                LMAX,
            )
            w2 = diag.hilc_weights(
                hilc_output_dir(name, masked=masked, real=2, deproj=deproj),
                deproj.wtag,
                LMAX,
            )
            y_cib1, y_cib2 = _y_alms(w1, w2, cib_alms, False)
            m_cib1, m_cib2 = _to_map(y_cib1), _to_map(y_cib2)
            del y_cib1, y_cib2
            y_cmb1, y_cmb2 = _y_alms(w1, w2, (cmb_alm,), True)
            m_cmb1, m_cmb2 = _to_map(y_cmb1), _to_map(y_cmb2)
            del y_cmb1, y_cmb2
            cl_cib = ps.deconv_per_ell(
                ps.decoupled_cross(m_cib1, m_cib2, mask_use, ws, bins), trans_b
            )
            cl_cmb = ps.deconv_per_ell(
                ps.decoupled_cross(m_cmb1, m_cmb2, mask_use, ws, bins), trans_b
            )
            del m_cmb1, m_cmb2
            np.savez_compressed(
                path,
                ell=ps.ELL_EFF,
                dl_cib=ps.bin_dl_18(ell, cl_cib),
                dl_cmb=ps.bin_dl_18(ell, cl_cmb),
                cl_cib=cl_cib,
                cl_cmb=cl_cmb,
                fsky_eff=fsky,
                prescription=name,
                deproj=deproj.key,
                kind=kind,
                lmax=LMAX,
                apodized_sht=bool(masked),
            )
            print("wrote", path, flush=True)
            if with_tsz_cib:
                y_tsz1, y_tsz2 = _y_alms(w1, w2, tsz_alms, False)
                m_tsz1, m_tsz2 = _to_map(y_tsz1), _to_map(y_tsz2)
                del y_tsz1, y_tsz2
                cl_x = ps.deconv_per_ell(
                    ps.decoupled_cross(m_tsz1, m_cib2, mask_use, ws, bins)
                    + ps.decoupled_cross(m_cib1, m_tsz2, mask_use, ws, bins),
                    trans_b,
                )
                del m_tsz1, m_tsz2
                _merge_npz(
                    path,
                    dl_tsz_cib=ps.bin_dl_18(ell, cl_x),
                    cl_tsz_cib=cl_x,
                )
                print("wrote tSZ×CIB", path, flush=True)
            del m_cib1, m_cib2


def main() -> None:
    ones = np.ones(hp.nside2npix(ps.NSIDE), dtype=np.float64)
    ell = np.arange(LMAX + 1, dtype=np.float64)
    trans_pix = _pixwin2()
    trans_b = _beam2()

    print("coupling full-sky ...", flush=True)
    ws_full, bins_full = ps.coupling(ones)
    for name in TRUTH_RUNS:
        _write_truth(
            name,
            ones=ones,
            ws_full=ws_full,
            bins_full=bins_full,
            trans_pix=trans_pix,
            ell=ell,
        )

    _write_residuals(
        "L1_m9", ALL_DEPROJ,
        ones=ones, ws_full=ws_full, bins_full=bins_full, trans_b=trans_b, ell=ell,
        with_tsz_cib=True,
    )
    for name in ALL_RUNS:
        if name == "L1_m9":
            continue
        _write_residuals(
            name, (DEPROJ_NONE,),
            ones=ones, ws_full=ws_full, bins_full=bins_full, trans_b=trans_b, ell=ell,
        )


if __name__ == "__main__":
    main()
