#!/usr/bin/env python3
"""NaMaster truth y and HILC-weighted CIB/CMB r1×r2 residuals.

Truth: ud-grade Compton-y to Nside=2048, MASTER nlb=1, pixwin-only deconv.
Written for L1_m9, fgas-8sigma, and Mstar-1sigma (own q>5 mask).
fgas-8sigma and Mstar-1sigma store residuals for every deprojection.
Residuals: channel-beam the signal, multiply the mask, map2alm, then
pyILC weights × (B_10/B_ν taper). Noise is not in these files. Deconv is
B_10^2 only, lmax=4096, same MASTER setup as the y-map p18 spectra.
Full-sky mean is not removed; NaMaster subtracts the mask-weighted monopole.
L1_m9 also stores tSZ×CIB.
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
    DEPROJ_NONE,
    ILC,
    cib_dir,
    cluster_mask_apo,
    cmb_path,
    hilc_output_dir,
    tsz_dir,
)

LMAX = ps.LMAX
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


OPERATOR = "beam_then_mask"


def _cache_path(name: str, *, masked: bool) -> Path:
    tag = "_q5apo" if masked else ""
    return ILC / "plot_cache" / f"signal_alms_{name}_beamed_lmax{LMAX}{tag}.npz"


def _k(path: Path) -> np.ndarray:
    return diag.load_map(path) * 1e-6


def _beam(m: np.ndarray, fwhm_arcmin: float) -> np.ndarray:
    if hp.get_nside(m) != ps.NSIDE:
        m = hp.ud_grade(m, ps.NSIDE)
    return hp.smoothing(m, fwhm=np.radians(fwhm_arcmin / 60.0), pol=False)


def _alm(m: np.ndarray, wapo: np.ndarray | None) -> np.ndarray:
    if wapo is None:
        if hp.get_nside(m) != ps.NSIDE:
            m = hp.ud_grade(m, ps.NSIDE)
        return hp.map2alm(m, lmax=LMAX, iter=0)
    return hp.map2alm(wapo * m, lmax=LMAX, iter=0)


def _apo_mask(name: str) -> np.ndarray:
    return np.asarray(hp.read_map(str(cluster_mask_apo(name)), dtype=np.float64))


def _component_alms(name: str, *, masked: bool) -> dict[str, tuple[np.ndarray, ...]]:
    cache = _cache_path(name, masked=masked)
    n = len(diag.FREQS)
    if cache.is_file():
        z = np.load(cache)
        print("loaded", cache, flush=True)
        return {
            key: tuple(z[f"{key}_{i}"] for i in range(n))
            for key in ("cmb", "cib", "tsz")
        }
    wapo = _apo_mask(name) if masked else None
    print(f"beam then map2alm lmax={LMAX} ({name}{' q5apo' if masked else ''})", flush=True)
    cmb0 = _k(cmb_path(name))
    out = {"cmb": [], "cib": [], "tsz": []}
    for f in diag.FREQS:
        fwhm = float(diag.BEAM_FWHM_ARCMIN[int(f)])
        out["cmb"].append(_alm(_beam(cmb0, fwhm), wapo))
        out["tsz"].append(
            _alm(_beam(_k(tsz_dir(name) / f"tSZ_deltaT_{f}GHz_nside4096.fits"), fwhm), wapo)
        )
        out["cib"].append(
            _alm(_beam(_k(cib_dir(name) / f"CIB_deltaT_{f}GHz_nside4096.fits"), fwhm), wapo)
        )
        print(f"  {f} GHz", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{f"{key}_{i}": a for key, alms in out.items() for i, a in enumerate(alms)})
    print("cached", cache, flush=True)
    return {key: tuple(alms) for key, alms in out.items()}


def _y_alms(w1: np.ndarray, w2: np.ndarray, alms: tuple[np.ndarray, ...]):
    n = w1.shape[1]
    bl10 = diag.bl10()[:n]
    taper = diag.taper()[:n]
    bnu = diag.bnu()[:, :n]
    y1 = y2 = None
    for a in range(len(diag.FREQS)):
        filt = bl10 / np.maximum(bnu[a], 1e-30) * taper
        c1 = hp.almxfl(alms[a], w1[a] * filt)
        c2 = hp.almxfl(alms[a], w2[a] * filt)
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
    full = _component_alms(name, masked=False)
    q5 = _component_alms(name, masked=True)
    skies = (
        ("total", False, ones, ws_full, bins_full, 1.0),
        ("masked", True, mask, ws_mask, bins_mask, fsky_eff),
    )
    for deproj in deprojs:
        for kind, masked, mask_use, ws, bins, fsky in skies:
            path = OUT / f"{name}_{deproj.key}_{kind}_residuals.npz"
            have = path.is_file()
            have_cross = False
            operator = ""
            if have:
                with np.load(path) as z:
                    have_cross = "dl_tsz_cib" in z
                    operator = str(z["operator"]) if "operator" in z.files else ""
            complete = have and operator == OPERATOR and (have_cross or not with_tsz_cib)
            if complete:
                print("skip", path, flush=True)
                continue
            print(f"residuals {name} {deproj.key} {kind} ...", flush=True)
            comp = q5 if masked else full
            cib_alms, cmb_alms = comp["cib"], comp["cmb"]
            tsz_alms = comp["tsz"] if with_tsz_cib else None
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
            y_cib1, y_cib2 = _y_alms(w1, w2, cib_alms)
            m_cib1, m_cib2 = _to_map(y_cib1), _to_map(y_cib2)
            del y_cib1, y_cib2
            y_cmb1, y_cmb2 = _y_alms(w1, w2, cmb_alms)
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
                operator=OPERATOR,
            )
            print("wrote", path, flush=True)
            if with_tsz_cib:
                y_tsz1, y_tsz2 = _y_alms(w1, w2, tsz_alms)
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

    only = sys.argv[1:]
    if not only or "L1_m9" in only:
        _write_residuals(
            "L1_m9", ALL_DEPROJ,
            ones=ones, ws_full=ws_full, bins_full=bins_full, trans_b=trans_b, ell=ell,
            with_tsz_cib=True,
        )
    # fgas / M* need every deprojection for the recovered-residual figure.
    # LS8 stays no-deprojection only.
    variant_deproj = {
        "fgas-8sigma": ALL_DEPROJ,
        "Mstar-1sigma": ALL_DEPROJ,
        "LS8": (DEPROJ_NONE,),
    }
    names = tuple(variant_deproj) if not only else tuple(a for a in only if a != "L1_m9")
    for name in names:
        _write_residuals(
            name, variant_deproj[name],
            ones=ones, ws_full=ws_full, bins_full=bins_full, trans_b=trans_b, ell=ell,
        )


if __name__ == "__main__":
    main()
