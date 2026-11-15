#!/usr/bin/env python3
"""NaMaster 18-bin HILC component budget (tSZ, CIB, CMB, noise, all crosses).

Same operator as the q>5 HILC map: multiply the apodized cluster mask, then
apply saved pyILC weights, then the MASTER workspace of
compute_hilc_y_namaster_ps.py. Also writes the fixed-weight CIB+δβ swap
(L1_m9 weights on M*-1σ maps and vice versa), with the fiducial mask applied
before the filter.
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

from flamingo_mock.powerspectra import sigma_dl_cross_binned
from hilc_prescriptions import (
    ALL_DEPROJ,
    DEPROJ_CIB_DBETA,
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
RUNS = ("L1_m9", "Mstar-1sigma")
FREQS = diag.FREQS
ELL_MIN = ps.ELL_MIN
ELL_MAX_EXCL = ps.ELL_MAX + 1
NOISE = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/planck_noise/homogeneous")


def _slice(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr[: LMAX + 1], dtype=np.float64)


def _cache(name: str, tag: str) -> Path:
    return ILC / "plot_cache" / f"budget_alms_{name}_lmax{LMAX}_{tag}.npz"


def _premask(m: np.ndarray, wapo: np.ndarray) -> np.ndarray:
    if hp.get_nside(m) != ps.NSIDE:
        m = hp.ud_grade(m, ps.NSIDE)
    return m * wapo


def _load_alms(name: str, mask: np.ndarray, tag: str) -> dict[str, tuple[np.ndarray, ...]]:
    path = _cache(name, tag)
    if path.is_file():
        z = np.load(path)
        print("loaded", path, flush=True)
        return {
            "cmb": (z["cmb"],),
            "tsz": tuple(z[f"tsz_{i}"] for i in range(len(FREQS))),
            "cib": tuple(z[f"cib_{i}"] for i in range(len(FREQS))),
            "n1": tuple(z[f"n1_{i}"] for i in range(len(FREQS))),
            "n2": tuple(z[f"n2_{i}"] for i in range(len(FREQS))),
        }
    print(f"map2alm budget alms {name} {tag} lmax={LMAX} ...", flush=True)
    cmb = hp.map2alm(_premask(diag.load_uk_to_k(cmb_path(name)), mask), lmax=LMAX, iter=0)
    tsz, cib, n1, n2 = [], [], [], []
    for f in FREQS:
        tsz.append(
            hp.map2alm(
                _premask(
                    diag.load_uk_to_k(tsz_dir(name) / f"tSZ_deltaT_{f}GHz_nside4096.fits"),
                    mask,
                ),
                lmax=LMAX,
                iter=0,
            )
        )
        cib.append(
            hp.map2alm(
                _premask(
                    diag.load_uk_to_k(cib_dir(name) / f"CIB_deltaT_{f}GHz_nside4096.fits"),
                    mask,
                ),
                lmax=LMAX,
                iter=0,
            )
        )
        n1.append(
            hp.map2alm(
                _premask(
                    diag.load_uk_to_k(NOISE / f"{f}GHz" / f"white_noise_{f}GHz_nside2048_uK.fits"),
                    mask,
                ),
                lmax=LMAX,
                iter=0,
            )
        )
        n2.append(
            hp.map2alm(
                _premask(
                    diag.load_uk_to_k(
                        NOISE / f"{f}GHz" / f"white_noise_{f}GHz_nside2048_uK_r2.fits"
                    ),
                    mask,
                ),
                lmax=LMAX,
                iter=0,
            )
        )
        print(f"  {f} GHz", flush=True)
    payload = {"cmb": cmb}
    for i in range(len(FREQS)):
        payload[f"tsz_{i}"] = tsz[i]
        payload[f"cib_{i}"] = cib[i]
        payload[f"n1_{i}"] = n1[i]
        payload[f"n2_{i}"] = n2[i]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    print("cached", path, flush=True)
    return {
        "cmb": (cmb,),
        "tsz": tuple(tsz),
        "cib": tuple(cib),
        "n1": tuple(n1),
        "n2": tuple(n2),
    }


def _sig_filt(w_nu: np.ndarray) -> np.ndarray:
    return w_nu * _slice(diag.bl10()) * _slice(diag.taper())


def _noi_filt(w_nu: np.ndarray, a: int) -> np.ndarray:
    bnu = np.maximum(_slice(diag.bnu()[a]), 1e-30)
    return w_nu * _slice(diag.bl10()) / bnu * _slice(diag.taper())


def _one(w, alms, same, filt):
    y = None
    for a in range(len(FREQS)):
        alm = alms[0] if same else alms[a]
        c = hp.almxfl(alm, filt(w[a], a))
        y = c if y is None else y + c
    return y


def _to_map(alm: np.ndarray) -> np.ndarray:
    return hp.alm2map(alm, nside=ps.NSIDE, lmax=LMAX)


def _cl(m1, m2, mask, ws, bins, trans):
    return ps.deconv_per_ell(ps.decoupled_cross(m1, m2, mask, ws, bins), trans)


def _weights(name, deproj, real):
    w = diag.hilc_weights(
        hilc_output_dir(name, masked=True, real=real, deproj=deproj),
        deproj.wtag,
        diag.LMAX,
    )
    return np.asarray(w[:, : LMAX + 1], dtype=np.float64)


def _maps(w1, w2, alms):
    sig = lambda w, a: _sig_filt(w)
    noi = lambda w, a: _noi_filt(w, a)
    t1 = _one(w1, alms["tsz"], False, sig)
    t2 = _one(w2, alms["tsz"], False, sig)
    c1 = _one(w1, alms["cib"], False, sig)
    c2 = _one(w2, alms["cib"], False, sig)
    b1 = _one(w1, alms["cmb"], True, sig)
    b2 = _one(w2, alms["cmb"], True, sig)
    n1 = _one(w1, alms["n1"], False, noi)
    n2 = _one(w2, alms["n2"], False, noi)
    out = {
        "tsz": (_to_map(t1), _to_map(t2)),
        "cib": (_to_map(c1), _to_map(c2)),
        "cmb": (_to_map(b1), _to_map(b2)),
        "n": (_to_map(n1), _to_map(n2)),
    }
    del t1, t2, c1, c2, b1, b2, n1, n2
    return out


def _dls(maps, mask, ws, bins, trans, ell):
    cls = {}
    for k, (a, b) in maps.items():
        cls[k] = _cl(a, b, mask, ws, bins, trans)
    pairs = (("tsz", "cib"), ("tsz", "cmb"), ("cib", "cmb"), ("tsz", "n"), ("cib", "n"), ("cmb", "n"))
    for a, b in pairs:
        cls[f"{a}_{b}"] = _cl(maps[a][0], maps[b][1], mask, ws, bins, trans) + _cl(
            maps[b][0], maps[a][1], mask, ws, bins, trans
        )
    rec1 = maps["tsz"][0] + maps["cib"][0] + maps["cmb"][0] + maps["n"][0]
    rec2 = maps["tsz"][1] + maps["cib"][1] + maps["cmb"][1] + maps["n"][1]
    cls["recon"] = _cl(rec1, rec2, mask, ws, bins, trans)
    cls["n11"] = _cl(maps["n"][0], maps["n"][0], mask, ws, bins, trans)
    cls["n22"] = _cl(maps["n"][1], maps["n"][1], mask, ws, bins, trans)
    dls = {k: ps.bin_dl_18(ell, v) for k, v in cls.items()}
    dls["sum"] = (
        dls["tsz"]
        + dls["cib"]
        + dls["cmb"]
        + dls["n"]
        + dls["tsz_cib"]
        + dls["tsz_cmb"]
        + dls["cib_cmb"]
        + dls["tsz_n"]
        + dls["cib_n"]
        + dls["cmb_n"]
    )
    return dls


def _meas(name, deproj, ell, fsky):
    split = OUT / f"{name}_{deproj.key}_masked_splitcross.npz"
    p18 = OUT / f"{name}_{deproj.key}_masked_p18.npz"
    if split.is_file():
        z = np.load(split)
        dl = ps.bin_dl_18(ell, np.asarray(z["cl_12"]))
        sig_g = sigma_dl_cross_binned(
            np.asarray(z["cl_11"]),
            np.asarray(z["cl_22"]),
            np.asarray(z["cl_12"]),
            ELL_MIN,
            ELL_MAX_EXCL,
            float(z["fsky_eff"]),
        )
        return dl, sig_g
    z = np.load(p18)
    return np.asarray(z["dl"], dtype=np.float64), np.full(len(ps.ELL_EFF), np.nan)


def _save_budget(path, name, deproj, dls, dl_meas, sig_g, fsky):
    np.savez_compressed(
        path,
        ell=ps.ELL_EFF,
        dl_meas=dl_meas,
        dl_sum=dls["sum"],
        dl_recon=dls["recon"],
        dl_tsz=dls["tsz"],
        dl_cib=dls["cib"],
        dl_cmb=dls["cmb"],
        dl_n=dls["n"],
        dl_n11=dls["n11"],
        dl_n22=dls["n22"],
        dl_tsz_cib=dls["tsz_cib"],
        dl_tsz_cmb=dls["tsz_cmb"],
        dl_cib_cmb=dls["cib_cmb"],
        dl_tsz_n=dls["tsz_n"],
        dl_cib_n=dls["cib_n"],
        dl_cmb_n=dls["cmb_n"],
        dl_n_sigma=sig_g,
        fsky_eff=fsky,
        prescription=name,
        deproj=deproj.key,
    )


def main() -> None:
    trans = ps.transfer(ps.NSIDE, LMAX)
    ell = np.arange(LMAX + 1, dtype=np.float64)
    work = {}
    for name in RUNS:
        mask = np.asarray(hp.read_map(str(cluster_mask_apo(name)), dtype=np.float64))
        print(f"coupling masked {name} ...", flush=True)
        work[name] = (mask, *ps.coupling(mask), float(np.mean(mask**2)))

    alms = {name: _load_alms(name, work[name][0], "q5apo") for name in RUNS}
    for name in RUNS:
        mask, ws, bins, fsky = work[name]
        for deproj in ALL_DEPROJ:
            out = OUT / f"{name}_{deproj.key}_masked_budget.npz"
            if out.is_file():
                print("skip", out, flush=True)
                continue
            print(f"budget {name} {deproj.key} ...", flush=True)
            w1, w2 = _weights(name, deproj, 1), _weights(name, deproj, 2)
            maps = _maps(w1, w2, alms[name])
            dls = _dls(maps, mask, ws, bins, trans, ell)
            del maps
            dl_meas, sig_g = _meas(name, deproj, ell, fsky)
            rel = np.abs(dls["sum"] - dl_meas) / np.maximum(np.abs(dl_meas), 1e-30)
            print(
                f"  |sum-meas|/meas  ell10={rel[0]:.3f} ell40={rel[5]:.3f} "
                f"ell335={rel[13]:.3f}  D_n/meas ell40={dls['n'][5] / dl_meas[5]:.3f}",
                flush=True,
            )
            _save_budget(out, name, deproj, dls, dl_meas, sig_g, fsky)
            print("wrote", out, flush=True)

    # Exp 2: CIB+δβ weight swap, fiducial q>5 mask.
    fx = OUT / "fixedw_deproj_cib_dbeta_masked.npz"
    if fx.is_file():
        print("skip", fx, flush=True)
        return
    print("fixed-weight CIB+dbeta swap ...", flush=True)
    mask, ws, bins, fsky = work["L1_m9"]
    deproj = DEPROJ_CIB_DBETA
    alms_fid = _load_alms("Mstar-1sigma", mask, "fidmask")
    w_fid = (_weights("L1_m9", deproj, 1), _weights("L1_m9", deproj, 2))
    w_ms = (_weights("Mstar-1sigma", deproj, 1), _weights("Mstar-1sigma", deproj, 2))
    cases = {
        "fid_fidw": _dls(_maps(*w_fid, alms["L1_m9"]), mask, ws, bins, trans, ell),
        "mstar_fidw": _dls(_maps(*w_fid, alms_fid), mask, ws, bins, trans, ell),
        "fid_mstarw": _dls(_maps(*w_ms, alms["L1_m9"]), mask, ws, bins, trans, ell),
        "mstar_mstarw": _dls(_maps(*w_ms, alms_fid), mask, ws, bins, trans, ell),
    }
    payload = {"ell": ps.ELL_EFF, "fsky_eff": fsky}
    for key, dls in cases.items():
        payload[f"{key}_dl"] = dls["recon"]
        payload[f"{key}_dl_sum"] = dls["sum"]
    np.savez_compressed(fx, **payload)
    print("wrote", fx, flush=True)
    r_own = cases["mstar_mstarw"]["recon"] / np.maximum(cases["fid_fidw"]["recon"], 1e-30)
    r_fix = cases["mstar_fidw"]["recon"] / np.maximum(cases["fid_fidw"]["recon"], 1e-30)
    print(
        f"  own-w ratio ell10={r_own[0]:.3f}  fid-w ratio ell10={r_fix[0]:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
