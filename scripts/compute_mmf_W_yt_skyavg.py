#!/usr/bin/env python3
"""Sky-averaged MMF w(x; theta_500) from W = IFFT(N^{-1} y_t)/𝒩.

25-point log theta_500 grid [0.5', 32'] matching sigma_y0.
Output: .../mmf_aperture/W_theta_yt_skyavg_25pt.npz

Optional: --count-q runs L1_m9 q>5 count (no catalogue written).
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("SZIFI_ARRAY_BACKEND", "numpy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.ndimage import map_coordinates

from flamingo_mock.szifi.paths import TILE_L_DEG, TILE_NX, SZiFiPaths
from flamingo_mock.szifi.run import default_params
from flamingo_mock.szifi.tiles import select_footprint_tile_ids

THETA_MIN = 0.5
THETA_MAX = 32.0
N_THETA = 25
N_R = 201
N_PHI = 180
N_TILE_WORKERS = 8
N_PHOT_WORKERS = 16
X_MAX = 5.0

INTER_DIR = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues/"
    "sigma_per_tile_flamingo_immf_it_splitA/intermediates"
)
SKYFR = Path("/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy")
OUT_NPZ = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues/"
    "mmf_aperture/W_theta_yt_skyavg_25pt.npz"
)
CACHE = OUT_NPZ.parent / "W_yt_radial_cache_25pt"
CAT = Path(
    "/rds/rds-lxu/flamingo/L1_m9/catalogues/"
    "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommz.csv"
)
YMAP = Path("/rds/rds-lxu/flamingo/L1_m9/maps/y_unlensed_L1_m9_lc0_nside4096.fits")
NOISE = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues/"
    "noise_curve_skyavg_flamingo_immf.npz"
)
ARCMIN_PER_RAD = 180.0 * 60.0 / np.pi
DX = TILE_L_DEG / TILE_NX / 180.0 * np.pi
TE = np.linspace(0.0, 5.0, N_R)

_WPayload = None
_YMAP = _NSIDE = _WGRID = _LOGTH = _TLUT = None
_EXP = _LRANGE = _BEAM = _PROFILE_TYPE = None


def theta_grid() -> np.ndarray:
    return np.exp(np.linspace(np.log(THETA_MIN), np.log(THETA_MAX), N_THETA))


def polar_ring(w2d: np.ndarray, r_arcmin: np.ndarray) -> np.ndarray:
    nx = w2d.shape[0]
    cx = cy = nx // 2
    r_pix = (r_arcmin / ARCMIN_PER_RAD) / DX
    phi = np.linspace(0.0, 2.0 * np.pi, N_PHI, endpoint=False)
    cphi, sphi = np.cos(phi), np.sin(phi)
    out = np.empty(r_arcmin.size, dtype=np.float64)
    out[0] = float(w2d[cx, cy])
    for i in range(1, r_arcmin.size):
        col = r_pix[i] * cphi + cy
        row = r_pix[i] * sphi + cx
        out[i] = float(
            map_coordinates(w2d, np.vstack([row, col]), order=3, mode="nearest").mean()
        )
    return out


def _init_tile_worker(payload: dict) -> None:
    global _WPayload, _EXP, _LRANGE, _BEAM, _PROFILE_TYPE
    _WPayload = payload
    import szifi

    paths = SZiFiPaths()
    params_szifi, params_data, params_model = default_params(
        paths, [int(payload["field_id_ref"])], split="A"
    )
    _LRANGE = params_szifi["lrange"]
    _BEAM = params_szifi["beam"]
    _PROFILE_TYPE = params_model["profile_type"]
    _EXP = szifi.input_data(params_szifi=params_szifi, params_data=params_data).data["experiment"]


def _one_tile(field_id: int) -> tuple[int, np.ndarray] | None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["SZIFI_ARRAY_BACKEND"] = "numpy"
    from szifi import maps, model, utils

    ckpt = CACHE / f"field_{field_id}.npy"
    if ckpt.is_file() and not _WPayload.get("overwrite", False):
        return field_id, np.load(ckpt)

    src = INTER_DIR / f"field_{field_id}.npz"
    if not src.is_file():
        return None
    inv_cov = np.asarray(np.load(src)["inv_cov"], dtype=np.complex128)
    theta_vec = np.asarray(_WPayload["theta_vec"], dtype=np.float64)
    a_sz = np.asarray(_WPayload["a_sz"], dtype=np.float64)
    cosmology = _WPayload["cosmology"]
    z = 0.2
    pix = maps.pixel(TILE_NX, DX)

    profiles = np.empty((theta_vec.size, N_R), dtype=np.float64)
    for j, th in enumerate(theta_vec):
        m500 = model.get_m_500(float(th), z, cosmology)
        nfw = model.gnfw(m500, z, cosmology, type=_PROFILE_TYPE)
        theta_cart = [(0.5 * pix.nx) * pix.dx, (0.5 * pix.nx) * pix.dx]
        t_tem = nfw.get_t_map_convolved(
            pix, _EXP, beam=_BEAM, theta_cart=theta_cart, get_nc=False, sed=False,
        )
        t_tem = t_tem / nfw.get_y_norm("centre")
        tem = maps.filter_tmap(t_tem, pix, _LRANGE)
        yt_fft = maps.get_tmap_times_fvec(maps.get_fft_f(tem, pix), a_sz)
        yt = maps.reshape_ell_matrix(yt_fft, inv_cov.shape[:2])
        psi_ell = utils.get_inv_cov_conjugate(yt, inv_cov)
        psi = np.asarray(maps.get_ifft_f(psi_ell, pix).real)
        yt_r = np.asarray(maps.get_ifft_f(yt, pix).real)
        norm = float(np.sum(yt_r * psi) * pix.dx * pix.dy)
        wy = np.tensordot(psi, a_sz, axes=([2], [0])) / norm
        wr = polar_ring(wy, TE * float(th))
        w0 = wr[0]
        profiles[j] = wr / w0 if w0 != 0.0 else wr

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    np.save(ckpt, profiles)
    return field_id, profiles


def build_w_table(overwrite: bool = False) -> None:
    import szifi
    from szifi import model

    theta_vec = theta_grid()
    paths = SZiFiPaths()
    params_szifi, params_data, _ = default_params(paths, [0], split="A")
    data = szifi.input_data(params_szifi=params_szifi, params_data=params_data)
    exp = data.data["experiment"]
    if params_szifi["a_matrix"] is None:
        a_matrix = np.zeros((len(exp.tsz_sed), 1))
        a_matrix[:, 0] = exp.tsz_sed
        params_szifi["a_matrix"] = a_matrix
    a_sz = np.asarray(params_szifi["a_matrix"][:, 0], dtype=np.float64)
    cosmology = model.cosmological_model(params_szifi).cosmology

    footprint = set(select_footprint_tile_ids(paths.masks_fits, min_ftile=0.3))
    have = sorted(
        int(p.stem.split("_")[1])
        for p in INTER_DIR.glob("field_*.npz")
        if int(p.stem.split("_")[1]) in footprint
    )
    skyfr = np.load(SKYFR).ravel()
    print(f"theta_500: {N_THETA} pts [{theta_vec[0]:g}, {theta_vec[-1]:g}]'  tiles={len(have)}")

    if overwrite and CACHE.is_dir():
        for p in CACHE.glob("field_*.npy"):
            p.unlink()

    payload = {
        "theta_vec": theta_vec,
        "a_sz": a_sz,
        "cosmology": cosmology,
        "field_id_ref": 0,
        "overwrite": overwrite,
    }

    profiles: dict[int, np.ndarray] = {}
    with get_context("fork").Pool(
        N_TILE_WORKERS, initializer=_init_tile_worker, initargs=(payload,)
    ) as pool:
        n = 0
        for out in pool.imap_unordered(_one_tile, have, chunksize=1):
            if out is None:
                continue
            fid, prof = out
            profiles[fid] = prof
            n += 1
            if n % 20 == 0 or n == len(have):
                print(f"  tiles {n}/{len(have)}", flush=True)

    num = np.zeros((N_THETA, N_R), dtype=np.float64)
    den = 0.0
    for fid, prof in profiles.items():
        w = float(skyfr[int(fid)]) if int(fid) < skyfr.size else 0.0
        if w <= 0:
            continue
        num += w * prof
        den += w
    w_sky = num / max(den, 1e-300)
    print(f"skyfrac weight={den:.4f}")
    for j in (0, N_THETA // 2, N_THETA - 1):
        print(f"  θ={theta_vec[j]:.3g}'  w(1)/w(0)={np.interp(1.0, TE, w_sky[j]):.4f}")

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        theta_500_arcmin=theta_vec,
        theta_eff=TE,
        w_skyavg=w_sky,
        n_tiles=len(profiles),
        skyfrac_weight=den,
    )
    print("Wrote", OUT_NPZ)


def _gnfw_t_profile() -> np.ndarray:
    import szifi
    from szifi import model

    paths = SZiFiPaths()
    params_szifi, params_data, params_model = default_params(paths, [0], split="A")
    szifi.input_data(params_szifi=params_szifi, params_data=params_data)
    cosmology = model.cosmological_model(params_szifi).cosmology
    nfw = model.gnfw(
        model.get_m_500(5.0, 0.2, cosmology), 0.2, cosmology, type=params_model["profile_type"]
    )
    y0 = nfw.get_y_norm("centre")
    th5 = 5.0 / ARCMIN_PER_RAD
    t = np.empty(TE.size, dtype=np.float64)
    t[0] = 1.0
    for i in range(1, TE.size):
        t[i] = nfw.get_y_at_angle(float(TE[i] * th5)) / y0
    return t


def _build_t_lut(log_th: np.ndarray) -> np.ndarray:
    """t(x; theta_500) on the same 25-point grid."""
    import szifi
    from szifi import model

    paths = SZiFiPaths()
    params_szifi, params_data, params_model = default_params(paths, [0], split="A")
    szifi.input_data(params_szifi=params_szifi, params_data=params_data)
    cosmology = model.cosmological_model(params_szifi).cosmology
    z = 0.2
    t_lut = np.empty((N_THETA, N_R), dtype=np.float64)
    for i, th in enumerate(np.exp(log_th)):
        nfw = model.gnfw(
            model.get_m_500(float(th), z, cosmology), z, cosmology, type=params_model["profile_type"]
        )
        y0 = nfw.get_y_norm("centre")
        th_rad = float(th) / ARCMIN_PER_RAD
        t_lut[i, 0] = 1.0
        for k in range(1, N_R):
            t_lut[i, k] = nfw.get_y_at_angle(float(TE[k] * th_rad)) / y0
    return t_lut


def _init_phot(ymap, nside, wsky, logth, t_lut):
    global _YMAP, _NSIDE, _WGRID, _LOGTH, _TLUT
    _YMAP, _NSIDE, _WGRID, _LOGTH, _TLUT = ymap, nside, wsky, logth, t_lut


def _interp_row(table: np.ndarray, theta500_arcmin: float) -> np.ndarray:
    x = np.log(theta500_arcmin)
    return np.array([np.interp(x, _LOGTH, table[:, j]) for j in range(N_R)])


def _y0_hat(th_rad: float, ph_rad: float, theta500_arcmin: float) -> float:
    import healpy as hp

    w_shape = _interp_row(_WGRID, theta500_arcmin)
    t_prof = _interp_row(_TLUT, theta500_arcmin)
    th_c_rad = theta500_arcmin / ARCMIN_PER_RAD
    iw = 2.0 * np.pi * th_c_rad**2 * np.trapezoid(TE * t_prof * w_shape, TE)
    w = w_shape / max(iw, 1e-30)
    r_max = (X_MAX * theta500_arcmin) / ARCMIN_PER_RAD
    vec = hp.ang2vec(th_rad, ph_rad)
    pix = hp.query_disc(_NSIDE, vec, r_max)
    if pix.size == 0:
        return 0.0
    x = hp.rotator.angdist(vec, hp.pix2vec(_NSIDE, pix)) * ARCMIN_PER_RAD / theta500_arcmin
    wf = np.interp(x, TE, w, left=w[0], right=0.0)
    return float(hp.nside2pixarea(_NSIDE) * np.dot(_YMAP[pix].astype(np.float64, copy=False), wf))


def _chunk(payload):
    th, ph, th500 = payload
    out = np.empty(th.size, dtype=np.float64)
    for i in range(th.size):
        out[i] = _y0_hat(float(th[i]), float(ph[i]), float(th500[i]))
    return out


def count_q_gt5() -> None:
    import sys
    import healpy as hp
    import pandas as pd

    sys.path.insert(0, "/scratch/scratch-lxu/flamingo_repo/src")
    from flamingo.catalogue import theta_500 as theta_500_fn

    wdata = np.load(OUT_NPZ)
    w_sky = np.asarray(wdata["w_skyavg"], dtype=np.float64)
    th_tab = np.asarray(wdata["theta_500_arcmin"], dtype=np.float64)
    log_th = np.log(th_tab)
    t_lut = _build_t_lut(log_th)

    frame = pd.read_csv(
        CAT, comment="#", usecols=["theta_rot_rad", "phi_rot_rad", "R_500c_Mpc", "z"]
    )
    n = len(frame)
    th500 = (
        np.rad2deg(theta_500_fn(frame["R_500c_Mpc"].to_numpy(np.float64), frame["z"].to_numpy(np.float64)))
        * 60.0
    )
    th = frame["theta_rot_rad"].to_numpy(np.float64)
    ph = frame["phi_rot_rad"].to_numpy(np.float64)
    print(f"N={n:,}  theta_500 [{th500.min():.3g}, {th500.max():.3g}]'", flush=True)

    ymap = np.asarray(hp.read_map(str(YMAP), dtype=np.float32), dtype=np.float32)
    nside = hp.npix2nside(ymap.size)
    chunk = max(1, n // (N_PHOT_WORKERS * 16))
    slices = [(th[i : i + chunk], ph[i : i + chunk], th500[i : i + chunk]) for i in range(0, n, chunk)]
    y_ap = np.empty(n, dtype=np.float64)
    done = 0
    with get_context("fork").Pool(
        N_PHOT_WORKERS, initializer=_init_phot, initargs=(ymap, nside, w_sky, log_th, t_lut)
    ) as pool:
        for part in pool.imap(_chunk, slices, chunksize=1):
            y_ap[done : done + part.size] = part
            done += part.size
            if done % 200000 < chunk or done == n:
                print(f"  {done:,}/{n:,}", flush=True)

    noise = np.load(NOISE)
    coeff = np.polyfit(np.log(noise["theta_500_arcmin"]), np.log(noise["sigma_y0_flamingo_mock"]), 3)
    sigma_y0 = np.exp(np.polyval(coeff, np.log(th500)))
    q = y_ap / sigma_y0
    ok = np.isfinite(q)
    n_out = int(np.sum((th500 < th_tab[0]) | (th500 > th_tab[-1])))
    print(f"theta_500 outside [{th_tab[0]:g},{th_tab[-1]:g}]' (linear extrap): {n_out:,}")
    print(f"N(q>5)={int(np.sum(ok & (q > 5))):,}")
    print(f"median Y_ap={np.median(y_ap[ok]):.4g}  median q={np.median(q[ok]):.4g}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--overwrite", action="store_true", help="rebuild tile cache + table")
    p.add_argument("--count-q", action="store_true", help="only count q>5")
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args()
    if args.count_q:
        count_q_gt5()
        return
    build_w_table(overwrite=args.overwrite)
    if not args.build_only:
        count_q_gt5()


if __name__ == "__main__":
    main()
