#!/usr/bin/env python3
"""Sky-averaged W(theta/theta_500) vs top-hat.

Per tile (cached iterative intermediates): W = IFFT(S/y), S = y_t^H N^{-1} d.
Radial profiles are cubic polar rings, then skyfrac-weighted (same as immf6).
Complex d(ell) is not averaged across tiles.
"""

from __future__ import annotations

import os
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("SZIFI_ARRAY_BACKEND", "numpy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

from flamingo_mock.szifi.paths import TILE_L_DEG, TILE_NSIDE, TILE_NX, SZiFiPaths
from flamingo_mock.szifi.tiles import select_footprint_tile_ids

INTER_DIR = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues/"
    "sigma_per_tile_flamingo_immf_it_splitA/intermediates"
)
YMAP = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc/inputs_nside2048_npipe/"
    "compton_y_nside2048.fits"
)
SKYFR = Path("/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy")
CACHE = INTER_DIR.parent / "W_radial_cache"
THETAS = (2.0, 5.0, 10.0)
N_R = 201
N_PHI = 180
N_WORKERS = 8
TE = np.linspace(0.0, 5.0, N_R)

_YMAP = None
_DX = TILE_L_DEG / TILE_NX / 180.0 * np.pi


def polar_ring(w2d: np.ndarray, r_arcmin: np.ndarray) -> np.ndarray:
    nx = w2d.shape[0]
    cx = nx // 2
    r_pix = (r_arcmin / (180.0 * 60.0 / np.pi)) / _DX
    phi = np.linspace(0.0, 2.0 * np.pi, N_PHI, endpoint=False)
    cphi, sphi = np.cos(phi), np.sin(phi)
    out = np.empty(r_arcmin.size, dtype=np.float64)
    out[0] = float(w2d[cx, cx])
    for i in range(1, r_arcmin.size):
        col = r_pix[i] * cphi + cx
        row = r_pix[i] * sphi + cx
        out[i] = float(
            map_coordinates(w2d, np.vstack([row, col]), order=3, mode="nearest").mean()
        )
    return out


def _init(ymap):
    global _YMAP
    _YMAP = ymap


def _one_tile(field_id: int) -> tuple[int, np.ndarray] | None:
    from szifi import maps, utils
    from szifi.sphere import get_cutout
    import healpy as hp

    cache = CACHE / f"field_{field_id}.npy"
    if cache.is_file():
        return field_id, np.load(cache)

    src = INTER_DIR / f"field_{field_id}.npz"
    if not src.is_file():
        return None
    data = np.load(src)
    inv_cov = np.asarray(data["inv_cov"], dtype=np.complex128)
    d_fft = np.asarray(data["d_fft"], dtype=np.complex128)
    yt = np.asarray(data["yt_fft"], dtype=np.complex128)
    ths = np.asarray(data["theta_yt_arcmin"], dtype=np.float64)

    lon, lat = hp.pix2ang(TILE_NSIDE, int(field_id), lonlat=True)
    y_cut = np.asarray(get_cutout(_YMAP, [lon, lat], TILE_NX, TILE_L_DEG), dtype=np.float64)
    pix = maps.pixel(TILE_NX, _DX)
    y_fft = maps.reshape_ell_matrix(maps.get_fft(y_cut, pix)[..., None], inv_cov.shape[:2])[..., 0]
    ninv_d = utils.get_inv_cov_conjugate(d_fft, inv_cov)

    profiles = np.empty((len(THETAS), N_R), dtype=np.float64)
    for j, th in enumerate(THETAS):
        k = int(np.argmin(np.abs(ths - th)))
        S = np.einsum("ijk,ijk->ij", np.conjugate(yt[k]), ninv_d)
        W_ell = np.divide(S, y_fft, out=np.zeros_like(S), where=np.abs(y_fft) > 1e-12)
        W_ell = np.where(np.isfinite(W_ell), W_ell, 0.0)
        w2d = np.asarray(maps.get_ifft(W_ell, pix).real, dtype=np.float64)
        w_r = polar_ring(w2d, TE * th)
        w0 = w_r[0]
        profiles[j] = w_r / w0 if w0 != 0.0 else w_r
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, profiles)
    return field_id, profiles


def main() -> None:
    import healpy as hp

    paths = SZiFiPaths()
    footprint = set(select_footprint_tile_ids(paths.masks_fits, min_ftile=0.3))
    have = [
        int(p.stem.split("_")[1])
        for p in INTER_DIR.glob("field_*.npz")
        if int(p.stem.split("_")[1]) in footprint
    ]
    skyfr = np.load(SKYFR).ravel()
    print(f"tiles with d,N,y_t: {len(have)}  footprint={len(footprint)}")

    print("reading Compton-y")
    ymap = np.asarray(hp.read_map(str(YMAP), dtype=np.float32), dtype=np.float64)

    profiles = {}
    with get_context("fork").Pool(N_WORKERS, initializer=_init, initargs=(ymap,)) as pool:
        n = 0
        for out in pool.imap_unordered(_one_tile, have, chunksize=1):
            if out is None:
                continue
            fid, prof = out
            profiles[fid] = prof
            n += 1
            if n % 20 == 0 or n == len(have):
                print(f"  {n}/{len(have)}", flush=True)

    num = np.zeros((len(THETAS), N_R), dtype=np.float64)
    den = 0.0
    for fid, prof in profiles.items():
        w = float(skyfr[int(fid)]) if int(fid) < skyfr.size else 0.0
        if w <= 0:
            continue
        num += w * prof
        den += w
    mean = num / max(den, 1e-300)
    print(f"skyfrac weight sum={den:.4f}  (full skyfracs sum={skyfr.sum():.4f})")
    for j, th in enumerate(THETAS):
        print(f"  skyavg theta_500={th:.0f}'  W(1)/W(0)={np.interp(1.0, TE, mean[j]):.4f}")

    out_npz = Path(
        "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues/"
        "mmf_aperture/W_theta_skyavg.npz"
    )
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        theta_eff=TE,
        W_skyavg=mean,
        thetas=np.array(THETAS),
        n_tiles=len(profiles),
        skyfrac_weight=den,
    )
    print("Wrote", out_npz)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot([0, 1, 1, 5], [1, 1, 0, 0], color="0.25", ls="--", lw=1.6,
            label=r"top-hat $\theta<\theta_{500}$")
    colours = {2.0: "C0", 5.0: "C1", 10.0: "C3"}
    for j, th in enumerate(THETAS):
        ax.plot(TE, mean[j], color=colours[th], lw=2.0, label=rf"$\theta_{{500}}={th:.0f}'$")
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_xlabel(r"$\theta/\theta_{500}$")
    ax.set_ylabel(r"$\langle W(\theta)/W(0)\rangle$")
    ax.set_title(rf"Sky-avg $W$, $N_{{\rm tile}}={len(profiles)}$")
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.4, 1.15)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig_base = Path("figures/mmf_W_theta_vs_tophat")
    fig.savefig(fig_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(fig_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Wrote", fig_base.with_suffix(".png"))


if __name__ == "__main__":
    main()
