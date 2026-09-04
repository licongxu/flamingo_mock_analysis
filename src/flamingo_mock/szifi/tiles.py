"""Flat-sky tile cutouts and PR4 mask projection for SZiFi."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .paths import (
    FREQS_GHZ,
    TILE_L_DEG,
    TILE_NSIDE,
    TILE_NX,
    SZiFiPaths,
)
from .skies import load_total_maps_uK, stack_maps_nxnxnf


def select_pilot_tile_ids(
    n: int = 8,
    b_min_deg: float = 40.0,
    nside_tile: int = TILE_NSIDE,
    rng_seed: int = 0,
) -> list[int]:
    """Pick ``n`` tile centres with |b| >= b_min_deg (Galactic)."""
    import healpy as hp

    npix = hp.nside2npix(nside_tile)
    lon, lat = hp.pix2ang(nside_tile, np.arange(npix), lonlat=True)
    candidates = np.where(np.abs(lat) >= b_min_deg)[0]
    if len(candidates) < n:
        raise ValueError(
            f"Only {len(candidates)} tiles with |b|>={b_min_deg}; need {n}"
        )
    # Prefer highest |b|, break ties stably.
    order = np.argsort(-np.abs(lat[candidates]))
    chosen = candidates[order[:n]]
    return [int(i) for i in chosen]


def select_all_tile_ids(nside_tile: int = TILE_NSIDE) -> list[int]:
    """All \(N_\mathrm{side}=8\) tiles (768), no Galactic / PS footprint cut."""
    import healpy as hp

    return list(range(int(hp.nside2npix(nside_tile))))


def unmasked_mask_stack(field_id: int, nside_map: int) -> np.ndarray:
    """``(mask_galaxy, mask_point, mask_tile)`` with GAL=PS=1."""
    import healpy as hp
    from szifi.sphere import get_cutout

    lon, lat = hp.pix2ang(TILE_NSIDE, field_id, lonlat=True)
    tile_hp = _tile_membership_map(field_id, nside_map, TILE_NSIDE)
    mask_tile = np.asarray(
        get_cutout(tile_hp, [lon, lat], TILE_NX, TILE_L_DEG), dtype=np.float64
    )
    mask_tile = (mask_tile > 0.5).astype(np.float64)
    ones = np.ones((TILE_NX, TILE_NX), dtype=np.float64)
    return np.stack([ones, ones, mask_tile], axis=0)


def select_footprint_tile_ids(
    masks_fits: Path,
    *,
    min_ftile: float = 0.3,
    nside_map: int = 2048,
    nside_tile: int = TILE_NSIDE,
) -> list[int]:
    """Tiles whose GAL×PS unmasked fraction is >= ``min_ftile`` (Planck footprint)."""
    import healpy as hp

    gal, ps = load_pr4_gal_ps(masks_fits, nside=nside_map)
    keep = (gal * ps).astype(np.float64)
    # Mean of child pixels = unmasked sky fraction inside each coarse tile.
    frac = hp.ud_grade(keep, nside_tile)
    ids = np.where(frac >= min_ftile)[0]
    return [int(i) for i in ids]


def load_pr4_gal_ps(masks_fits: Path, nside: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Load PR4 GAL (field 1) and PS (field 2); return binary float maps."""
    import healpy as hp

    gal = np.asarray(hp.read_map(str(masks_fits), field=1, dtype=np.float64))
    ps = np.asarray(hp.read_map(str(masks_fits), field=2, dtype=np.float64))
    if hp.npix2nside(gal.size) != nside:
        gal = hp.ud_grade(gal, nside)
        ps = hp.ud_grade(ps, nside)
    # Soft GAL edges → binary for SZiFi peak-finding; PS is already binary.
    gal_bin = (gal > 0.5).astype(np.float64)
    ps_bin = (ps > 0.5).astype(np.float64)
    return gal_bin, ps_bin


def _tile_membership_map(field_id: int, nside_map: int, nside_tile: int = TILE_NSIDE) -> np.ndarray:
    """HEALPix map = 1 inside the nside_tile pixel ``field_id``."""
    import healpy as hp

    m = np.zeros(hp.nside2npix(nside_tile), dtype=np.float64)
    m[field_id] = 1.0
    return hp.ud_grade(m, nside_map)


def cutout_stack(
    maps_list: list[np.ndarray],
    field_id: int,
    nx: int = TILE_NX,
    l_deg: float = TILE_L_DEG,
    nside_tile: int = TILE_NSIDE,
) -> np.ndarray:
    """Stack frequency cutouts into (nx, nx, n_freq) float32."""
    import healpy as hp
    from szifi.sphere import get_cutout

    lon, lat = hp.pix2ang(nside_tile, field_id, lonlat=True)
    stack = np.zeros((nx, nx, len(maps_list)), dtype=np.float32)
    for i, m in enumerate(maps_list):
        cut = np.asarray(get_cutout(m, [lon, lat], nx, l_deg), dtype=np.float64)
        cut = np.nan_to_num(cut, nan=0.0, posinf=0.0, neginf=0.0)
        stack[:, :, i] = cut.astype(np.float32)
    return stack


def cutout_mask_triplet(
    gal: np.ndarray,
    ps: np.ndarray,
    field_id: int,
    nside_map: int,
    nx: int = TILE_NX,
    l_deg: float = TILE_L_DEG,
    nside_tile: int = TILE_NSIDE,
) -> np.ndarray:
    """Return (mask_galaxy, mask_point, mask_tile) each (nx, nx)."""
    import healpy as hp
    from szifi.sphere import get_cutout

    lon, lat = hp.pix2ang(nside_tile, field_id, lonlat=True)
    tile_hp = _tile_membership_map(field_id, nside_map, nside_tile)
    mask_galaxy = np.asarray(get_cutout(gal, [lon, lat], nx, l_deg), dtype=np.float64)
    mask_point = np.asarray(get_cutout(ps, [lon, lat], nx, l_deg), dtype=np.float64)
    mask_tile = np.asarray(get_cutout(tile_hp, [lon, lat], nx, l_deg), dtype=np.float64)
    mask_galaxy = (mask_galaxy > 0.5).astype(np.float64)
    mask_point = (mask_point > 0.5).astype(np.float64)
    mask_tile = (mask_tile > 0.5).astype(np.float64)
    return np.stack([mask_galaxy, mask_point, mask_tile], axis=0)


def prepare_tile(
    paths: SZiFiPaths,
    field_id: int,
    maps_uK: dict[int, np.ndarray],
    gal: np.ndarray,
    ps: np.ndarray,
    split: str = "A",
    overwrite: bool = False,
    *,
    write_mask: bool = True,
    unmasked: bool = False,
) -> tuple[Path, Path]:
    """Write Planck-like tmap.npy and mask.npy for one tile."""
    paths.make_dirs(split)
    tmap_path = paths.tmap_path(split, field_id)
    mask_path = paths.mask_path(split, field_id)
    tmap_ok = tmap_path.exists() and not overwrite
    mask_ok = mask_path.exists() and not overwrite and not unmasked
    if tmap_ok and mask_ok:
        return tmap_path, mask_path

    if not tmap_ok:
        maps_list = stack_maps_nxnxnf(maps_uK, FREQS_GHZ)
        tmap = cutout_stack(maps_list, field_id)
        # Leading singleton so `[tmap] = np.load(...)` matches data_planck.
        np.save(tmap_path, tmap[np.newaxis, ...])
    if write_mask and not mask_ok:
        if unmasked:
            np.save(mask_path, unmasked_mask_stack(field_id, paths.nside))
        else:
            mask_galaxy, mask_point, mask_tile = cutout_mask_triplet(
                gal, ps, field_id, nside_map=paths.nside
            )
            np.save(mask_path, np.stack([mask_galaxy, mask_point, mask_tile], axis=0))
    return tmap_path, mask_path


_PREP_STATE = None
_MASK_STATE = None


def _prepare_one_tile_worker(fid: int) -> int:
    """Process-pool worker for ``prepare_tiles`` (uses fork COW ``_PREP_STATE``)."""
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    paths, maps_uK, gal, ps, split, overwrite, unmasked = _PREP_STATE
    prepare_tile(
        paths,
        fid,
        maps_uK,
        gal,
        ps,
        split=split,
        overwrite=overwrite,
        write_mask=not unmasked,
        unmasked=unmasked,
    )
    return int(fid)


def _write_unmasked_mask_worker(fid: int) -> int:
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    paths, split = _MASK_STATE
    np.save(
        paths.mask_path(split, fid),
        unmasked_mask_stack(int(fid), paths.nside),
    )
    return int(fid)


def prepare_tiles(
    paths: SZiFiPaths,
    field_ids: list[int],
    split: str = "A",
    overwrite: bool = False,
    n_workers: int | None = None,
    unmasked: bool = False,
    threads_per_worker: int | None = None,
) -> list[int]:
    """Load skies once; write cutouts for all ``field_ids`` (CPU-parallel, half-machine cap)."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from flamingo_mock.szifi.run import half_machine_pool_limits, _init_worker_threads

    global _PREP_STATE, _MASK_STATE

    if unmasked:
        workers, threads = half_machine_pool_limits(
            n_workers, threads_per_worker=threads_per_worker
        )
        workers = min(workers, max(1, len(field_ids)))
        print(
            f"prepare: writing unmasked (GAL=PS=1) masks for {len(field_ids)} tiles; "
            f"workers={workers}",
            flush=True,
        )
        _MASK_STATE = (paths, split)
        ctx = get_context("fork")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_worker_threads,
            initargs=(threads,),
        ) as pool:
            for i, fid in enumerate(
                pool.map(_write_unmasked_mask_worker, field_ids, chunksize=1), start=1
            ):
                if i % 50 == 0 or i == len(field_ids):
                    print(f"  unmasked mask {i}/{len(field_ids)} (last tile {fid})", flush=True)
        _MASK_STATE = None

    todo = []
    for fid in field_ids:
        tmap_missing = overwrite or not paths.tmap_path(split, fid).exists()
        mask_missing = (not unmasked) and (
            overwrite or not paths.mask_path(split, fid).exists()
        )
        if tmap_missing or mask_missing:
            todo.append(int(fid))
    if not todo:
        print(f"prepare: all {len(field_ids)} tmaps already on disk")
        return list(field_ids)

    print(f"Loading total maps split={split} ...")
    maps_uK = load_total_maps_uK(paths, split=split)
    gal, ps = None, None
    if not unmasked:
        print(f"Loading PR4 GAL/PS masks from {paths.masks_fits} ...")
        gal, ps = load_pr4_gal_ps(paths.masks_fits, nside=paths.nside)
    paths.make_dirs(split)

    workers, threads = half_machine_pool_limits(
        n_workers, threads_per_worker=threads_per_worker
    )
    workers = min(workers, len(todo))
    print(
        f"prepare: {len(todo)} tiles to cut ({len(field_ids) - len(todo)} exist); "
        f"workers={workers}, threads/worker={threads}",
        flush=True,
    )

    _PREP_STATE = (paths, maps_uK, gal, ps, split, overwrite, unmasked)
    ctx = get_context("fork")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker_threads,
        initargs=(threads,),
    ) as pool:
        for i, fid in enumerate(
            pool.map(_prepare_one_tile_worker, todo, chunksize=1), start=1
        ):
            if i % 20 == 0 or i == len(todo):
                print(f"  prepared {i}/{len(todo)} (last tile {fid})", flush=True)
    _PREP_STATE = None
    return list(field_ids)
