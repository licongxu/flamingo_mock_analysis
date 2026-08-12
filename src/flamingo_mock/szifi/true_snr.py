"""SZiFi fixed-mode true SNR (q-bar_t) at truth halo positions / theta_500."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import healpy as hp
import numpy as np
import szifi
from szifi import maps

from flamingo_mock.szifi.paths import TILE_L_DEG, TILE_NSIDE, TILE_NX, SZiFiPaths
from flamingo_mock.szifi.run import default_params, half_machine_pool_limits
from flamingo_mock.szifi.validate import DEFAULT_TRUTH_CATALOGUE, load_truth_qfrommap


def footprint_tile_ids(paths: SZiFiPaths, split: str = "A") -> list[int]:
    """Tile ids that have prepared tmap cutouts on disk."""
    return sorted(
        int(p.name.split("_")[2])
        for p in paths.tiles_dir(split).glob("flamingo_field_*_tmap.npy")
    )


def load_parent_truth(
    paths: SZiFiPaths,
    *,
    truth_csv: Path = DEFAULT_TRUTH_CATALOGUE,
    split: str = "A",
    z_max: float = 1.0,
    q_ap_min: float = 2.0,
) -> dict[str, np.ndarray]:
    """Footprint truth parent sample for fixed-mode extraction.

    Pre-selects by aperture SNR (``q_ap_min``) only to keep the parent sample
    tractable; the completeness x-axis is the MMF true SNR ``q_true_mmf``.
    """
    truth = load_truth_qfrommap(
        truth_csv, paths, z_max=z_max, q_th_truth=q_ap_min
    )
    sel = truth["q_from_aperture"] >= q_ap_min
    lon = truth["lon_rot_deg"][sel]
    lat = truth["lat_rot_deg"][sel]
    field_id = hp.ang2pix(TILE_NSIDE, lon, lat, lonlat=True).astype(np.int32)
    have = np.asarray(footprint_tile_ids(paths, split), dtype=np.int32)
    in_tiles = np.isin(field_id, have)
    return {
        "lon": lon[in_tiles].astype(np.float64),
        "lat": lat[in_tiles].astype(np.float64),
        "theta_500": truth["theta_500_arcmin"][sel][in_tiles].astype(np.float64),
        "z": truth["z"][sel][in_tiles].astype(np.float64),
        "M_500c_Msun": truth["M_500c_Msun"][sel][in_tiles].astype(np.float64),
        "q_from_aperture": truth["q_from_aperture"][sel][in_tiles].astype(np.float64),
        "field_id": field_id[in_tiles],
    }


def _make_input_catalogue(
    lon: np.ndarray,
    lat: np.ndarray,
    theta_500: np.ndarray,
    z: np.ndarray,
    m500: np.ndarray,
    field_id: int,
) -> szifi.cat.cluster_catalogue:
    """Build a fixed-mode input catalogue in tile flat-sky coordinates.

    Halos are already assigned by ``ang2pix`` to ``field_id``; convert lon/lat
    to SZiFi ``theta_x/y`` (radians, origin at tile corner) as in
    ``cluster_catalogue.select_tile(..., type='field')``.
    """
    from szifi import sphere

    cat = szifi.cluster_catalogue()
    lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    theta_500 = np.atleast_1d(np.asarray(theta_500, dtype=np.float64))
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))
    m500 = np.atleast_1d(np.asarray(m500, dtype=np.float64))
    n = len(lon)
    dx = TILE_L_DEG / TILE_NX / 180.0 * np.pi
    pix = maps.pixel(TILE_NX, dx)
    lx = pix.nx * pix.dx
    theta_x, theta_y = sphere.get_xy(int(field_id), lon, lat, TILE_NSIDE)
    # n=1 tiles: get_xy can return scalars — SZiFi needs 1d arrays.
    theta_x = np.atleast_1d(np.asarray(theta_x, dtype=np.float64))
    theta_y = np.atleast_1d(np.asarray(theta_y, dtype=np.float64))
    cat.catalogue["lon"] = lon
    cat.catalogue["lat"] = lat
    cat.catalogue["theta_500"] = theta_500
    cat.catalogue["z"] = z
    cat.catalogue["m_500"] = m500
    cat.catalogue["q_opt"] = np.zeros(n, dtype=np.float64)
    cat.catalogue["y0"] = np.zeros(n, dtype=np.float64)
    cat.catalogue["theta_x"] = theta_x + 0.5 * lx
    cat.catalogue["theta_y"] = theta_y + 0.5 * lx
    return cat


def _tile_checkpoint_path(paths: SZiFiPaths, field_id: int) -> Path:
    return Path(paths.catalogues_dir()) / "true_snr_tiles" / f"field_{int(field_id)}.npz"


def _empty_tile_result(
    lon, lat, th, z, m500, q_ap, fid: int, *, q_true: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    n = len(lon)
    if q_true is None:
        q_true = np.full(n, np.nan, dtype=np.float64)
    return {
        "lon": np.asarray(lon, dtype=np.float64),
        "lat": np.asarray(lat, dtype=np.float64),
        "theta_500": np.asarray(th, dtype=np.float64),
        "z": np.asarray(z, dtype=np.float64),
        "M_500c_Msun": np.asarray(m500, dtype=np.float64),
        "q_from_aperture": np.asarray(q_ap, dtype=np.float64),
        "q_true_mmf": np.asarray(q_true, dtype=np.float64),
        "field_id": np.full(n, int(fid), dtype=np.int32),
    }


def _run_fixed_tile(payload: dict) -> dict[str, np.ndarray]:
    """Worker: extract q_true_mmf for one tile (non-iterative fixed mode)."""
    # Force CPU-only before any array backend init (multi-worker GPU hangs).
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["SZIFI_ARRAY_BACKEND"] = "numpy"
    os.environ.setdefault("MPLBACKEND", "Agg")
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(payload["threads"])

    paths = SZiFiPaths(out_root=payload["out_root"])
    fid = int(payload["field_id"])
    ckpt = _tile_checkpoint_path(paths, fid)
    if ckpt.is_file():
        data = np.load(ckpt)
        return {k: np.asarray(data[k]) for k in data.files}

    lon = np.asarray(payload["lon"])
    lat = np.asarray(payload["lat"])
    th = np.asarray(payload["theta_500"])
    z = np.asarray(payload["z"])
    m500 = np.asarray(payload["M_500c_Msun"])
    q_ap = np.asarray(payload["q_from_aperture"])
    if len(lon) == 0:
        out = _empty_tile_result(lon, lat, th, z, m500, q_ap, fid, q_true=np.zeros(0))
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ckpt, **out)
        return out

    sub = _make_input_catalogue(lon, lat, th, z, m500, fid)

    params_szifi, params_data, params_model = default_params(
        paths, [fid], split=payload["split"]
    )
    params_szifi["extraction_mode"] = "fixed"
    params_szifi["iterative"] = False
    params_szifi["get_lonlat"] = False
    params_szifi["array_backend"] = "numpy"

    data = szifi.input_data(params_szifi=params_szifi, params_data=params_data)
    data.data["catalogue_input"] = {fid: sub}
    # rank!=0 suppresses per-cluster prints
    cf = szifi.cluster_finder(
        params_szifi=params_szifi,
        params_model=params_model,
        data_file=data,
        rank=1,
    )
    import contextlib
    import io

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            cf.find_clusters()
    except KeyError as err:
        # SZiFi bug: fixed + non-iterative still indexes catalogue_find_0.
        if "catalogue_find" not in str(err):
            raise

    # Empty mask_select tiles break before writing catalogue_fixed_0.
    fixed_cat = getattr(cf, "results", None)
    catalogues = getattr(fixed_cat, "catalogues", {}) if fixed_cat is not None else {}
    if "catalogue_fixed_0" not in catalogues:
        print(
            f"  warn field={fid}: no catalogue_fixed_0 "
            f"(likely empty mask_select); q_true=nan",
            flush=True,
        )
        out = _empty_tile_result(lon, lat, th, z, m500, q_ap, fid)
    else:
        fixed = catalogues["catalogue_fixed_0"]
        q_true = np.asarray(fixed.catalogue["q_opt"], dtype=np.float64).ravel()
        if len(q_true) != len(lon):
            raise RuntimeError(
                f"field {fid}: fixed catalogue length {len(q_true)} != input {len(lon)}"
            )
        out = _empty_tile_result(lon, lat, th, z, m500, q_ap, fid, q_true=q_true)

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ckpt, **out)
    return out


def extract_true_snr(
    paths: SZiFiPaths | None = None,
    *,
    truth_csv: Path = DEFAULT_TRUTH_CATALOGUE,
    split: str = "A",
    z_max: float = 1.0,
    q_ap_min: float = 2.0,
    n_workers: int | None = None,
    threads_per_worker: int | None = None,
    out_path: Path | None = None,
) -> dict[str, np.ndarray]:
    """Run SZiFi fixed-mode extraction for the parent truth sample.

    Uses non-iterative MMF (``iterative=False``) at true (lon, lat, theta_500).
    This is the SZiFi ``extraction_mode='fixed'`` true SNR ``q_true_mmf``.
    """
    paths = paths or SZiFiPaths()
    parent = load_parent_truth(
        paths,
        truth_csv=truth_csv,
        split=split,
        z_max=z_max,
        q_ap_min=q_ap_min,
    )
    workers, threads = half_machine_pool_limits(
        n_workers, threads_per_worker=threads_per_worker
    )
    field_ids = np.unique(parent["field_id"])
    print(
        f"true_snr: n_truth={len(parent['lon'])} n_tiles={len(field_ids)} "
        f"workers={workers} threads/worker={threads} q_ap_min={q_ap_min}"
    )

    payloads = []
    for fid in field_ids:
        m = parent["field_id"] == fid
        payloads.append(
            {
                "out_root": str(paths.out_root),
                "split": split,
                "field_id": int(fid),
                "threads": threads,
                "lon": parent["lon"][m],
                "lat": parent["lat"][m],
                "theta_500": parent["theta_500"][m],
                "z": parent["z"][m],
                "M_500c_Msun": parent["M_500c_Msun"][m],
                "q_from_aperture": parent["q_from_aperture"][m],
            }
        )

    chunks: list[dict[str, np.ndarray]] = []
    progress_path = Path(paths.catalogues_dir()) / "true_snr_extract_progress.txt"
    tile_dir = Path(paths.catalogues_dir()) / "true_snr_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    n_cached = sum(1 for pl in payloads if _tile_checkpoint_path(paths, pl["field_id"]).is_file())
    print(f"true_snr: resume checkpoints={n_cached}/{len(payloads)} in {tile_dir}")
    # Parent must not hold GPU before forking workers.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["SZIFI_ARRAY_BACKEND"] = "numpy"

    def _progress(done: int, total: int, fid: int | None = None) -> None:
        extra = f" field={fid}" if fid is not None else ""
        msg = f"  tile {done}/{total}{extra}"
        print(msg, flush=True)
        progress_path.write_text(f"{done} {total}\n")

    if workers == 1:
        for i, pl in enumerate(payloads):
            chunks.append(_run_fixed_tile(pl))
            _progress(i + 1, len(payloads), int(pl["field_id"]))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_fixed_tile, pl): pl["field_id"] for pl in payloads}
            done = 0
            for fut in as_completed(futs):
                fid = futs[fut]
                chunks.append(fut.result())
                done += 1
                _progress(done, len(futs), int(fid))

    keys = [
        "lon",
        "lat",
        "theta_500",
        "z",
        "M_500c_Msun",
        "q_from_aperture",
        "q_true_mmf",
        "field_id",
    ]
    out = {k: np.concatenate([c[k] for c in chunks]) for k in keys}
    # Drop truth where fixed extraction was impossible (e.g. empty mask_select).
    ok = np.isfinite(out["q_true_mmf"])
    n_drop = int((~ok).sum())
    if n_drop:
        print(f"true_snr: dropping {n_drop} truth with non-finite q_true_mmf")
        out = {k: v[ok] for k, v in out.items()}

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **out)
        print(f"Wrote {out_path} (n={len(out['lon'])})")
    return out


def load_true_snr(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: np.asarray(data[k]) for k in data.files}
