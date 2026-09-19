"""Run SZiFi iMMF / sciMMF and write q>5 catalogues."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import szifi

from .paths import SZiFiPaths


def _survey_file_path() -> str:
    return str(Path(__file__).resolve().parent / "survey.py")


def half_machine_pool_limits(
    n_workers: int | None = None,
    *,
    threads_per_worker: int | None = None,
) -> tuple[int, int]:
    """Cap process pool at half the host CPUs (and a modest worker count).

    Returns ``(n_workers, threads_per_worker)`` with
    ``n_workers * threads_per_worker <= nproc // 2``.
    Defaults are conservative so BLAS/healpy overhead stays under half.
    If both ``n_workers`` and ``threads_per_worker`` are set, use up to the
    full half-machine budget (still never above ``nproc // 2``).
    """
    nproc = os.cpu_count() or 4
    half = max(1, nproc // 2)
    if n_workers is not None and threads_per_worker is not None:
        budget = half
        workers = max(1, min(int(n_workers), half))
        threads = max(1, int(threads_per_worker))
    else:
        # Leave headroom inside the half-cap for non-OMP threads / the parent.
        budget = max(1, (half * 3) // 4)  # 75% of half → ~37% of machine
        # Memory-safe default: each SZiFi worker is heavy (~several GB).
        default_workers = min(6, max(1, budget // 10))
        workers = int(n_workers) if n_workers is not None else default_workers
        cap = half if n_workers is not None else min(budget, 8)
        workers = max(1, min(workers, cap))
        if threads_per_worker is None:
            threads = max(1, budget // workers)
        else:
            threads = max(1, int(threads_per_worker))
    while workers * threads > budget and threads > 1:
        threads -= 1
    while workers * threads > budget and workers > 1:
        workers -= 1
    return workers, threads


def _limit_compute_threads(threads: int) -> None:
    """Cap BLAS/OpenMP/XLA threads. Must run before JAX is imported in a process."""
    threads = max(1, int(threads))
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
        "NPROC",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[key] = str(threads)
    os.environ["JAX_NUM_CPU_DEVICES"] = "1"
    flags = os.environ.get("XLA_FLAGS", "")
    extra = "--xla_cpu_multi_thread_eigen=false"
    if extra not in flags:
        os.environ["XLA_FLAGS"] = (flags + " " + extra).strip()
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import jax

        jax.config.update("jax_num_cpu_devices", 1)
    except Exception:
        pass


def _init_worker_threads(threads: int) -> None:
    """Process-pool initializer: pin compute libraries to ``threads`` cores."""
    _limit_compute_threads(threads)


def default_params(
    paths: SZiFiPaths,
    field_ids: list[int],
    split: str = "A",
    *,
    mmf_type: str = "standard",
    deproject_cib: list[str] | None = None,
    compute_coupling_matrix: bool = True,
) -> tuple[dict, dict, dict]:
    """Build params_szifi / params_data / params_model for FLAMINGO mocks."""
    params_szifi = deepcopy(szifi.params_szifi_default)
    params_data = deepcopy(szifi.params_data_default)
    params_model = deepcopy(szifi.params_model_default)

    params_szifi["path"] = str(
        Path(szifi.__file__).resolve().parent.parent
    ) + "/"
    params_szifi["path_data"] = str(paths.out_root) + "/"
    params_szifi["survey_file"] = _survey_file_path()
    params_szifi["flamingo_out_root"] = str(paths.out_root)
    params_szifi["save_and_load_template"] = False
    params_szifi["beam"] = "gaussian"
    params_szifi["integrate_bandpass"] = False
    params_szifi["array_backend"] = os.environ.get("SZIFI_ARRAY_BACKEND", "jax")
    params_szifi["mmf_type"] = mmf_type
    params_szifi["deproject_cib"] = deproject_cib
    # Pilot default: fsky avoids NaMaster coupling-matrix build (very slow on 1024^2).
    params_szifi["decouple_type"] = "fsky"
    params_szifi["compute_coupling_matrix"] = False
    params_szifi["save_coupling_matrix"] = False
    params_szifi["coupling_matrix_needed"] = False
    # SZiFi params.py misspells this as snr_weigthing; mmf.py reads snr_weighting.
    params_szifi["snr_weighting"] = False
    params_szifi["theta_500_vec_arcmin"] = np.exp(
        np.linspace(np.log(0.5), np.log(32.0), 25)
    )

    params_data["data_set"] = "flamingo_mock"
    params_data["field_ids"] = list(field_ids)
    params_data["other_params"] = {
        "components": ["tSZ", "kSZ", "CIB", "CMB", "noise"],
        "npipe_split": split,
    }

    # CIB SED params for sciMMF (Planck-like defaults from SZiFi constrained example).
    params_model["alpha_cib"] = 0.36
    params_model["T0_cib"] = 20.7
    params_model["beta_cib"] = 1.6
    params_model["z_eff_cib"] = 0.2

    return params_szifi, params_data, params_model


def _method_tag(mmf_type: str, method: str | None = None) -> str:
    if method is not None:
        return str(method)
    if mmf_type == "spectrally_constrained":
        return "scimmf"
    return "immf"


def sigma_per_tile_dir(
    paths: SZiFiPaths,
    *,
    method: str = "immf",
    split: str = "A",
) -> Path:
    """Directory for per-tile sigma_y0(theta) arrays written during MMF runs."""
    return paths.catalogues_dir() / f"sigma_per_tile_{method}_split{split}"


def save_per_tile_sigma(
    results_dict: dict,
    theta_500_arcmin,
    cache_dir: Path,
) -> int:
    """Persist iterative and non-iterative per-tile MMF noise curves."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    theta_path = cache_dir / "theta_500_arcmin.npy"
    theta = np.asarray(theta_500_arcmin, dtype=np.float64)
    if theta_path.is_file():
        cached_theta = np.load(theta_path)
        if cached_theta.shape != theta.shape or not np.allclose(cached_theta, theta):
            raise ValueError(f"theta grid does not match existing cache: {theta_path}")
    else:
        np.save(theta_path, theta)

    n_written = 0
    for field_id, res in results_dict.items():
        sigma_vec = getattr(res, "sigma_vec", None) or {}
        fid = int(field_id)
        if "find_0" in sigma_vec:
            np.save(
                cache_dir / f"field_{fid}_noit.npy",
                np.asarray(sigma_vec["find_0"], dtype=np.float64),
            )
        primary = sigma_vec.get("find_1", sigma_vec.get("find_0"))
        if primary is None:
            continue
        np.save(cache_dir / f"field_{fid}.npy", np.asarray(primary, dtype=np.float64))
        n_written += 1
    return n_written


def run_mmf(
    paths: SZiFiPaths,
    field_ids: list[int],
    *,
    split: str = "A",
    mmf_type: str = "standard",
    deproject_cib: list[str] | None = None,
    q_th_final: float = 5.0,
    merge_radius_arcmin: float = 10.0,
    compute_coupling_matrix: bool = True,
    method: str | None = None,
    save_sigma: bool = True,
) -> szifi.cat.cluster_catalogue:
    """Run SZiFi on prepared tiles; return merged catalogue with q >= q_th_final."""
    params_szifi, params_data, params_model = default_params(
        paths,
        field_ids,
        split=split,
        mmf_type=mmf_type,
        deproject_cib=deproject_cib,
        compute_coupling_matrix=compute_coupling_matrix,
    )

    data = szifi.input_data(params_szifi=params_szifi, params_data=params_data)
    cluster_finder = szifi.cluster_finder(
        params_szifi=params_szifi,
        params_model=params_model,
        data_file=data,
        rank=0,
    )
    cluster_finder.find_clusters()

    results = cluster_finder.results_dict
    if save_sigma:
        cache = sigma_per_tile_dir(
            paths, method=_method_tag(mmf_type, method), split=split
        )
        n_sig = save_per_tile_sigma(
            results, params_szifi["theta_500_vec_arcmin"], cache
        )
        print(f"  saved per-tile sigma_y0 for {n_sig} tiles -> {cache}", flush=True)

    detection_processor = szifi.detection_processor(results, params_szifi)

    # Iterative catalogue when iterative=True (catalogue_find_1); else find_0.
    cat_key = (
        "catalogue_find_1"
        if params_szifi.get("iterative", True)
        else "catalogue_find_0"
    )
    if cat_key not in detection_processor.results.catalogues:
        cat_key = "catalogue_find_0"
    catalogue = detection_processor.results.catalogues[cat_key]

    catalogue = szifi.get_catalogue_q_th(catalogue, q_th_final)
    n = len(catalogue.catalogue.get("q_opt", []))
    if n > 1:
        catalogue = szifi.merge_detections(
            catalogue,
            radius_arcmin=merge_radius_arcmin,
            return_merge_flag=False,
            mode="fof",
        )
    return catalogue


def catalogue_to_dict(catalogue) -> dict[str, np.ndarray]:
    """Extract numpy arrays from a SZiFi catalogue object."""
    out = {}
    for key, val in catalogue.catalogue.items():
        if val is None:
            continue
        arr = np.asarray(val)
        if arr.size == 0:
            continue
        out[key] = arr
    return out


def save_catalogue_npz(
    catalogue,
    path: Path,
    meta: dict | None = None,
) -> Path:
    """Save catalogue columns (+ optional meta JSON sidecar)."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = catalogue_to_dict(catalogue)
    np.savez_compressed(path, **cols)
    if meta is not None:
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


def run_imf_and_scimmf(
    paths: SZiFiPaths,
    field_ids: list[int],
    *,
    split: str = "A",
    q_th_final: float = 5.0,
    out_dir: Path | None = None,
    tag: str = "pilot",
    methods: tuple[str, ...] = ("immf", "scimmf"),
) -> dict[str, Path]:
    """Run iMMF and/or sciMMF; write catalogues under out_dir (default pilot/)."""
    out_dir = Path(out_dir) if out_dir is not None else paths.pilot_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    method_specs = {
        "immf": ("standard", None),
        "scimmf": ("spectrally_constrained", ["cib"]),
    }
    for name in methods:
        mmf_type, deproj = method_specs[name]
        print(f"=== Running {name} (mmf_type={mmf_type}) on {len(field_ids)} tiles ===")
        cat = run_mmf(
            paths,
            field_ids,
            split=split,
            mmf_type=mmf_type,
            deproject_cib=deproj,
            q_th_final=q_th_final,
            method=name,
        )
        n = len(cat.catalogue.get("q_opt", []))
        print(f"  {name}: {n} detections with q>={q_th_final}")
        out = out_dir / f"{tag}_split{split}_{name}_q{q_th_final:g}.npz"
        save_catalogue_npz(
            cat,
            out,
            meta={
                "mmf": name,
                "mmf_type": mmf_type,
                "deproject_cib": deproj,
                "split": split,
                "field_ids": [int(i) for i in field_ids],
                "q_th_final": q_th_final,
                "n_detections": int(n),
            },
        )
        written[name] = out
        print(f"  wrote {out}")

    return written


def merge_catalogue_npzs(
    npz_paths: list[Path],
    out_path: Path,
    *,
    q_th_final: float = 5.0,
    merge_radius_arcmin: float = 10.0,
    meta: dict | None = None,
) -> Path:
    """Concatenate partial catalogues, re-threshold, FoF-merge, and save."""
    import json

    import szifi

    cat = szifi.cat.cluster_catalogue()
    keys_seen: set[str] | None = None
    chunks: dict[str, list[np.ndarray]] = {}
    for path in npz_paths:
        data = np.load(path)
        keys = set(data.files)
        if keys_seen is None:
            keys_seen = keys
            for k in keys:
                chunks[k] = []
        for k in keys_seen:
            if k in data.files and np.asarray(data[k]).size:
                chunks[k].append(np.asarray(data[k]))
    if not chunks or not any(chunks.values()):
        np.savez_compressed(out_path)
        if meta is not None:
            out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        return out_path

    for k, parts in chunks.items():
        if parts:
            cat.catalogue[k] = np.concatenate(parts)

    cat = szifi.get_catalogue_q_th(cat, q_th_final)
    if len(cat.catalogue.get("q_opt", [])) > 0:
        cat = szifi.merge_detections(
            cat,
            radius_arcmin=merge_radius_arcmin,
            return_merge_flag=False,
            mode="fof",
        )
    n = len(cat.catalogue.get("q_opt", []))
    meta_out = dict(meta or {})
    meta_out["n_detections"] = int(n)
    meta_out["n_partial_files"] = len(npz_paths)
    return save_catalogue_npz(cat, out_path, meta=meta_out)


def run_mmf_batched(
    paths: SZiFiPaths,
    field_ids: list[int],
    *,
    split: str = "A",
    method: str = "immf",
    q_th_final: float = 5.0,
    batch_size: int = 4,
    out_dir: Path | None = None,
    tag: str = "footprint",
    n_workers: int | None = None,
    threads_per_worker: int | None = None,
    array_backend: str = "jax",
) -> Path:
    """Run one MMF method in tile batches (resume-friendly); return merged catalogue.

    Process-pool workers are capped at half the host CPUs. ``array_backend='jax'``
    uses JAX on CPU so many workers do not contend for the two GPUs.
    """
    out_dir = Path(out_dir) if out_dir is not None else paths.catalogues_dir()
    partial_dir = out_dir / f"partial_{tag}_split{split}_{method}"
    partial_dir.mkdir(parents=True, exist_ok=True)

    method_specs = {
        "immf": ("standard", None),
        "scimmf": ("spectrally_constrained", ["cib"]),
    }
    mmf_type, deproj = method_specs[method]
    ids = list(field_ids)

    jobs: list[tuple] = []
    partials: list[Path] = []
    n_batch = (len(ids) + batch_size - 1) // batch_size
    for b, start in enumerate(range(0, len(ids), batch_size)):
        batch = ids[start : start + batch_size]
        part = partial_dir / f"batch_{b:04d}_q{q_th_final:g}.npz"
        partials.append(part)
        if part.exists() and part.stat().st_size > 100:
            print(f"[resume] {method} batch {b+1}/{n_batch} exists: {part.name}", flush=True)
            continue
        jobs.append(
            (
                str(paths.out_root),
                batch,
                split,
                mmf_type,
                deproj,
                q_th_final,
                str(part),
                b,
                n_batch,
                method,
            )
        )

    workers, threads = half_machine_pool_limits(
        n_workers, threads_per_worker=threads_per_worker
    )
    backend = str(array_backend).lower()
    print(
        f"{method}: {len(jobs)} batches to run, {n_batch - len(jobs)} resumed; "
        f"workers={workers}, threads/worker={threads}, backend={backend}",
        flush=True,
    )
    if jobs:
        _limit_compute_threads(threads)
        os.environ["SZIFI_ARRAY_BACKEND"] = backend
        if backend == "jax":
            os.environ["JAX_PLATFORMS"] = "cpu"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # spawn so workers re-import JAX with the thread caps above (fork
        # inherits an already-initialized 192-thread XLA pool from the parent).
        ctx = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_worker_threads,
            initargs=(threads,),
        ) as pool:
            for part_path, n, b, n_batch_ in pool.map(_run_one_batch_job, jobs, chunksize=1):
                print(
                    f"  [{method}] batch {b+1}/{n_batch_}: {n} detections → {part_path}",
                    flush=True,
                )

    out = out_dir / f"{tag}_split{split}_{method}_q{q_th_final:g}.npz"
    merge_catalogue_npzs(
        partials,
        out,
        q_th_final=q_th_final,
        meta={
            "mmf": method,
            "mmf_type": mmf_type,
            "deproject_cib": deproj,
            "split": split,
            "n_tiles": len(ids),
            "batch_size": batch_size,
            "n_workers": workers,
            "threads_per_worker": threads,
            "q_th_final": q_th_final,
            "tag": tag,
            "array_backend": backend,
        },
    )
    print(f"merged {method} → {out}", flush=True)
    return out


def _run_one_batch_job(args: tuple) -> tuple[str, int, int, int]:
    """Worker entry: run one tile batch and write a partial npz."""
    (
        out_root,
        batch,
        split,
        mmf_type,
        deproj,
        q_th_final,
        part_path,
        b,
        n_batch,
        method,
    ) = args
    os.environ.setdefault(
        "SZIFI_ARRAY_BACKEND", os.environ.get("SZIFI_ARRAY_BACKEND", "jax")
    )
    os.environ.setdefault("MPLBACKEND", "Agg")
    paths = SZiFiPaths(out_root=out_root)
    print(
        f"=== {method} batch {b+1}/{n_batch}: tiles {batch[0]}..{batch[-1]} "
        f"(n={len(batch)}) pid={os.getpid()} ===",
        flush=True,
    )
    cat = run_mmf(
        paths,
        list(batch),
        split=split,
        mmf_type=mmf_type,
        deproject_cib=deproj,
        q_th_final=q_th_final,
        method=method,
    )
    n = len(cat.catalogue.get("q_opt", []))
    save_catalogue_npz(
        cat,
        Path(part_path),
        meta={
            "mmf": method,
            "batch": b,
            "field_ids": list(batch),
            "n_detections": int(n),
            "q_th_final": q_th_final,
            "split": split,
        },
    )
    return part_path, int(n), int(b), int(n_batch)
