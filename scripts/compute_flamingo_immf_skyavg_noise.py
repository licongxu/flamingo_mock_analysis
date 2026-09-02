#!/usr/bin/env python3
"""Sky-averaged iMMF noise curve on FLAMINGO mock tiles vs Planck immf6.

Per-tile σ_y0(θ) from SZiFi standard MMF (6 HFI channels, NPIPE split A),
sky-fraction weighted average (same recipe as hmfast/tszpower), then compare
to the Planck-collaboration ``immf6`` curve in ``sigma_dict_szifi.npy``.

Outputs
-------
- ``.../szifi/catalogues/sigma_per_tile_flamingo_immf_splitA.npz``
- ``.../szifi/catalogues/noise_curve_skyavg_flamingo_immf.npz``
- ``figures/noise_curve_immf6_vs_flamingo_mock.{png,pdf}``
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Worker subprocess: CPU-only SZiFi before any CUDA init.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("SZIFI_ARRAY_BACKEND", "numpy")
os.environ.setdefault("MPLBACKEND", "Agg")

from flamingo_mock.szifi.paths import (  # noqa: E402
    DEFAULT_OUT_ROOT_HOMOG,
    DEFAULT_TOTAL_MAPS_HOMOG,
    SZiFiPaths,
    TILE_NX,
)
from flamingo_mock.szifi.run import (  # noqa: E402
    default_params,
    half_machine_pool_limits,
    save_per_tile_sigma,
    sigma_per_tile_dir,
)
from flamingo_mock.szifi.tiles import select_all_tile_ids, select_footprint_tile_ids  # noqa: E402

THETA_MIN = 0.5
THETA_MAX = 32.0
N_THETA = 25
I_Y500 = 0.06728373215772082  # SZiFi y0_to_Y_500 integral factor

DEFAULT_SIGMA_OBJ = Path("/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy")
DEFAULT_SKYFR = Path("/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy")


def theta_grid() -> np.ndarray:
    return np.exp(np.linspace(np.log(THETA_MIN), np.log(THETA_MAX), N_THETA))


def _tile_ckpt_path(cache_dir: Path, field_id: int) -> Path:
    return cache_dir / f"field_{int(field_id)}.npy"


def _tile_intermediate_path(cache_dir: Path, field_id: int) -> Path:
    return cache_dir / "intermediates" / f"field_{int(field_id)}.npz"


def _yt_theta_indices(theta_vec: np.ndarray) -> np.ndarray:
    """Keep 2D y_t(ell) at ~2', 5', 10' (enough for the aperture, not 25 full maps)."""
    return np.array([int(np.argmin(np.abs(theta_vec - t))) for t in (2.0, 5.0, 10.0)])


def _save_mmf_intermediates(cf, theta_vec: np.ndarray, out_path: Path, field_id: int) -> None:
    """Write N(ell), d(ell), and y_t(ell) from the last (iterative) covariance."""
    from szifi import maps, model

    pix = cf.pix
    inv_cov = np.asarray(cf.inv_cov)
    d_fft = maps.reshape_ell_matrix(maps.get_fft_f(cf.t_obs, pix), inv_cov.shape[:2])
    N_ell = np.asarray(cf.cspec.spec_tensor)
    ell_vec = np.asarray(cf.cspec.ell_vec)
    invN_ell = np.linalg.inv(N_ell)

    a_sz = np.asarray(cf.params_szifi["a_matrix"][:, 0])
    idx = _yt_theta_indices(theta_vec)
    yt = []
    z = 0.2
    for th in theta_vec[idx]:
        m500 = model.get_m_500(float(th), z, cf.cosmology)
        nfw = model.gnfw(m500, z, cf.cosmology, type=cf.params_model["profile_type"])
        theta_cart = [(0.5 * pix.nx) * pix.dx, (0.5 * pix.nx) * pix.dx]
        t_tem = nfw.get_t_map_convolved(
            pix, cf.exp, beam=cf.params_szifi["beam"],
            theta_cart=theta_cart, get_nc=False, sed=False,
        )
        t_tem = t_tem / nfw.get_y_norm(cf.params_szifi["norm_type"])
        tem = maps.filter_tmap(t_tem, pix, cf.params_szifi["lrange"])
        yt_fft = maps.get_tmap_times_fvec(maps.get_fft_f(tem, pix), a_sz)
        yt.append(maps.reshape_ell_matrix(yt_fft, inv_cov.shape[:2]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ell_vec=ell_vec.astype(np.float32),
        N_ell=N_ell.astype(np.float32),
        invN_ell=invN_ell.astype(np.float32),
        inv_cov=inv_cov.astype(np.float32),
        d_fft=np.asarray(d_fft, dtype=np.complex64),
        yt_fft=np.stack(yt, axis=0).astype(np.complex64),
        theta_yt_arcmin=theta_vec[idx].astype(np.float32),
        field_id=int(field_id),
    )


def _sigma_vec_one_tile(payload: dict) -> tuple[int, np.ndarray]:
    """Worker: return (field_id, sigma_y0[ntheta])."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["SZIFI_ARRAY_BACKEND"] = "numpy"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(payload["threads"])

    import szifi
    from szifi import maps, mmf, model, spec

    field_id = int(payload["field_id"])
    split = payload["split"]
    theta_vec = np.asarray(payload["theta_vec"], dtype=np.float64)
    paths = SZiFiPaths(
        out_root=payload["out_root"],
        total_maps_dir=payload.get("total_maps_dir", SZiFiPaths.total_maps_dir),
        kind=payload.get("kind", "npipe"),
    )

    ckpt = _tile_ckpt_path(Path(payload["cache_dir"]), field_id)
    if ckpt.is_file() and not payload.get("overwrite", False):
        return field_id, np.load(ckpt)

    iterative = bool(payload.get("iterative", False))
    params_szifi, params_data, params_model = default_params(
        paths, [field_id], split=split, mmf_type="standard"
    )
    params_szifi["iterative"] = iterative
    params_szifi["inpaint"] = False
    params_szifi["array_backend"] = "numpy"
    params_szifi["theta_500_vec_arcmin"] = theta_vec
    params_szifi["extraction_mode"] = "find"

    if iterative:
        import contextlib
        import io

        data = szifi.input_data(params_szifi=params_szifi, params_data=params_data)
        cf = szifi.cluster_finder(
            params_szifi=params_szifi,
            params_model=params_model,
            data_file=data,
            rank=1,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cf.find_clusters()
        sigma = np.asarray(
            cf.results_dict[field_id].sigma_vec[
                "find_1"
                if "find_1" in cf.results_dict[field_id].sigma_vec
                else "find_0"
            ],
            dtype=np.float64,
        )
        if payload.get("use_runner_cache"):
            save_per_tile_sigma(cf.results_dict, theta_vec, Path(payload["cache_dir"]))
        else:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            np.save(ckpt, sigma)
        if hasattr(cf, "inv_cov") and cf.inv_cov is not None and hasattr(cf, "cspec"):
            try:
                _save_mmf_intermediates(
                    cf, theta_vec, _tile_intermediate_path(Path(payload["cache_dir"]), field_id),
                    field_id,
                )
            except Exception as err:
                print(f"  warn field={field_id}: intermediates not saved ({err})", flush=True)
        return field_id, sigma

    data = szifi.input_data(params_szifi=params_szifi, params_data=params_data)
    d = data.data
    exp = d["experiment"]
    cosmology = model.cosmological_model(params_szifi).cosmology
    dx = d["dx_arcmin"][field_id] / 60.0 / 180.0 * np.pi
    pix = maps.pixel(TILE_NX, dx)
    t_noi = np.asarray(d["t_noi"][field_id], dtype=np.float32)
    mask_ps = np.asarray(d["mask_ps"][field_id], dtype=np.float64)
    mask_point = np.asarray(d["mask_point"][field_id], dtype=np.float64)

    if params_szifi["a_matrix"] is None:
        a_matrix = np.zeros((len(exp.tsz_sed), 1))
        a_matrix[:, 0] = exp.tsz_sed
        params_szifi["a_matrix"] = a_matrix

    ps = spec.power_spectrum(
        pix,
        mask=mask_ps,
        cm_compute=False,
        cm_compute_scratch=False,
        cm_save=False,
        cm_name=None,
        bin_fac=params_szifi["powspec_bin_fac"],
    )
    cspec = spec.cross_spec(np.arange(len(params_szifi["freqs"])))
    cspec.get_cross_spec(
        pix,
        t_map=t_noi,
        ps=ps,
        decouple_type=params_szifi["decouple_type"],
        inpaint_flag=False,
        mask_point=mask_point,
        lsep=params_szifi["lsep"],
        bin_fac=params_szifi["powspec_bin_fac"],
    )
    inv_cov = cspec.get_inv_cov(
        pix,
        t_map=t_noi,
        interp_type=params_szifi["interp_type"],
        bin_fac=params_szifi["powspec_bin_fac"],
        mask=mask_ps,
        cov_type=params_szifi["cov_type"],
        cov_kernel_shape=params_szifi["cov_kernel_shape"],
    )
    params_szifi["cmmf_type"] = "standard_mmf"
    cmmf = mmf.scmmf_precomputation(
        pix=pix,
        freqs=params_szifi["freqs"],
        inv_cov=inv_cov,
        lrange=params_szifi["lrange"],
        beam_type=params_szifi["beam"],
        exp=exp,
        cmmf_type=params_szifi["cmmf_type"],
        a_matrix=params_szifi["a_matrix"],
        comp_to_calculate=params_szifi["comp_to_calculate"],
        mmf_type="standard",
    )

    sigma = np.zeros(len(theta_vec), dtype=np.float64)
    t_dummy = np.zeros_like(t_noi)
    z = 0.2
    for j, th in enumerate(theta_vec):
        m500 = model.get_m_500(float(th), z, cosmology)
        nfw = model.gnfw(m500, z, cosmology, type=params_model["profile_type"])
        theta_cart = [(0.5 * pix.nx) * pix.dx, (0.5 * pix.nx) * pix.dx]
        t_tem = nfw.get_t_map_convolved(
            pix,
            exp,
            beam=params_szifi["beam"],
            theta_cart=theta_cart,
            get_nc=False,
            sed=False,
        )
        t_tem = t_tem / nfw.get_y_norm(params_szifi["norm_type"])
        tem = maps.filter_tmap(t_tem, pix, params_szifi["lrange"])
        _, _, std = mmf.get_mmf_q_map(
            t_dummy,
            tem,
            inv_cov,
            pix,
            mmf_type="standard",
            cmmf_prec=cmmf,
            tem_norm=None,
        )
        sigma[j] = float(std)

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    np.save(ckpt, sigma)
    return field_id, sigma


def compute_per_tile(
    paths: SZiFiPaths,
    field_ids: list[int],
    *,
    split: str = "A",
    n_workers: int | None = None,
    threads_per_worker: int = 2,
    overwrite: bool = False,
    iterative: bool = False,
    use_runner_cache: bool = False,
) -> dict[int, np.ndarray]:
    if use_runner_cache and iterative:
        cache_dir = sigma_per_tile_dir(paths, method="immf", split=split)
    else:
        tag = f"flamingo_immf{'_it' if iterative else ''}_split{split}"
        cache_dir = paths.catalogues_dir() / f"sigma_per_tile_{tag}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    theta_vec = theta_grid()
    workers, threads = half_machine_pool_limits(
        n_workers, threads_per_worker=threads_per_worker
    )

    def _tile_done(fid: int) -> bool:
        primary = cache_dir / f"field_{int(fid)}.npy"
        if use_runner_cache and iterative:
            return primary.is_file()
        return _tile_ckpt_path(cache_dir, fid).is_file()

    todo = [fid for fid in field_ids if overwrite or not _tile_done(fid)]
    done = {}
    for fid in field_ids:
        if fid in todo:
            continue
        primary = cache_dir / f"field_{int(fid)}.npy"
        path = primary if primary.is_file() else _tile_ckpt_path(cache_dir, fid)
        done[fid] = np.load(path)
    print(
        f"tiles: total={len(field_ids)} cached={len(done)} todo={len(todo)} "
        f"workers={workers} threads/worker={threads} "
        f"(~{workers * threads} CPU threads)",
        flush=True,
    )

    if todo:
        payloads = [
            {
                "field_id": int(fid),
                "split": split,
                "theta_vec": theta_vec,
                "out_root": str(paths.out_root),
                "total_maps_dir": str(paths.total_maps_dir),
                "kind": paths.kind,
                "cache_dir": str(cache_dir),
                "threads": threads,
                "overwrite": overwrite,
                "iterative": iterative,
                "use_runner_cache": use_runner_cache,
            }
            for fid in todo
        ]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_sigma_vec_one_tile, pl): pl["field_id"] for pl in payloads}
            n = 0
            for fut in as_completed(futs):
                fid, sigma = fut.result()
                done[int(fid)] = sigma
                n += 1
                if n % 20 == 0 or n == len(todo):
                    print(f"  finished {n}/{len(todo)}", flush=True)

    return {int(fid): np.asarray(done[int(fid)], dtype=np.float64) for fid in field_ids}


def sky_average(
    per_tile: dict[int, np.ndarray],
    skyfr_file: Path = DEFAULT_SKYFR,
    *,
    equal_weight: bool = False,
) -> np.ndarray:
    skyfr = None if equal_weight else np.load(skyfr_file).ravel()
    num = np.zeros(N_THETA, dtype=np.float64)
    den = 0.0
    for fid, arr in per_tile.items():
        w = 1.0 if equal_weight else float(skyfr[int(fid)])
        if w <= 0:
            continue
        num += w * np.asarray(arr, dtype=np.float64)
        den += w
    return num / max(den, 1e-300)


def load_reference_immf6(
    filter_name: str = "immf6",
    sigma_obj_file: Path = DEFAULT_SIGMA_OBJ,
    skyfr_file: Path = DEFAULT_SKYFR,
) -> np.ndarray:
    sigma_obj = np.load(sigma_obj_file, allow_pickle=True).item()
    skyfr = np.load(skyfr_file).ravel()
    data = sigma_obj[filter_name]
    num = np.zeros(N_THETA, dtype=np.float64)
    den = 0.0
    for tile, arr in data.items():
        w = float(skyfr[int(tile)])
        num += w * np.asarray(arr, dtype=np.float64)
        den += w
    return num / max(den, 1e-300)


def plot_comparison(
    theta: np.ndarray,
    sigma_ref: np.ndarray,
    sigma_mock: np.ndarray,
    out_base: Path,
    *,
    ref_label: str = "Planck SZiFi immf6 (theory)",
    mock_label: str = "FLAMINGO mock iMMF (sky avg)",
) -> None:
    sigma_y500_ref = sigma_ref * (theta**2) * np.pi * I_Y500
    sigma_y500_mock = sigma_mock * (theta**2) * np.pi * I_Y500
    ratio = sigma_mock / np.clip(sigma_ref, 1e-30, None)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    ax.loglog(theta, sigma_ref, "k-", lw=2, label=ref_label)
    ax.loglog(theta, sigma_mock, "C1--", lw=2, label=mock_label)
    ax.set_xlabel(r"$\theta_{500}$ [arcmin]")
    ax.set_ylabel(r"$\sigma_{y_0}$")
    ax.set_title(r"MMF noise on $y_0$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.loglog(theta, sigma_y500_ref, "k-", lw=2, label=ref_label)
    ax.loglog(theta, sigma_y500_mock, "C1--", lw=2, label=mock_label)
    ax.set_xlabel(r"$\theta_{500}$ [arcmin]")
    ax.set_ylabel(r"$\sigma_{Y_{500}}$")
    ax.set_title(r"Propagated to $Y_{500}$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    ax.semilogx(theta, ratio, "C0-", lw=2)
    ax.axhline(1.0, color="0.3", ls="--")
    ax.set_xlabel(r"$\theta_{500}$ [arcmin]")
    ax.set_ylabel(r"$\sigma_{\rm mock} / \sigma_{\rm immf6}$")
    ax.set_title("Ratio (mock / Planck immf6)")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Sky-averaged SZiFi iMMF noise: FLAMINGO mock (NPIPE A) vs Planck immf6",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_base.with_suffix('.png')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("A", "B"), default="A")
    p.add_argument(
        "--kind",
        choices=("npipe", "homog"),
        default="npipe",
        help="Tile set: NPIPE footprint (default) or homog full-sky",
    )
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--total-maps-dir", type=Path, default=None)
    p.add_argument("--full-sky", action="store_true")
    p.add_argument(
        "--n-workers",
        type=int,
        default=12,
        help="Parallel tile workers (default 12)",
    )
    p.add_argument(
        "--threads-per-worker",
        type=int,
        default=2,
        help="BLAS/OpenMP threads per worker (default 2 → 24 cores total)",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--iterative",
        action="store_true",
        help="Full SZiFi iMMF with iterative covariance (find_1); separate cache",
    )
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument(
        "--runner-cache",
        action="store_true",
        help="use sigma_per_tile_immf_split{split}/ like flamingo_mock.szifi.run",
    )
    p.add_argument(
        "--fig",
        type=Path,
        default=None,
    )
    args = p.parse_args()

    if args.kind == "homog":
        paths = SZiFiPaths(
            out_root=args.out_root or DEFAULT_OUT_ROOT_HOMOG,
            total_maps_dir=args.total_maps_dir or DEFAULT_TOTAL_MAPS_HOMOG,
            kind="homog",
        )
    else:
        overrides = {}
        if args.out_root is not None:
            overrides["out_root"] = args.out_root
        if args.total_maps_dir is not None:
            overrides["total_maps_dir"] = args.total_maps_dir
        paths = SZiFiPaths(**overrides)
    paths.make_dirs(args.split)
    theta = theta_grid()
    full_sky = bool(args.full_sky or args.kind == "homog")
    field_ids = (
        select_all_tile_ids()
        if full_sky
        else select_footprint_tile_ids(paths.masks_fits, min_ftile=0.3)
    )

    if not args.plot_only:
        per_tile = compute_per_tile(
            paths,
            field_ids,
            split=args.split,
            n_workers=args.n_workers,
            threads_per_worker=args.threads_per_worker,
            overwrite=args.overwrite,
            iterative=args.iterative,
            use_runner_cache=args.runner_cache,
        )
        if args.runner_cache and args.iterative:
            tag = f"immf_split{args.split}"
            cache_dir = sigma_per_tile_dir(paths, method="immf", split=args.split)
            per_tile = {
                int(p.stem.split("_")[1]): np.load(p)
                for p in sorted(cache_dir.glob("field_*.npy"))
                if "_noit" not in p.stem
            }
        else:
            tag = f"flamingo_immf{'_it' if args.iterative else ''}_split{args.split}"
        np.savez_compressed(
            paths.catalogues_dir() / f"sigma_per_tile_{tag}.npz",
            theta_500_arcmin=theta,
            **{f"field_{k}": v for k, v in per_tile.items()},
        )
    else:
        if args.runner_cache and args.iterative:
            tag = f"immf_split{args.split}"
            cache_dir = sigma_per_tile_dir(paths, method="immf", split=args.split)
        else:
            tag = f"flamingo_immf{'_it' if args.iterative else ''}_split{args.split}"
            cache_dir = paths.catalogues_dir() / f"sigma_per_tile_{tag}"
        per_tile = {
            int(p.stem.split("_")[1]): np.load(p)
            for p in sorted(cache_dir.glob("field_*.npy"))
            if "_noit" not in p.stem
        }

    sigma_mock = sky_average(per_tile, equal_weight=full_sky)
    sigma_ref = load_reference_immf6()
    out_npz = (
        paths.catalogues_dir() / "noise_curve_skyavg_flamingo_immf_it.npz"
        if args.iterative
        else paths.catalogues_dir() / "noise_curve_skyavg_flamingo_immf.npz"
    )
    mock_key = "sigma_y0_flamingo_mock_it" if args.iterative else "sigma_y0_flamingo_mock"
    np.savez(
        out_npz,
        theta_500_arcmin=theta,
        sigma_y0_immf6_planck=sigma_ref,
        **{mock_key: sigma_mock},
        sigma_y500_immf6_planck=sigma_ref * (theta**2) * np.pi * I_Y500,
        sigma_y500_flamingo_mock=sigma_mock * (theta**2) * np.pi * I_Y500,
        n_tiles=len(per_tile),
        split=args.split,
        kind=args.kind,
        equal_weight=full_sky,
    )
    fig = args.fig or Path("figures") / (
        f"noise_curve_immf6_vs_flamingo_homog_{paths.out_root.name}"
        if args.kind == "homog"
        else "noise_curve_immf6_vs_flamingo_mock"
    )
    if not args.no_plot:
        plot_comparison(theta, sigma_ref, sigma_mock, fig)
    med_ratio = np.median(sigma_mock / np.clip(sigma_ref, 1e-30, None))
    print(f"Wrote {out_npz}")
    print(f"median(mock/immf6) = {med_ratio:.3f} over {len(per_tile)} tiles")


if __name__ == "__main__":
    main()
