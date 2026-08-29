#!/usr/bin/env python3
"""Full-sky homog iMMF with szifi_jax on GPU1; write catalogue + mollview.

Reproduces figures/szifi_homog_immf_mollview.png (N=2364, q>=5, unmasked).
Does not overwrite the original NumPy-szifi catalogue.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
os.environ["XLA_FLAGS"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "10")
os.environ.setdefault("MKL_NUM_THREADS", "10")
os.environ.setdefault("MPLBACKEND", "Agg")
# Do not import flamingo_mock.szifi.cli: it blanks CUDA_VISIBLE_DEVICES.

import tensorflow as tf  # noqa: E402

tf.config.set_visible_devices([], "GPU")

import healpy as hp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from flamingo_mock.szifi.paths import DEFAULT_OUT_ROOT_HOMOG, SZiFiPaths  # noqa: E402
from flamingo_mock.szifi.run import (  # noqa: E402
    default_params,
    merge_catalogue_npzs,
    save_catalogue_npz,
)
from flamingo_mock.szifi.tiles import select_all_tile_ids  # noqa: E402

import szifi_jax  # noqa: E402

HOMOG_ROOT = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi_homog")
REF_CAT = HOMOG_ROOT / "catalogues" / "homog_immf_fullsky_splitA_immf_q5.npz"
YMAP = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/tsz/"
    "compton_y_nside4096.fits"
)
FIG_OUT = Path("figures/szifi_homog_immf_mollview.png")
PRESCRIPTIONS = ("L1_m9", "fgas-8sigma", "Mstar-1sigma", "LS8")
Q_EDGES = np.geomspace(5.0, 40.0, 6)
JAX_ONLY = {
    "powspec_implementation": "nmt_jax",  # GPU; matches NaMaster coupled cells at ~1e-16
    "inv_cov_backend": "numpy",  # unique-bin LAPACK then GPU scatter (CNC-exact)
    "inpaint_type": "diffusive",  # Gauss-Seidel; Jacobi is not CNC-exact
    "mmf_batched": True,
    "coupling_matrix_backend": "pymaster",
    "gpu_tile_batch": 16,
}


def _jax_params(paths, field_ids, split="A", gpu_tile_batch=16):
    params_szifi, params_data, params_model = default_params(
        paths, field_ids, split=split, mmf_type="standard", deproject_cib=None
    )
    params_szifi.update(JAX_ONLY)
    params_szifi["gpu_tile_batch"] = int(gpu_tile_batch)
    params_szifi["save_and_load_template"] = False
    params_szifi["snr_weighting"] = False
    # Gauss-Legendre is exact for q_opt / positions (template-norm invariant);
    # much faster than scipy.quad. y0 / sigma_vec move at ~4e-6.
    params_model["los_integral"] = "gauss_legendre"
    params_data["field_ids"] = list(field_ids)
    return params_szifi, params_data, params_model


def run_mmf_jax(paths, field_ids, split="A", q_th_final=5.0, merge_radius_arcmin=10.0,
                gpu_tile_batch=16):
    params_szifi, params_data, params_model = _jax_params(
        paths, field_ids, split=split, gpu_tile_batch=gpu_tile_batch
    )
    data = szifi_jax.input_data(params_szifi=params_szifi, params_data=params_data)
    cf = szifi_jax.cluster_finder(
        params_szifi=params_szifi, params_model=params_model, data_file=data, rank=0
    )
    cf.find_clusters()
    dp = szifi_jax.detection_processor(cf.results_dict, params_szifi)
    cat_key = "catalogue_find_1" if params_szifi.get("iterative", True) else "catalogue_find_0"
    if cat_key not in dp.results.catalogues:
        cat_key = "catalogue_find_0"
    catalogue = dp.results.catalogues[cat_key]
    catalogue = szifi_jax.get_catalogue_q_th(catalogue, q_th_final)
    n = len(catalogue.catalogue.get("q_opt", []))
    if n > 1:
        catalogue = szifi_jax.merge_detections(
            catalogue,
            radius_arcmin=merge_radius_arcmin,
            return_merge_flag=False,
            mode="fof",
        )
    return catalogue


def match_lonlat(lon_a, lat_a, lon_b, lat_b, radius_arcmin=1.0):
    if len(lon_a) == 0 or len(lon_b) == 0:
        return (
            np.zeros(len(lon_a), dtype=bool),
            np.full(len(lon_a), np.inf),
            np.full(len(lon_a), -1, dtype=int),
        )
    va = np.asarray(hp.ang2vec(lon_a, lat_a, lonlat=True))
    vb = np.asarray(hp.ang2vec(lon_b, lat_b, lonlat=True))
    from scipy.spatial import cKDTree

    d, idx = cKDTree(vb).query(va, k=1)
    dist = np.degrees(d) * 60.0
    return dist <= radius_arcmin, dist, idx


def plot_homog_mollview(cat, out_path: Path, ymap_path: Path, nside: int = 2048) -> None:
    y4096 = np.asarray(hp.read_map(str(ymap_path), dtype=np.float64))
    ymap = hp.ud_grade(y4096, nside) if hp.npix2nside(y4096.size) != nside else y4096
    good = ymap[np.isfinite(ymap) & (ymap > 0)]
    vmax = float(np.percentile(good, 99.5))
    vmin = float(np.percentile(good, 1.0))
    lon = np.asarray(cat["lon"])
    lat = np.asarray(cat["lat"])
    theta_500 = np.asarray(cat["theta_500"])
    n_det = len(lon)

    hp.mollview(
        ymap,
        min=vmin,
        max=vmax,
        title=rf"Truth Compton-y + full-sky iMMF (N={n_det}, $q \ge 5$, unmasked)",
        unit=r"$y$",
        cmap="hot",
        cbar=True,
        hold=True,
    )
    hp.graticule(dmer=30, dpar=30, alpha=0.25, verbose=False)
    s = np.clip((theta_500 / 60.0) ** 2 * 120.0, 4.0, 400.0)
    hp.projscatter(
        lon,
        lat,
        lonlat=True,
        s=s,
        facecolors="none",
        edgecolors="cyan",
        linewidths=0.45,
        alpha=0.85,
        label=f"iMMF ($n={n_det}$)",
    )
    plt.legend(loc="lower right", fontsize=10, framealpha=0.9)
    plt.gcf().text(
        0.02,
        0.02,
        f"{n_det} clusters detected (q >= 5, full sky)",
        transform=plt.gcf().transFigure,
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}", flush=True)


def compare_to_ref(cat, ref_path: Path, radius_arcmin=1.0, field_ids=None) -> dict:
    ref = np.load(ref_path)
    lon_r = np.asarray(ref["lon"])
    lat_r = np.asarray(ref["lat"])
    q_r = np.asarray(ref["q_opt"]) if "q_opt" in ref.files else None
    if field_ids is not None and "pixel_ids" in ref.files:
        keep = np.isin(np.asarray(ref["pixel_ids"]).astype(int), list(field_ids))
        lon_r, lat_r = lon_r[keep], lat_r[keep]
        if q_r is not None:
            q_r = q_r[keep]
    lon = np.asarray(cat["lon"])
    lat = np.asarray(cat["lat"])
    q = np.asarray(cat["q_opt"])
    ok, dist, idx = match_lonlat(lon, lat, lon_r, lat_r, radius_arcmin)
    ok_r, dist_r, _ = match_lonlat(lon_r, lat_r, lon, lat, radius_arcmin)
    nq_jax = np.histogram(q, bins=Q_EDGES)[0].astype(int).tolist()
    nq_ref = (
        np.histogram(q_r, bins=Q_EDGES)[0].astype(int).tolist()
        if q_r is not None
        else None
    )
    out = {
        "n_jax": int(len(q)),
        "n_ref": int(len(lon_r)),
        "n_matched": int(ok.sum()),
        "n_ref_matched": int(ok_r.sum()),
        "median_sep_arcmin": float(np.median(dist[ok])) if ok.any() else None,
        "max_sep_matched_arcmin": float(np.max(dist[ok])) if ok.any() else None,
        "n_unmatched_jax": int((~ok).sum()),
        "n_unmatched_ref": int((~ok_r).sum()),
        "median_abs_dq": None,
        "max_abs_dq": None,
        "nq_jax": nq_jax,
        "nq_ref": nq_ref,
        "nq_match": nq_jax == nq_ref,
    }
    if q_r is not None and ok.any():
        dq = np.abs(q[ok] - q_r[idx[ok]])
        out["median_abs_dq"] = float(np.median(dq))
        out["max_abs_dq"] = float(np.max(dq))
    print(
        f"vs ref {ref_path.name}: N jax={out['n_jax']} ref={out['n_ref']}  "
        f"matched {out['n_matched']}/{out['n_jax']} jax, "
        f"{out['n_ref_matched']}/{out['n_ref']} ref  "
        f"median sep {out['median_sep_arcmin']} arcmin"
        + (
            f"  |Δq| median {out['median_abs_dq']:.3e} max {out['max_abs_dq']:.3e}"
            if out["max_abs_dq"] is not None
            else ""
        ),
        flush=True,
    )
    print(f"  N(q) jax={out['nq_jax']}  ref={out['nq_ref']}  bins={out['nq_match']}", flush=True)
    return out


def all_clusters_match(stats: dict) -> bool:
    """True if every cluster matches: same N, all positions, identical N(q) bins."""
    if stats["n_jax"] != stats["n_ref"]:
        return False
    if stats["n_unmatched_jax"] or stats["n_unmatched_ref"]:
        return False
    if stats.get("nq_match") is False:
        return False
    return True


def numpy_ref_cat(out_root: Path) -> Path:
    p = Path(out_root) / "catalogues" / "fullsky_splitA_immf_q5.npz"
    if p.is_file():
        return p
    return REF_CAT


def run_one(
    *,
    out_root: Path,
    field_ids: list[int] | None,
    batch_size: int,
    q_th: float,
    tag: str,
    ref_cat: Path,
    skip_plot: bool,
    fig: Path,
) -> dict:
    paths = SZiFiPaths(out_root=out_root, kind="homog")
    ids = field_ids if field_ids is not None else select_all_tile_ids()
    out_dir = paths.catalogues_dir()
    partial_dir = out_dir / f"partial_{tag}_splitA_immf"
    partial_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"tiles n={len(ids)} batch={batch_size}  GPU={os.environ['CUDA_VISIBLE_DEVICES']}  "
        f"out_root={out_root}",
        flush=True,
    )
    partials: list[Path] = []
    n_batch = (len(ids) + batch_size - 1) // batch_size
    t_all = time.time()
    for b, start in enumerate(range(0, len(ids), batch_size)):
        batch = ids[start : start + batch_size]
        part = partial_dir / f"batch_{b:04d}_q{q_th:g}.npz"
        partials.append(part)
        if part.exists() and part.stat().st_size > 100:
            print(f"[resume] batch {b+1}/{n_batch} {part.name}", flush=True)
            continue
        t0 = time.time()
        cat = run_mmf_jax(
            paths, batch, q_th_final=q_th, gpu_tile_batch=batch_size
        )
        n = len(cat.catalogue.get("q_opt", []))
        save_catalogue_npz(
            cat,
            part,
            meta={
                "mmf": "immf",
                "backend": "szifi_jax",
                "batch": b,
                "field_ids": list(batch),
                "n_detections": int(n),
                "q_th_final": q_th,
            },
        )
        dt = time.time() - t0
        print(
            f"  batch {b+1}/{n_batch} tiles {batch[0]}..{batch[-1]}  "
            f"n={n}  {dt:.1f}s  ({dt/len(batch):.2f}s/tile)",
            flush=True,
        )

    out = out_dir / f"{tag}_splitA_immf_q{q_th:g}.npz"
    merge_catalogue_npzs(
        partials,
        out,
        q_th_final=q_th,
        meta={
            "mmf": "immf",
            "mmf_type": "standard",
            "backend": "szifi_jax",
            "split": "A",
            "n_tiles": len(ids),
            "batch_size": batch_size,
            "q_th_final": q_th,
            "tag": tag,
        },
    )
    elapsed = time.time() - t_all
    cat = {k: np.asarray(v) for k, v in np.load(out).items()}
    n = len(cat.get("q_opt", []))
    print(f"merged N={n} → {out}  wall {elapsed/60:.1f} min", flush=True)

    stats = None
    if ref_cat.is_file() and "lon" in cat:
        stats = compare_to_ref(
            cat, ref_cat, field_ids=None if field_ids is None else ids
        )
        sidecar = out.with_suffix(".json")
        payload = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        payload.update({"n_detections": n, "vs_ref": stats, "wall_s": elapsed})
        sidecar.write_text(json.dumps(payload, indent=2) + "\n")

    if not skip_plot and n:
        try:
            plot_homog_mollview(cat, fig, YMAP)
        except FileNotFoundError as exc:
            print(f"skip plot ({exc})", flush=True)

    return {"out": out, "n": n, "stats": stats, "elapsed": elapsed}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--field-ids", type=int, nargs="+", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--q-th", type=float, default=5.0)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument(
        "--prescription",
        choices=PRESCRIPTIONS,
        default=None,
        help="Run one L1 feedback variation (tiles + NumPy ref under szifi_homog/<name>)",
    )
    p.add_argument(
        "--prescriptions",
        nargs="+",
        choices=PRESCRIPTIONS,
        default=None,
        help="Run several prescriptions in one process (JIT stays warm)",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Catalogue name tag (does not overwrite the original NumPy catalogue)",
    )
    p.add_argument("--ref", type=Path, default=None)
    p.add_argument("--fig", type=Path, default=FIG_OUT)
    p.add_argument("--skip-plot", action="store_true")
    args = p.parse_args()

    print(f"jax devices: {jax.devices()}", flush=True)

    names = list(args.prescriptions) if args.prescriptions else (
        [args.prescription] if args.prescription else [None]
    )
    failed = []
    for name in names:
        if name is None:
            out_root = args.out_root or DEFAULT_OUT_ROOT_HOMOG
            tag = args.tag or "homog_immf_fullsky_szifi_jax"
            ref = args.ref or numpy_ref_cat(out_root)
            if not ref.is_file():
                ref = REF_CAT
            skip_plot = args.skip_plot
        else:
            out_root = HOMOG_ROOT / name
            tag = args.tag or "szifi_jax"
            ref = args.ref or numpy_ref_cat(out_root)
            skip_plot = True
        print(f"=== {name or 'homog'}  ref={ref} ===", flush=True)
        result = run_one(
            out_root=out_root,
            field_ids=args.field_ids,
            batch_size=args.batch_size,
            q_th=args.q_th,
            tag=tag,
            ref_cat=ref,
            skip_plot=skip_plot,
            fig=args.fig,
        )
        stats = result["stats"]
        fullsky = args.field_ids is None
        if stats is None:
            print("WARNING: no reference catalogue; cannot test ALL-cluster match", flush=True)
            failed.append(name or "homog")
        elif fullsky and not all_clusters_match(stats):
            print(
                "FAIL: catalogue does not match ALL reference clusters "
                f"(N jax={stats['n_jax']} ref={stats['n_ref']}, "
                f"unmatched jax={stats['n_unmatched_jax']} ref={stats['n_unmatched_ref']})",
                flush=True,
            )
            failed.append(name or "homog")
        elif fullsky:
            print("PASS: ALL clusters match (N, positions, N(q) bins)", flush=True)
            if stats.get("max_abs_dq") is not None and stats["max_abs_dq"] > 1e-4:
                print(
                    f"  note: max |Δq|={stats['max_abs_dq']:.3e} "
                    "(positions and N(q) still identical)",
                    flush=True,
                )

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
