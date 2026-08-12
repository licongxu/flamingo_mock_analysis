"""Benchmark SZiFi catalogues against the L2p8_m9 lightcone0 truth (qfrommap)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from flamingo_mock.szifi.paths import SZiFiPaths
from flamingo_mock.szifi.tiles import load_pr4_gal_ps

DEFAULT_TRUTH_CATALOGUE = Path(
    "/rds/rds-lxu/flamingo/L2p8_m9/lightcone0/catalogues/"
    "halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_qfrommap.csv"
)

_TRUTH_COLS = (
    "z",
    "lon_rot_deg",
    "lat_rot_deg",
    "M_500c_Msun",
    "theta_500_arcmin",
    "q_from_aperture",
)


@dataclass
class BenchmarkResult:
    """Completeness / purity for one observed catalogue."""

    catalogue: str
    truth_catalogue: str
    n_detected: int
    n_detected_catalogue: int
    n_truth_all: int
    n_truth_detectable: int
    n_true_positives: int
    n_false_positives: int
    n_undetected: int
    purity: float
    completeness_all: float
    completeness_detectable: float
    match_radius_arcmin: float
    q_th_obs: float
    q_th_truth: float
    z_max: float
    footprint: str
    n_detected_excluded_mask: int


def load_planck_unmasked_mask(
    paths: SZiFiPaths,
) -> tuple[np.ndarray, int]:
    """PR4 GAL×PS binary mask (1 = unmasked Planck footprint)."""
    gal, ps = load_pr4_gal_ps(paths.masks_fits, nside=paths.nside)
    return (gal * ps).astype(np.float64), paths.nside


def unmasked_at_lonlat(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    mask: np.ndarray,
    nside: int,
) -> np.ndarray:
    ipix = hp.ang2pix(nside, lon_deg, lat_deg, lonlat=True, nest=True)
    return mask[ipix] > 0.5


def detection_q_mask(det: dict[str, np.ndarray], q_th_obs: float) -> np.ndarray:
    """Catalogue rows with q >= threshold."""
    return det["q_opt"] >= q_th_obs


def detection_in_footprint_mask(
    det: dict[str, np.ndarray],
    mask: np.ndarray,
    nside: int,
    q_th_obs: float,
) -> np.ndarray:
    """q-threshold and PR4 GAL×PS unmasked (same sky as truth for purity)."""
    return detection_q_mask(det, q_th_obs) & unmasked_at_lonlat(
        det["lon"], det["lat"], mask, nside
    )


def _lonlat_to_vec(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    clat = np.cos(lat)
    return np.column_stack([clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)])


def load_truth_qfrommap(
    truth_csv: Path,
    paths: SZiFiPaths,
    *,
    z_max: float = 1.0,
    q_th_truth: float = 5.0,
    chunk_size: int = 500_000,
    mask: np.ndarray | None = None,
    nside: int | None = None,
) -> dict[str, np.ndarray]:
    """Load truth halos in the Planck footprint (PR4 GAL×PS unmasked, yang26rot)."""
    if mask is None or nside is None:
        mask, nside = load_planck_unmasked_mask(paths)

    chunks: dict[str, list[np.ndarray]] = {c: [] for c in _TRUTH_COLS}
    for chunk in pd.read_csv(
        truth_csv, comment="#", usecols=_TRUTH_COLS, chunksize=chunk_size
    ):
        z = chunk["z"].to_numpy()
        lon = chunk["lon_rot_deg"].to_numpy()
        lat = chunk["lat_rot_deg"].to_numpy()
        ok = (
            (z <= z_max)
            & unmasked_at_lonlat(lon, lat, mask, nside)
        )
        if not ok.any():
            continue
        for col in _TRUTH_COLS:
            chunks[col].append(chunk[col].to_numpy()[ok])

    if not chunks["z"]:
        empty = np.array([], dtype=np.float64)
        return {c: empty for c in _TRUTH_COLS}

    out = {c: np.concatenate(chunks[c]) for c in _TRUTH_COLS}
    out["detectable"] = out["q_from_aperture"] >= q_th_truth
    return out


def load_detection_catalogue(cat_path: Path) -> dict[str, np.ndarray]:
    data = np.load(cat_path)
    return {k: np.asarray(data[k]) for k in data.files}


def cross_match_greedy(
    det_lon: np.ndarray,
    det_lat: np.ndarray,
    det_q: np.ndarray,
    det_theta: np.ndarray,
    truth_lon: np.ndarray,
    truth_lat: np.ndarray,
    truth_theta: np.ndarray,
    *,
    match_radius_arcmin: float = 10.0,
    use_theta_500: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one match (highest-q detections first).

    Follows Zubeldia et al. (2024) §3.4.1 / §4.1: sort by SNR, associate within
    ``match_radius_arcmin`` (default 10'), exclusive one-to-one. For truth
    catalogues without SNR we take the nearest neighbour inside the radius.
    """
    n_det = len(det_lon)
    n_truth = len(truth_lon)
    det_hit = np.zeros(n_det, dtype=bool)
    truth_hit = np.zeros(n_truth, dtype=bool)
    if n_det == 0 or n_truth == 0:
        return det_hit, truth_hit

    tree = cKDTree(_lonlat_to_vec(truth_lon, truth_lat))
    order = np.argsort(-det_q)

    for i in order:
        if use_theta_500:
            r = max(match_radius_arcmin, float(det_theta[i]))
        else:
            r = match_radius_arcmin
        cos_r = np.cos(np.deg2rad(r / 60.0))
        dist, j = tree.query(_lonlat_to_vec(det_lon[i : i + 1], det_lat[i : i + 1])[0], k=1)
        # cKDTree euclidean distance on unit sphere: d = 2*sin(angle/2)
        # angle = 2*arcsin(d/2); cos(angle) = 1 - d^2/2 for small angles
        cos_ang = 1.0 - 0.5 * dist * dist
        if cos_ang >= cos_r and not truth_hit[j]:
            det_hit[i] = True
            truth_hit[j] = True

    return det_hit, truth_hit


DEFAULT_SNR_BIN_EDGES: tuple[float, ...] = (5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, np.inf)
# SZiFi-style completeness bins in fixed-mode true SNR q-bar_t (include below q_th).
DEFAULT_QTRUE_BIN_EDGES: tuple[float, ...] = (
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    10.0,
    12.0,
    15.0,
    20.0,
    30.0,
    np.inf,
)


@dataclass
class MatchResult:
    """Cross-match between detections and truth (truth already footprint-cut)."""

    det_lon: np.ndarray
    det_lat: np.ndarray
    det_q: np.ndarray
    det_theta: np.ndarray
    det_hit: np.ndarray
    truth_q: np.ndarray
    truth_hit: np.ndarray
    truth_q_label: str = "q_from_aperture"


@dataclass
class SnrBinnedBenchmark:
    """Completeness in truth-SNR bins; purity vs cumulative detection threshold."""

    bin_edges: list[float]
    bin_centers: list[float]
    completeness: list[float]
    completeness_err: list[float]
    completeness_n: list[int]
    purity: list[float]
    purity_err: list[float]
    purity_n: list[int]
    purity_thresholds: list[float]
    completeness_erf: list[float] | None = None
    truth_q_label: str = "q_from_aperture"
    q_th_obs: float = 5.0


def _binomial_err(p: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    return float(np.sqrt(max(p * (1.0 - p) / n, 0.0)))


def erf_completeness(
    q_true: np.ndarray | float,
    *,
    q_th: float = 5.0,
    opt_bias: bool = True,
) -> np.ndarray:
    """SZiFi / Planck analytic completeness (see ``szifi.cat.get_erf_completeness``).

    Blind optimisation bias: \(\\bar{q}\\to\\sqrt{\\bar{q}^2+3}\) when ``opt_bias``.
    """
    from scipy import special as sp

    q = np.asarray(q_true, dtype=np.float64)
    if opt_bias:
        q = np.sqrt(np.maximum(q, 0.0) ** 2 + 3.0)
    return 0.5 * (1.0 - sp.erf((q_th - q) / np.sqrt(2.0)))


def run_match(
    cat_path: Path,
    *,
    truth_csv: Path = DEFAULT_TRUTH_CATALOGUE,
    paths: SZiFiPaths | None = None,
    q_th_obs: float = 5.0,
    q_th_truth: float = 5.0,
    z_max: float = 1.0,
    match_radius_arcmin: float = 10.0,
    use_theta_500: bool = False,
    true_snr: dict[str, np.ndarray] | None = None,
    apply_footprint_to_detections: bool = True,
) -> tuple[MatchResult, dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Load catalogues, cross-match; return match arrays + truth table + mask info.

    If ``true_snr`` is provided (fixed-mode catalogue with ``q_true_mmf``), that
    sample is used as the truth parent and ``truth_q`` is the MMF true SNR.

    By default detections are cut to the same PR4 GAL×PS unmasked footprint as
    truth (Zubeldia-style purity inside the survey/cosmology mask). Set
    ``apply_footprint_to_detections=False`` only for diagnostics.
    """
    paths = paths or SZiFiPaths()
    mask, nside = load_planck_unmasked_mask(paths)
    if true_snr is not None:
        truth = {
            "lon_rot_deg": np.asarray(true_snr["lon"]),
            "lat_rot_deg": np.asarray(true_snr["lat"]),
            "theta_500_arcmin": np.asarray(true_snr["theta_500"]),
            "q_from_aperture": np.asarray(true_snr["q_from_aperture"]),
            "q_true_mmf": np.asarray(true_snr["q_true_mmf"]),
            "z": np.asarray(true_snr["z"]),
            "detectable": np.asarray(true_snr["q_true_mmf"]) >= q_th_truth,
        }
        truth_q = truth["q_true_mmf"]
        truth_q_label = "q_true_mmf"
        truth_lon = truth["lon_rot_deg"]
        truth_lat = truth["lat_rot_deg"]
        truth_theta = truth["theta_500_arcmin"]
    else:
        truth = load_truth_qfrommap(
            truth_csv, paths, z_max=z_max, q_th_truth=q_th_truth, mask=mask, nside=nside
        )
        truth_q = truth["q_from_aperture"]
        truth_q_label = "q_from_aperture"
        truth_lon = truth["lon_rot_deg"]
        truth_lat = truth["lat_rot_deg"]
        truth_theta = truth["theta_500_arcmin"]

    det = load_detection_catalogue(cat_path)
    det_q_ok = detection_q_mask(det, q_th_obs)
    n_in_masked = int(
        (det_q_ok & ~unmasked_at_lonlat(det["lon"], det["lat"], mask, nside)).sum()
    )
    if apply_footprint_to_detections:
        det_ok = detection_in_footprint_mask(det, mask, nside, q_th_obs)
    else:
        det_ok = det_q_ok
    det_lon = det["lon"][det_ok]
    det_lat = det["lat"][det_ok]
    det_q = det["q_opt"][det_ok]
    det_theta = det["theta_500"][det_ok]
    det_hit, truth_hit = cross_match_greedy(
        det_lon,
        det_lat,
        det_q,
        det_theta,
        truth_lon,
        truth_lat,
        truth_theta,
        match_radius_arcmin=match_radius_arcmin,
        use_theta_500=use_theta_500,
    )
    match = MatchResult(
        det_lon=det_lon,
        det_lat=det_lat,
        det_q=det_q,
        det_theta=det_theta,
        det_hit=det_hit,
        truth_q=truth_q,
        truth_hit=truth_hit,
        truth_q_label=truth_q_label,
    )
    return match, truth, det, n_in_masked


def benchmark_snr_bins(
    match: MatchResult,
    *,
    bin_edges: tuple[float, ...] | None = None,
    q_th_truth: float = 5.0,
    q_th_obs: float = 5.0,
    restrict_truth_to_qth: bool = True,
    erf_opt_bias: bool = True,
) -> SnrBinnedBenchmark:
    """Bin completeness by truth SNR; purity vs cumulative q_opt threshold.

    For SZiFi-style completeness (``truth_q_label='q_true_mmf'``), set
    ``restrict_truth_to_qth=False`` and pass ``DEFAULT_QTRUE_BIN_EDGES`` so bins
    below ``q_th`` show the selection-function rise; overlay is the ERF model.
    """
    if bin_edges is None:
        bin_edges = (
            DEFAULT_QTRUE_BIN_EDGES
            if match.truth_q_label == "q_true_mmf"
            else DEFAULT_SNR_BIN_EDGES
        )
    edges = np.asarray(bin_edges, dtype=np.float64)
    n_bin = len(edges) - 1
    centers: list[float] = []
    comp, comp_err, comp_n = [], [], []
    pur, pur_err, pur_n, pur_th = [], [], [], []
    erf_vals: list[float] = []

    truth_q = match.truth_q
    parent = (truth_q >= q_th_truth) if restrict_truth_to_qth else np.ones_like(truth_q, dtype=bool)
    for i in range(n_bin):
        lo, hi = edges[i], edges[i + 1]
        if np.isfinite(hi):
            t_in = parent & (truth_q >= lo) & (truth_q < hi)
            centers.append(float(0.5 * (lo + hi)))
        else:
            t_in = parent & (truth_q >= lo)
            centers.append(float(lo * 1.15))

        nt = int(t_in.sum())
        kt = int(match.truth_hit[t_in].sum())
        # SZiFi-style: recovered if matched AND detection q_opt >= q_th_obs.
        # With catalogue already cut at q_th_obs, match == recovery.
        pc = kt / nt if nt else float("nan")
        comp.append(pc)
        comp_err.append(_binomial_err(pc, nt))
        comp_n.append(nt)
        erf_vals.append(
            float(erf_completeness(centers[-1], q_th=q_th_obs, opt_bias=erf_opt_bias))
            if nt
            else float("nan")
        )

    # Purity uses detection-SNR thresholds (>= q_th_obs), independent of truth bins.
    pur_edges = np.asarray(
        [e for e in DEFAULT_SNR_BIN_EDGES if e >= q_th_obs or not np.isfinite(e)],
        dtype=np.float64,
    )
    for lo in pur_edges[:-1]:
        sel = match.det_q >= lo
        nd = int(sel.sum())
        kd = int(match.det_hit[sel].sum())
        pp = kd / nd if nd else float("nan")
        pur.append(pp)
        pur_err.append(_binomial_err(pp, nd))
        pur_n.append(nd)
        pur_th.append(float(lo))

    return SnrBinnedBenchmark(
        bin_edges=[float(e) if np.isfinite(e) else None for e in edges],
        bin_centers=centers,
        completeness=comp,
        completeness_err=comp_err,
        completeness_n=comp_n,
        purity=pur,
        purity_err=pur_err,
        purity_n=pur_n,
        purity_thresholds=pur_th,
        completeness_erf=erf_vals,
        truth_q_label=match.truth_q_label,
        q_th_obs=q_th_obs,
    )


def benchmark_catalogue(
    cat_path: Path,
    *,
    truth_csv: Path = DEFAULT_TRUTH_CATALOGUE,
    paths: SZiFiPaths | None = None,
    q_th_obs: float = 5.0,
    q_th_truth: float = 5.0,
    z_max: float = 1.0,
    match_radius_arcmin: float = 10.0,
    use_theta_500: bool = False,
    true_snr: dict[str, np.ndarray] | None = None,
    apply_footprint_to_detections: bool = True,
) -> BenchmarkResult:
    """Cross-match detections to truth; return purity and completeness.

    With ``true_snr`` (SZiFi fixed-mode), ``completeness_detectable`` is the
    cumulative recovery fraction among truth with \(\\bar{q}_t\\ge q_th_truth\).
    Detections default to the same GAL×PS footprint as truth (purity in-mask).
    """
    match, truth, _det, n_in_masked = run_match(
        cat_path,
        truth_csv=truth_csv,
        paths=paths,
        q_th_obs=q_th_obs,
        q_th_truth=q_th_truth,
        z_max=z_max,
        match_radius_arcmin=match_radius_arcmin,
        use_theta_500=use_theta_500,
        true_snr=true_snr,
        apply_footprint_to_detections=apply_footprint_to_detections,
    )

    n_det = len(match.det_q)
    n_truth_all = len(truth["z"])
    det_mask = truth["detectable"]
    n_truth_det = int(det_mask.sum())
    n_tp = int(match.det_hit.sum())
    n_fp = n_det - n_tp
    # Undetected among the detectable parent (SZiFi cumulative completeness).
    n_miss = int((det_mask & ~match.truth_hit).sum()) if n_truth_det else int((~match.truth_hit).sum())
    n_tp_det = int((det_mask & match.truth_hit).sum())

    purity = n_tp / n_det if n_det else float("nan")
    comp_all = n_tp / n_truth_all if n_truth_all else float("nan")
    comp_det = n_tp_det / n_truth_det if n_truth_det else float("nan")
    n_catalogue = n_det + n_in_masked if apply_footprint_to_detections else n_det

    return BenchmarkResult(
        catalogue=str(cat_path),
        truth_catalogue=str(truth_csv) if true_snr is None else "true_snr_fixed_mmf",
        n_detected=n_det,
        n_detected_catalogue=n_catalogue,
        n_truth_all=n_truth_all,
        n_truth_detectable=n_truth_det,
        n_true_positives=n_tp,
        n_false_positives=n_fp,
        n_undetected=n_miss,
        purity=purity,
        completeness_all=comp_all,
        completeness_detectable=comp_det,
        match_radius_arcmin=match_radius_arcmin,
        q_th_obs=q_th_obs,
        q_th_truth=q_th_truth,
        z_max=z_max,
        footprint=(
            "planck_gal_x_ps_unmasked_det+truth"
            if apply_footprint_to_detections
            else "planck_gal_x_ps_unmasked_truth_only"
        ),
        n_detected_excluded_mask=n_in_masked,
    )


def write_benchmark_json(result: BenchmarkResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), indent=2) + "\n")


def load_benchmark_json(path: Path) -> BenchmarkResult:
    data = json.loads(path.read_text())
    data.setdefault("footprint", "planck_gal_x_ps_unmasked")
    data.setdefault("n_detected_excluded_mask", 0)
    data.setdefault("n_detected_catalogue", data.get("n_detected", 0))
    data.pop("min_ftile", None)
    return BenchmarkResult(**data)


def load_snr_bins_json(path: Path) -> SnrBinnedBenchmark:
    data = json.loads(path.read_text())
    data.setdefault("purity_thresholds", data.get("bin_edges", [])[:-1])
    data.setdefault("completeness_erf", None)
    data.setdefault("truth_q_label", "q_from_aperture")
    data.setdefault("q_th_obs", 5.0)
    return SnrBinnedBenchmark(**data)


def match_detection_flags(
    cat_path: Path,
    *,
    truth_csv: Path = DEFAULT_TRUTH_CATALOGUE,
    paths: SZiFiPaths | None = None,
    q_th_obs: float = 5.0,
    q_th_truth: float = 5.0,
    z_max: float = 1.0,
    match_radius_arcmin: float = 10.0,
    use_theta_500: bool = False,
    apply_footprint_to_detections: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Return (catalogue, det_ok mask, det_matched_bool on det_ok rows)."""
    match, _truth, det, _ = run_match(
        cat_path,
        truth_csv=truth_csv,
        paths=paths,
        q_th_obs=q_th_obs,
        q_th_truth=q_th_truth,
        z_max=z_max,
        match_radius_arcmin=match_radius_arcmin,
        use_theta_500=use_theta_500,
        apply_footprint_to_detections=apply_footprint_to_detections,
    )
    paths = paths or SZiFiPaths()
    mask, nside = load_planck_unmasked_mask(paths)
    if apply_footprint_to_detections:
        det_ok = detection_in_footprint_mask(det, mask, nside, q_th_obs)
    else:
        det_ok = detection_q_mask(det, q_th_obs)
    return det, det_ok, match.det_hit



def write_snr_bins_json(binned: SnrBinnedBenchmark, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(binned), indent=2) + "\n")
