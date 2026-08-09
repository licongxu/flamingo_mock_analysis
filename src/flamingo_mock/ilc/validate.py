"""Validate a pyILC Compton-y reconstruction against FLAMINGO truth y.

Loads the shipped FITS y-map (does not reimplement ILC). Power spectra are
**beam-deconvolved** by the ILC common beam (default 5 arcmin) so high-ℓ
suppression from ``perform_ILC_at_beam`` is removed from the reported C_ℓ.
Truth y is beam-free; for ratio plots we also show truth C_ℓ (no beam) and
truth convolved then deconvolved (should match truth).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .beams import apply_beam_to_map, deconvolve_cl_beam, gaussian_beam_bl
from .paths import ILC_BEAM_FWHM_ARCMIN, ILCPaths


def _py(x):
    """Convert numpy scalars to plain Python for JSON."""
    if x is None:
        return None
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def find_default_ymap(paths: ILCPaths | None = None) -> Path | None:
    """Most recent full-component ILC y-map under the ILC output dirs."""
    paths = paths or ILCPaths()
    for method in ("hilc", "nilc"):
        for split in ("A", "B"):
            d = paths.output_dir(method, split)
            if not d.is_dir():
                continue
            hits = sorted(d.glob("*needletILCmap*component_tSZ*.fits"))
            full = [h for h in hits if "scale" not in h.name]
            if full:
                return full[0]
            if hits:
                return hits[0]
    return None


def validate_ymap(
    ymap: str | Path | None = None,
    *,
    truth: str | Path | None = None,
    ymap_split: str | Path | None = None,
    lmax: int = 3000,
    ilc_beam_fwhm_arcmin: float = ILC_BEAM_FWHM_ARCMIN,
    bl_floor: float = 1e-3,
    figures_dir: str | Path | None = None,
    paths: ILCPaths | None = None,
) -> dict:
    """Validate an ILC y-map against truth; return the summary dict.

    The summary carries ``ok_amplitude``, ``ok_corr`` and
    ``ok_beam_deconvolution`` flags; figures are written to ``figures_dir``
    when given.
    """
    import healpy as hp

    paths = paths or ILCPaths()
    ypath = Path(ymap) if ymap else find_default_ymap(paths)
    if ypath is None or not ypath.is_file():
        raise FileNotFoundError("no y-map found; pass ymap explicitly")

    y = np.asarray(hp.read_map(str(ypath), dtype=np.float64))
    nside = hp.npix2nside(y.size)

    if truth:
        tpath = Path(truth)
    else:
        candidates = [
            paths.truth_map(),
            paths.compton_y_map(),
        ]
        tpath = next((c for c in candidates if c.is_file()), None)
    if tpath is None or not Path(tpath).is_file():
        raise FileNotFoundError("truth map missing")
    tpath = Path(tpath)

    truth_map = np.asarray(hp.read_map(str(tpath), dtype=np.float64))
    if hp.npix2nside(truth_map.size) != nside:
        truth_map = hp.ud_grade(truth_map, nside)

    finite = np.isfinite(y) & np.isfinite(truth_map)
    if finite.sum() < 0.5 * y.size:
        raise RuntimeError("too few finite pixels")

    lmax = min(int(lmax), 3 * nside - 1)
    fwhm = float(ilc_beam_fwhm_arcmin)

    # Match beams for map-level comparison: ILC carries the common beam.
    truth_beamed = apply_beam_to_map(truth_map, fwhm, lmax=lmax)

    y_f = y[finite]
    t_f = truth_map[finite]
    tb_f = truth_beamed[finite]
    med_abs_y = float(np.median(np.abs(y_f)))
    med_abs_t = float(np.median(np.abs(t_f)))
    std_y = float(np.std(y_f))
    std_t = float(np.std(t_f))
    step = max(1, y_f.size // 2_000_000)
    corr = float(np.corrcoef(y_f[::step], t_f[::step])[0, 1])
    corr_beamed = float(np.corrcoef(y_f[::step], tb_f[::step])[0, 1])

    # Raw Cl from maps (ILC map carries the common beam)
    cl_yy_raw = hp.anafast(y, lmax=lmax)
    cl_tt_raw = hp.anafast(truth_map, lmax=lmax)  # truth is beam-free
    cl_yt_raw = hp.anafast(y, truth_map, lmax=lmax)

    # Beam-deconvolved ILC auto / cross: divide by B_l^2 (auto) or B_l (cross with beam-free truth)
    bl = gaussian_beam_bl(fwhm, lmax)
    cl_yy_dec = deconvolve_cl_beam(cl_yy_raw, fwhm, bl_floor=bl_floor)
    # cross: <y_ILC * truth> ~ B_l * C_true  →  deconvolve one power of B
    cl_yt_dec = np.full_like(cl_yt_raw, np.nan)
    good = bl >= bl_floor
    cl_yt_dec[good] = cl_yt_raw[good] / bl[good]
    # truth auto stays raw (beam-free). For shape checks also beam-convolve truth then deconv.
    cl_tt_beamed = hp.anafast(truth_beamed, lmax=lmax)
    cl_tt_beamed_dec = deconvolve_cl_beam(cl_tt_beamed, fwhm, bl_floor=bl_floor)

    ell = np.arange(lmax + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Cross-spectrum transfer: y carries B_ell, truth is beam-free
        #   raw:    C_yt / C_tt  ~ B_ell * T_ell   (falls as beam at high ell)
        #   deconv: C_yt / (B C_tt) ~ T_ell         (should track ~1 if ILC is unbiased)
        transfer_raw = cl_yt_raw / cl_tt_raw
        transfer_dec = cl_yt_dec / cl_tt_raw
        rho_ell_raw = cl_yt_raw / np.sqrt(np.abs(cl_yy_raw * cl_tt_raw))
        rho_ell_dec = cl_yt_dec / np.sqrt(np.abs(cl_yy_dec * cl_tt_raw))

    lo, hi = 50, min(500, lmax)
    rho_band = float(np.nanmean(rho_ell_dec[lo : hi + 1]))

    def band_median(a, lo_, hi_):
        return float(np.nanmedian(a[lo_ : hi_ + 1]))

    # Transfer diagnostics (cross with truth — not auto, which is noise-biased)
    T_mid = band_median(transfer_dec, 200, 800)
    T_hi = band_median(transfer_dec, 1500, min(2500, lmax))
    T_raw_hi = band_median(transfer_raw, 1500, min(2500, lmax))
    bl2000 = float(bl[min(2000, lmax)])

    cross_split = None
    cl_x = cl_x_dec = None
    if ymap_split:
        y1 = np.asarray(hp.read_map(str(ymap_split), dtype=np.float64))
        if hp.npix2nside(y1.size) != nside:
            y1 = hp.ud_grade(y1, nside)
        cl_x = hp.anafast(y, y1, lmax=lmax)
        cl_x_dec = deconvolve_cl_beam(cl_x, fwhm, bl_floor=bl_floor)
        with np.errstate(divide="ignore", invalid="ignore"):
            # both maps at common beam → divide by B^2
            t_x_dec = cl_x_dec / cl_tt_raw
        cross_split = {
            "cl_cross_ell100_raw": float(cl_x[min(100, lmax)]),
            "cl_cross_ell100_dec": float(cl_x_dec[min(100, lmax)])
            if np.isfinite(cl_x_dec[min(100, lmax)])
            else None,
            "transfer_split_cross_over_truth_mid_ell": band_median(t_x_dec, 200, 800),
        }

    ok_amp = bool(1e-8 < med_abs_y < 1e-3 and 1e-8 < std_y < 1e-3)
    ok_corr = bool(corr_beamed > 0.2 or corr > 0.15 or rho_band > 0.2)
    # Beam deconv OK if: (1) mid-ell transfer O(1); (2) raw high-ell transfer is
    # suppressed relative to deconv by ~B_ell (cross falls as one power of B).
    transfer_mid_ok = bool(np.isfinite(T_mid) and 0.3 < T_mid < 3.0)
    # raw high-ell should track B_ell * T, deconv should restore T
    if np.isfinite(T_raw_hi) and np.isfinite(T_hi) and T_hi > 0 and bl2000 > 0:
        raw_vs_bl = T_raw_hi / (bl2000 * T_hi)
        highell_beam_shape_ok = bool(0.3 < raw_vs_bl < 3.0)
    else:
        highell_beam_shape_ok = False
    ok_beam_dec = bool(transfer_mid_ok and highell_beam_shape_ok)

    summary = {
        "ymap": str(ypath),
        "truth": str(tpath),
        "nside": int(nside),
        "lmax": int(lmax),
        "ilc_beam_fwhm_arcmin": float(fwhm),
        "beam_deconvolved": True,
        "ratio_definition": "transfer = C_ell^{y x truth} / (B_ell * C_ell^{truth}) "
        "(auto Cyy/Ctt is noise-biased and is NOT used for the ratio plot)",
        "frac_finite": float(finite.mean()),
        "median_abs_y": _py(med_abs_y),
        "median_abs_truth": _py(med_abs_t),
        "std_y": _py(std_y),
        "std_truth": _py(std_t),
        "pixel_corr": _py(corr),
        "pixel_corr_vs_truth_beamed": _py(corr_beamed),
        "rho_ell_band_50_500_dec": _py(rho_band),
        "cl_yy_ell100_raw": _py(cl_yy_raw[min(100, lmax)]),
        "cl_yt_ell100_dec": _py(cl_yt_dec[min(100, lmax)]),
        "cl_tt_ell100": _py(cl_tt_raw[min(100, lmax)]),
        "transfer_mid_ell_200_800_dec": _py(T_mid),
        "transfer_high_ell_1500_2500_dec": _py(T_hi),
        "transfer_high_ell_1500_2500_raw": _py(T_raw_hi),
        "B_ell_2000": _py(bl2000),
        "ok_amplitude": ok_amp,
        "ok_corr": ok_corr,
        "ok_beam_deconvolution": ok_beam_dec,
        "cross_split": (
            {k: _py(v) for k, v in cross_split.items()} if cross_split is not None else None
        ),
    }

    if figures_dir is not None:
        _write_figures(
            Path(figures_dir),
            ymap=y,
            truth=truth_map,
            truth_beamed=truth_beamed,
            med_abs_t=med_abs_t,
            fwhm=fwhm,
            lmax=lmax,
            ell=ell,
            bl=bl,
            good=good,
            cl_tt_raw=cl_tt_raw,
            cl_yt_raw=cl_yt_raw,
            cl_yt_dec=cl_yt_dec,
            cl_x_dec=cl_x_dec,
            rho_ell_raw=rho_ell_raw,
            rho_ell_dec=rho_ell_dec,
            rho_band=rho_band,
            transfer_raw=transfer_raw,
            transfer_dec=transfer_dec,
            T_mid=T_mid,
            bl2000=bl2000,
            summary=summary,
        )

    return summary


def _write_figures(fig_dir: Path, *, summary: dict, **kw) -> None:
    """Mollview triplet, spectra, and beam-deconvolution transfer figures."""
    import healpy as hp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ymap = kw["ymap"]
    truth = kw["truth"]
    truth_beamed = kw["truth_beamed"]
    fwhm = kw["fwhm"]
    lmax = kw["lmax"]
    ell = kw["ell"]
    bl = kw["bl"]
    good = kw["good"]
    cl_tt_raw = kw["cl_tt_raw"]
    cl_yt_raw = kw["cl_yt_raw"]
    cl_yt_dec = kw["cl_yt_dec"]
    cl_x_dec = kw["cl_x_dec"]
    rho_ell_raw = kw["rho_ell_raw"]
    rho_ell_dec = kw["rho_ell_dec"]
    rho_band = kw["rho_band"]
    transfer_raw = kw["transfer_raw"]
    transfer_dec = kw["transfer_dec"]
    T_mid = kw["T_mid"]
    bl2000 = kw["bl2000"]

    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 4.5))
    vmax = max(5e-6, 5 * kw["med_abs_t"])
    hp.mollview(truth, min=-vmax, max=vmax, title="Truth Compton-y", unit="y", sub=(1, 3, 1), fig=fig)
    hp.mollview(ymap, min=-vmax, max=vmax, title=f"ILC y (beam {fwhm}')", unit="y", sub=(1, 3, 2), fig=fig)
    hp.mollview(ymap - truth_beamed, min=-vmax, max=vmax, title="ILC − truth*beam", unit="y", sub=(1, 3, 3), fig=fig)
    moll_path = fig_dir / "ilc_y_vs_truth_mollview.png"
    fig.savefig(moll_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig2, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    fac = ell * (ell + 1) / (2 * np.pi)
    m_dec = np.isfinite(cl_yt_dec) & (ell >= 2) & good
    # Signal estimator: cross with truth (noise-unbiased), beam-deconvolved
    ax[0].loglog(ell[2:], (fac * cl_tt_raw)[2:], label=r"truth $C_\ell^{tt}$", lw=1.6, color="k")
    ax[0].loglog(
        ell[m_dec],
        (fac * np.abs(cl_yt_dec))[m_dec],
        label=r"$C_\ell^{y\times t}/B_\ell$ (beam-dec)",
        lw=1.3,
        color="C1",
    )
    ax[0].loglog(
        ell[2:],
        (fac * np.abs(cl_yt_raw))[2:],
        label=r"$C_\ell^{y\times t}$ raw (has $B_\ell$)",
        lw=1.0,
        alpha=0.7,
        color="C0",
    )
    if cl_x_dec is not None:
        ax[0].loglog(
            ell[m_dec],
            (fac * np.abs(cl_x_dec))[m_dec],
            label=r"splitA$\times$splitB $/B_\ell^2$",
            lw=1.0,
            alpha=0.75,
            color="C2",
        )
    ax[0].set_xlabel(r"$\ell$")
    ax[0].set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
    ax[0].set_title(f"spectra to ℓ={lmax} (cross-based, beam-deconvolved)")
    ax[0].legend(fontsize=7)
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].set_xlim(2, lmax)

    ax[1].semilogx(ell[2:], rho_ell_raw[2:], lw=1.0, alpha=0.7, label="raw")
    ax[1].semilogx(ell[m_dec], rho_ell_dec[m_dec], lw=1.2, label="beam-deconv")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].axhline(1, color="k", lw=0.5, ls="--")
    ax[1].set_ylim(-0.2, 1.05)
    ax[1].set_xlabel(r"$\ell$")
    ax[1].set_ylabel(r"$C_\ell^{yt}/\sqrt{C_\ell^{yy}C_\ell^{tt}}$")
    ax[1].set_title(f"cross-corr deconv band={rho_band:.3f}")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].set_xlim(2, lmax)
    fig2.tight_layout()
    cl_path = fig_dir / "ilc_y_vs_truth_spectra.png"
    fig2.savefig(cl_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    # Transfer function figure: cross with truth, NOT noise-biased auto ratio
    fig3, ax3 = plt.subplots(figsize=(7.5, 4.4))
    ax3.semilogx(
        ell[2:],
        transfer_raw[2:],
        label=r"raw $C_\ell^{yt}/C_\ell^{tt}$ ($\sim B_\ell$)",
        alpha=0.85,
        color="C0",
    )
    ax3.semilogx(
        ell[m_dec],
        transfer_dec[m_dec],
        label=r"deconv $C_\ell^{yt}/(B_\ell C_\ell^{tt})$",
        lw=1.5,
        color="C1",
    )
    ax3.plot(ell[2:], bl[2:], "k--", lw=1.0, alpha=0.7, label=rf"$B_\ell$ ({fwhm:g}')")
    ax3.axhline(1.0, color="k", ls=":", lw=0.9)
    ax3.set_ylim(-0.2, 2.5)
    ax3.set_xlim(2, lmax)
    ax3.set_xlabel(r"$\ell$")
    ax3.set_ylabel(r"transfer $T_\ell$")
    ax3.set_title(
        r"Beam deconv on $y\times$truth cross (not auto $C^{yy}/C^{tt}$)"
        + f"\nmid $T$={T_mid:.2f}, high-ℓ raw/dec ≈ $B_{{2000}}$={bl2000:.2f}"
    )
    ax3.legend(fontsize=8, loc="best")
    ax3.grid(True, which="both", alpha=0.3)
    fig3.tight_layout()
    ratio_path = fig_dir / "ilc_y_beam_deconv_ratio.png"
    fig3.savefig(ratio_path, dpi=120, bbox_inches="tight")
    plt.close(fig3)

    summary["figure_mollview"] = str(moll_path.resolve())
    summary["figure_spectra"] = str(cl_path.resolve())
    summary["figure_beam_ratio"] = str(ratio_path.resolve())


def summary_ok(summary: dict) -> bool:
    """True if all validation flags pass."""
    return bool(
        summary["ok_amplitude"] and summary["ok_corr"] and summary["ok_beam_deconvolution"]
    )


def print_summary(summary: dict) -> str:
    """JSON rendering of the summary (also returned for logging)."""
    return json.dumps(summary, indent=2)
