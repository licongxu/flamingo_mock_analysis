#!/usr/bin/env python3
"""Validate a pyILC Compton-y reconstruction against FLAMINGO truth y.

Loads the shipped FITS y-map (does not reimplement ILC). Power spectra are
**beam-deconvolved** by the ILC common beam (default 5 arcmin) so high-ℓ
suppression from perform_ILC_at_beam is removed from the reported C_ℓ.
Truth y is beam-free; for ratio plots we also show truth C_ℓ (no beam) and
optionally truth convolved then deconvolved (should match truth).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# local helper
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from beam_utils import apply_beam_to_map, deconvolve_cl_beam, gaussian_beam_bl  # noqa: E402


def find_default_ymap() -> Path | None:
    roots = [
        Path("/home/ext_andyxlcnb_gmail_com/cosmology_data/flamingo_ilc/hilc_output_noise_split0"),
        Path("/home/ext_andyxlcnb_gmail_com/cosmology_data/flamingo_ilc/hilc_output_noise_split1"),
        Path("/home/ext_andyxlcnb_gmail_com/cosmology_data/flamingo_ilc/nilc_output_noise_split0"),
    ]
    for d in roots:
        if not d.is_dir():
            continue
        hits = sorted(d.glob("*needletILCmap*component_tSZ*.fits"))
        # prefer full component map over per-scale
        full = [h for h in hits if "scale" not in h.name]
        if full:
            return full[0]
        if hits:
            return hits[0]
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate ILC y-map vs truth (beam-deconvolved Cl)")
    p.add_argument("--ymap", type=str, default=None, help="ILC y-map FITS path")
    p.add_argument("--truth", type=str, default=None, help="Truth Compton-y FITS")
    p.add_argument("--figures-dir", type=str, default="figures")
    p.add_argument("--log", type=str, default=None)
    p.add_argument("--lmax", type=int, default=3000, help="Maximum multipole for Cl")
    p.add_argument(
        "--ilc-beam-fwhm-arcmin",
        type=float,
        default=5.0,
        help="Common ILC beam FWHM used in perform_ILC_at_beam (deconvolved from Cl)",
    )
    p.add_argument(
        "--bl-floor",
        type=float,
        default=1e-3,
        help="B_ell floor for safe deconvolution (NaN above where B_ell < floor)",
    )
    p.add_argument(
        "--ymap-split1",
        type=str,
        default=None,
        help="Optional second-split ILC y-map for cross-split C_ell",
    )
    args = p.parse_args(argv)

    ypath = Path(args.ymap) if args.ymap else find_default_ymap()
    if ypath is None or not ypath.is_file():
        print("ERROR: no y-map found; pass --ymap", file=sys.stderr)
        return 2

    ymap = np.asarray(hp.read_map(str(ypath), dtype=np.float64))
    nside = hp.npix2nside(len(ymap))
    if args.truth:
        tpath = Path(args.truth)
    else:
        candidates = [
            Path(
                f"/home/ext_andyxlcnb_gmail_com/cosmology_data/flamingo_ilc/"
                f"inputs_nside{nside}_noise/compton_y_nside{nside}.fits"
            ),
            Path(
                "/home/ext_andyxlcnb_gmail_com/cosmology_data/flamingo_ilc/"
                "inputs_nside2048_noise/compton_y_nside2048.fits"
            ),
            Path(
                "/home/ext_andyxlcnb_gmail_com/flamingo_mock_analysis/"
                "maps_100_143_353/raw/compton_y_nside4096.fits"
            ),
        ]
        tpath = next((c for c in candidates if c.is_file()), None)
    if tpath is None or not Path(tpath).is_file():
        print("ERROR: truth map missing", file=sys.stderr)
        return 2
    tpath = Path(tpath)

    truth = np.asarray(hp.read_map(str(tpath), dtype=np.float64))
    if hp.npix2nside(len(truth)) != nside:
        truth = hp.ud_grade(truth, nside)

    finite = np.isfinite(ymap) & np.isfinite(truth)
    if finite.sum() < 0.5 * len(ymap):
        print("ERROR: too few finite pixels", file=sys.stderr)
        return 1

    lmax = min(int(args.lmax), 3 * nside - 1)
    fwhm = float(args.ilc_beam_fwhm_arcmin)

    # Match beams for map-level comparison: ILC carries common beam.
    truth_beamed = apply_beam_to_map(truth, fwhm, lmax=lmax)

    y_f = ymap[finite]
    t_f = truth[finite]
    tb_f = truth_beamed[finite]
    med_abs_y = float(np.median(np.abs(y_f)))
    med_abs_t = float(np.median(np.abs(t_f)))
    std_y = float(np.std(y_f))
    std_t = float(np.std(t_f))
    step = max(1, len(y_f) // 2_000_000)
    corr = float(np.corrcoef(y_f[::step], t_f[::step])[0, 1])
    corr_beamed = float(np.corrcoef(y_f[::step], tb_f[::step])[0, 1])

    # Raw Cl from maps (ILC map carries common beam)
    cl_yy_raw = hp.anafast(ymap, lmax=lmax)
    cl_tt_raw = hp.anafast(truth, lmax=lmax)  # truth is beam-free
    cl_yt_raw = hp.anafast(ymap, truth, lmax=lmax)

    # Beam-deconvolved ILC auto / cross: divide by B_l^2 (auto) or B_l (cross with beam-free truth)
    bl = gaussian_beam_bl(fwhm, lmax)
    cl_yy_dec = deconvolve_cl_beam(cl_yy_raw, fwhm, bl_floor=args.bl_floor)
    # cross: <y_ILC * truth> ~ B_l * C_true  →  deconvolve one power of B
    cl_yt_dec = np.full_like(cl_yt_raw, np.nan)
    good = bl >= args.bl_floor
    cl_yt_dec[good] = cl_yt_raw[good] / bl[good]
    # truth auto stays raw (beam-free). For shape checks also beam-convolve truth then deconv.
    cl_tt_beamed = hp.anafast(truth_beamed, lmax=lmax)
    cl_tt_beamed_dec = deconvolve_cl_beam(cl_tt_beamed, fwhm, bl_floor=args.bl_floor)

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
    if args.ymap_split1:
        y1 = np.asarray(hp.read_map(args.ymap_split1, dtype=np.float64))
        if hp.npix2nside(len(y1)) != nside:
            y1 = hp.ud_grade(y1, nside)
        cl_x = hp.anafast(ymap, y1, lmax=lmax)
        cl_x_dec = deconvolve_cl_beam(cl_x, fwhm, bl_floor=args.bl_floor)
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

    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 4.5))
    vmax = max(5e-6, 5 * med_abs_t)
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
            label=r"split0$\times$split1 $/B_\ell^2$",
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

    text = json.dumps(summary, indent=2)
    print(text)
    if args.log:
        logp = Path(args.log)
        logp.parent.mkdir(parents=True, exist_ok=True)
        logp.write_text(text + "\n")
        print(f"wrote log {logp}")

    if not ok_amp:
        print("FAIL: y-map amplitude not in Compton-y range", file=sys.stderr)
        return 1
    if not ok_corr:
        print("FAIL: correlation with truth too low", file=sys.stderr)
        return 1
    if not ok_beam_dec:
        print("FAIL: beam-deconvolved Cl ratio unhealthy", file=sys.stderr)
        return 1
    print("PASS: y-map amplitude, correlation, and beam-deconvolved Cl OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
