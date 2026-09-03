"""HILC r1×r2 plots for each hydro/cosmology variant (no deprojection).

Same layout as the L1_m9 fiducial Fig. 9 scripts: truth y, split-cross,
CIB/CMB/noise residuals, ILC-bias curve. Full sky and q>5-masked (each
prescription uses its own SZiFi catalogue mask). Writes under figures/hilc/.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flamingo_mock.powerspectra import (
    compute_cl,
    dl_from_cl,
    ilc_bias_fraction,
    n_modes_tophat_hilc,
    sigma_dl_cross_binned,
)

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from hilc_prescriptions import (  # noqa: E402
    ALL_RUNS,
    LABELS,
    REPO,
    catalogue_path,
    cib_dir,
    cluster_mask_apo,
    cmb_path,
    hilc_output_dir,
    hilc_ymap,
    tsz_dir,
)

_spec = importlib.util.spec_from_file_location(
    "hilc_r1xr2_diag", _SCRIPTS / "plot_hilc_homog_r1xr2_split_diagnostics.py"
)
diag = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["hilc_r1xr2_diag"] = diag
_spec.loader.exec_module(diag)

LMAX = diag.LMAX
NSIDE = diag.NSIDE
N_FREQ = 6
HILC_BINSIZE = 50
ELL_PLOT_MAX = diag.ELL_PLOT_MAX
ELL_MIN, ELL_MAX, ELL_EFF = diag.ELL_MIN, diag.ELL_MAX, diag.ELL_EFF
FREQS = diag.FREQS
YLIM = (1.0e-17, 3.0e-9)
FIG_ROOT = REPO / "figures" / "hilc"


def _load_uk_to_k(path: Path) -> np.ndarray:
    m = diag.load_map(path) * 1e-6
    return m - np.mean(m)


def signal_alms(name: str) -> dict[str, tuple]:
    print(f"map2alm CMB / tSZ / CIB ({name}) ...", flush=True)
    cmb = _load_uk_to_k(cmb_path(name))
    tsz = [_load_uk_to_k(tsz_dir(name) / f"tSZ_deltaT_{f}GHz_nside4096.fits") for f in FREQS]
    cib = [_load_uk_to_k(cib_dir(name) / f"CIB_deltaT_{f}GHz_nside4096.fits") for f in FREQS]
    return {
        "cmb": (hp.map2alm(cmb, lmax=LMAX, iter=0),),
        "tsz": tuple(diag._map2alm_list(tsz)),
        "cib": tuple(diag._map2alm_list(cib)),
    }


def masked_cl(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    a0 = (a - np.sum(w * a) / np.sum(w)) * w
    b0 = (b - np.sum(w * b) / np.sum(w)) * w
    return hp.anafast(a0, b0, lmax=LMAX, iter=0) / float(np.mean(w**2))


def alm_pair_to_cl(y1, y2, w: np.ndarray | None) -> np.ndarray:
    if w is None:
        return hp.alm2cl(y1, y2)
    m1 = hp.alm2map(y1, nside=NSIDE, lmax=LMAX)
    m2 = hp.alm2map(y2, nside=NSIDE, lmax=LMAX)
    return masked_cl(m1, m2, w)


def load_pack(name: str, *, masked: bool, sig: dict, bl, good, w_apo) -> dict:
    y1p = hilc_ymap(name, masked=masked, real=1)
    y2p = hilc_ymap(name, masked=masked, real=2)
    wdir1 = hilc_output_dir(name, masked=masked, real=1)
    wdir2 = hilc_output_dir(name, masked=masked, real=2)
    cache_p = wdir2 / f"hilc_{name}_{'q5masked' if masked else 'fullsky'}_r1xr2_cl.npz"
    stored: dict = {}
    if cache_p.is_file():
        z = np.load(cache_p)
        stored = {k: z[k] for k in z.files}
        print("loaded", cache_p)

    fsky = float(np.mean(w_apo**2)) if w_apo is not None else 1.0
    need_yy = any(k not in stored for k in ("cl_11", "cl_22", "cl_12", "cl_tt"))
    if need_yy:
        y1, y2 = diag.load_map(y1p), diag.load_map(y2p)
        yt = diag.load_map(tsz_dir(name) / "compton_y_nside4096.fits")
        if w_apo is None:
            stored["cl_11"] = compute_cl(y1, lmax=LMAX, deconv_pixel_window=False)
            stored["cl_22"] = compute_cl(y2, lmax=LMAX, deconv_pixel_window=False)
            stored["cl_12"] = compute_cl(y1, y2, lmax=LMAX, deconv_pixel_window=False)
            stored["cl_tt"] = compute_cl(yt, lmax=LMAX, deconv_pixel_window=False)
        else:
            stored["cl_11"] = masked_cl(y1, y1, w_apo)
            stored["cl_22"] = masked_cl(y2, y2, w_apo)
            stored["cl_12"] = masked_cl(y1, y2, w_apo)
            stored["cl_tt"] = masked_cl(yt, yt, w_apo)
        del y1, y2, yt

    need_res = any(k not in stored for k in ("cl_cib_w", "cl_cmb_w", "cl_n_w"))
    if need_res:
        print(f"  weighted residuals {name} {'q5masked' if masked else 'fullsky'} ...", flush=True)
        w1 = diag.hilc_weights(wdir1, "", LMAX)
        w2 = diag.hilc_weights(wdir2, "", LMAX)
        y_cib1, y_cib2 = diag.y_alms_signal(w1, w2, sig["cib"], same_all_freq=False)
        y_cmb1, y_cmb2 = diag.y_alms_signal(w1, w2, sig["cmb"], same_all_freq=True)
        y_n1, y_n2 = diag.y_alms_noise(w1, w2)
        stored["cl_cib_w"] = alm_pair_to_cl(y_cib1, y_cib2, w_apo)
        stored["cl_cmb_w"] = alm_pair_to_cl(y_cmb1, y_cmb2, w_apo)
        stored["cl_n_w"] = alm_pair_to_cl(y_n1, y_n2, w_apo)
        del y_cib1, y_cib2, y_cmb1, y_cmb2, y_n1, y_n2

    cache_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_p, **stored)
    n_modes = n_modes_tophat_hilc(LMAX, HILC_BINSIZE, fsky)
    frac = ilc_bias_fraction(0, N_FREQ, n_modes)
    cl_tt = np.asarray(stored["cl_tt"], dtype=np.float64)
    ells = np.arange(cl_tt.size, dtype=np.float64)
    pack = {
        "cl_tt": cl_tt,
        "cl_12_d": diag.deconv_auto(stored["cl_12"], bl, good),
        "cl_11_d": diag.deconv_auto(stored["cl_11"], bl, good),
        "cl_22_d": diag.deconv_auto(stored["cl_22"], bl, good),
        "cl_cib_d": diag.deconv_auto(stored["cl_cib_w"], bl, good),
        "cl_cmb_d": diag.deconv_auto(stored["cl_cmb_w"], bl, good),
        "cl_n_d": diag.deconv_auto(stored["cl_n_w"], bl, good),
        "fsky": fsky,
        "d_bias": frac * np.abs(dl_from_cl(ells, cl_tt)),
    }
    pack["dl_cross"] = diag.bin_mean_dl(pack["cl_12_d"])
    pack["dl_cib"] = diag.bin_mean_dl(pack["cl_cib_d"])
    pack["dl_cmb"] = diag.bin_mean_dl(pack["cl_cmb_d"])
    pack["dl_n"] = diag.bin_mean_dl(pack["cl_n_d"])
    pack["dl_cross_sigma"] = sigma_dl_cross_binned(
        pack["cl_11_d"], pack["cl_22_d"], pack["cl_12_d"], ELL_MIN, ELL_MAX, fsky
    )
    return pack


def _curve_and_bins(ax, ells, sl, cl, dl, *, color, marker, label, lw=1.3):
    ax.plot(ells[sl], np.abs(dl_from_cl(ells, cl))[sl], color=color, lw=lw, alpha=0.55, zorder=2)
    mag = np.abs(dl)
    pos = np.asarray(dl) >= 0
    ax.plot(ELL_EFF[pos], mag[pos], marker, color=color, ms=5.0, ls="none", zorder=4, label=label)
    if np.any(~pos):
        ax.plot(
            ELL_EFF[~pos], mag[~pos], marker, color=color, ms=5.5, ls="none",
            mfc="none", mew=1.5, zorder=4,
        )


def plot_fig9(rows: list[tuple[str, dict]], *, masked: bool, out: Path) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(8.6, 3.15 * n), sharex=True)
    if n == 1:
        axes = [axes]
    ells = np.arange(LMAX + 1, dtype=np.float64)
    sl = slice(2, ELL_PLOT_MAX + 1)
    for ax, (name, d) in zip(axes, rows):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ELL_EFF[0], ELL_PLOT_MAX)
        ax.set_ylim(*YLIM)
        ax.plot(ells[sl], np.abs(dl_from_cl(ells, d["cl_tt"]))[sl], color="k", lw=1.7, label=r"input truth $y$", zorder=3)
        ax.plot(ells[sl], d["d_bias"][sl], color="crimson", lw=1.6, ls="--", zorder=5, label=r"$|\Delta C_\ell^{y_1 y_2}|$")
        ax.plot(ells[sl], np.abs(dl_from_cl(ells, d["cl_12_d"]))[sl], color="C1", lw=1.8, alpha=0.45, zorder=2)
        y = np.abs(d["dl_cross"])
        sig = np.asarray(d["dl_cross_sigma"], dtype=np.float64)
        lo = np.maximum(y - sig, y * 1.0e-3)
        ax.fill_between(ELL_EFF, lo, y + sig, color="C1", alpha=0.28, zorder=1, lw=0)
        ax.errorbar(
            ELL_EFF, y, yerr=[y - lo, sig], fmt="o", color="C1", ms=6.5,
            elinewidth=2.2, capsize=4.5, capthick=1.8, zorder=6,
            label=r"HILC $y$ $r_1\times r_2$",
        )
        _curve_and_bins(ax, ells, sl, d["cl_cib_d"], d["dl_cib"], color="C2", marker="^", label=r"CIB residual")
        _curve_and_bins(ax, ells, sl, d["cl_cmb_d"], d["dl_cmb"], color="C4", marker="v", label=r"CMB residual")
        _curve_and_bins(ax, ells, sl, d["cl_n_d"], d["dl_n"], color="0.35", marker="+", label=r"noise residual", lw=1.0)
        ax.set_ylabel(r"$D_\ell$")
        nq = int(np.load(catalogue_path(name))["q_opt"].size)
        hole = rf"$q>5$ holes ($N={nq}$)" if masked else "full sky"
        ax.set_title(f"{LABELS[name]}  ({hole}, $N_\\mathrm{{deproj}}=0$)")
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    axes[-1].set_xlabel(r"$\ell$")
    sky = r"$q>5$ cluster holes" if masked else "full sky"
    fig.suptitle(
        rf"HILC $y$ $r_1\times r_2$ ({sky}, independent noise, shared CMB+tSZ+CIB)"
        "\n"
        r"lines: unbinned $D_\ell$; points: Planck 2015 XXII bins",
        y=1.01, fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_y_vs_truth(name: str, *, masked: bool, pack: dict, out: Path) -> None:
    ells = np.arange(LMAX + 1, dtype=np.float64)
    sl = slice(2, ELL_PLOT_MAX + 1)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(ells[sl], np.abs(dl_from_cl(ells, pack["cl_tt"]))[sl], color="k", lw=1.8, label=r"truth $y$")
    ax.plot(ells[sl], np.abs(dl_from_cl(ells, pack["cl_12_d"]))[sl], color="C1", lw=1.6, label=r"HILC $r_1\times r_2$")
    ax.set_xlim(ELL_EFF[0], ELL_PLOT_MAX)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell$")
    sky = r"$q>5$ masked" if masked else "full sky"
    ax.set_title(f"{LABELS[name]} HILC $y$ vs truth ({sky})")
    ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_compare(rows: list[tuple[str, dict]], *, masked: bool, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ells = np.arange(LMAX + 1, dtype=np.float64)
    sl = slice(2, ELL_PLOT_MAX + 1)
    colors = {"L1_m9": "k", "fgas-8sigma": "C0", "Mstar-1sigma": "C3", "LS8": "C2"}
    for name, d in rows:
        ax.plot(
            ells[sl], np.abs(dl_from_cl(ells, d["cl_tt"]))[sl],
            color=colors[name], lw=1.0, ls=":", alpha=0.7,
        )
        ax.plot(
            ELL_EFF, np.abs(d["dl_cross"]), color=colors[name], lw=1.8,
            marker="o", ms=4.5, label=LABELS[name],
        )
    ax.set_xlim(ELL_EFF[0], ELL_PLOT_MAX)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell$")
    sky = r"$q>5$ masked" if masked else "full sky"
    ax.set_title(rf"HILC $y$ $r_1\times r_2$ ({sky})")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    names = list(ALL_RUNS)
    bl, good = diag.bl10(), diag.bl10() >= 1e-3
    full_rows = []
    mask_rows = []
    for name in names:
        sig = signal_alms(name)
        pack_f = load_pack(name, masked=False, sig=sig, bl=bl, good=good, w_apo=None)
        w_apo = diag.load_map(cluster_mask_apo(name))
        pack_m = load_pack(name, masked=True, sig=sig, bl=bl, good=good, w_apo=w_apo)
        sub = FIG_ROOT / name
        plot_fig9([(name, pack_f)], masked=False, out=sub / "r1xr2_fig9_fullsky.png")
        plot_fig9([(name, pack_m)], masked=True, out=sub / "r1xr2_fig9_q5masked.png")
        plot_y_vs_truth(name, masked=False, pack=pack_f, out=sub / "y_vs_truth_fullsky.png")
        plot_y_vs_truth(name, masked=True, pack=pack_m, out=sub / "y_vs_truth_q5masked.png")
        full_rows.append((name, pack_f))
        mask_rows.append((name, pack_m))
        del sig
    plot_fig9(full_rows, masked=False, out=FIG_ROOT / "r1xr2_fig9_fullsky_all.png")
    plot_fig9(mask_rows, masked=True, out=FIG_ROOT / "r1xr2_fig9_q5masked_all.png")
    plot_compare(full_rows, masked=False, out=FIG_ROOT / "r1xr2_compare_fullsky.png")
    plot_compare(mask_rows, masked=True, out=FIG_ROOT / "r1xr2_compare_q5masked.png")


if __name__ == "__main__":
    main()
