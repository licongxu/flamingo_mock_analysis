# HILC deprojection follow-up (2026-09-08)

Why the \(M_\ast-1\sigma\) CIB+\(\delta\beta\) ratio is ~0.72 at \(\ell=10\) while no-deproj stays ~1.0, and whether that ratio is a tSZ response.

**It is not a tSZ response.** \(M_\ast-1\sigma\) Compton-\(y\) is slightly *higher* (rms ×1.02). No-deproj HILC is 1.006. Full-sky CIB+\(\delta\beta\) HILC is **1.031**. The 0.72 is the masked spectrum of a noisy constrained-ILC map, not a gas-pressure decrement.

Branch: `cib_deproj_diagonosis`. HILC runs used **CPU ducc** (`PYILC_BACKEND=numpy`, `ilc_backend: numpy`, `sht_backend: ducc0`, `CUDA_VISIBLE_DEVICES=`). Log: `logs/hilc_followup_ducc_cpu.log`.

Ratios are \(D_\ell^{\mathrm{case}}/D_\ell^{\mathrm{L1\_m9}}\) for the same deprojection, Planck-18 bin at \(\ell=10\) (and \(\ell=40\) in parentheses).

## Conclusion

The published 0.72 is \(D_\ell\) of the **original \(q>5\) CIB+\(\delta\beta\) y-maps**, each analysed with **its own** cluster-hole mask.

CIB+\(\delta\beta\) uses two extra ILC constraints (preserve tSZ, null model CIB and \(\delta\beta\)). With six channels that leaves a noisy leftover mode. On the **masked** \(q>5\) spectra that leftover is 3–4× true tSZ at \(\ell\le 40\). On the **full sky** it is not: at \(\ell=40\), full-sky \(\delta\beta\) \(10^{12}D=0.0589\) vs no-deproj 0.0569.

What the leftover is, from cross-spectra of the \(q>5\) r1 maps at \(\ell=10\) (apodised fiducial mask, anafast/\(f_{\mathrm{sky}}\)):

| pair | \(r\) | notes |
|---|---|---|
| \(\delta\beta\) × truth \(y\) | 0.32 | ~10% of power is tSZ |
| \(\delta\beta\) × CMB | 0.07 | adding CMB deproj leaves \(D_\ell\) unchanged (0.0023 → 0.0022) |
| \(\delta\beta\) × CIB at every \(\nu\) | 0.20 | model CIB SED is already nulled |
| fiducial \(\delta\beta\) × \(M_\ast\) \(\delta\beta\) | **0.98** | same field, slightly different amplitude |
| r1×r2 split-cross vs auto | consistent | extra is common to the splits, not white noise |
| CIB shuffle \(\delta\beta\) ratio | ~1 | not leftover tSZ–CIB spatial correlation |

The leftover amplitude tracks CIB, not pressure. \(M_\ast\) CIB \(D_\ell(\ell=10)\) is **0.68** of fiducial (amplitude ~0.82; all-scale rms was ×0.84). That is the same size as 0.72.

The **0.72 rather than ~1** is then produced by measuring this noisy map through cluster holes — including two *different* hole sets. Same original \(q>5\) y-maps, only the analysis mask changes (anafast r1×r2):

| analysis | no deproj \(\ell=10\) (40) | CIB+\(\delta\beta\) \(\ell=10\) (40) |
|---|---|---|
| no holes | 1.033 (1.008) | **0.92 (1.03)** |
| both on fiducial \(q>5\) mask | 0.996 (1.005) | 0.83 (0.82) |
| own masks (the plot) | 1.006 (1.003) | **0.74 (0.84)** |
| full-sky HILC (`*_total_p18.npz`) | 1.048 (1.038) | **1.031 (1.043)** |

No-deproj barely moves. \(\delta\beta\) masked \(D_\ell\) at \(\ell=40\) *rises* relative to the unmasked analysis of the same maps (0.0063 → 0.016): the extra is hole-coupled small-scale constrained-ILC power, not more \(y\). Fiducial holes: 7.21M pixels; \(M_\ast\): 7.30M; overlap 13.7% of the sky.

Pixel check: \(q>5\) vs full-sky ILC agree **in the survey** (nodeproj \(r=0.99\), \(\delta\beta\) \(r=0.96\)) and disagree **in the holes** (\(r=0.16\)). Full-sky ILC holes are brighter (real clusters). \(q>5\) ILC holes are harmonic inpainting, not Compton-\(y\).

Hybrids (fid tSZ + \(M_\ast\) CIB and the reverse) break the tSZ–CIB coupling the same way a CIB shuffle does; they are on disk but are **not** used to interpret 0.72. Re-running HILC with a shared covariance mask is a different y-map and is also not the explanation above — the table uses the **original** \(q>5\) products and only changes how \(D_\ell\) is measured, except the last row (full-sky HILC).

**Paper:** do not read the CIB+\(\delta\beta\) \(M_\ast\) curve on the existing ratio figure as tSZ feedback. Use no-deproj, or full-sky \(\delta\beta\) (both ~1 and slightly above 1).

## What was built

| Exp | What | Scripts / products |
|---|---|---|
| 0 | Shared CMB seed 42 and homog noise r1/r2 | `tests/test_shared_cmb_noise_seeds.py` |
| 1 | Full component budget (tSZ, CIB, CMB, noise, all crosses) | `scripts/compute_hilc_component_budget.py` → `ps_namaster/{L1_m9,Mstar-1sigma}_*_masked_budget.npz` |
| 1 plot | Stacked budget vs measured \(D_\ell\) | `figures/l1_m9/l1_m9_hilc_component_budget.{png,pdf}` |
| 2 | L1_m9 \(\delta\beta\) weights on \(M_\ast\) ILC inputs (and reverse) | `scripts/compute_hilc_fixed_weights_inputs.py` → `ps_namaster/fixedw_inputs_deproj_cib_dbeta_masked.npz` |
| 2 plot | Fixed-weight ratio | `figures/l1_m9/l1_m9_hilc_feedback_ratio_fixed_weights.{png,pdf}` |
| weights | Five schemes × L1_m9 / \(M_\ast\) / \(f_{\mathrm{gas}}\), \(q>5\) | `figures/l1_m9/l1_m9_hilc_weights_q5_schemes.{png,pdf}` |
| 3 | Hybrid skies (not used for the conclusion) | `scripts/build_hybrid_tSZ_CIB_totals.py`, `run_hilc_followup.py` |
| 4 | \(M_\ast\) and \(f_{\mathrm{gas}}\) HILC with fiducial cluster mask (`extra_tag=fidmask`) | same runner; does **not** overwrite own-mask outputs |
| 5 | HILC nodeproj and \(\delta\beta\) on `L1_m9_cibshuffle` | same runner |
| 3–5 plot | Follow-up ratios | `figures/l1_m9/l1_m9_hilc_followup_ratios.{png,pdf}` |

Shared path helpers: `scripts/hilc_prescriptions.py` (`mask_name`, `extra_tag`, follow-up run names). YAML writer emits `ilc_backend: numpy`. `scripts/run_hilc.py` defaults `PYILC_BACKEND` to `numpy`.

## How to rerun

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
cd /scratch/scratch-lxu/flamingo_mock_analysis
export CUDA_VISIBLE_DEVICES="" PYILC_BACKEND=numpy

python -m pytest tests/test_shared_cmb_noise_seeds.py tests/test_hilc_residual_products.py tests/test_hilc_prescriptions.py -q

python scripts/compute_hilc_component_budget.py
python scripts/compute_hilc_fixed_weights_inputs.py
python scripts/plot_hilc_component_budget.py
python scripts/plot_hilc_fixed_weights_ratio.py
python scripts/plot_hilc_weights_fullsky_vs_q5masked.py   # also writes the q>5 scheme grid

python scripts/build_hybrid_tSZ_CIB_totals.py
for rx in hybrid_fidtSZ_MstarCIB hybrid_MstartSZ_fidCIB L1_m9_cibshuffle; do
  python scripts/build_homog_r2_test_maps.py --prescription "$rx"
done
python scripts/run_hilc_followup.py          # skips y-maps already on disk
python scripts/compute_hilc_followup_ps.py
python scripts/plot_hilc_followup_ratios.py
```

## Exp 0 — shared CMB and noise

L1_m9, \(f_{\mathrm{gas}}-8\sigma\), and \(M_\ast-1\sigma\) are pixel-identical in CMB (seed 42). Homogeneous noise files are shared; FITS `SEED` headers are `42+\nu` (r1) and `1042+\nu` (r2). LS8 CMB is allowed to differ. `total_r2-(total_r1-n1)` recovers `n2` to float32 rounding (857 GHz is large).

Hydro ICs are also shared (same 1 Gpc L1_m9 box / lightcone0 / `same_rot`). Maps are not identical: vs L1_m9, \(M_\ast\) has \(r_y=0.999\), \(r_{\mathrm{CIB}\,353}=0.992\), \(y\) rms ×1.02, CIB rms ×0.84.

A CIB+\(\delta\beta\) ratio \(\neq 1\) is not a different CMB, noise, or initial-condition draw. \(M_\ast\) *should* enhance tSZ; no-deproj and full-sky \(\delta\beta\) do.

## Exp 1 — component budget

Weights from disk, applied with the pyILC 10′ rebeam + taper, then the same NaMaster 18-bin workspace as the measured r1×r2 \(D_\ell\). Algebra identity \(\sum\) terms = recon holds to \(\sim 10^{-15}\).

L1_m9, \(q>5\), \(10^{12}D_\ell\) at \(\ell=40\):

| scheme | measured | \(\sum\) | tSZ | CIB | tSZ×CIB | \(\lvert\sum-\mathrm{meas}\rvert/\mathrm{meas}\) |
|---|---|---|---|---|---|---|
| no deproj | 0.0064 | 0.0062 | 0.0075 | 0.0045 | **−0.0061** | 2.4% |
| CIB | 0.0064 | 0.0063 | 0.0075 | 0.0014 | −0.0030 | 2.0% |
| CIB+\(\delta\beta\) | **0.0233** | **0.0067** | 0.0076 | 0.0005 | −0.0006 | **71%** |
| CIB+\(\delta\beta\)+CMB | 0.0234 | 0.0066 | 0.0076 | 0.0005 | −0.0006 | 72% |
| CIB+\(\delta\beta\)+\(\delta T\) | 0.0025 | −0.0046 | 0.0076 | 0.0008 | +0.0008 | does not close |

No-deproj (and CIB-only) **close**. The old incomplete budget looked open because it omitted the large negative tSZ×CIB.

CIB+\(\delta\beta\) **does not close**: saved weights × components or ILC inputs give recon ≈ tSZ, while the measured masked \(\delta\beta\) \(D_\ell\) is several times larger. That gap is the leftover constrained-ILC mode (plus hole coupling into the MASTER bins). It is not recovered as weighted CIB with those saved weights. Noise r1×r2 is consistent with zero (\(\lvert D_n\rvert < 3\sigma_G\) on all five schemes).

At \(\ell=10\) the moments (+\(\delta T\)) scheme is *larger* than \(\delta\beta\) (0.0051 vs 0.0023), so \(\delta T\) deprojection does not remove the \(\ell=10\) excess. Adding CMB deprojection does nothing.

Tests: `test_budget_closes` (algebra all schemes; match-to-y for nodeproj and CIB only), `test_noise_cross_consistent_with_zero`.

## Exp 2 — fixed weights

Apply L1_m9 CIB+\(\delta\beta\) weights to \(M_\ast\) ILC inputs (fiducial mask), and the reverse.

At \(\ell=10\), relative to fiducial inputs + fiducial weights:

| | ratio |
|---|---|
| \(M_\ast\) maps, \(M_\ast\) weights | 0.993 |
| \(M_\ast\) maps, **fiducial** weights | **1.029** |
| fiducial maps, \(M_\ast\) weights | 0.963 |
| measured y-map \(M_\ast/\)fid (own masks) | **0.716** |

The 0.72 is **not** this linear weight×map product. Fiducial weights on the \(M_\ast\) sky give ~1. The operator also fails to reproduce the fiducial \(\delta\beta\) y-map (33% at \(\ell=10\), 71% at \(\ell=40\)), consistent with Exp 1: the leftover is not in the saved-weight reconstruction.

## Weights vs \(\ell\)

`l1_m9_hilc_weights_q5_schemes`: 5 schemes × 6 channels, L1_m9 / \(M_\ast\) / \(f_{\mathrm{gas}}\) on the \(q>5\) sky. 217 GHz takes most of the weight under \(\delta\beta\) (~0.69 fiducial vs ~0.63 \(M_\ast\) at \(\ell=10\)). \(\sum_\nu w_\nu \sim -3\times 10^{-4}\), so CMB leak from \(\sum w\) is negligible. Hydro variants track the fiducial curves; the 217 GHz shift does not produce 0.72 (Exp 2).

## Exp 5 — CIB shuffle

L1_m9 CIB tile-permuted; tSZ/CMB/noise and the fiducial \(q>5\) mask fixed.

| | no deproj \(\ell=10\) (40) | CIB+\(\delta\beta\) \(\ell=10\) (40) |
|---|---|---|
| L1_m9 CIB shuffle | **1.81 (1.32)** | 0.98 (1.01) |

No-deproj jumps once tSZ–CIB spatial correlation is destroyed (negative tSZ×CIB in Exp 1). \(\delta\beta\) stays ~1: the 0.72 is not leftover tSZ-correlated CIB in the \(\delta\beta\) map.

## Exp 4 HILC reruns (different y-maps)

These recompute ILC with the fiducial cluster mask in the YAML (`extra_tag=fidmask`). That is a **different** y-map from the original own-mask products. Do not read “own 0.72 → fidmask 0.85” as a fraction of the original 0.72. The conclusion table above holds the original y-maps fixed.

NaMaster of those reruns, for the record:

| case | no deproj \(\ell=10\) (40) | CIB+\(\delta\beta\) \(\ell=10\) (40) |
|---|---|---|
| \(M_\ast\) own mask (original) | 1.006 (1.002) | **0.716 (0.825)** |
| \(M_\ast\) fiducial-mask HILC | 1.063 (1.029) | 0.847 (0.823) |
| \(f_{\mathrm{gas}}\) own mask | 1.230 (1.113) | 1.069 (1.116) |
| \(f_{\mathrm{gas}}\) fiducial-mask HILC | 1.024 (0.975) | 0.977 (0.983) |

## Tests

```text
tests/test_shared_cmb_noise_seeds.py          # Exp 0
tests/test_hilc_residual_products.py          # budget algebra + noise cross
tests/test_hilc_prescriptions.py              # fidmask YAML does not clobber own-mask outputs
```

`test_l1_m9_fiducial_catalogue_has_2509_detections` looks for `szifi_homog/L1_m9/catalogues/fullsky_splitA_immf_q5.npz`, which is not the current catalogue filename; that failure is unrelated to this follow-up.
