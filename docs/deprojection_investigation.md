# Deprojection investigation: FLAMINGO mock HILC/NILC vs McCarthy & Hill (pyILC)

**Date:** 2026-08-09  
**Scope:** Constrained-ILC (“deprojection”) options for Compton-*y* on the FLAMINGO synthetic Planck-like sky.  
**References:** McCarthy & Hill (2024) `reference_paper/pyilc.pdf` (arXiv:2307.01043); installed pyILC `agent_evolve` branch; tracked configs under `configs/`.

---

## 1. Current pipeline status: **no deprojection**

All tracked production YAMLs are **unconstrained / nodeproj** baselines:

| Config | `ILC_preserved_comp` | `N_deproj` | `ILC_deproj_comps` |
|--------|----------------------|------------|--------------------|
| `configs/hilc_y_flamingo_npipe_splitA.yml` | `tSZ` | **0** | `[]` |
| `configs/hilc_y_flamingo_npipe_splitB.yml` | `tSZ` | **0** | `[]` |
| `configs/nilc_y_flamingo_npipe_splitA.yml` | `tSZ` | **0** | `[]` |

The package writer (`src/flamingo_mock/ilc/config.py`) previously hard-coded `N_deproj: 0` and `ILC_deproj_comps: []`. Default baseline remains nodeproj; deprojection is now optional via `ILCConfig(n_deproj=…, deproj_comps=…)`.

**Contrast with the paper sample configs:**

| Sample YAML | `N_deproj` | `ILC_deproj_comps` |
|-------------|------------|--------------------|
| `sample_run_Planck_tsz_nodeproj.yml` | 0 | `[]` |
| `sample_run_Planck_tsz_deproj.yml` | **2** | `['CIB', 'CIB_dbeta']` |

Paper abstract (pyILC): maps are constructed with various constrained ILC (“deprojection”) options to minimize CIB contamination, including **moment-based** deprojection of the CIB SED.

Evidence: `{SCRATCH}/nodeproj_status.log`.

---

## 2. What pyILC supports

From `pyilc.input.COMP_TYPES` (installed package):

```
CMB, kSZ, tSZ, rSZ, mu, CIB, CIB_dbeta, CIB_dT, radio, radio_dbeta, radio2
```

YAML keys (parsed in `pyilc/input.py`):

- `ILC_preserved_comp` — component with unit response (here always `tSZ` / Compton-*y*).
- `N_deproj` — integer (or per-needlet-scale list).
- `ILC_deproj_comps` — list of length `N_deproj` of names from `COMP_TYPES`.
- Constraint: `N_deproj + 1 ≤ N_freqs` (one preserve + deprojections).
- SED parameters for CIB/moments come from `param_dict_file` (`fg_SEDs_default_params.yml`: default `Tdust_CIB=24 K`, `beta_CIB=1.2`, pivot 353 GHz).

**Mechanics (paper §III.C):** constrained ILC enforces \(w\cdot a_{\rm tSZ}=1\) and \(w\cdot a_{\rm fg}=0\) for each deprojected SED. This **nulls** that foreground at the cost of **higher variance** (noise penalty). Multi-component deprojection uses an \((N_{\rm deproj}+1)\times(N_{\rm deproj}+1)\) constraint matrix \(Q\).

Moment deprojections:

- `CIB` — modified blackbody SED (β, T).
- `CIB_dbeta` — ∂/∂β of CIB MBB (reduces sensitivity to β).
- `CIB_dT` — ∂/∂T of CIB MBB.

Paper Figs. 9–12 show that adding deprojections **raises** auto-power (noise) while **suppressing** residual CIB in \(C_\ell^{y\times857}\).

Dry-parse of deproj YAMLs on our 6-HFI paths: all of  
`nodeproj`, `CMB`, `CIB`, `CIB+CMB`, `CIB+CIB_dbeta`, `CIB+CIB_dbeta+CIB_dT`  
parse successfully via `ILCInfo` — see `{SCRATCH}/deproj_config_parse.log`.

---

## 3. Relevance on the FLAMINGO mock sky

Mock components (`data_description.md`): **CMB + tSZ + kSZ + CIB** only. **No** Galactic dust, synchrotron, free-free, or radio point sources in the signal model (masks only for sky cuts).

| Deproj component | On mock sky? | Recommended for y? | Notes |
|------------------|--------------|--------------------|--------|
| **CIB** | Yes (strong at 353–857 GHz) | **Primary candidate** | Main residual bias for *y* after multi-freq ILC; paper’s focus |
| **CIB_dbeta** | Yes (if CIB SED ≠ assumed MBB) | Useful follow-up | Mock CIB uses Yang 3-param greybody (β_d=1.65, T0=35.14 K, z_eff); pyILC default β=1.2, T=24 K — **SED mismatch** |
| **CIB_dT** | Yes | Heavier follow-up | Same SED-mismatch caveat; large noise penalty (paper Figs. 9–10) |
| **CMB** | Yes | Optional | Blackbody; unconstrained ILC already down-weights CMB somewhat; paper finds CMB deproj ≈10% noise change on real data |
| **kSZ** | Yes | Rarely for *y* | kSZ is blackbody-like (same SED as CMB in µK_CMB); deprojecting CMB nulls both CMB and kSZ in thermo units |
| **rSZ, mu** | No (not in mock) | N/A | Relativistic SZ / chemical potential not injected |
| **radio / radio_dbeta / radio2** | No (not in signal model) | N/A | Only if adding radio PS later |

**Channel budget:** 6 HFI → at most `N_deproj = 5`. High-resolution needlet scales drop low-res channels (beam criterion), so aggressive multi-deproj may only be possible on large scales (paper “CMB5” = CMB deproj on first five needlet scales only).

---

## 4. Recommendation for this pipeline

### Baseline (current)
Keep **`N_deproj: 0`** as the default tracked configs for HILC A/B and NILC A:

- Clean recovery vs **known** truth *y* is the primary validation metric.
- Unconstrained ILC already achieves mid-ℓ transfer \(T\sim0.85\)–0.91 with 6 HFI + GAL×PS mask.
- Matches paper’s “no deprojection” control maps (Figs. 6–9 blue curves).

### Next science-grade steps (when residual CIB in *y* is the target)
1. **NILC, `N_deproj=1`, `ILC_deproj_comps: [CIB]`** — first constrained run; closest to “standard CIB deprojection” in the paper. Expect **higher** auto-\(C_\ell^{yy}\) (noise) and **lower** correlation with the mock CIB map.
2. **NILC, `N_deproj=2`, `[CIB, CMB]`** — null CMB (and kSZ in µK_CMB); paper-style for cross-correlations.
3. **NILC, `N_deproj=2`, `[CIB, CIB_dbeta]`** — paper sample `sample_run_Planck_tsz_deproj.yml`; only after aligning CIB SED params in `fg_SEDs_default_params.yml` with the mock (or using mock-matched β, T).
4. **HILC** with the same deproj sets for fast A/B cross-spectra (HILC ~20–30 s vs NILC ~12 min).

### Cost / noise tradeoff (from paper)
- Each added deprojection uses one more degree of freedom → **noise penalty** grows roughly with \(|1+N_{\rm deproj}-N_{\rm freq}|/N_{\rm modes}\).
- Moment deproj (CIB+δβ+δT) can raise low-ℓ power by **orders of magnitude** without 857 GHz (Fig. 11); with 545/857 the penalty is smaller but still large (Fig. 10).
- For mock validation vs truth *y*, unconstrained is preferred; for **CIB-clean** *y* maps for LSS cross-correlations, use CIB (±moments) deproj.

### YAML change for a follow-up run
```yaml
ILC_preserved_comp: tSZ
N_deproj: 1
ILC_deproj_comps: [CIB]
# or: N_deproj: 2, ILC_deproj_comps: [CIB, CIB_dbeta]
```
Package API:
```python
ILCConfig(method="nilc", split="A", n_deproj=1, deproj_comps=("CIB",))
```

---

## 5. Optional trial run (executed as evidence)

- **Method:** HILC split A, **CIB deprojected** (`N_deproj=1`, `ILC_deproj_comps=[CIB]`).
- **Config:** `configs/hilc_y_flamingo_npipe_splitA_deproj_CIB.yml`
- **Why HILC not full NILC matrix:** HILC is ~30× faster; sufficient to prove constrained-ILC path and compare noise/transfer to nodeproj baseline. Full NILC deproj matrix is a non-goal (plan §Non-goals).
- **Result:** SUCCESS in 20.5 s; y-map written under `hilc_output_npipe_splitA_deproj_CIB/`.
- **Validation (vs truth, GAL×PS, 10′ deconv):**

| metric | nodeproj HILC A | deproj CIB HILC A |
|--------|-----------------|-------------------|
| `std_y` | 1.807e-06 | 1.838e-06 |
| pixel corr (beamed truth) | 0.473 | 0.496 |
| mid-ℓ transfer | 0.909 | 0.960 |
| `cl_yy` @ ℓ=100 (raw) | 8.927e-17 | 8.947e-17 |
| summary_ok | True | True |

- Noise penalty: deproj auto-power / std_y is higher than nodeproj (as expected from constrained ILC).
- Figure: `figures/hilc_nodeproj_vs_deproj_CIB_power_spectrum.png`
- Logs: `{SCRATCH}/run_hilc_deproj_CIB.log`, `validate_hilc_deproj_CIB.json`.

---

## 6. Conclusions

1. **We had not investigated deprojection operationally** — all shipped runs used `N_deproj: 0`. That matches the paper’s *nodeproj* control, not the paper’s science *deproj* suite.
2. pyILC already implements constrained ILC for CMB/CIB/moments/kSZ/…; wiring is YAML-only.
3. On FLAMINGO mocks, **CIB deprojection** is the physically motivated first step; moment deproj needs SED alignment with the mock CIB.
4. **Recommendation:** keep nodeproj as default validation baseline; add optional CIB (then CIB+CMB / CIB+δβ) deproj configs for residual-CIB studies; prefer NILC for final maps, HILC for quick A×B tests.

---

## 7. Full paper-style deprojection suite (executed)

HILC split A, all McCarthy & Hill Fig. 9-style combinations (preserve tSZ):

| Label | `N_deproj` | `ILC_deproj_comps` |
|-------|------------|--------------------|
| No deprojection | 0 | `[]` |
| CMB | 1 | `[CMB]` |
| CIB | 1 | `[CIB]` |
| CIB + CMB | 2 | `[CIB, CMB]` |
| CIB + δβ | 2 | `[CIB, CIB_dbeta]` |
| CIB + δβ + CMB | 3 | `[CIB, CIB_dbeta, CMB]` |
| CIB + δβ + δT | 3 | `[CIB, CIB_dbeta, CIB_dT]` |
| CIB + δβ + δT + CMB | 4 | `[CIB, CIB_dbeta, CIB_dT, CMB]` |

Configs: `configs/deproj_suite/hilc_splitA_*.yml`  
Metrics: `docs/deproj_suite_metrics.md`  
Figures (paper-style scale):
- `figures/deproj_suite_hilc_auto_power.png` — auto \(C_\ell^{yy}\) (noise penalty)
- `figures/deproj_suite_hilc_cross_truth.png` — × truth (signal recovery)
- `figures/deproj_suite_hilc_transfer.png` — transfer functions

**Findings on the mock:** mild CIB deproj improves transfer/corr slightly; moment deproj (δβ+δT) incurs a large noise penalty (std_y ×2–3) and lower correlation, matching paper qualitative behaviour. Full NILC deproj matrix not re-run (HILC suite covers the paper combination set; NILC nodeproj remains the needlet baseline).
