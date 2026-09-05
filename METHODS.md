# METHODS — MMF cluster detection and Compton-*y* reconstruction on FLAMINGO mock skies

Working notes for the paper (`tsz_cnc_paper_plots/6a4738926d5991d919e1a0c8/main.tex`,
`\subsubsection{Cluster detection and Compton-y reconstruction}`).

Everything below is read out of the code and products in this repo
(`/scratch/scratch-lxu/flamingo_mock_analysis`, branch `szifi_branch`) plus the on-disk
products under `/rds/rds-lxu/flamingo/integrated_maps_synthetic/`. Numbers marked
**[fiducial]** are the ones behind the current FLAMINGO work; anything else is a
variant/diagnostic. Every value carries a `file:line` pointer so it can be re-checked.

> **Two different things are called *q*.** Easiest way to get the text wrong:
> - `q_opt` — the **true MMF detection significance** from the SZiFi run
>   (*ŷ*₀/σ_ŷ₀, blind peak finding). §3–4.
> - `q_from_aperture` (a.k.a. *q*_map) — an **empirical aperture significance** assigned to
>   *known* SOAP/HBT+ halo positions: *Y*₅₀₀^cyl / σ_{Y₅₀₀}(θ₅₀₀), from a map-measured aperture
>   flux and the analytic *Planck* `immf6` MMF noise curve. §5. This is what the paper text as
>   currently written (and the CNC fits in `flamingo_repo`) uses; the blind MMF catalogue of §3–4
>   is the pipeline-level replacement for it.

---

## 1. Multifrequency input skies

### 1.1 Channel set **[fiducial]**

Six *Planck* HFI channels, in SZiFi / `Planck_simple` order
(`src/flamingo_mock/config.py:64`, `src/flamingo_mock/szifi/paths.py:13`):

| ν [GHz] | 100 | 143 | 217 | 353 | 545 | 857 |
|---|---|---|---|---|---|---|
| Gaussian FWHM [arcmin] | 9.66 | 7.22 | 4.90 | 4.92 | 4.67 | 4.22 |
| white noise *w*⁻¹ᐟ² [µK·arcmin] | 77.4 | 33.0 | 46.8 | 154.0 | 806.7 | 19115.0 |
| *N*_ℓ = (*w*⁻¹ᐟ² → rad)² [µK²] | 5.07e-4 | 9.21e-5 | 1.85e-4 | 2.00e-3 | 5.51e-2 | 30.9 |

- FWHMs: Table I values in `BEAM_FWHM_ARCMIN` (`config.py:70-77`).
- Noise levels and *N*_ℓ: table in `notebooks/homogeneous_planck_white_noise.ipynb` (cell 4).
  Pixel rms is σ_pix = *w*⁻¹ᐟ² / √Ω_pix → 45.06 / 19.21 / 27.24 / 89.67 / 469.6 / 11125.5 µK
  per pixel at *N*_side = 2048 (measured from the on-disk maps).

### 1.2 Total-map recipe **[fiducial, the "homog" set]**

`scripts/build_l1_prescription_totals.py`:

```
T_ν = B_ν ⊛ ( CMB + ΔT_ν^tSZ + ΔT_ν^CIB ) + noise_ν ,      ν ∈ {100 … 857} GHz
```

- All components in **µK_CMB**, **N_side = 2048**, RING, float32, Galactic coordinates.
  Components are built at *N*_side = 4096 (native lightcone resolution) and `ud_grade`-downgraded
  (`build_l1_prescription_totals.py:25-26,78-81`).
- **Beam**: Gaussian Table I FWHM, applied to the **signal only** —
  `signal = cmb + tsz + cib` then `hp.smoothing(..., fwhm)`
  (`build_l1_prescription_totals.py:105-109`; FITS keys `FWHM`, `BEAMON=1` at `:122-123`).
- **Noise added after beaming** (`build_l1_prescription_totals.py:109`), i.e. noise is *not* beamed.
- **HEALPix pixel window is deliberately not deconvolved** (`PIXWIN=0`,
  `build_l1_prescription_totals.py:126`; restated in `total_maps/README.txt`). The MMF does not
  deconvolve it either, so map-making and filter are self-consistent.
- **No kSZ** (`NKSZ=1` = "kSZ not included", `build_l1_prescription_totals.py:125`).
- Output: `total_maps/<prescription>/sky_CMB_tSZ_CIB_homog_{ν}GHz_nside2048_uK.fits`
  (`build_l1_prescription_totals.py:115`, path pattern in `szifi/paths.py:94-103`).
- Prescriptions kept side by side: **`L1_m9`** (fiducial hydro, D3A), `fgas-8sigma`,
  `Mstar-1sigma`, `LS8` (`build_l1_prescription_totals.py:24`). **`L1_m9_cibshuffle`** is a
  CIB-only nested nside=8 tile permutation of the fiducial CIB (seed 20260831); CMB, tSZ and
  white noise stay pixel-aligned (`scripts/build_l1_m9_cibshuffle_totals.py`). `total_maps/test/`
  is the old L2p8 demo, unused (r2 HILC inputs are rebuilt from `total_maps/L1_m9`, not from `test/`).
- Default homog roots: total maps `total_maps/L1_m9`, outputs `szifi_homog`
  (`szifi/paths.py:32-37`).

**Alternate set (not fiducial):** *NPIPE* ILC coadds, K_CMB, A/B splits, at
`ilc/inputs_nside2048_npipe/sky_CMB_tSZ_kSZ_CIB_npipe_split{A,B}_{ν}GHz_nside2048_K.fits`
(`szifi/paths.py:31,56,94-103`, `kind="npipe"`; ×1e6 K→µK in `szifi/skies.py:11,32-34`).
Used for the Planck-footprint validation runs (§3.2, §6) where realistic scanning-anisotropic
noise and the PR4 footprint matter.

### 1.3 Components — feeds `\LX{Describe we add CMB as GRF, and Planck instrumental noise.}`

| Component | How it is built | Code |
|---|---|---|
| **tSZ** | Yang et al. Compton-*y* lightcone shells (`lensed_tSZ_rot_same_rot.hdf5`) summed to a full-sky *y* map, then converted per channel with the non-relativistic *f*(*x*) = *x* coth(*x*/2) − 4, ΔT = T_CMB·*y*·*f*(*x*), T_CMB = 2.7255 K | `src/flamingo_mock/tsz.py`; `spectral.py:36-51`; `config.py:18,21` |
| **CIB** | Released bandpass-convolved maps at 217/353/545/857 GHz (Jy/sr). 100/143 GHz are *approximated*: exact released band → use as is; between two released bands → log-*ν* interpolation of *I*_ν; outside the released range → scale the nearest band by the three-parameter greybody at *z*_eff | `cib.py:1-15,27-42,80`; `spectral.py:86,137` |
| CIB SED | β_d = 1.65, *T*₀ = 35.14 K, α = 0.0, *z*_eff = 1.90 (L1_m9 mean-ratio fit) | `config.py:45-50` |
| **CMB** | **Gaussian random field**: CAMB **unlensed** scalar *C*_ℓ^TT at the FLAMINGO cosmology → `hp.synalm` realisation (**seed 42**, *N*_side = 4096, no pixel window) → lensed with the FLAMINGO convergence map κ via `pixell.lensing.lens_map_curved` (κ → φ with φ_ℓₘ = 2κ_ℓₘ/[ℓ(ℓ+1)], ℓ ≤ 2 zeroed; intermediate CAR grid res = π/(2·N_side); back to HEALPix with `method="harm"`) | `cmb.py:27-45` (CAMB), `:48-56` (κ→φ), `:58-86` (realise+lens; `:68-69` seed/synalm, `:74` lensing, `:83` reproject); `scripts/make_lensed_cmb_ls8.py:23` |
| CMB cosmology | D3A: *h*=0.681, Ω_m=0.306, Ω_b=0.0486, Σm_ν=0.06 eV, *A*_s=2.099e-9, *n*_s=0.967 · LS8: *h*=0.682, Ω_m=0.305, Ω_b=0.0473, *A*_s=1.836e-9, *n*_s=0.965 | `config.py:23-41` |
| **Noise** | Homogeneous **pixel-white** Gaussian map per channel, `rng.normal(0, σ_pix)`, independent seed 42 + ν per frequency (channels uncorrelated). Verified flat: median *C*_ℓ over 1000 ≤ ℓ ≤ 2000 is 1.000 ± 0.004 of the table *N*_ℓ | `notebooks/homogeneous_planck_white_noise.ipynb` cells 4, 6, 12, 13 |
| kSZ | Implemented (`ksz.py`) but **excluded** from the fiducial totals | `build_l1_prescription_totals.py:125` |
| Galactic FGs, radio point sources | **Not in the signal** — only *used* as analysis masks (PR4 GAL×PS, §3.2) | AGENTS.md §Sky model |

A second independent noise realisation (`*_r2.fits`) exists for every channel and feeds the
r1×r2 cross-check (`scripts/build_homog_r2_test_maps.py`, `scripts/plot_hilc_homog_r1xr2_split_diagnostics.py`).

---

## 2. The two reconstructions at a glance

| | Cluster detection | *y*-map reconstruction |
|---|---|---|
| Tool | **SZiFi** (`/scratch/scratch-lxu/agent_dev/auto_research_agent/szifi`, `agent_evolve` branch) | **pyILC / HILC** (harmonic needlet ILC, code on `needlet_ilc`) |
| Domain | flat-sky tiles, 768 × 1024² | full-sky harmonic |
| Product | `q_opt, y0, theta_500, lon, lat, …` catalogue | Compton-*y* map + needlet bandpowers |
| Code | `src/flamingo_mock/szifi/` | `configs/hilc_y_flamingo_homog*.yml`, `scripts/run_hilc.py` |
| Papers | Zubeldia+ 2023a (iMMF, arXiv:2204.13780), 2023b (sciMMF, arXiv:2212.07410) | — (see §8) |

---

## 3. MMF cluster detection (SZiFi iMMF) **[fiducial]**

Science driver: `scripts/run_szifi_jax_homog_immf.py` (`szifi_jax` on GPU 1, 768 tiles,
batch 16). Algorithm parameters still come from `szifi/run.py::default_params`
(`run.py:94-144`), which deep-copies `szifi.params_szifi_default` / `params_model_default`
and overrides only what is listed below. The CPU CLI `flamingo-szifi run --kind homog
--full-sky --method immf` (`cli.py`) shares those params; it is the slower resume-friendly
path, not the fiducial catalogue.

### 3.1 Flat-sky tiling

| Quantity | Value | Ref |
|---|---|---|
| Tile grid | HEALPix, `TILE_NSIDE = 8` → **768 tiles** | `szifi/paths.py:14`; `tiles.py:41-45` |
| Tile pixel grid | `TILE_NX = 1024` (1024 × 1024) | `paths.py:15` |
| Tile side | `TILE_L_DEG = 14.8`° | `paths.py:16` |
| Tile pixel size | 14.8/1024 deg = **0.8672 arcmin** | `szifi/survey.py:59` |
| Tile type | `tile_type = "healpix"` (SZiFi default) | `szifi/params.py` |

Cutouts via `szifi.sphere.get_cutout`, cached as
`szifi_homog/<prescription>/tiles/splitA/flamingo_field_{id}_tmap.npy` with a *leading singleton
axis* so `[tmap] = np.load(...)` matches SZiFi's `data_planck.py` (`tiles.py:173-174`), plus
`..._mask.npy` holding the triplet `(mask_galaxy, mask_point, mask_tile)`
(`tiles.py:125-155`, `prepare_tile` at `:149`).

### 3.2 Masks **[fiducial = unmasked full sky]**

- **[fiducial]** `--full-sky`: all 768 tiles with `GAL = PS = 1`; only the *tile-boundary* mask is
  non-trivial (`tiles.py:48-60 unmasked_mask_stack`). No Galactic cut, no point-source holes, no
  inpainting — justified because the mock signal contains no Galactic foregrounds and no radio
  point sources.
- Footprint variant (`--footprint`, used for the NPIPE runs in §6): keep tiles whose GAL×PS
  unmasked fraction ≥ **0.3** (`tiles.py:63-79`), with PR4 NILC `Masks.fits` field 1 (`GAL-MASK`)
  and field 2 (`PS-MASK`) binarised at > 0.5 and *N*_side = 2048 (`tiles.py:81-95`).
  Archive: `masks/pr4_nilc/Masks.fits` (`paths.py:25-27`).
- Masks derived inside the survey adapter (`szifi/survey.py:65-100`):

| SZiFi key | Definition | Value / Ref |
|---|---|---|
| `mask_ps` = `mask_map` | apodised `mask_galaxy` | `apotype="Smooth"`, `aposcale=0.2` (`survey.py:84-86`) |
| `mask_peak_finding` | `GAL × PS [× tile]` | `survey.py:88,92` |
| `mask_select` | FFT-buffered `GAL × PS`, × tile, then `get_fsky_criterion_mask` | buffer **10 arcmin** (`survey.py:65,89-91`), criterion `min_ftile = 0.3` (`survey.py:94-96`) |
| `mask_tile` | hard tile boundary | `tiles.py:145-152` |
| mode | `tilemask_mode = "field"` (SZiFi default) → `mask_select_buffer = 0` | `survey.py:99-105` |
| mode-coupling | `decouple_type = "fsky"`; coupling matrix neither computed, saved, nor needed | `run.py:121-124` |

### 3.3 Filter and noise-covariance construction **[fiducial]**

| Parameter | Value | Ref / note |
|---|---|---|
| Channels | all six (`freqs = [0..5]`) | SZiFi default |
| Multipole range | `lrange = [100, 2500]` | SZiFi default (`params.py`) |
| Beam model | `beam = "gaussian"`, `integrate_bandpass = False` | `run.py:115-116` |
| Instrument object | `expt.experiment("Planck_simple")`, then **`exp.FWHM` overwritten with the Table I vector** so SZiFi uses exactly the beams that were applied to the maps | `survey.py:118-120` |
| Search scales | **θ₅₀₀ ∈ 25 log-spaced points, 0.5′–32′** | `run.py:127-129` (same grid as `compute_flamingo_immf_skyavg_noise.py:47-49`) |
| Pressure profile | GNFW `profile_type = "arnaud"` (Arnaud+2010), `concentration = 1.177` | `params_model_default`, unmodified (`run.py:106`) |
| Amplitude convention | `norm_type = "centre"` → template normalised to **central Compton-*y*** *y*₀ | `params.py`; `nfw.get_y_norm(...)` at `compute_flamingo_immf_skyavg_noise.py:96,260` |
| *m*(θ₅₀₀) mapping | templates built at *z* = 0.2 via `model.get_m_500(θ₅₀₀, z=0.2)` | `compute_flamingo_immf_skyavg_noise.py:87,247` |
| SED matrix | `a_matrix[:,0] = exp.tsz_sed` (single tSZ component for iMMF) | `compute_flamingo_immf_skyavg_noise.py:197-200` |
| Power spectra | `powspec_bin_fac = 4`, `lsep = 3000` | `params.py` |
| Point sources | `inpaint = True`, `inpaint_type = "diffusive"`, `n_inpaint = 100` (inert in the fiducial run: PS mask ≡ 1) | `params.py` |
| Noise covariance | `cov_type = "isotropic"`; **N(ℓ) from masked residuals**, re-estimated after removing detections | `params.py`; iMMF of Zubeldia+2023a |
| Iteration | `iterative = True`, `max_it = 1`, `q_th = 4`, `q_th_noise = 4`, `mask_radius = 3` (in units of θ₅₀₀) → catalogue `catalogue_find_1` used | `params.py`; `run.py:248-255` |
| Peak finding | `extraction_mode = "find"`, `detection_method = "maxima_lomem"`, `apod_type = "old"` | `params.py` |
| SNR weighting | off — note SZiFi's `params.py` misspells the key (`snr_weigthing`); the repo sets the spelling `mmf.py` actually reads | `run.py:126` + its inline comment |
| Cosmology inside SZiFi | `cosmology = "Planck15"` (only used for the θ₅₀₀ ↔ *M*₅₀₀ template mapping) | `params.py` |

This is exactly the estimator already written in the paper as Eqs.
`\eqref{eq:mmf_estimator}`–`\eqref{eq:mmf_snr}`:
*ŷ*₀ = *N*⁻¹ ∫ **y**_t† **C**⁻¹(ℓ) **d**(ℓ), σ²_ŷ₀ = *N*⁻¹, *q* = *ŷ*₀/σ_ŷ₀ — with **C**(ℓ) the
6×6 frequency covariance measured from the (masked, then residual-based, iterated) maps, and
**y**_t the tSZ-SED × A10-profile × Gaussian-beam template, band-limited to 100 ≤ ℓ ≤ 2500.

### 3.4 Catalogue formation **[fiducial]**

1. GPU `szifi_jax` in **batches of 16 tiles** (`run_szifi_jax_homog_immf.py`, `gpu_tile_batch=16`;
   sidecar `batch_size: 16, n_tiles: 768, backend: "szifi_jax"`). ~8–9 min wall per full-sky run
   on CUDA:1. Partial batches under `catalogues/partial_{tag}_splitA_{immf|scimmf}/`.
2. Keep `q_opt ≥ q_th_final = 5.0` (`run.py:257`).
3. **FoF merge** of duplicates, `merge_radius_arcmin = 10.0` (`run.py:260-264`), re-applied to the
   concatenated all-sky catalogue (`run.py:389-394`).
4. Write npz + JSON sidecar (filename keeps `_immf_q5` even for sciMMF, matching the archive):
   `szifi_homog/<prescription>/catalogues/szifi_jax{,_scimmf}_splitA_immf_q5.npz`.
   Columns: `q_opt, y0, theta_500, theta_x, theta_y, lon, lat, pixel_ids`.
5. Per-tile σ_{*y*₀}(θ) on the same 25-point grid is written during the run
   (`run.py:165 save_per_tile_sigma`, hooked from `run_szifi_jax_homog_immf.py:97-102`) →
   `catalogues/sigma_per_tile_{immf|scimmf}_splitA/{theta_500_arcmin.npy, field_{id}.npy,
   field_{id}_noit.npy}` (768 tiles).

**Detection counts, *q* ≥ 5, full sky (768 tiles), post *z*_eff=1.90 regen (2026-09-05):**

| Sky | Method | *N*(*q* ≥ 5) | Sidecar |
|---|---|---|---|
| **L1_m9 correlated CIB** | iMMF | **2602** | `szifi_homog/L1_m9/catalogues/szifi_jax_splitA_immf_q5.json` |
| L1_m9 correlated CIB | sciMMF | 2867 | `…/szifi_jax_scimmf_splitA_immf_q5.json` |
| L1_m9 shuffled CIB | iMMF | 3119 | `szifi_homog/L1_m9_cibshuffle/catalogues/szifi_jax_splitA_immf_q5.json` |
| L1_m9 shuffled CIB | sciMMF | 2979 | `…/szifi_jax_scimmf_splitA_immf_q5.json` |

L1_m9 iMMF distributions (read from the npz): *q* = 5.00–99.6, median 6.62;
*y*₀ = 9.69e-6 – 1.97e-2, median 7.54e-5; θ₅₀₀ median 6.73′ with pile-up at the grid edges
(0.5′: 109; 32′: 142) — worth one sentence about the discrete scale grid.
Pre-regen 8-point (1′–10′) L1_m9 iMMF had *N* = 2509; hydro variants
(`fgas-8sigma`, `Mstar-1sigma`, `LS8`) are not yet re-run on the new maps
(`szifi_homog/archive/2026-09-05_pre_regen/`).

**CNC *q* binning:** `np.geomspace(5, 40, 6)` → 5 log-spaced bins, deliberately identical to the
5 *q* bins of the synthetic-data section (`scripts/plot_szifi_homog_binned_Nq.py:22,57`);
figure `figures/szifi/szifi_homog_cnc_binned_Nq_qgt5_immf_scimmf_l1m9_cibshuffle.{png,pdf}`.
Bin totals omit *q* > 40 (L1_m9 iMMF: 2591 of 2602).

### 3.5 sciMMF (CIB-deprojected) **[run on homog L1_m9]**

- `mmf_type = "spectrally_constrained"`, `deproject_cib = ["cib"]`, internal `cmmf_type="one_dep"`
  (`run.py:119,315-318,427-430`; `--mmf-type spectrally_constrained` on the JAX driver).
- The deprojected CIB SED uses the **SZiFi/*Planck* defaults, not the FLAMINGO CIB model**:
  α_cib = 0.36, *T*₀_cib = 20.7 K, β_cib = 1.6, *z*_eff_cib = 0.2 (`run.py:139-142`) — compare the
  FLAMINGO values β_d = 1.65, *T*₀ = 35.14 K, *z*_eff = 1.90 (`config.py:45-50`).
- Full-sky homog catalogues: §3.4 table. NPIPE-footprint sciMMF
  (`szifi/catalogues/footprint_splitA_scimmf_q5.npz`) is a separate validation product, archived.

---

## 4. MMF noise curve σ_{*y*₀}(θ₅₀₀) and the *Y*₅₀₀ aperture

`scripts/compute_flamingo_immf_skyavg_noise.py`, `scripts/compute_mmf_W_yt_skyavg.py`.

- Per-tile σ_{*y*₀} on a **25-point log grid, θ₅₀₀ ∈ [0.5′, 32′]**
  (`compute_flamingo_immf_skyavg_noise.py:47-49`), obtained by pushing a **null map** through
  `mmf.get_mmf_q_map` with that tile's own *N*(ℓ) (`:243-265`) — i.e. the *analysis* noise,
  not a theory prediction.
- Sky average = **sky-fraction-weighted** mean over tiles using the *Planck* tile sky fractions
  `skyfracs_szifi_cosmology.npy` (`:351-364`); `--full-sky`/`--kind homog` switches to equal
  weights (`:523,564`).
- Cross-check against the *Planck* **`immf6`** curve
  (`/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy`), weighted identically
  (`:367-382`); the run prints `median(mock/immf6)` (`:596`) → the natural validation sentence
  *"the MMF noise curve of the mocks agrees with the Planck immf6 curve to X%"* — re-run to fill X.
- *Y*₅₀₀ conversion used throughout:

  ```
  σ_{Y₅₀₀}(θ₅₀₀) = σ_{y₀}(θ₅₀₀) · θ₅₀₀² · π · I,   I = 0.06728373215772082
  ```
  (`compute_flamingo_immf_skyavg_noise.py:50` — SZiFi's `y0_to_Y_500` integral factor.)
- Filter aperture *W*(x; θ₅₀₀) = IFFT(*N*⁻¹**y**_t)/𝒩, sky-averaged on the same 25-point θ grid,
  radial grid *x* ∈ [0, 5] with 201 radii × 180 azimuths, cubic polynomial in log θ₅₀₀
  (`compute_mmf_W_yt_skyavg.py:29-37`) →
  `szifi/catalogues/mmf_aperture/W_theta_yt_skyavg_25pt.npz`.
- σ_{*y*₀} is **saved per tile during every MMF run** on the same 25-point search-scale grid
  (`run.py:165 save_per_tile_sigma` → `catalogues/sigma_per_tile_{immf|scimmf}_splitA/field_{id}.npy`,
  plus `field_{id}_noit.npy` for the non-iterated covariance). The optional
  `compute_flamingo_immf_skyavg_noise.py --kind homog --iterative --full-sky` path still writes
  `sigma_per_tile_flamingo_immf_it_splitA/` plus the sky-averaged vs-`immf6` npz; it is not
  needed to obtain per-tile σ after a `szifi_jax` run.

---

## 5. Empirical aperture *q* on known halos (`q_from_aperture`, a.k.a. *q*_map)

This is the *q* behind the FLAMINGO numbers currently in the paper
(the `l1_m9_cnc_binned_Nq_Nz_qgt5…` / `…_qfrommap` figures).

- Built in the **sibling repo** `/scratch/scratch-lxu/flamingo_repo`,
  `src/flamingo/aperture_snr.py`: for each SOAP/HBT+ halo it measures the cylindrical Compton-*y*
  `Y_500cyl_arcmin2` inside θ₅₀₀ on the map, takes `sigma_Y500_arcmin2` from a **cubic polynomial
  fit of log σ_{Y₅₀₀} vs log θ₅₀₀** to the sky-fraction-weighted *Planck* `immf6` tile curves
  (`fit_sigma_y500(..., filter_name="immf6", theta_min_arcmin=0.5, theta_max_arcmin=32,
  poly_deg=3)`), and sets `q_from_aperture = Y_500cyl / sigma_Y500`
  (`aperture_snr.py:14-45`).
- Catalogues consumed here use **rotation-consistent** coordinates (`lon_rot_deg`, `lat_rot_deg`
  — "yang26rot", same shell rotations as the maps):
  - `/rds/flamingo/L2p8_m9/lightcone0/catalogues/halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_qfrommap.csv`
    (`szifi/validate.py:17-20`)
  - `/rds/flamingo/L1_m9/catalogues/halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrom{map,mz}.csv`
    (`compute_mmf_W_yt_skyavg.py:50-52`, `plot_l1_m9_mmf_y0_vs_soap.py:39-41`)
- Truth columns used: `z, lon_rot_deg, lat_rot_deg, M_500c_Msun, theta_500_arcmin,
  q_from_aperture` (`validate.py:22-29`); `detectable ≡ q_from_aperture ≥ q_th_truth`
  (`validate.py:133`).
- **Limitation to state in the paper**: *q*_map is evaluated at *known* positions with an assumed
  noise model, so it contains no profile mismatch, no deblending and no peak-finding selection
  effects. The blind MMF catalogue of §3 is precisely what removes that caveat.

---

## 6. Validation against truth (completeness / purity)

`src/flamingo_mock/szifi/validate.py` (`flamingo-szifi benchmark`),
`scripts/benchmark_szifi_completeness_szifi.py`, `src/flamingo_mock/szifi/true_snr.py`
(`flamingo-szifi true-snr`).

- Cross-match: **greedy, one-to-one, highest-*q* detection first**, association radius
  **10 arcmin** (`validate.py:142-184`; documented as following Zubeldia et al. 2024 §3.4.1/§4.1).
  Detections and truth are both cut to the PR4 GAL×PS footprint
  (`footprint = "planck_gal_x_ps_unmasked_det+truth"`), `z_max = 1.0`.
- SZiFi-style completeness uses the **fixed-mode** true SNR q̄_t: re-extract at truth position and
  truth θ₅₀₀ (`true_snr.py`; parent sample pre-cut at aperture *q* ≥ 2, `true_snr.py:31-56`).
- Bins — detection SNR `(5,6,7,8,10,12,15,20,30,∞)`, true SNR
  `(2,3,4,5,6,7,8,10,12,15,20,30,∞)` (`validate.py:187-194`).
- **Result on file** (`szifi/catalogues/footprint_splitA_immf_q5_benchmark.json`; NPIPE split A,
  L2p8_m9 lightcone0 truth, *q*_obs ≥ 5, *q*_truth ≥ 5, *z* < 1):
  731 detections (240 more excluded by the footprint cut), **691 true positives, 40 false
  positives → purity 0.945**, and **completeness 0.233** of the 1840 detectable truth halos
  (0.0012 of all 569 290 truth halos with *M*₅₀₀c > 5e13 *M*_☉). Per-SNR-bin values in
  `..._benchmark_snr_bins.json`.
  → This is the ready-made "our detection pipeline reaches purity … completeness …" sentence, but
  note it is a *footprint + NPIPE-noise* result, **not** the fiducial full-sky homogeneous catalogue.

---

## 7. Cluster masking for the tSZ power spectrum **[ties MMF → power spectrum]**

`scripts/build_szifi_q5_cluster_mask.py` — the pipeline version of the masking geometry already
described in `subsec:syntheticdata`:

- Source catalogue expected by the mask script: `szifi_homog/<name>/catalogues/fullsky_splitA_immf_q5.npz`
  (`build_szifi_q5_cluster_mask.py`, default `--prescription L1_m9`). The current fiducial
  detections are `szifi_jax_splitA_immf_q5.npz` (**2602**); copy or symlink that file to the
  `fullsky_splitA_immf_q5.npz` name before rebuilding HILC cluster masks. Hydro-variant
  catalogues are not yet regenerated.
  Legacy `homog_immf_fullsky` (2364) is L2p8_m9 test-only (`--l2p8-test`).
- Hole radius **max(4 θ₅₀₀, 2 × FWHM)** with FWHM = 10 arcmin (`:16,42-45`) — identical in form to
  θ_max = max(4θ₅₀₀, 2θ_FWHM) in the synthetic section.
- Apodisation: `nmt.mask_apodization(mask, 0.25, apotype="C2")` → 0.25° cosine taper (`:17,56`),
  matching the synthetic-data analysis.
- Outputs (`ilc/`): `szifi_immf_q5_cluster_mask_nside2048.fits` (binary) and
  `szifi_immf_q5_cluster_mask_c2_025deg_nside2048.fits` (apodised), with *f*_sky(raw), ⟨*W*²⟩ and
  the soft-edge fraction printed (`:58-63`).
- Masked bandpowers then go through MASTER/`NaMaster` with the same 18-bin *D*_ℓ scheme as the
  synthetic analysis (`flamingo_mock/powerspectra.py`, `scripts/plot_hilc_homog_*.py`).

---

## 8. HILC Compton-*y* reconstruction

**Status: everything below is read directly from `configs/hilc_y_flamingo_homog*.yml`, the
`scripts/run_hilc.py` driver, the saved weight vectors and `logs/hilc_homog*.log`. Two
pyILC-internal semantics (`taper_width`, and exactly how the per-needlet frequency covariance is
estimated) are being confirmed against the pyILC source — marked ⚠ below.**

Implementation: **pyILC** (harmonic needlet ILC, McCarthy & Hill 2024), package on the
`needlet_ilc` branch at `/scratch/scratch-lxu/flamingo_needlet_ilc/src/flamingo_mock/ilc/`
(imported by `sys.path` insertion, e.g. `scripts/run_hilc.py:19-24`), with
`run_ilc(config, backend="jax")` as the single entry point.

### 8.1 Configuration **[fiducial = `hilc_y_flamingo_homog.yml`, full sky]**

| Key | Value | Note |
|---|---|---|
| `ELLMAX` | **4096** | = 2·N_side |
| `N_side` | **2048** | same grid as the total maps |
| `taper_width` | **200** | ⚠ apodisation of the harmonic needlet window at band edges |
| `wavelet_type` | **`TopHatHarmonic`** | sharp needlets in ℓ |
| `BinSize` | 50 | ℓ binning for the covariance |
| number of needlet scales | **81** (`scale0 … scale80`) | counted from `flamingo_weightvector_scale{J}_component_tSZ.txt` |
| `N_freqs` / `bandpass_type` | 6 / `DeltaBandpasses` | monochromatic |
| `freqs_delta_ghz` | **[100, 143, 353, 217, 545, 857]** | note 353 before 217 |
| `beam_type` / `beam_FWHM_arcmin` | `Gaussians` / **[9.66, 7.22, 4.92, 4.9, 4.67, 4.22]** | consistent with the 353/217 swap (4.92 ↔ 353, 4.90 ↔ 217) — not a bug |
| `perform_ILC_at_beam` | **10.0 arcmin** | all channels rescaled to a common 10′ beam |
| `ILC_preserved_comp` | **`tSZ`** | the single constraint **w**ᵀ**a**_tSZ = 1 |
| `N_deproj` / `ILC_deproj_comps` | **0 / []** | **no CIB (or any) deprojection** — minimum-variance HILC with the tSZ constraint only |
| `param_dict_file` | `pyilc/input/fg_SEDs_default_params.yml` | component SEDs |
| `save_weights` / `save_as` | yes / fits | |
| `ilc_backend` | `jax` | `PYILC_BACKEND` env override |
| `work_in_healpix` | yes | HEALPix, not CAR |

Input maps: `ilc/inputs_nside2048_homog/sky_CMB_tSZ_CIB_homog_{ν}GHz_nside2048_K.fits` — the same
beamed CMB+tSZ+CIB **L1_m9** totals as §1.2 but in **K_CMB**. r2 variant reads
`ilc/inputs_nside2048_homog_r2/…` (independent noise realisation, same L1_m9 beamed signal;
`scripts/build_homog_r2_test_maps.py` from `total_maps/L1_m9`, not `total_maps/test`).

**The four configs differ only in inputs and masks:**

| Config | Inputs | Cluster mask | Output suffix |
|---|---|---|---|
| `hilc_y_flamingo_homog.yml` | r1 | none | `_hilc_y_homog_fullsky` |
| `hilc_y_flamingo_homog_r2.yml` | r2 | none | `_hilc_y_homog_fullsky_r2` |
| `…_q5masked.yml` | r1 | q>5 iMMF holes | `_hilc_y_homog_q5masked` |
| `…_q5masked_r2.yml` | r2 | q>5 iMMF holes | `_hilc_y_homog_q5masked_r2` |

The masked variants set the **q>5 iMMF cluster mask of §7**
(`ilc/szifi_immf_q5_cluster_mask_nside2048.fits`, binary, radius max(4θ₅₀₀, 2·10′)) **both** as
`mask_before_covariance_computation` and `mask_before_wavelet_computation`, i.e. the covariance
*and* the needlet transform see the masked maps — a genuine "re-run component separation on the
masked multifrequency maps" step, which is exactly the caveat the Discussion currently raises
about foreground templates being assumed unchanged after masking.

### 8.2 Products and measured performance

Output dir `ilc/hilc_output_homog_q5masked/`:
`flamingo_needletILCmap_component_tSZ_hilc_y_homog_q5masked.fits` (402.7 MB, *N*_side 2048)
plus one 6-number weight vector per needlet scale.

Example weight vectors (channel order 100/143/**353**/**217**/545/857):

| scale | 100 | 143 | 353 | 217 | 545 | 857 |
|---|---|---|---|---|---|---|
| 0 (largest scale) | −0.101 | −0.158 | +0.036 | **+0.228** | −0.0050 | +6e-6 |
| 20 | −0.033 | −0.192 | +0.063 | **+0.161** | −0.0038 | −2e-5 |
| 40 | −0.0019 | −0.145 | **+0.087** | +0.016 | +0.0036 | −1.2e-4 |
| 60 | −1.6e-4 | −0.076 | **+0.109** | −0.013 | +0.0080 | −1.9e-4 |

Low needlet scales are dominated by **217 GHz** (+0.23) with negative 100/143 GHz weights; going to
small scales the weight moves from 217 to **353 GHz** (+0.12 at *J* = 60), while 857 GHz stays
≈ 0. Run log confirms the constraint holds throughout: `g = Σ w·a_tSZ = 1.0000` at ℓ = 50 and 300.

**Channel drop at small scales:** the saved weight vector has 6 entries for needlet scales
*J* = 0–61 but only **5** for *J* = 62–80 — one frequency is removed from the ILC in the highest-ℓ
bands (⚠ which one is not recoverable from the `.txt` files; needs the pyILC weight-saving code).
Do not write "all six channels are combined at every scale" until this is checked.

Measured from `logs/hilc_homog_q5masked.log` (q5masked, r1):

| Quantity | Value |
|---|---|
| mask *f*_sky (raw / ⟨*W*²⟩) | 0.9538 / 0.9257 |
| HILC *y* rms / truth *y* rms | 1.404e-6 / 1.872e-6 |
| median transfer function, 50 ≤ ℓ ≤ 500 | 0.818 |
| median correlation coefficient ρ vs truth, 50 ≤ ℓ ≤ 500 | 0.440 |
| runtime | ~85 s (4 threads, `taskset 16-19`) |

Masked *D*_ℓ vs theory / CIB-map / noise power (`logs/hilc_homog_q5masked.log`):

| ℓ | *D*_ℓ^HILC | *D*_ℓ^theory | *D*_ℓ^CIB(map) | *D*_ℓ^noise |
|---|---|---|---|---|
| 54 | 3.20e-14 | 2.10e-14 | 9.59e-15 | 1.02e-14 |
| 306 | 5.01e-13 | 1.44e-13 | 8.72e-14 | 3.36e-13 |
| 999 | 6.06e-12 | 5.91e-13 | 1.00e-12 | 4.69e-12 |
| 2007 | 8.78e-11 | 1.03e-12 | 1.10e-11 | 7.66e-11 |

→ the reconstructed *y* spectrum is **dominated by CIB + noise residuals at ℓ ≳ 300** (HILC ≫
theory), and ρ vs truth falls to ~0.4. That is a real result to report, not a bug, but it means
the HILC *y*-map is not a clean tSZ power-spectrum input on small scales with this noise and no CIB
deprojection. Cross-checks built for exactly this: r1×r2 cross-spectra
(`scripts/plot_hilc_homog_r1xr2_split_diagnostics.py` — noise-independent signal test) and
the pyILC-convention plot `scripts/plot_hilc_homog_pyilc_convention.py`.

Masked power spectra from these maps use NaMaster decoupling with the same geometry as §7
(`logs/hilc_homog_q5masked.log`: "NaMaster decoupling … computing coupling matrix").

⚠ **Still to confirm from the pyILC source** (background explorer running): the exact harmonic
window function of `TopHatHarmonic` and how `taper_width = 200` shapes it; and whether the
per-needlet frequency covariance **C**_J is built from the input maps themselves (masked) plus an
analyic beam+noise term, or purely from the data. The paper's Eq. `\eqref{eq:hilc_weights}` is the
right formula either way; the sentence describing *how* **C**_ℓ is estimated must wait for this.

---

## 9. Reproduce the fiducial numbers

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
cd /scratch/scratch-lxu/flamingo_mock_analysis

# 1) per-component maps (tSZ ΔT, CIB ΔT) and the total maps for every L1_m9 prescription
python scripts/build_l1_m9_fiducial_components.py
python scripts/build_l1_prescription_totals.py

# 2) tiles + GPU iMMF / sciMMF (see docs/regeneration_pipeline.md Step 3)
flamingo-szifi prepare --kind homog --full-sky --split A --n-workers 6 \
    --out-root       /rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi_homog/L1_m9 \
    --total-maps-dir /rds/rds-lxu/flamingo/integrated_maps_synthetic/total_maps/L1_m9
python scripts/run_szifi_jax_homog_immf.py --prescription L1_m9 --mmf-type standard --no-ref
python scripts/run_szifi_jax_homog_immf.py --prescription L1_m9 --mmf-type spectrally_constrained --no-ref

# 3) N(q) CNC figure (iMMF/sciMMF × correlated/shuffled) and the q>5 cluster mask
python scripts/plot_szifi_homog_binned_Nq.py \
    --cats /rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi_homog/L1_m9/catalogues/szifi_jax_splitA_immf_q5.npz \
           /rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi_homog/L1_m9_cibshuffle/catalogues/szifi_jax_splitA_immf_q5.npz \
           /rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi_homog/L1_m9/catalogues/szifi_jax_scimmf_splitA_immf_q5.npz \
           /rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi_homog/L1_m9_cibshuffle/catalogues/szifi_jax_scimmf_splitA_immf_q5.npz \
    --labels "iMMF correlated" "iMMF shuffled CIB" "sciMMF correlated" "sciMMF shuffled CIB" \
    --stem szifi_homog_cnc_binned_Nq_qgt5_immf_scimmf_l1m9_cibshuffle
python scripts/build_szifi_q5_cluster_mask.py
```

Logs of the 2026-09-05 regen: `logs/prepare_{L1_m9,L1_m9_cibshuffle}_regen.log`,
`logs/szifi_jax_{l1_m9,scimmf_l1_m9,l1_m9_cibshuffle,scimmf_l1_m9_cibshuffle}_regen.log`.

---

## 10. Things to settle before writing this section

1. **Fiducial vs L2p8 test catalogues.**
   L1_m9 fiducial detections are `szifi_homog/L1_m9/catalogues/szifi_jax_splitA_immf_q5.npz` (**2602**).
   Root `homog_immf_fullsky` (**2364**) is the old L2p8_m9 smoke-test catalogue only
   (`build_szifi_q5_cluster_mask.py --l2p8-test`); it must not drive fiducial masks or HILC.
   The mask script still defaults to `fullsky_splitA_immf_q5.npz` — point it at the JAX
   catalogue before regenerating HILC q5 masks.
2. **Which *q* goes into the CNC?** §5's `q_from_aperture` (current text and figures) vs §3's blind
   `q_opt`. If both appear, the paper must state explicitly that they are different estimators —
   the counts and the physics claims differ substantially.
3. **Fiducial noise choice.** The fiducial totals use *homogeneous* white noise, while the
   validated purity/completeness numbers were measured on *NPIPE* (anisotropic, position-tied)
   noise restricted to the PR4 footprint. Either re-run the benchmark on the fiducial set or scope
   the purity claim accordingly.
4. **sciMMF deprojection SED is the *Planck* default**, not the FLAMINGO CIB model (§3.5).
   Homog full-sky sciMMF catalogues exist; the SED mismatch is the remaining caveat.
5. **"Multifrequency" here means CMB + tSZ + CIB + white noise**: no kSZ, no Galactic foregrounds,
   no radio point sources. Say so plainly rather than implying a full *Planck*-like foreground model.
6. **θ₅₀₀ is a discrete 25-point grid (0.5′–32′)** and `theta_500` still piles up at both edges;
   the mask radius max(4θ₅₀₀, 10′) is therefore partly quantised.
7. **Mask FWHM = 10′** in `build_szifi_q5_cluster_mask.py:16` is the *synthetic-section* beam, not
   any *Planck* channel FWHM (2.5× the smallest). Fine, but describe it as an analysis choice.
8. **Pixel window is not deconvolved anywhere** (neither map-making nor MMF). Self-consistent, but
   worth one clause so a referee does not read it as an oversight.
