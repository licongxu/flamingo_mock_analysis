# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## Project-specific instructions

**Branch:** `szifi_branch` — MMF galaxy-cluster finding on FLAMINGO mock Planck skies with **SZiFi** (iterative / spectrally constrained matched filters). Needlet ILC work lives on the `needlet_ilc` branch; both branches share the same on-disk mock data.

### Environment

Activate the Python environment before running notebooks, the CLI, or tests:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
```

Key packages: `healpy`, `numpy`, `matplotlib`, `astropy`, `camb`, `pixell`, `pymaster`, `scikit-learn`. The `flamingo-mock` package is installed editable from this repo (`pip install -e .`). **SZiFi** is installed editable from `/scratch/scratch-lxu/agent_dev/auto_research_agent/szifi` (`pip install -e .` in that directory; already in `cmbagent_env`).

Optional SZiFi deps: `SZpack` (relativistic tSZ), `pixell` / `orphics` (CAR tiles). GPU acceleration: set `SZIFI_ARRAY_BACKEND=jax` (see `szifi/backend.py`).

### SZiFi

Galaxy-cluster detection via multi-frequency matched filters on flat-sky cut-outs (HEALPix tiles or CAR).

| Method | Paper | What it does |
|---|---|---|
| **iMMF** | Zubeldia et al. (2023a) | Iterative MMF with unbiased noise covariance from masked residuals |
| **sciMMF** | Zubeldia et al. (2023b) | Spectrally constrained MMF — CIB (and other SED) deprojection |

Local papers: `/scratch/scratch-lxu/agent_dev/auto_research_agent/szifi/ref_paper/szifi_refa.pdf` ([arXiv:2204.13780](https://arxiv.org/abs/2204.13780)), `szifi_refb.pdf` ([arXiv:2212.07410](https://arxiv.org/abs/2212.07410)). Upstream repo: [inigozubeldia/szifi](https://github.com/inigozubeldia/szifi) (`agent_evolve` branch on this machine).

Reference Planck driver: `/scratch/scratch-lxu/agent_dev/auto_research_agent/szifi/test_files/run_szifi_planck.py` and survey stub `szifi/surveys/data_planck.py`. FLAMINGO mocks need a new survey data module that reads our coadded maps, noise, and masks.

**Algorithm parameters:** `szifi/params.py` (`params_szifi_default`, `params_data_default`, `params_model_default`).

### Mock sky + MMF pipeline (this branch)

Everything runs through the `flamingo_mock` package (`src/flamingo_mock/`); notebooks are for visualization only. Detailed data provenance is in `data_description.md`; Planck noise in `noise_description.md`.

| Step | Command / module | Output |
|------|----------|--------|
| 0 | `notebooks/build_synthetic_component_maps.ipynb` | Inspect / rebuild component maps |
| 1 | `flamingo-mock-maps build` (`flamingo_mock.{cmb,tsz,ksz,cib}`) | Per-component maps, µK_CMB, $N_\mathrm{side}=4096$ |
| 2 | *(planned)* `flamingo_mock.szifi.prepare` | Coadd (beam-smoothed CMB+tSZ+kSZ+CIB) + NPIPE noise, Planck masks → SZiFi tile inputs |
| 3 | *(planned)* `flamingo_mock.szifi.run` | SZiFi iMMF / sciMMF cluster catalogues |
| 4 | *(planned)* `flamingo_mock.szifi.validate` | Completeness / purity vs FLAMINGO halo lightcone |

Steps 2–4 are the work of this branch. Until packaged, follow the SZiFi test scripts and adapt `surveys/data_planck.py`.

**Sky model (four components):** lensed CMB + tSZ + kSZ + CIB — no Galactic foregrounds, no radio point sources in the signal. See `data_description.md` §5.

**Truth catalogues for validation:** `/rds/flamingo/halo_cat/lightcone0/lightcone_halos_{0000..0078}.hdf5` (SOAP/HBTplus; see `data_description.md` §4). Match detected clusters to `Lightcone/HaloCentre` and `BoundSubhalo/TotalMass`.

### Large data on disk

Large files live on the **26 TB `/rds` volume**. Do not commit FITS/HDF5 map data.

| Location | Contents |
|----------|----------|
| `/rds/flamingo/L2800N5040/HYDRO_FIDUCIAL/lightcone0_shells/` | Raw FLAMINGO integrated HEALPix maps (Yang et al. 2026), $N_\mathrm{side}=4096$ |
| `/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/` | Synthetic component maps (`{cmb,tsz,ksz,cib}/`, µK_CMB, $N_\mathrm{side}=4096$) |
| `/rds/rds-lxu/flamingo/integrated_maps_synthetic/planck_noise/npipe/` | Planck NPIPE noise MCs (`{freq}GHz/{A,B,full}/`, K_CMB for 100–353 GHz, $N_\mathrm{side}=2048$) — see `noise_description.md` |
| `/rds/rds-lxu/flamingo/integrated_maps_synthetic/masks/pr4_nilc/` | Planck PR4 NILC mask archive (`Masks.fits`) |
| `/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc/` | ILC products from `needlet_ilc` branch (optional y-maps); shared `gal_ps_mask_nside2048.fits` |
| `/rds/flamingo/halo_cat/` | FLAMINGO halo lightcone catalogues for cluster validation |

Remote source for the FLAMINGO integrated maps:
[COSMA DataWeb — yang26 / lightcone0_shells](https://dataweb.cosma.dur.ac.uk:8443/flamingo/viewer.html?path=FLAMINGO%2FL2p8_m9%2FL2p8_m9%2Fintegrated_maps%2Fyang26%2Flightcone0_shells)

### Planck masks

PR4 NILC multi-field archive (downloaded from PLA):

```
/rds/rds-lxu/flamingo/integrated_maps_synthetic/masks/pr4_nilc/Masks.fits
```

| healpy field | Name | Use |
|---|---|---|
| 0 | `NILC-MASK` | Near full-sky confidence mask — **not** used alone for cluster finding |
| 1 | `GAL-MASK` | Galactic plane cut (soft edges) |
| 2 | `PS-MASK` | Point-source holes (binary) |

For ILC (on `needlet_ilc`), the pipeline builds a binary **GAL × PS** product at $N_\mathrm{side}=2048$:

```
/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc/gal_ps_mask_nside2048.fits
```

SZiFi expects apodised flat-sky masks per tile (`mask_galaxy`, `mask_point`, tile boundary masks). Project the PR4 fields onto each SZiFi cut-out and apodise (see `szifi/maps.py`, `surveys/data_planck.py`). Masks are **not** baked into the NPIPE noise maps (`noise_description.md` §3).

### Planck noise pitfalls (read before coadding)

From `noise_description.md`:

1. Only **100** NPIPE realisations on the PLA (mc_00200–00299); index is **5-digit** zero-padded.
2. **Units:** 100–353 GHz in K_CMB (`×1e6` → µK); 545/857 GHz in MJy/sr (`×1e6 × jysr2uk(freq)`).
3. Noise is **anisotropic** and **position-tied** — do not rotate or relocate patches.
4. Use NPIPE **A/B splits** for null tests; full-frequency maps omit the coverage token.

Download more noise: `notebooks/download_planck_noise.ipynb`.

### Reference papers

| Paper | Path |
|-------|------|
| Yang et al. (2026) — FLAMINGO mock maps & lightcone | `reference_paper/yang26flamingo.pdf` ([arXiv:2512.09891](https://doi.org/10.48550/arxiv.2512.09891)) |
| Zubeldia et al. (2023a) — iMMF / SZiFi | `…/szifi/ref_paper/szifi_refa.pdf` ([arXiv:2204.13780](https://arxiv.org/abs/2204.13780)) |
| Zubeldia et al. (2023b) — sciMMF / CIB mitigation | `…/szifi/ref_paper/szifi_refb.pdf` ([arXiv:2212.07410](https://arxiv.org/abs/2212.07410)) |

On-disk data dictionaries: `data_description.md`, `noise_description.md`.

### Related branch: `needlet_ilc`

Harmonic / needlet ILC Compton-$y$ maps (pyILC, McCarthy & Hill 2024) are built on `needlet_ilc`. That branch adds `src/flamingo_mock/ilc/` and `flamingo-ilc {prepare,config,run,validate}`. ILC $y$-maps on disk can be an alternative cluster-detection input, but **this branch targets direct MMF on multi-frequency temperature maps**, as in the SZiFi Planck catalogues.

### Repo layout

| Path | Purpose |
|------|---------|
| `src/flamingo_mock/` | Package: map making (+ planned `szifi/` subpackage) |
| `data_description.md` | FLAMINGO inputs, sky model, halo cats, compsep context |
| `noise_description.md` | Planck NPIPE noise: units, splits, download, pitfalls |
| `notebooks/` | Component maps, noise download, Yang et al. spectra |
| `figures/` | Saved plots |
| `reference_paper/` | PDF references |
| `reference_tables/planck_info.png` | Planck HFI beam FWHMs (Table I) |

---

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
