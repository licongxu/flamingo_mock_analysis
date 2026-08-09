# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## Project-specific instructions

### Environment

Activate the Python environment before running notebooks or scripts:

```bash
source ~/envs/cosmo_env/bin/activate
```

Key packages: `healpy`, `h5py`, `numpy`, `matplotlib`, `astropy`, `camb`, `pixell`.

### pyILC

Needlet / harmonic ILC uses **[pyILC](https://github.com/licongxu/pyilc/tree/agent_evolve)** (`agent_evolve` branch) installed at `~/software_packages/pyilc`.

**Priority:** use the **JAX backend on TPU/GPU** for ILC weight solves. Set `ilc_backend: jax` in the YAML config, or export `PYILC_BACKEND=jax`. On TPU hosts use a JAX build with TPU support (`jax[tpu]`). Fall back to `numba` or `numpy` only on CPU-only machines.

Method paper: **McCarthy & Hill (2024)** — `reference_paper/pyilc.pdf` ([arXiv:2307.01043](https://doi.org/10.48550/arxiv.2307.01043)).  
Reference implementation: `~/software_packages/pyilc/notebooks/Planck_CMB_HILC.ipynb`.  
Project notebook: `notebooks/nilc_y_flamingo_pyilc.ipynb` (pyILC Compton-$y$ on coadds + FFP10 noise splits).

### Mock sky map pipeline

Mock frequency maps are built in a **sequential notebook pipeline**. Read these notebooks before changing map generation or ILC inputs:

| Step | Notebook | Output |
|------|----------|--------|
| 0 | `notebooks/visualize_yang26_lightcone0_maps.ipynb` | Inspect raw FLAMINGO `.hdf5` integrated maps |
| 1 | `notebooks/simulate_lensed_primary_cmb.ipynb` | Lensed primary CMB $T$ map (CAMB + pixell + FLAMINGO $\kappa$) |
| 2 | `notebooks/simulate_tsz_frequency_maps.ipynb` | $\Delta T_\mathrm{tSZ}(\nu)$ from lensed Compton-$y$ |
| 3 | `notebooks/simulate_cib_frequency_maps.ipynb` | $\Delta T_\mathrm{CIB}(\nu)$ from released bandpass maps |
| 4 | `notebooks/coadd_multifrequency_sky_maps.ipynb` | Coadded skies $T_\nu = T_\mathrm{CMB} + \Delta T_\mathrm{tSZ} + \Delta T_\mathrm{CIB}$ |
| 5 | `notebooks/nilc_y_flamingo_pyilc.ipynb` | pyILC Compton-$y$ on coadds + FFP10 noise splits (ELLMAX≥3000) |

Step 4 writes FITS maps to `maps_100_143_353/raw/` (components) and `maps_100_143_353/coadd/` (multi-frequency skies). Step 5 builds noise-added splits under `~/cosmology_data/flamingo_ilc/` and runs pyILC (see `configs/*noise_split*.yml`).

Optional: `notebooks/visualize_planck_ffp10_noise.ipynb` for Planck FFP10 noise maps; `notebooks/yang26_power_spectra.ipynb` for power-spectrum validation.

### Large data on disk

Large files live on the **196 GB NVMe volume** mounted at `/mnt/data/` (not on the ~97 GB root disk). Do not commit FITS/HDF5 map data.

Two paths under `$HOME` are bind-mounted from `/mnt/data/` for convenience:

| Bind mount | Canonical path on `/mnt/data` |
|------------|-------------------------------|
| `~/cosmology_data/` | `/mnt/data/cosmology_data/` |
| `~/flamingo_mock_analysis/maps_100_143_353/` | `/mnt/data/flamingo_mock_analysis/maps_100_143_353/` |

Either path works; prefer `/mnt/data/...` when writing docs or scripts so the storage location is explicit.

| Location | Contents |
|----------|----------|
| `/mnt/data/cosmology_data/flamingo/L2p8_m9/integrated_maps/yang26/lightcone0_shells/` | FLAMINGO L2p8_m9 integrated HEALPix maps (yang26 / lightcone0). Eight `.hdf5` shell maps at $N_\mathrm{side}=4096$ (~1.5 GB each; ~13 GB total). |
| `/mnt/data/cosmology_data/planck_noise/` | Planck FFP10 simulated instrumental noise maps (`.fits`, multiple frequencies). |
| `/home/ext_andyxlcnb_gmail_com/flamingo_mock_analysis/maps_100_143_353/` | Derived coadds/raw (gitignored). HEALPix FITS at $N_\mathrm{side}=4096$. |
| `/home/ext_andyxlcnb_gmail_com/cosmology_data/flamingo_ilc/` | Noise-split ILC inputs/outputs (outside git). |

Remote source for the FLAMINGO integrated maps:
[COSMA DataWeb — yang26 / lightcone0_shells](https://dataweb.cosma.dur.ac.uk:8443/flamingo/viewer.html?path=FLAMINGO%2FL2p8_m9%2FL2p8_m9%2Fintegrated_maps%2Fyang26%2Flightcone0_shells)

### Reference papers

| Paper | Path |
|-------|------|
| Yang et al. (2026) — FLAMINGO mock maps & lightcone | `reference_paper/yang26flamingo.pdf` ([arXiv:2512.09891](https://doi.org/10.48550/arxiv.2512.09891)) |
| McCarthy & Hill (2024) — pyILC needlet / harmonic ILC | `reference_paper/pyilc.pdf` ([arXiv:2307.01043](https://doi.org/10.48550/arxiv.2307.01043)) |

### Repo layout

| Path | Purpose |
|------|---------|
| `notebooks/` | Analysis notebooks (visualization, simulation, coadd, ILC) |
| `figures/` | Saved plots |
| `reference_paper/` | PDF references |
| `maps_100_143_353/` | Local derived map products (gitignored) |

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
