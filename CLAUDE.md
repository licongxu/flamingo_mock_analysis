# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## Project-specific instructions

### Environment

Activate the Python environment before running notebooks, the CLI, or tests:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
```

Key packages: `healpy`, `numpy`, `matplotlib`, `astropy`, `camb`, `pixell`, `jax` (CUDA), `pyilc` (editable). The `flamingo-mock` package itself is installed editable from this repo (`pip install -e .`).

### pyILC

Needlet / harmonic ILC uses **[pyILC](https://github.com/licongxu/pyilc/tree/agent_evolve)** (`agent_evolve` branch) installed editable at `/scratch/scratch-lxu/agent_dev/auto_research_agent/pyilc`.

**Priority:** use the **JAX backend on GPU** for ILC weight solves (this host has 2× RTX PRO 6000). The generated YAMLs carry `ilc_backend: jax`; override with `flamingo-ilc run --backend {jax,numba,numpy,cupy,auto}` or `PYILC_BACKEND`. Fall back to `numba` or `numpy` only on CPU-only machines.

Method paper: **McCarthy & Hill (2024)** — `reference_paper/pyilc.pdf` ([arXiv:2307.01043](https://doi.org/10.48550/arxiv.2307.01043)).  
Reference implementation: `/scratch/scratch-lxu/agent_dev/auto_research_agent/pyilc/notebooks/Planck_CMB_HILC.ipynb`.  
Project notebook: `notebooks/nilc_y_flamingo_pyilc.ipynb` (drives the `flamingo-ilc` pipeline).

### Mock sky map + ILC pipeline

Everything runs through the `flamingo_mock` package (`src/flamingo_mock/`); notebooks are for visualization only.

| Step | Command / module | Output |
|------|----------|--------|
| 0 | `notebooks/visualize_yang26_lightcone0_maps.ipynb` | Inspect raw FLAMINGO integrated maps |
| 1 | `flamingo-mock-maps build` (`flamingo_mock.{cmb,tsz,ksz,cib}`) | Component maps, µK_CMB, $N_\mathrm{side}=4096$ |
| 2 | `flamingo-ilc prepare` (`flamingo_mock.ilc.prepare`) | Coadd (K_CMB) + channel beams + NPIPE noise splits A/B, $N_\mathrm{side}=2048$ |
| 3 | `flamingo-ilc config` (`flamingo_mock.ilc.config`) | pyILC YAMLs in `configs/` (generated; do not hand-edit) |
| 4 | `flamingo-ilc run` (`flamingo_mock.ilc.run`) | pyILC Compton-$y$ (HILC/NILC, ELLMAX=3000, JAX backend) |
| 5 | `flamingo-ilc validate` (`flamingo_mock.ilc.validate`) | Beam-deconvolved $C_\ell$ vs truth $y$ |

Optional: `notebooks/download_planck_noise.ipynb` for fetching Planck NPIPE noise; `notebooks/reproduce_yang26_power_spectra.ipynb` for power-spectrum validation.

### Large data on disk

Large files live on the **26 TB `/rds` volume**. Do not commit FITS/HDF5 map data.

| Location | Contents |
|----------|----------|
| `/rds/rds-lxu/flamingo/integrated_maps_synthetic/L2800N5040/HYDRO_FIDUCIAL/lightcone0_shells/` | FLAMINGO L2p8_m9 integrated HEALPix maps (Yang et al. 2026), $N_\mathrm{side}=4096$ |
| `/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/` | Synthetic component maps (`{cmb,tsz,ksz,cib}/`, µK_CMB, $N_\mathrm{side}=4096$) |
| `/rds/rds-lxu/flamingo/integrated_maps_synthetic/planck_noise/npipe/` | Planck NPIPE noise MCs (`{freq}GHz/{A,B,full}/`, K_CMB for 100–353 GHz, $N_\mathrm{side}=2048$) — see `noise_description.md` |
| `/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc/` | ILC inputs (`inputs_nside2048_npipe/`) and pyILC outputs (outside git) |

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
| `src/flamingo_mock/` | Package: map making + `ilc/` subpackage (prepare/config/run/validate) |
| `configs/` | Generated pyILC YAMLs (tracked; regenerate with `flamingo-ilc config`) |
| `tests/` | pytest suite (beam helpers, backends, configs, noise, validation) |
| `notebooks/` | Visualization notebooks |
| `figures/` | Saved plots |
| `reference_paper/` | PDF references |

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
