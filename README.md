# FLAMINGO mock analysis — pyILC Compton-$y$ with Planck NPIPE noise

McCarthy & Hill (2024) style needlet / harmonic ILC ([arXiv:2307.01043](https://doi.org/10.48550/arxiv.2307.01043)) on **FLAMINGO mock** multi-frequency skies **plus Planck NPIPE (PR4) noise splits**, not real Planck PR4 maps.

The pipeline is a Python package (`src/flamingo_mock/`), driven by two console commands:

| Command | Role |
|---------|------|
| `flamingo-mock-maps build` | Component maps (lensed CMB, tSZ, kSZ, CIB) from the FLAMINGO integrated maps |
| `flamingo-ilc` | ILC pipeline: `prepare` → `config` → `run` → `validate` |

## Environment

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
```

pyILC is the local [`agent_evolve`](https://github.com/licongxu/pyilc/tree/agent_evolve) checkout at `/scratch/scratch-lxu/agent_dev/auto_research_agent/pyilc` (installed editable). Its constrained-ILC weight solves are multi-backend (numpy / numba / **JAX** / CuPy); on this GPU host the **JAX backend is the default** (`ilc_backend: jax` in the generated YAMLs; override with `--backend` or `PYILC_BACKEND`).

## 1. Component maps (already on disk)

`flamingo-mock-maps build` writes µK_CMB components at $N_\mathrm{side}=4096$ under `/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/{cmb,tsz,ksz,cib}/`.

## 2. Prepare noise-added split maps

```bash
flamingo-ilc prepare --nside 2048
```

Per channel (100 / 143 / 353 GHz): coadd CMB+tSZ+kSZ+CIB (→ K_CMB), smooth with the Planck HFI Gaussian beam (9.66′ / 7.22′ / 4.92′), downgrade to $N_\mathrm{side}=2048$, then add NPIPE detector-set noise splits **A** and **B** (`mc_00200`, K_CMB; independently destriped, so A×B is noise-decoupled).

Writes under `/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc/inputs_nside2048_npipe/`:

- `sky_CMB_tSZ_kSZ_CIB_npipe_split{A,B}_{100,143,353}GHz_nside2048_K.fits`
- `sky_CMB_tSZ_kSZ_CIB_signal_*` (beam-smoothed coadd, no noise)
- `compton_y_nside2048.fits` (truth)

## 3. pyILC configs

```bash
flamingo-ilc config --out-dir configs          # HILC splits A,B + NILC split A
flamingo-ilc config --backend numba            # CPU fallback
```

YAML settings: `ELLMAX: 3000`, `N_side: 2048`, `beam_FWHM_arcmin: [9.66, 7.22, 4.92]`, `perform_ILC_at_beam: 5.0`, `ILC_preserved_comp: tSZ`, `N_deproj: 0`, `ilc_backend: jax`. The tracked YAMLs under `configs/` are generated artifacts — regenerate them with this command rather than editing by hand (a test checks they match the writer).

## 4. Run the ILC (ELLMAX=3000)

**HILC (primary, fast):**

```bash
flamingo-ilc run configs/hilc_y_flamingo_npipe_splitA.yml
flamingo-ilc run configs/hilc_y_flamingo_npipe_splitB.yml   # second split for cross spectra
```

**NILC (paper needlets; heavier):**

```bash
flamingo-ilc run configs/nilc_y_flamingo_npipe_splitA.yml
```

Outputs under `/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc/{hilc,nilc}_output_npipe_split{A,B}/`.

## 5. Validate (beam-deconvolved $C_\ell$ to $\ell=3000$)

```bash
flamingo-ilc validate \
  --ymap .../hilc_output_npipe_splitA/flamingo_needletILCmap_component_tSZ_hilc_y_npipe_splitA.fits \
  --ymap-split .../hilc_output_npipe_splitB/flamingo_needletILCmap_component_tSZ_hilc_y_npipe_splitB.fits \
  --lmax 3000 --ilc-beam-fwhm-arcmin 5.0 --figures-dir figures
```

This:

1. Deconvolves the 5′ ILC common beam from ILC auto- and cross-spectra ($C_\ell / B_\ell^2$).
2. Plots raw vs deconv vs truth to $\ell=3000$.
3. Uses splitA×splitB for a noise-decoupled spectrum.
4. Writes `figures/ilc_y_beam_deconv_ratio.png` showing that high-$\ell$ is no longer an undeconvolved $B_\ell^2$ cliff.

## Beam convention

- **Input maps:** coadds smoothed with channel Gaussians, then + NPIPE noise.
- **pyILC:** rebeams channels to `perform_ILC_at_beam=5′` using `beam_fac = B_common / B_channel`.
- **Reported spectra:** divide by $B_{5'}(\ell)^2$ so comparisons to beam-free truth $y$ are beam-deconvolved.

## Tests

```bash
pytest tests/ -v
```

Covers: beam deconvolution helpers, pyILC multi-backend weight solves (JAX/numba parity with numpy, preserved-component response), config writer ↔ tracked YAML sync, NPIPE split independence, and (once products exist) prepared-input and y-map validation.

## Layout

| Path | Role |
|------|------|
| `src/flamingo_mock/` | Map-making package (components, coadd) |
| `src/flamingo_mock/ilc/paths.py` | Filesystem layout (components, NPIPE noise, ILC products) |
| `src/flamingo_mock/ilc/beams.py` | $B_\ell$ / $C_\ell$ deconvolution helpers |
| `src/flamingo_mock/ilc/noise.py` | NPIPE noise paths/loading (K_CMB channels only) |
| `src/flamingo_mock/ilc/prepare.py` | Coadd + beams + noise splits |
| `src/flamingo_mock/ilc/config.py` | pyILC YAML writer |
| `src/flamingo_mock/ilc/run.py` | pyILC entry |
| `src/flamingo_mock/ilc/validate.py` | Beam-deconvolved validation |
| `configs/` | Generated pyILC YAMLs (tracked) |
| `/rds/rds-lxu/flamingo/integrated_maps_synthetic/` | Large I/O (outside git) |

## Reference

- McCarthy & Hill (2024) — `reference_paper/pyilc.pdf` ([arXiv:2307.01043](https://doi.org/10.48550/arxiv.2307.01043))
- Yang et al. FLAMINGO maps — `reference_paper/yang26flamingo.pdf` ([arXiv:2512.09891](https://doi.org/10.48550/arxiv.2512.09891))
- Planck noise reference — `noise_description.md`
