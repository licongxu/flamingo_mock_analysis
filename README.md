# FLAMINGO mock analysis — pyILC Compton-$y$ with Planck noise

McCarthy & Hill (2024) style needlet / harmonic ILC ([arXiv:2307.01043](https://doi.org/10.48550/arxiv.2307.01043)) on **FLAMINGO mock** multi-frequency skies **plus Planck FFP10 noise splits**, not real Planck PR4 maps.

## What was wrong before (and is fixed)

| Issue | Fix |
|-------|-----|
| `ELLMAX=2048` only | `ELLMAX=3000`, $N_\mathrm{side}=2048$ |
| Signal-only coadds | Coadd + FFP10 `mc_00000` / `mc_00001` noise splits |
| `perform_ILC_at_beam: 5` without deconvolution → high-$\ell$ cliff | Validation reports **beam-deconvolved** $C_\ell / B_\ell^2$; channel beams match prep |
| Fake 1′ beams | Planck HFI FWHMs 9.66′ / 7.22′ / 4.92′ applied to coadds |

## Environment

```bash
source ~/envs/cosmo_env/bin/activate
export PYILC_BACKEND=numba
```

## 1. Prepare noise-added split maps

```bash
python scripts/prepare_mock_ilc_inputs.py --nside 2048
```

Writes under `~/cosmology_data/flamingo_ilc/inputs_nside2048_noise/`:

- `sky_CMB_tSZ_CIB_noise_split{0,1}_{100,143,353}GHz_nside2048_K.fits` (K_CMB)
- `sky_CMB_tSZ_CIB_signal_*` (beam-smoothed coadd, no noise)
- `compton_y_nside2048.fits` (truth)

Channel beams: 100→9.66′, 143→7.22′, 353→4.92′. Noise from `~/cosmology_data/planck_noise/ffp10_noise_*_mc_0000{0,1}.fits` (already K_CMB, nside 2048).

## 2. Run ILC (ELLMAX=3000)

**HILC (primary, fast):**

```bash
python scripts/run_pyilc_y.py configs/hilc_y_flamingo_noise_split0.yml
python scripts/run_pyilc_y.py configs/hilc_y_flamingo_noise_split1.yml   # second noise realization
```

**NILC (paper needlets; heavier):**

```bash
python scripts/run_pyilc_y.py configs/nilc_y_flamingo_noise_split0.yml
```

YAML settings: `ELLMAX: 3000`, `N_side: 2048`, `beam_FWHM_arcmin: [9.66, 7.22, 4.92]`, `perform_ILC_at_beam: 5.0`, `ILC_preserved_comp: tSZ`, `N_deproj: 0`.

Outputs under `~/cosmology_data/flamingo_ilc/hilc_output_noise_split{0,1}/` (and `nilc_output_noise_split0/`).

## 3. Validate (beam-deconvolved $C_\ell$ to $\ell=3000$)

```bash
python scripts/validate_ymap_vs_truth.py \
  --ymap ~/cosmology_data/flamingo_ilc/hilc_output_noise_split0/flamingo_needletILCmap_component_tSZ_hilc_y_noise_split0.fits \
  --ymap-split1 ~/cosmology_data/flamingo_ilc/hilc_output_noise_split1/flamingo_needletILCmap_component_tSZ_hilc_y_noise_split1.fits \
  --truth ~/cosmology_data/flamingo_ilc/inputs_nside2048_noise/compton_y_nside2048.fits \
  --lmax 3000 --ilc-beam-fwhm-arcmin 5.0 \
  --figures-dir figures
```

This:

1. Deconvolves the 5′ ILC common beam from ILC auto- and cross-spectra (`C_\ell / B_\ell^2`).
2. Plots raw vs deconv vs truth to $\ell=3000$.
3. Optionally uses split0×split1 for a noise-decoupled spectrum.
4. Writes `figures/ilc_y_beam_deconv_ratio.png` showing that high-$\ell$ is no longer an undeconvolved $B_\ell^2$ cliff.

## Beam convention

- **Input maps:** coadds smoothed with channel Gaussians, then + FFP10 noise.
- **pyILC:** rebeams channels to `perform_ILC_at_beam=5′` using `beam_fac = B_common / B_channel`.
- **Reported spectra:** divide by $B_5'(ℓ)^2$ so comparisons to beam-free truth $y$ are beam-deconvolved.

## Tests

```bash
pytest tests/test_pyilc_flamingo_config.py -v
```

## Layout

| Path | Role |
|------|------|
| `configs/hilc_y_flamingo_noise_split{0,1}.yml` | Active HILC YAMLs (ELLMAX 3000 + noise) |
| `configs/nilc_y_flamingo_noise_split0.yml` | NILC YAML (same physics) |
| `scripts/prepare_mock_ilc_inputs.py` | Beams + FFP10 splits |
| `scripts/beam_utils.py` | $B_ℓ$ / Cl deconvolution helpers |
| `scripts/run_pyilc_y.py` | pyILC entry |
| `scripts/validate_ymap_vs_truth.py` | Beam-deconvolved validation |
| `~/cosmology_data/flamingo_ilc/` | Large I/O (outside git) |

## Reference

- McCarthy & Hill (2024) — `reference_paper/pyilc.pdf`
- Yang et al. FLAMINGO maps — `reference_paper/yang26flamingo.pdf`
