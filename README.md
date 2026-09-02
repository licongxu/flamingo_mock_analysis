# flamingo_mock_analysis

Synthetic multi-frequency CMB sky maps from the FLAMINGO (Yang et al. 2026)
integrated lightcone maps. The exploratory analysis notebooks have been
packaged into importable source code under `src/flamingo_mock/`; the notebook
`notebooks/reproduce_yang26_power_spectra.ipynb` reproduces the paper's
power-spectrum figures (Fig. 8 left, Table 3, Fig. 10, component autos) at the
native Nside=4096 using this package.

## Sky model (components stored separately)

Four components, each kept as its own map — **not summed together** at this
stage (beams, coaddition, and noise come later):

* **CMB** — Gaussian realization of the CAMB unlensed TT spectrum (FLAMINGO
  D3A / DES Y3 cosmology), lensed with the FLAMINGO kappa map via pixell.
* **tSZ** — lensed Compton-y map × non-relativistic f(nu) = x coth(x/2) - 4.
* **kSZ** — lensed Doppler-b map, dT = -T_CMB * b (frequency independent).
* **CIB** — released bandpass maps at 217/353/545/857 GHz (Jy/sr → uK_CMB);
  100/143 GHz via greybody-SED scaling from the nearest band at z_eff = 1.5.

Instrumental noise and beam convolution are **not** applied yet.

## Data

* Input: `/rds/flamingo/L2800N5040/HYDRO_FIDUCIAL/lightcone0_shells`
  (HEALPix Nside=4096 RING FITS; see `data_description.md`).
* Output: `/rds/rds-lxu/flamingo/integrated_maps_synthetic`
  (`components/{cmb,cib,tsz,ksz}/` — per-component, beam-unconvolved,
  Planck frequencies 100/143/217/353/545/857 GHz).

## Install

```bash
pip install -e .          # core (tSZ, kSZ, CIB component builders)
pip install -e ".[cmb]"   # + camb, pixell for the lensed CMB step
```

## Usage

```bash
# Build all four components (no coaddition), Planck 6-channel, Nside=4096
flamingo-mock-maps build

# Or use the notebook:
#   notebooks/build_synthetic_component_maps.ipynb

# Quick low-resolution test
flamingo-mock-maps build --nside 256 --frequencies 100 217 857

# Skip CMB lensing (reuse cached lensed CMB if present)
flamingo-mock-maps build --steps tsz ksz cib
```

Or from Python:

```python
from flamingo_mock import MockConfig
from flamingo_mock.config import PLANCK_FREQUENCIES_GHZ
from flamingo_mock import cmb, tsz, ksz, cib

cfg = MockConfig(frequencies=PLANCK_FREQUENCIES_GHZ, nside=4096)
cfg.make_dirs()
cmb.make_lensed_cmb(cfg, out_dir=cfg.components_dir / "cmb")
tsz.make_tsz_maps(cfg, out_dir=cfg.components_dir / "tsz")
ksz.make_ksz_map(cfg, out_dir=cfg.components_dir / "ksz")
cib.make_cib_maps(cfg, out_dir=cfg.components_dir / "cib")
```

## Layout

```
src/flamingo_mock/
  config.py    paths, constants, cosmology, beams, MockConfig
  spectral.py  tSZ/kSZ responses, Jy/sr <-> uK, CIB greybody SED
  io.py        HEALPix map I/O, copy/link helpers
  cmb.py       CAMB spectrum + pixell lensing (optional deps)
  tsz.py       Compton-y -> dT(nu)
  ksz.py       Doppler-b -> dT
  cib.py       released bands + interpolation/SED scaling
  sky.py       optional coadd helper (not used in default workflow)
  powerspectra.py  anafast-based C_ell estimators, binning, decorrelation
  cli.py       flamingo-mock-maps entry point
  szifi/       iMMF / sciMMF cluster finding (flamingo-szifi)

scripts/       one-off drivers; names are the grouping
  build_l1_* / download_* / make_lensed_* / check_l1_*
               L1_m9 component + total-map making
  run_hilc.py / plot_hilc_* / configs/hilc_y_flamingo_homog*.yml
               homog HILC y-maps (pyILC on needlet_ilc)
  run_szifi_* / plot_szifi_* / benchmark_szifi_* / backfill_*
               SZiFi catalogues, N(q), purity, zooms
  compute_* / plot_l1_m9_*
               MMF noise curves, SOAP photometry, feedback ratios
  regenerate_* / pub_style.py
               paper figures
```
