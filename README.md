# flamingo_mock_analysis

Synthetic multi-frequency CMB sky maps from the FLAMINGO (Yang et al. 2026)
integrated lightcone maps. The exploratory analysis notebooks have been
packaged into importable source code under `src/flamingo_mock/`; the notebook
`notebooks/reproduce_yang26_power_spectra.ipynb` reproduces the paper's
power-spectrum figures (Fig. 8 left, Table 3, Fig. 10, component autos) at the
native Nside=4096 using this package.

## Sky model

Four components, summed in thermodynamic units (uK_CMB) on the full sky:

```
T_nu = CMB_lensed + dT_tSZ(nu) + dT_kSZ + dT_CIB(nu)
```

* **CMB** — Gaussian realization of the CAMB unlensed TT spectrum (FLAMINGO
  D3A / DES Y3 cosmology), lensed with the FLAMINGO kappa map via pixell.
* **tSZ** — lensed Compton-y map x non-relativistic f(nu) = x coth(x/2) - 4.
* **kSZ** — lensed Doppler-b map, dT = -T_CMB * b (frequency independent).
* **CIB** — released bandpass maps at 217/353/545/857 GHz (Jy/sr -> uK_CMB);
  other frequencies by log-frequency interpolation or greybody-SED scaling
  from the nearest band at z_eff = 1.5.

Instrumental noise is **not** included yet; it will be added per frequency
at a later stage.

## Data

* Input: `/rds/flamingo/L2800N5040/HYDRO_FIDUCIAL/lightcone0_shells`
  (HEALPix Nside=4096 RING FITS; see `data_description.md`).
* Output: `/rds/rds-lxu/flamingo/integrated_maps_synthetic`
  (`components/{cmb,cib,tsz,ksz}/` for per-component maps at Planck
  frequencies 100/143/217/353/545/857 GHz; beam-unconvolved).

## Install

```bash
pip install -e .          # core (tSZ, kSZ, CIB, coadd)
pip install -e ".[cmb]"   # + camb, pixell for the lensed CMB step
```

## Usage

```bash
# Full pipeline, Planck 6-channel (100–857 GHz), Nside=4096
flamingo-mock-maps build

# Or use the notebook (archives FLAMINGO maps + builds multifrequency products):
#   notebooks/build_synthetic_component_maps.ipynb

# Quick low-resolution test (inputs downgraded with ud_grade)
flamingo-mock-maps build --nside 256 --frequencies 90 217 857

# Skip the expensive CMB lensing (reuses cached components for the coadd)
flamingo-mock-maps build --steps tsz ksz cib coadd

# Also smooth coadds with per-frequency Gaussian beams
flamingo-mock-maps build --smooth
```

Or from Python:

```python
from flamingo_mock import MockConfig
from flamingo_mock import tsz, ksz, cib
from flamingo_mock.sky import make_coadd_maps

cfg = MockConfig(nside=256, frequencies=(90.0, 217.0, 857.0))
cfg.make_dirs()
tsz_uK = tsz.make_tsz_maps(cfg)
ksz_uK = ksz.make_ksz_map(cfg)
cib_uK = cib.make_cib_maps(cfg)
```

## Layout

```
src/flamingo_mock/
  config.py    paths, constants, cosmology, beams, MockConfig
  spectral.py  tSZ/kSZ responses, Jy/sr <-> uK, CIB greybody SED
  io.py        HEALPix map I/O
  cmb.py       CAMB spectrum + pixell lensing (optional deps)
  tsz.py       Compton-y -> dT(nu)
  ksz.py       Doppler-b -> dT
  cib.py       released bands + interpolation/SED scaling
  sky.py       per-frequency coaddition (+ optional beam smoothing)
  powerspectra.py  anafast-based C_ell estimators, binning, decorrelation
  cli.py       flamingo-mock-maps entry point
```
