# FLAMINGO Synthetic Maps — Data Description

All data relevant to the ILC pipeline on this machine lives under a single root:

```
/rds/rds-lxu/flamingo/integrated_maps_synthetic/
```

This document describes what is stored there, where it comes from, and how it
was produced. Nothing outside that tree is required to run the pipeline, except
that four CIB intensity archives under `components/cib/` are symlinks to the
Yang et al. FLAMINGO release (see §2).

**References**

| Topic | Source |
|---|---|
| FLAMINGO integrated maps & lightcone physics | Yang et al. (2026), [arXiv:2512.09891](https://doi.org/10.48550/arxiv.2512.09891) — `reference_paper/yang26flamingo.pdf` |
| Planck HFI channels & beams | Table I, `reference_tables/planck_info.png` |
| Planck NPIPE noise | `noise_description.md` |
| Needlet / harmonic ILC | McCarthy & Hill (2024), [arXiv:2307.01043](https://doi.org/10.48550/arxiv.2307.01043) — `reference_paper/pyilc.pdf` |

**Pipeline commands:** `flamingo-mock-maps build` → `flamingo-ilc prepare` → `config` → `run` → `validate`.

---

## 1. On-disk layout

```
/rds/rds-lxu/flamingo/integrated_maps_synthetic/
├── components/              # unbeamed per-component maps (§4–§5)
│   ├── cmb/
│   ├── tsz/
│   ├── ksz/
│   └── cib/
├── planck_noise/npipe/      # Planck NPIPE noise MCs (§7)
└── ilc/                     # ILC products (§6, §8; created by flamingo-ilc)
    ├── inputs_nside2048_npipe/
    ├── hilc_output_npipe_split{A,B}/
    └── nilc_output_npipe_split{A,B}/
```

| Subtree | Size (approx.) | Status |
|---|---|---|
| `components/` | 15 GB | populated |
| `planck_noise/npipe/` | 5 GB | populated |
| `ilc/` | — | not yet created |

---

## 2. Provenance — where these maps come from

The on-disk products are built in two stages from three distinct sources.

### 2.1 FLAMINGO simulation (Yang et al. 2026)

The secondary anisotropies and CIB originate in the **FLAMINGO** hydrodynamical
simulation (`L2p8_m9`, comoving box 2.8 Gpc, DES Y3 fiducial cosmology,
`HYDRO_FIDUCIAL` feedback). Yang et al. integrate particle/lightcone data into
full-sky HEALPix maps at *N*<sub>side</sub> = 4096:

- Lightcone shells are stacked from *z* = 0 to 4.5 with random shell rotations
  (the `_rot` suffix) applied consistently across all quantities, preserving
  physical cross-correlations between tSZ, kSZ, CIB, and lensing.
- Maps are **lensed** upstream (deflected shell-by-shell with the integrated κ
  map; paper §3.7) and stored at native integrated-map resolution — **no
  instrument beam** has been applied.
- We use **lightcone 0** (`lightcone0_shells`).

The upstream release files read during the build (not copied into
`integrated_maps_synthetic` except where noted) are:

| Upstream file | Physical quantity | Native unit |
|---|---|---|
| `lensed_tSZ_rot.fits` | Compton *y* | dimensionless |
| `lensed_kSZ_rot.fits` | Doppler *b* (Δ*T*/T = −*b*) | dimensionless |
| `kappa_rot.fits` | CMB lensing convergence κ | dimensionless |
| `lensed_CIB_rot_BANDPASS_F{217,353,545,857}_three_params.fits` | CIB specific intensity | Jy/sr |

These are the Yang et al. "integrated maps" described in their Section 3. FITS
headers carry only `NSIDE` and `ORDERING`; coordinates are Galactic.

### 2.2 `flamingo-mock-maps build` — component conversion

Command: `flamingo-mock-maps build` (`src/flamingo_mock/`).

This step converts the FLAMINGO physics maps (and a separately generated primary
CMB) into **thermodynamic temperature fluctuations** in µK<sub>CMB</sub> at six
Planck HFI frequencies (100, 143, 217, 353, 545, 857 GHz). Output goes to
`components/`.

| Component | Input | Transformation |
|---|---|---|
| **CMB** | FLAMINGO κ + CAMB *C*<sub>ℓ</sub><sup>TT</sup> | Draw unlensed Gaussian *T*, lens with κ via `pixell` (seed 42, FLAMINGO cosmology) |
| **tSZ** | Compton *y* | Δ*T* = *T*<sub>CMB</sub> · *y* · *f*(*x*), *f*(*x*) = *x* coth(*x*/2) − 4 |
| **kSZ** | Doppler *b* | Δ*T* = −*T*<sub>CMB</sub> · *b* (frequency-independent) |
| **CIB** | Released *I*<sub>ν</sub> [Jy/sr] | Exact at 217/353/545/857 GHz; log-interp or SED-scaled at 100/143 GHz; then Jy/sr → µK<sub>CMB</sub> |

The primary CMB is **not** taken from FLAMINGO — it is an independent Gaussian
realisation lensed with the simulation's κ field, matching the approach in
Yang et al. and the compsep mock-sky pipeline.

**What this step deliberately does not do:** coadd the four components, smooth
with Planck beams, or add noise. Those happen later (§6).

### 2.3 Planck NPIPE noise — external archive, local copy

The noise under `planck_noise/npipe/` is **not simulated locally**. It consists
of official Planck Legacy Archive NPIPE end-to-end residual maps (PR4), fetched
by `notebooks/download_planck_noise.ipynb` and stored under this tree for the
ILC pipeline. See `noise_description.md` for provenance, units, and coverage.

### 2.4 End-to-end chain

```
FLAMINGO L2p8_m9 simulation (Yang et al. 2026)
  └─ integrated lightcone maps: y, b, κ, CIB_I  [Nside 4096, lensed, unbeamed]
        │
        ▼  flamingo-mock-maps build
components/{cmb,tsz,ksz,cib}/  ← YOU ARE HERE (on disk)
  └─ per-component ΔT maps       [Nside 4096, µK_CMB, unbeamed, multifrequency]
        │
        ▼  flamingo-ilc prepare  (+ planck_noise/npipe/)
ilc/inputs_nside2048_npipe/
  └─ coadd + channel beam + NPIPE noise  [Nside 2048, K_CMB]
        │
        ▼  flamingo-ilc config → run → validate
ilc/{hilc,nilc}_output_npipe_split*/
  └─ Compton-y ILC maps, validated against compton_y truth
```

---

## 3. Sky model

Four extragalactic components; no Galactic foregrounds, radio sources, or
relativistic tSZ corrections.

| Component | Origin | Frequency dependence |
|---|---|---|
| Lensed primary CMB | CAMB + FLAMINGO κ | CMB blackbody (same in µK<sub>CMB</sub> at all ν) |
| tSZ | FLAMINGO Compton *y* | Non-relativistic *f*(*x*) |
| kSZ | FLAMINGO Doppler *b* | **None** |
| CIB | FLAMINGO bandpass maps + SED approx. | Per-frequency |

Cosmology (FLAMINGO DES Y3, Yang et al. Table 1): *h* = 0.681, Ω<sub>m</sub> = 0.306,
Ω<sub>b</sub> = 0.0486, *m*<sub>ν</sub> = 0.06 eV, *A*<sub>s</sub> = 2.099×10<sup>−9</sup>,
*n*<sub>s</sub> = 0.967, *T*<sub>CMB</sub> = 2.7255 K. CMB seed: **42**.

Instrumental noise is added from `planck_noise/npipe/` at ILC-prepare time, not
during component-map construction.

---

## 4. Component maps (`components/`)

**[verified on disk, 2026-08-09]**

### Format

| Property | Value |
|---|---|
| Pixelisation | HEALPix *N*<sub>side</sub> = 4096, RING |
| Dtype | float32 (~0.81 GB/map); CMB maps float64 (~1.6 GB) |
| Coordinates | Galactic |
| Beam | **Unconvolved** — native integrated-map resolution |
| Units | µK<sub>CMB</sub> for all Δ*T* maps |

These are **unbeamed multi-frequency component maps**. Each component is stored
separately at every Planck HFI frequency; there is no per-channel sky coadd and
no Planck beam smoothing at this stage.

Planck HFI parameters (Table I, `reference_tables/planck_info.png`):

| Freq (GHz) | Beam FWHM (′) | ILC channel |
|---|---|---|
| 100 | 9.66 | yes |
| 143 | 7.22 | yes |
| 217 | 4.90 | no |
| 353 | 4.92 | yes |
| 545 | 4.67 | no |
| 857 | 4.22 | no |

### File inventory

**CMB** — generated locally from CAMB + FLAMINGO κ:

| File | Quantity |
|---|---|
| `primary_CMB_T_lensed_nside4096_seed42.fits` | Lensed primary *T* |
| `primary_CMB_T_unlensed_nside4096_seed42.fits` | Unlensed primary *T* |
| `camb_cltt_unlensed_nside4096_seed42.npz` | CAMB *C*<sub>ℓ</sub><sup>TT</sup> used |

**tSZ** — from FLAMINGO Compton *y*:

| File | Quantity |
|---|---|
| `compton_y_nside4096.fits` | Compton *y* (ILC ground truth) |
| `tSZ_deltaT_{100,143,217,353,545,857}GHz_nside4096.fits` | tSZ Δ*T* per frequency |

**kSZ** — from FLAMINGO Doppler *b*:

| File | Quantity |
|---|---|
| `doppler_b_nside4096.fits` | Doppler *b* |
| `kSZ_deltaT_nside4096.fits` | kSZ Δ*T* (identical at all frequencies) |

**CIB** — from FLAMINGO released intensities:

| File | Quantity | Notes |
|---|---|---|
| `CIB_deltaT_{100,…,857}GHz_nside4096.fits` | CIB Δ*T* per frequency | primary working maps |
| `CIB_I_{217,353,545,857}GHz_nside4096.fits` | Released *I*<sub>ν</sub> [Jy/sr] | symlinks to Yang release (archival) |

100 and 143 GHz CIB Δ*T* maps are SED-scaled from 217 GHz at *z*<sub>eff</sub> = 1.5
(subdominant at those frequencies).

### Measured RMS (µK<sub>CMB</sub>, full sky)

| Map | std |
|---|---|
| CMB (lensed) | 108 |
| tSZ 100 GHz | 8.5 |
| tSZ 857 GHz | 63 |
| kSZ | 4.2 |
| CIB 353 GHz | 171 |

Frequency-dependent tSZ and CIB RMS confirms genuine multi-frequency maps.

---

## 5. Build details (reference)

Implementation in `src/flamingo_mock/{cmb,tsz,ksz,cib}.py`, spectral conventions
in `spectral.py`.

**CMB:** CAMB unlensed *C*<sub>ℓ</sub><sup>TT</sup> → Gaussian draw → κ → φ<sub>ℓm</sub>
→ `pixell.lensing.lens_map_curved`.

**tSZ:** *y* map copied to `compton_y_nside4096.fits`; Δ*T*<sub>ν</sub> = *T*<sub>CMB</sub> *y f*(*x*<sub>ν</sub>).

**kSZ:** Δ*T* = −*T*<sub>CMB</sub> *b*; one map serves all frequencies.

**CIB:** released bands used directly; 100/143 GHz via three-parameter greybody
SED (β<sub>d</sub> = 1.65, *T*<sub>0</sub> = 35.14 K, α = 0); Jy/sr → µK<sub>CMB</sub>
via d*B*<sub>ν</sub>/d*T*.

At each frequency the total signal (formed in memory, not written here) is:

```
S_ν = T_CMB + ΔT_tSZ(ν) + ΔT_kSZ + ΔT_CIB(ν)     [µK_CMB]
```

---

## 6. ILC inputs (`ilc/inputs_nside2048_npipe/`)

Created by `flamingo-ilc prepare --nside 2048`. Reads from `components/` and
`planck_noise/npipe/`; writes back under `ilc/`.

Per channel (100, 143, 353 GHz):

1. Coadd CMB + tSZ + kSZ + CIB (µK → K).
2. Beam-smooth with Planck HFI Gaussian (FWHM from Table I; ℓ<sub>max</sub> = 6000).
3. Downgrade to *N*<sub>side</sub> = 2048.
4. Add NPIPE noise splits A and B (`mc_00200`).

| File | Contents | Beam |
|---|---|---|
| `sky_CMB_tSZ_kSZ_CIB_signal_{ν}GHz_nside2048_K.fits` | Coadd, no noise | Channel Gaussian |
| `sky_CMB_tSZ_kSZ_CIB_npipe_split{A,B}_{ν}GHz_nside2048_K.fits` | Coadd + noise | Channel Gaussian |
| `compton_y_nside2048.fits` | Truth Compton *y* | Unbeamed (ud-graded only) |
| `input_manifest.json` | Provenance metadata | — |

| Stage | *N*<sub>side</sub> | Beam |
|---|---|---|
| `components/` | 4096 | **None** |
| ILC prepare inputs | 2048 | Planck channel Gaussian |
| pyILC internal | 2048 | Common 5′ (`perform_ILC_at_beam`) |
| Validation spectra | — | Deconvolved to 5′ vs truth *y* |

545/857 GHz components exist but are excluded from ILC (NPIPE noise at those
frequencies is MJy/sr, not K<sub>CMB</sub>; see `noise_description.md`).

---

## 7. Planck noise (`planck_noise/npipe/`)

Official NPIPE PR4 end-to-end residual maps from the Planck Legacy Archive.
Full description in `noise_description.md`.

| Freq (GHz) | Splits | Units | ILC |
|---|---|---|---|
| 100 | A, B, full | K<sub>CMB</sub> | yes (A, B) |
| 143 | A, B, full | K<sub>CMB</sub> | yes (A, B) |
| 353 | A, B | K<sub>CMB</sub> | yes (A, B) |
| 545 | A, B | MJy/sr | no |
| 857 | A, B | MJy/sr | no |

Default MC: `mc_00200`. Splits A and B are independently destriped (noise-decoupled cross-spectrum).

---

## 8. ILC outputs (`ilc/{hilc,nilc}_output_*`)

Expected after `flamingo-ilc config` and `flamingo-ilc run`:

- `hilc_output_npipe_split{A,B}/` — harmonic ILC Compton-*y* maps
- `nilc_output_npipe_splitA/` — needlet ILC (heavier)

Settings (generated YAMLs in `configs/`): ℓ<sub>max</sub> = 3000, preserved
component = tSZ, no deprojection, JAX backend.

Validation compares beam-deconvolved *C*<sub>ℓ</sub> of the ILC *y*-map against
`compton_y_nside2048.fits` via split A × split B.
