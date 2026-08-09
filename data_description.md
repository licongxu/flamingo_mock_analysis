# FLAMINGO Mock CMB Data — Description of On-Disk Products

Reference document for the FLAMINGO-derived mock CMB datasets available on this
machine under `/rds/flamingo`, plus the derived products and noise archives held
in collaborator directories.

**Primary source document:** Yang et al., *"Self-consistent secondary cosmic
microwave background anisotropies and extragalactic foregrounds in the FLAMINGO
simulations"*, [arXiv:2512.09891](https://arxiv.org/abs/2512.09891) (v2, 30 Mar
2026). A local copy is in this repository at `reference_paper/yang26flamingo.pdf`.
Section 3 of that paper is the authoritative description of how every map was
constructed. No data-description file ships with the full-sky maps themselves.

Statements below are marked **[verified]** where checked directly against the
files on disk, and **[inferred]** where deduced from indirect evidence.

---

## 1. Directory inventory

| Path | Size | Contents |
|---|---|---|
| `/rds/flamingo/L2800N5040/` | 2.5 TB | Full-sky maps, 2.8 Gpc fiducial run |
| `/rds/flamingo/Jeger_rot/` | 20 GB | Full-sky maps, 1 Gpc `L1_m9` run |
| `/rds/flamingo/halo_cat/` | 677 GB | Halo lightcone catalogues |
| `/rds/flamingo/compsep_data/` | 239 GB | Derived flat-sky patches + noise |
| `/rds/rds-anilipour/planck_noise/` | 94 GB | Full-sky NPIPE noise realisations |

---

## 2. Provenance

`/rds/flamingo/L2800N5040` is a local copy of Tianyi Yang's derived-map
directory on COSMA8: `/cosma8/data/dp004/dc-yang3/maps/L2800N5040/HYDRO_FIDUCIAL/`.
The upstream raw FLAMINGO lightcone shells live at
`/cosma8/data/dp004/flamingo/Runs/L2800N5040/HYDRO_FIDUCIAL/neutrino_corrected_maps_downsampled_4096/lightcone0_shells/`.

The paper's Data Availability section states the maps "will be made publicly
available on the FLAMINGO data release website upon acceptance", so as of the v2
preprint there is no public release page; these local copies predate it.

### Simulation runs

`L2800N5040` is the FLAMINGO flagship large-volume run, denoted `L2p8_m9` in the
paper: comoving box 2.8 Gpc, 5040³ particles per species (2.8 × 10¹¹ total),
intermediate resolution `m9` (gas particle mass 1.1 × 10⁹ M☉), DES Y3
3×2pt+All Ext. ΛCDM cosmology, `HYDRO_FIDUCIAL` calibrated feedback. It was the
largest hydrodynamical simulation evolved to z = 0 at the time it was run.

`Jeger_rot/L1_m9` is the smaller 1 Gpc intermediate-resolution run under a
specific ("Jeger") rotation realisation. Lightcone outputs for the 1 Gpc boxes
reach only z = 3.0, versus z = 4.5 for the 2.8 Gpc run.

---

## 3. Full-sky map format

**[verified]** All full-sky maps are HEALPix `nside = 4096`, `RING` ordering,
single-column `float32`, ~805 MB each. FITS headers contain only `NSIDE` and
`ORDERING` — no units, no coordinate system, no comments. Coordinates are
Galactic (per the paper). Pixel window deconvolution via `healpy.pixwin` is
required when computing spectra.

### Lightcone construction (paper Section 3.1)

The observer's past lightcone is split into concentric spherical shells with
Δz = 0.05 from z = 0 to 3, and Δz = 0.25 above. Particle contributions are
accumulated per shell, then stacked. To avoid repeating structures along the
line of sight, shells are randomly rotated every box-length; **the same
rotations are applied to every quantity**, which is what preserves the physical
cross-correlations between CIB, tSZ, lensing etc. This is the meaning of the
`_rot` in the filenames; `kappa_non_rot.fits` and `CIB_nonrot_*` are unrotated
counterparts.

`/rds/flamingo/L2800N5040/HYDRO_FIDUCIAL/` contains `lightcone0_shells` through
`lightcone4_shells` (5 independent observers; the paper averages over 8).

### Physical quantities

| File stem | Quantity | Units |
|---|---|---|
| `ComptonY_rot`, `tSZ_rot` | Compton y (paper Eq. 7) | dimensionless |
| `DopplerB_rot`, `kSZ_rot` | Doppler b (Eq. 8), ΔT_kSZ/T_CMB = −b, T_CMB = 2.73 K | dimensionless |
| `kappa_rot`, `CMB_lensing_rot` | CMB lensing convergence κ | dimensionless |
| `TotalMass_rot`, `DM_rot` | total matter / dispersion measure | see Sec. 3.2 |
| `CIB_rot_BANDPASS_F{217,353,545,857}` | CIB, bandpass-integrated | Jy/sr |
| `CIB_rot_{90,150}_deltabp` | CIB, delta bandpass | Jy/sr |
| `StarFormationRate_rot_BANDPASS_F*` | per-shell SFR (CIB precursor) | — |
| `lensed_T_e_857_rot` | y-weighted electron temperature ⟨T_e,y⟩ | keV |
| `delta_I_rel_857_rot` | relativistic tSZ intensity correction (SZpack COMBO) | intensity |
| `lensed_radiops_rot_llim_*_MBHcut` | radio point sources, 3 Eddington-ratio cuts, z < 2.5 | — |

### Filename suffix conventions

- `lensed_` — deflected with `pixell`, lensed shell by shell using the
  integrated κ map up to that shell (paper Section 3.7).
- `_three_params` — the default CIB SED model, with β_d, T₀ and α free. The
  paper also describes four-parameter and "extended" variants (Appendix A);
  `_three_params` is the fiducial choice.
- `_seed{5,15,20,25,30}` — alternative CMB/realisation seeds.
- `_z3` — integrated only to z = 3.0 rather than z = 4.5.
- `_0`, `_1`, `_2` — **[inferred]** cumulative redshift depth, not chunks to be
  summed. Evidence below.
- `_all` — **[inferred]** a slightly more inclusive stack than the plain name.

**[verified]** Measured statistics for `lightcone0_shells`:

```
ComptonY_rot_0      mean=7.048e-07  std=1.742e-06
ComptonY_rot_1      mean=1.413e-06  std=2.062e-06
ComptonY_rot_2      mean=1.611e-06  std=2.076e-06
ComptonY_rot_0_z3   mean=7.048e-07  std=1.742e-06   <- bit-identical to _0
tSZ_rot             mean=1.611e-06  std=2.076e-06   <- bit-identical to _2
tSZ_rot_all         mean=1.622e-06  std=2.076e-06   <- marginally larger
DopplerB_rot_0/1/2  mean ~ -1.3e-07, std 4.66e-07 -> 8.34e-07 -> 1.53e-06
```

The index looks cumulative: mean and scatter grow monotonically, `_0` is
bit-identical to the explicitly labelled `_0_z3` (so index 0 is the z < 3
integration), and `_2` is bit-identical to `tSZ_rot.fits` (so index 2 is the
full-depth z < 4.5 map). For Doppler b the mean stays near zero while the
scatter triples, as expected when stacking more shells of a zero-mean field.

**Caveat:** `ComptonY_rot_0` has a strictly positive minimum (2.5 × 10⁻⁸) while
`_1` and `_2` go negative (−1.2 × 10⁻⁵), which is unphysical for Compton y and
does not fit a naive cumulative sum. It probably originates in the neutrino
correction or the downsampling/pixel-window deconvolution applied upstream.
**Confirm the exact index semantics with Tianyi Yang (t.yang@ljmu.ac.uk) before
relying on them.**

---

## 4. Halo lightcone catalogues

`/rds/flamingo/halo_cat/lightcone0/lightcone_halos_{0000..0078}.hdf5` (79 files).

**[verified]** These are self-documenting SOAP/HBTplus outputs: every dataset
carries a `Description` attribute plus CGS conversion factors and `a-scale` /
`h-scale` exponents. No external description is needed. Key fields include
`Lightcone/Redshift` ("Redshift of the halo at the time of lightcone crossing"),
`Lightcone/HaloCentre`, `Lightcone/SnapshotNumber`, `BoundSubhalo/TotalMass`,
`InputHalos/IsCentral`, `InputHalos/HBTplus/TrackId`. An `Index` group provides
`FirstHaloInPixel` / `NumHalosPerPixel` on an `nside = 16`, `nest` grid.

Note `lightcone_halos_0000.hdf5` is empty (all datasets have shape 0).

Construction (paper Section 3.1): structure finding is done in post-processing
with HBT-HERONS; for each halo a black hole particle is used as a tracer, and
wherever that tracer ID appears in the black hole particle lightcone a copy of
the halo is placed at that position.

---

## 5. The mock sky model

**The sky model has exactly four components: lensed CMB + tSZ + kSZ + CIB.**

Built by `lensed_full_data.py` (see Section 7) as:

```python
smap = cib*jysr2uk(freq) + tsz*tsz(freq) + ksz*ksz(freq) + lensed_cmb
smap = hp.sphtfunc.smoothing(smap, fwhm=beam_fwhm[freq])
```

Everything is converted to µK_CMB, summed on the full sky, smoothed with the
per-frequency instrument beam, then cut into patches. Instrumental noise is
**not** baked in; it is added at analysis time from separate files.

Component sources (all from the 1 Gpc `L1_m9` run, i.e. `Jeger_rot`):

- tSZ — `lensed_tSZ_rot_Jeger_rot.fits` (Compton y)
- kSZ — `lensed_kSZ_rot_Jeger_rot.fits` (Doppler b)
- CIB — `lensed_CIB_rot_{90,150}_deltabp_three_params_Jeger_rot.fits` and
  `lensed_CIB_rot_BANDPASS_F{217,353,545,857}_three_params_Jeger_rot.fits`
- CMB — Gaussian realisation from CLASS, lensed with the FLAMINGO κ map

### Two things that are *not* additive components

**SFR is not a sky component.** The `StarFormationRate_rot_BANDPASS_F*` maps are
the *precursor* of the CIB: upstream, per-shell SFR maps are converted into
infrared emission through the dust SED model (the β_d, T₀, α fit) to produce the
`CIB_rot_BANDPASS_F*_three_params` maps. Adding SFR on top of CIB would double
count the same physical emission. **[verified]** `StarFormationRate` and
`sfr_zmid` appear nowhere in the analysis pipeline scripts.

**κ is used but not added.** `CMB_lensing_rot_Jeger_rot.fits` is read, converted
to a lensing potential (`fl = 2/(ℓ(ℓ+1))`) and then a deflection field, and fed
to `lenspyx.alm2lenmap` to lens the CMB realisation. The tSZ, kSZ and CIB maps
were already lensed upstream (hence their `lensed_` prefixes), so lensing is
applied consistently to all four components.

### Deliberately excluded

Radio point sources (`lensed_radiops_rot_llim_*`, which do exist on disk),
relativistic tSZ corrections (`delta_I_rel_857`, `lensed_T_e_857_rot`), optical
depth / patchy screening, and total-matter / dispersion-measure maps.

**No Galactic emission of any kind** — no thermal dust, synchrotron or
free-free. This is a purely extragalactic sky. This matters for mock Planck work,
since real 545 and 857 GHz data is dominated by Galactic dust.

### CMB realisation

Gaussian realisation of the CLASS unlensed TT spectrum, then lensed with the
FLAMINGO κ via `lenspyx` at target accuracy ε = 10⁻⁶, `nside = 4096`.

Cosmology used in `lensed_full_data.py` (the FLAMINGO DES Y3 cosmology):
h = 0.681, Ω_m = 0.306, Ω_b = 0.0486, Ω_ν = 1.39e-3, A_s = 2.099e-9,
n_s = 0.967, τ_reio = 0.054, N_ur = 2.0328, N_ncdm = 1, m_ncdm = 0.06 eV,
T_CMB = 2.7255 K.

> **Discrepancy:** `/rds/flamingo/compsep_data/data_description.md` instead
> quotes a Planck-like cosmology (A_s = 2.1e-9, n_s = 0.965, h = 0.6736,
> ω_b = 0.02237, ω_cdm = 0.12, τ = 0.0544, Y_He = 0.2454) with seed 42. The
> generating script's values are the ones actually used. Treat that file's
> cosmology table as unreliable.

---

## 6. Instrumental noise

Every channel gets noise, but no channel gets both instruments.

### 90 / 150 / 217 GHz — simulated Simons Observatory LAT

Gaussian random fields drawn from the official SO noise curves
(`so_models_v3`, `SOLatV3point1`, mode 2, elevation 50°, f_sky = 0.4), using
bands 93/145/225 GHz. The 90 and 150 GHz channels are drawn jointly from the
2×2 covariance via Cholesky factorisation so they are correlated; 217 GHz is
independent. 3000 realisations, generated directly at the 1.17′ patch pixel
scale.

Because it is a stationary GRF, this noise is statistically homogeneous — there
is no hit-count variation across the sky. **There is no full-sky version**: SO
noise is synthesised directly as 256×256 flat patches, so nothing can be
projected. Full-sky SO noise would have to be generated from scratch.

### 353 / 545 / 857 GHz — real Planck NPIPE noise Monte Carlos

**[verified]** Source: `/rds/rds-anilipour/planck_noise/npipe6v20_noise_{freq}_mc_{00200..00299}.fits`

- 94 GB total, 300 files: 100 realisations at each of 353, 545, 857 GHz
- `nside = 2048`, `RING`, Galactic, projected at native resolution (no upsampling)
- 353 GHz files are 604 MB with three fields (`TEMPERATURE`, `Q_POLARISATION`,
  `U_POLARISATION`); the pipeline reads `field=0` only
- 545 and 857 GHz files are 201 MB, temperature only
- No `TUNIT` cards in the headers

These are spatially inhomogeneous and realistic, which is why the analysis code
restricts Planck noise patches to 35° < |b| < 70°, avoiding both the Galactic
plane and the poorly behaved poles.

**This full-sky archive is the recommended entry point for full-sky mock Planck
work** — it lets you bypass the patch-cut arrays and the unit ambiguity below,
since you apply the conversion yourself.

### Planck noise units — resolved

**[verified]** The cutting code saves projected noise with **no** unit
conversion, so the stored `.npy` arrays carry raw NPIPE units. Comparing one
patch against the total sky signal in the same patch:

| Channel | signal std (µK_CMB) | noise `×1e6` | noise `×jysr2uk×1e6` |
|---|---|---|---|
| 353 | 134 | 49.8 | 0.2 |
| 545 | 735 | 16792 | **294** |
| 857 | 61454 | 16314 | **11363** |

At 545 GHz a flat `×1e6` makes the noise 23× larger than the entire sky signal,
which is unphysical. Conclusion: **353 GHz is in K_CMB (scale by `1e6`); 545 and
857 GHz are in MJy/sr (scale by `utils.jysr2uk(freq)*1e6`).**

Conversion factors: `jysr2uk(353) = 3.371e-03`, `jysr2uk(545) = 1.751e-02`,
`jysr2uk(857) = 6.965e-01`.

> **Known bug:** `jax_compsep_multi.py` applies a flat `×1e6` to every frequency
> above 217 GHz, over-scaling 545 GHz by ~57× and 857 GHz by ~1.44×. The
> `so+planck_compsep.ipynb` notebook uses the correct per-frequency factors. The
> flat factor came from a switch to NPIPE accompanied by the comment "new npipe
> uses K_CMB for all frequencies", which the amplitude test above contradicts
> for 545 GHz. Verify before reusing either code path.

> **Also note:** `compsep_data/data_description.md` labels this noise as Planck
> **FFP10**. The generating code reads **NPIPE** (`npipe6v20`). The units
> statement in that file is right; the provenance label is wrong.

---

## 7. Derived flat-sky patches (`compsep_data`)

Produced by `/scratch/scratch-anilipour/comp_separation/lensed_full_data.py`,
with helpers in the adjacent `utils.py`. A companion write-up of the projection
scheme is at `.../comp_separation/projection_methodology.md`.

### Patch geometry

| Quantity | Value |
|---|---|
| Patch size | 5° × 5° (25 deg²) |
| Grid | 256 × 256 pixels |
| Pixel resolution | 1.171875′ |
| Lat/lon step Δθ | 5° |
| Galactic cut | 0° (full sky tiled) |
| Number of patches | 1523 |
| Coordinates | Galactic |
| Projection | Gnomonic tangent-plane, bilinear interpolation |
| Base resolution | nside 4096 → upsampled to 8192 for signal; 2048 for Planck noise |

Patch centres are placed at latitudes stepped by Δθ, with longitude spacing
Δℓ(b) = Δθ/cos b so centres are separated by ~Δθ in true angular distance at
every latitude. Ordering follows `utils.get_patch_centers(gal_cut=0°,
step_size=5°)`. Cutting uses `healpy.projector.GnomonicProj` plus
`healpy.get_interp_val` (`utils.proj_bilinear`).

### Beams

Individual component maps are smoothed with a 1′ FWHM Gaussian before cutting.
The stacked maps use survey-specific beams:

| Freq (GHz) | 90 | 150 | 217 | 353 | 545 | 857 |
|---|---|---|---|---|---|---|
| FWHM (′) | 2.2 | 1.4 | 1.0 | 4.5 | 4.72 | 4.42 |

### Three flavours of stacked map

| File | Contents | Purpose |
|---|---|---|
| `stacked_{freq}` | CIB + tSZ + kSZ + lensed CMB | the mock observation |
| `signal_free_{freq}` | CIB + kSZ + lensed CMB (**no tSZ**) | contamination ensemble for tSZ recovery |
| `ksz_signal_free_{freq}` | CIB + tSZ + lensed CMB (**no kSZ**) | contamination ensemble for kSZ recovery |

Plus individual components: `tsz.npy` (Compton y), `ksz.npy` (Doppler b),
`lensed_cmb.npy` (µK_CMB), `cib_{freq}.npy` (Jy/sr). Each is
`(1523, 256, 256)`, `float64`, ~798 MB.

> **Important:** only `stacked_*` and the individual components are in
> `/rds/flamingo/compsep_data`. The `signal_free_*` and `ksz_signal_free_*` maps
> exist **only** in `/scratch/scratch-anilipour/comp_separation/data/cut_maps`.
> Since `jax_compsep_multi.py` loads `signal_free_{freq}.npy` unconditionally,
> it cannot run against the `/rds/flamingo` copy as-is.

> **Possible staleness:** in the scratch directory, `stacked_*`, `cib_*`, `ksz`
> and `lensed_cmb` are dated 23 July while `signal_free_*` are dated 15 April,
> so the signal-free ensemble may predate a regeneration of the signal maps.
> Check before treating them as a matched pair.

> `compsep_data/data_description.md` describes the signal-free maps as
> "CMB + CIB + WN only, without tSZ", but the code shows kSZ is also included.

> That file also documents a `ps/` tree (`ps_fs/`, `ps_mean/`, `ps_std/`) of
> angular power spectra that does **not** exist under `/rds/flamingo`. It lives
> at `/scratch/scratch-anilipour/comp_separation/data/ps`.

### Unit conversions and response functions (`utils.py`)

- `jysr2uk(nu)` — Jy/sr → µK_CMB using `astropy` thermodynamic temperature
  equivalency with the Planck 2015 T_CMB.
- `tsz(nu)` = `(X/tanh(X/2) − 4) * TCMB_uK` with `X = hν/kT_CMB`, T_CMB = 2.726 K.
  Returns µK_CMB per unit Compton y, so an ILC preserving this response outputs
  a y-map.
- `ksz(nu)` = `−TCMB_uK` (frequency independent in thermodynamic units), so an
  ILC preserving it recovers τ·(v_z/c).

---

## 8. ILC setup as used in `comp_separation`

Implemented in `/scratch/scratch-anilipour/comp_separation/jax_compsep_multi.py`
(classes `simpleILC_flatSky` and `BeamAwareILC`), with earlier equivalents inside
the `so+planck_*.ipynb` notebooks.

**Domain.** Flat-sky harmonic space. Each patch is mean-subtracted, apodised
with a 2D Tukey window (α = 0.1), and FFT'd. Power is corrected by
`1/mean(window²)`.

**Binning.** The 2D ℓ grid `ℓ = 2π|k|` is binned into `n_ell_bins = 50` linear
bins up to `lmax = 10000`. The frequency-frequency covariance is estimated
empirically from the data itself in each bin, i.e. this is a per-bin harmonic
ILC, not a needlet ILC.

**Beam handling.** `BeamAwareILC` deconvolves to a common target resolution
(`TARGET_FWHM_ARCMIN = 2.0`) by multiplying each channel's response by the beam
ratio `exp(−½ ℓ² (σ_ν² − σ_target²))` averaged within each ℓ bin, rather than
pre-smoothing the maps.

**Weights.** Constrained ILC (CILC): with the preserved response `a` and
deprojected responses `b_i` stacked into `R`, and `Q = R C⁻¹ Rᵀ`, weights are
formed from the cofactor expansion of `Q`, giving unit response to the target
and null response to each deprojected component. Singular bins fall back to
zero weight.

**Component choices differ by target** — this is the key configuration knob:

| Script / notebook | Preserved | Deprojected |
|---|---|---|
| `jax_compsep_multi.py` | tSZ | CIB |
| `so+planck_multifreq_compsep.ipynb` | tSZ | CIB |
| `so+planck_compsep.ipynb` | tSZ | CMB |
| `so+planck_ksz_compsep.ipynb` | CMB (for kSZ) | tSZ |

**SEDs used for deprojection.** tSZ as above. The CIB is a modified blackbody
with fiducial `T_dust = 24 K`, `β = 1.2`, `ν₀ = 353 GHz`, converted to
thermodynamic units via `1/dB_ν/dT`. Note this analytic CIB SED is *not* the
same as the three-parameter SED used to build the maps, so CIB deprojection is
inherently approximate.

**Channels.** The ILC uses all six frequencies (90–857). The downstream
scattering-transform denoising stage uses only the lower four (90, 150, 217, 353).

**Downstream.** The CILC y-map is the initial estimate for a scattering
transform denoising step, whose loss matches ST statistics of the residual
against an ensemble of "contamination" maps built by pushing the signal-free
maps through the saved ILC weights. See `description.md` and
`projection_methodology.md` in the same directory.

---

## 9. Summary of known documentation errors

Collected here because they are easy to trip over. All refer to
`/rds/flamingo/compsep_data/data_description.md`:

1. Planck noise is **NPIPE** (`npipe6v20`), not FFP10.
2. The cosmology table is Planck-like; the code actually uses the FLAMINGO
   DES Y3 cosmology.
3. Signal-free maps contain kSZ, contrary to the "CMB + CIB + WN only" wording.
4. The documented `ps/` directory is absent from `/rds/flamingo`.
5. `signal_free_*` / `ksz_signal_free_*` are not mentioned at all, and are not
   present under `/rds/flamingo`.

Separately, in `jax_compsep_multi.py`: the flat `×1e6` Planck noise scaling is
wrong for 545 and 857 GHz (see Section 6).
