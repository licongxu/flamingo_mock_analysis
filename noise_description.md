# Planck Instrumental Noise — Description

Reference for the Planck noise available for mock analyses, how it is produced,
and how to extend coverage beyond what is already on disk.

**Primary references:** Planck Legacy Archive
[Simulation data](https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Simulation_data)
and [NPIPE Introduction](https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/NPIPE_Introduction)
wiki pages; the NPIPE paper, Planck Collaboration Int. LVII, *A&A* 643, A42
(2020), [arXiv:2007.04997](https://arxiv.org/abs/2007.04997); and the PR4
cosmology likelihood paper, Tristram et al., *A&A* 682, A37 (2024).

Companion notebook: `notebooks/download_planck_noise.ipynb` fetches from the PLA
into `/rds/rds-lxu/flamingo/integrated_maps_synthetic/planck_noise/npipe/`.

---

## 1. Where the noise comes from

The Planck noise is **not simulated locally**. Unlike the Simons Observatory
noise (a Gaussian random field drawn on the fly from analytic noise curves), the
Planck noise consists of official realisations downloaded from the Planck Legacy
Archive at `pla.esac.esa.int` through its AIO interface.

Canonical product URLs look like
`.../product-action?SIMULATED_MAP.FILE_ID=npipe6v20_noise_100_A_mc_00200.fits`.

The FITS headers carry NPIPE mapmaking metadata that confirms these are real
end-to-end pipeline products rather than hand-rolled simulations: a best-fit
dipole amplitude (`DIPOAMP`), an overall gain (`GAIN`), a temperature offset
(`OFFSET`), and an effective central-frequency difference (`DFREQ`).

## 2. What the NPIPE noise actually is

NPIPE noise MCs are **end-to-end residual maps**. Simulated time-ordered data
are pushed through the same calibration, Madam destriping and mapmaking as the
flight data, and the noise product is the *output minus input* difference. Two
consequences:

- It is **not white**. It retains correlated low-frequency (1/f) structure and
  scanning-direction striping that survive destriping.
- It is **not noise-only**. Being a residual, it also contains whatever
  systematics survive the pipeline, including signal-proportional gain errors.

The realism is inherited from the actual mission — noise levels, correlation
lengths and sky coverage all come from the real instrument, not from a model.

**Format.** HEALPix `nside = 2048`, RING ordering, Galactic coordinates, `float32`.
The 353 GHz files hold three fields (`TEMPERATURE`, `Q_POLARISATION`,
`U_POLARISATION`); the pipeline reads `field=0` (temperature) only. The 545 and
857 GHz files are temperature-only.

## 3. Sky coverage: full sky, but strongly inhomogeneous

The maps are **genuinely full sky with no mask applied**.

**[verified]** All 50,331,648 pixels of the on-disk 353/545/857 realisations:
`nan=0, unseen=0, exact_zero=0`, and headers declare `OBJECT='FULLSKY'` with
`INDXSCHM='IMPLICIT'` (every pixel present). This is expected — Planck was an
all-sky survey, and the noise MCs inherit that coverage. Masks (common
confidence, point-source, Galactic plane) are shipped as *separate* PLA
products and are **not** pre-applied.

But the depth varies enormously. Planck scanned in rings roughly about the
ecliptic axis, so integration time is strongly position-dependent. Measuring the
local RMS of the 545 GHz map within 3072 regions (nside = 16):

| Statistic | RMS (MJy/sr) |
|---|---|
| minimum | 1.77e-04 |
| 5th percentile | 3.73e-04 |
| median | 5.80e-04 |
| 95th percentile | 2.34e-03 |
| maximum | 4.15e-02 |

That is a **234× spread** between the quietest and noisiest regions (6.3× between
the 5th and 95th percentiles, so not driven by a few pathological pixels).
Splitting by ecliptic latitude: regions within 15° of the ecliptic plane are on
average **3× noisier** than those within 15° of the ecliptic poles.

### Practical consequences

- **A single global noise level is meaningless.** Any quoted Planck σ must be
  region-specific; forecasts using a sky-averaged value are optimistic in the
  ecliptic plane and pessimistic at the poles.
- **Noise realisations are tied to sky position.** You cannot rotate or relocate
  a noise patch without changing its statistics. This is why the analysis code
  restricts Planck patches to 35° < |b| < 70° and requires the noise index to
  match the sky patch index — whereas the homogeneous SO noise patches are
  freely interchangeable.
- **This couples to ILC weights.** A harmonic ILC estimating one global
  covariance per multipole bin implicitly assumes statistical homogeneity, which
  this noise badly violates. That is a large part of the motivation for working
  in patches, restricting to a clean sky fraction, or using a needlet ILC that
  localises in both scale and position.

To quantify the depth directly rather than inferring it from the noise, the PLA
provides matching hit-count maps (`npipe6v20_{freq}_hmap.fits`) and white-noise
covariance maps (`wcov_mcscaled` for 100–353, `wcov_hrscaled` for 545/857).

## 4. Noise realisation count — a hard limit on ILC variance

Only **100** NPIPE realisations (indices 00200–00299) are on the PLA. The full
release has 600 full-frequency and detector-set realisations (and 100 of those
also include single-detector and half-ring maps), but volume constraints mean
only the 100 are exposed on the PLA; the rest need a NERSC account
(`/global/cfs/cdirs/cmb/data/planck2020`).

100 samples matters more than it might seem. A standard ILC result is that
estimating the frequency covariance from the data itself biases the recovered
power by roughly **N_freq / N_noise** — for 6 channels and 100 realisations that
is about 6% scatter in the noise bias alone, before any sample variance. By
contrast FFP10 offers 300 realisations, and the SO pipeline uses 3000 cheap
Gaussian realisations, so the Planck side is the variance-limited part of any
joint SO+Planck mock. Mitigations: use NERSC access for more realisations, fall
back to FFP10 (300), or regularise the covariance estimate.

## 5. Frequency coverage: 100–857 GHz is fully available

The archive covers every Planck frequency. For the six HFI channels of interest:

| Frequency (GHz) | NPIPE | FFP10 | Native nside | Typical full-map size |
|---|---|---|---|---|
| 100 | yes | yes | 2048 | ~576 MB |
| 143 | yes | yes | 2048 | ~576 MB |
| 217 | yes | yes | 2048 | ~576 MB |
| 353 | yes | yes (353_psb also) | 2048 | ~576 MB |
| 545 | yes | yes | 2048 | ~192 MB |
| 857 | yes | yes | 2048 | ~192 MB |

Anilipour's existing setup uses 90/150/217 with **simulated SO noise** and only
353/545/857 with real Planck noise, because its goal was a joint SO+Planck
experiment. The present repo instead targets the six Planck HFI channels
(100–857, per `reference_tables/planck_info.png` Table I and
`flamingo_mock.config.PLANCK_FREQUENCIES_GHZ`). So for a Planck-only analysis,
**100/143/217 noise is needed but is not yet on disk** — downloading it is the
purpose of `notebooks/download_planck_noise.ipynb`.

## 6. Splits: NPIPE A/B detector sets vs FFP10 ring cuts

The two suites use fundamentally different split philosophies, and this drives
the null-test strategy.

**NPIPE (PR4) — detector-set A/B splits.** Template:

```
npipe6v20_noise_{frequency}_{coverage}_mc_{realization}.fits
```

- `{coverage}` is omitted for full-frequency maps, or `A` / `B` for detector-set
  splits (`npipe6v20_noise_100_mc_00200.fits` vs `…_100_A_mc_00200.fits`)
- `{realization}` runs 00200–00299 — **zero-padded to 5 digits**. The PLA wiki
  text says "four digits"; its own examples and the live archive use five.
  `mc_0200` returns HTTP 404; `mc_00200` returns 200.
- `{frequency}` is unpadded for HFI (`100`, not `0100`)

**[verified]** Live PLA probes (Aug 2026): full, `_A`, and `_B` all return
HTTP 200 for 100/143/217/353 at `mc_00200`. Alt names `_full_`, 4-digit
`mc_0200`, and zero-padded `0100` all 404.

NPIPE reprocesses A and B **independently** — destriping and systematics
template fitting are done separately — so the halves have genuinely reduced
correlations. PR4 likelihoods (e.g. CamSpec) use A×B cross-spectra for noise
estimation. The on-disk anilipour files
(`npipe6v20_noise_353_mc_00200.fits`, no coverage token) are the full-frequency
variant.

**FFP10 (PR3) — ring-based splits.** Template:

```
ffp10_noise_{frequency}_{ring_cut}_map_mc_{realization}.fits
```

- `{ring_cut}` is one of `full`, `hm1`, `hm2`, `oe1`, `oe2`
  (full mission, half-missions 1/2, odd/even rings)
- `{realization}` runs 00000–00299 (5-digit zero-padded), so **300** realisations

Quirk: due to the polarisation orientation of the 100 GHz bolometers, the
odd/even ring maps are badly conditioned at nside 2048 and are additionally
provided at nside 1024 (substitute `_map_1024_mc_`).

| | NPIPE (PR4) | FFP10 (PR3) |
|---|---|---|
| Split type | Detector sets A / B (or omit for full) | Ring cuts: full, hm1, hm2, oe1, oe2 |
| Realisations on PLA | 100 (00200–00299) | 300 (00000–00299) |
| Frequencies | all nine (030–857) | HFI 100–857 (+ 353_psb) |
| Index padding | **5 digits** | 5 digits |
| Processing | newer, lower noise/systematics | older PR3 |

**Which to use?** If 100 realisations is too few for covariance estimation, or
if you want a clean ring-based null test, FFP10 at 100/143/217 is the path of
least resistance: three times the realisations, directly downloadable. The
trade-off is that FFP10 is the older PR3 processing with somewhat higher noise
and systematics than NPIPE.

## 7. Units

This is the single most common source of error, and it already bit the existing
pipeline (see `data_description.md` §6 and §9).

The cutting code saves projected noise with **no unit conversion**, so stored
arrays carry raw NPIPE units:

- **K_CMB**: 100, 143, 217, 353 GHz → scale to µK_CMB with `×1e6`
- **MJy/sr**: 545, 857 GHz → scale with `×1e6 × jysr2uk(freq)`

`jysr2uk(freq)` is the Jy/sr → µK_CMB thermodynamic conversion at Planck 2015
T_CMB (= 1.751e-02 at 545, 6.965e-01 at 857; see `data_description.md` §6 for
the amplitude test that fixes this). Applying a flat `×1e6` to 545 GHz inflates
its noise by ~57× and is a known bug in `jax_compsep_multi.py`. The official
beam FWHMs needed for smoothing the noise are tabulated in
`flamingo_mock.config.BEAM_FWHM_ARCMIN`.

## 8. Download instructions

See `notebooks/download_planck_noise.ipynb` for an executable walkthrough. In
brief, each file is

```
https://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=npipe6v20_noise_{freq}_A_mc_{real:05d}.fits
```

with `{freq}` in {100, 143, 217, 353, 545, 857} and `{real}` in 00200–00299
(5-digit pad). Use `_A` / `_B` for detector-set splits, or omit the coverage
token for full-frequency maps.
Files are stored under

```
/rds/rds-lxu/flamingo/integrated_maps_synthetic/planck_noise/npipe/{freq}GHz/
```

using a `*.part` temp file that is renamed to the final name only on a
successful HTTP 200 — so the downloader is idempotent and resumable.

## 9. Known pitfalls (checklist)

1. **100 realisations max** on the PLA → ILC covariance bias; see §4.
2. **Units differ across bands** (K_CMB vs MJy/sr); see §7.
3. **"Noise" is a residual** (output−input), not pure noise; see §2.
4. **Anisotropic depth**, ~234× spread; see §3.
5. **Provenance is NPIPE, not FFP10** — the older `compsep_data/data_description.md`
   mislabels it FFP10.
6. **Tied to sky position** — do not rotate or relocate noise patches; see §3.
