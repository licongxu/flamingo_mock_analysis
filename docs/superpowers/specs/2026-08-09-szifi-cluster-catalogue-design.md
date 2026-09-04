# SZiFi Cluster Catalogue on FLAMINGO Mock Planck Skies — Design

**Date:** 2026-08-09  
**Branch:** `szifi_branch`  
**Status:** Approved (user)

## Goal

Produce galaxy-cluster catalogues at detection SNR \(q > 5\) from FLAMINGO mock multi-frequency skies using SZiFi (iMMF and sciMMF), with Planck HFI beams, NPIPE noise, and PR4 Galactic + point-source masks.

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| MMF products | Both **iMMF** and **sciMMF** (`deproject_cib=["cib"]`) |
| Noise | NPIPE detector-set **A** first, then **A+B** |
| Sky coverage | Pilot tiles (~4–8 high-\|b\|) → then all tiles surviving GAL×PS + `min_ftile` |
| Signal beam | Gaussian Table I FWHMs (`BEAM_FWHM_ARCMIN`) |
| SZiFi filter beam | `beam="gaussian"` matching the same FWHMs |
| Total maps | May load existing on-disk **total** sky products (coadd+beam+noise); **no ILC algorithm** |
| Redshift / halo match | Deferred (user will specify later) |
| Delivery order | Notebook pilot first, then package (`flamingo_mock.szifi` + CLI) |

## Architecture

```
components/ + planck_noise/  OR  existing total-map FITS (K_CMB, Nside=2048)
        │
        ▼
prepare: multi-freq tile cutouts + PR4 GAL/PS masks (apodised) + coupling matrices
        │
        ▼
run: SZiFi GPU (SZIFI_ARRAY_BACKEND=jax) × {iMMF, sciMMF}
        │
        ▼
catalogue: merge tiles, threshold q>5 → /rds/.../szifi/
```

SZiFi does **not** depend on `flamingo_mock.ilc` or pyILC. Total maps under `ilc/inputs_*` are optional storage of coadded skies, not ILC outputs used as y-maps.

## Inputs

- **Frequencies:** 100, 143, 217, 353, 545, 857 GHz  
- **Components (if rebuilding):** CMB + tSZ + kSZ + CIB, µK_CMB, Nside=4096 under `.../components/`  
- **Noise:** `.../planck_noise/npipe/{freq}GHz/{A,B,full}/` — **not** beam-convolved  
- **Masks:** `.../masks/pr4_nilc/Masks.fits` fields 1 (GAL) and 2 (PS)  
- **Working Nside:** 2048  
- **SZiFi fork:** `/scratch/scratch-lxu/agent_dev/auto_research_agent/szifi` with `SZIFI_ARRAY_BACKEND=jax`

## Outputs

Root: `/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/`

```
tiles/splitA/     # cutouts + masks per field_id
pilot/            # Phase 0 catalogues
catalogues/       # merged q>5 iMMF + sciMMF (A, later B)
```

Catalogue columns follow SZiFi (`q_opt`, `lon`, `lat`, `theta_500`, …). No redshift required for `extraction_mode="find"`.

## Phased delivery

### Phase 0 — Notebook pilot

`notebooks/szifi_pilot.ipynb` plus thin helpers under `src/flamingo_mock/szifi/` so Phase 1 is not a rewrite.

- Select ~4–8 high-|b| tiles (`nside_tile=8` geometry: 1024², 14.8°)  
- Split A only  
- Run iMMF + sciMMF  
- Write pilot catalogues under `.../szifi/pilot/`

### Phase 1 — Package + CLI

`flamingo-szifi {prepare,run,catalogue}` wrapping the same helpers; scale to all usable tiles; then split B.

## Non-goals (v1)

- Halo lightcone matching / redshift assignment  
- ILC / needlet y-map cluster finding  
- Galactic foregrounds or radio point sources in the sky model  
- Full-mission `full` noise as the primary product (A/B first)

## Success criteria

1. Pilot notebook produces non-empty iMMF and sciMMF catalogues with \(q > 5\) on split A.  
2. Package can regenerate those catalogues from CLI.  
3. Masked-sky run writes merged catalogues under `/rds/.../szifi/catalogues/`.  
4. No dependency on the ILC algorithm.
