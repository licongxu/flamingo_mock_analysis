# SZiFi Cluster Catalogue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build iMMF + sciMMF cluster catalogues (\(q > 5\)) from FLAMINGO mock Planck HFI skies with NPIPE noise and PR4 masks.

**Architecture:** Thin `flamingo_mock.szifi` helpers prepare tiles from existing total maps (or rebuild coadd+beam+noise), adapt SZiFi survey I/O, run GPU SZiFi, merge catalogues. Notebook pilot first; CLI second.

**Tech Stack:** Python, healpy, numpy, SZiFi (`SZIFI_ARRAY_BACKEND=jax`), existing `flamingo_mock` package.

## Global Constraints

- No ILC algorithm dependency; total-map FITS may be reused as storage only.
- Six HFI channels; Gaussian beams from `BEAM_FWHM_ARCMIN`.
- Noise not beam-convolved; NPIPE split A first, then A+B.
- SZiFi experiment: `Planck_simple` + `beam="gaussian"` (RIMO files absent → `Planck_real` unavailable); override `FWHM` to Table I values; `integrate_bandpass=False`.
- Output root: `/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/`.
- Environment: `source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate`.

## File structure

| Path | Responsibility |
|------|----------------|
| `src/flamingo_mock/szifi/__init__.py` | Package export |
| `src/flamingo_mock/szifi/paths.py` | On-disk paths, tile geometry constants |
| `src/flamingo_mock/szifi/skies.py` | Locate/load total maps [K_CMB]; optional rebuild |
| `src/flamingo_mock/szifi/tiles.py` | Cutouts, mask projection, save npy products |
| `src/flamingo_mock/szifi/survey.py` | SZiFi `input_data_survey` adapter |
| `src/flamingo_mock/szifi/run.py` | iMMF / sciMMF runners + catalogue merge |
| `src/flamingo_mock/szifi/cli.py` | `flamingo-szifi` CLI (Phase 1) |
| `notebooks/szifi_pilot.ipynb` | Phase 0 end-to-end pilot |
| `tests/test_szifi_paths.py` | Path / FWHM / tile-id helpers |

---

### Task 1: Paths + sky map loading

**Files:**
- Create: `src/flamingo_mock/szifi/__init__.py`
- Create: `src/flamingo_mock/szifi/paths.py`
- Create: `src/flamingo_mock/szifi/skies.py`
- Create: `tests/test_szifi_paths.py`
- Modify: `pyproject.toml` (entry point later in Task 4)

**Interfaces:**
- Produces: `SZiFiPaths`, `FREQS_GHZ`, `TILE_NSIDE`, `TILE_NX`, `TILE_L_DEG`, `BEAM_FWHM_ARCMIN` re-export
- Produces: `total_map_path(split, freq) -> Path`, `load_total_maps_uK(split) -> dict[int, ndarray]`

- [ ] **Step 1: Write failing test for paths and FWHM order**

```python
from flamingo_mock.szifi.paths import FREQS_GHZ, beam_fwhm_vec_arcmin
from flamingo_mock.config import BEAM_FWHM_ARCMIN

def test_freqs_and_fwhm_order():
    assert FREQS_GHZ == (100, 143, 217, 353, 545, 857)
    assert list(beam_fwhm_vec_arcmin()) == [BEAM_FWHM_ARCMIN[f] for f in FREQS_GHZ]
```

- [ ] **Step 2: Run test — expect fail**

Run: `pytest tests/test_szifi_paths.py::test_freqs_and_fwhm_order -v`

- [ ] **Step 3: Implement `paths.py` and `skies.py`**

`paths.py`: constants + `SZiFiPaths` dataclass pointing at components, noise, masks, szifi out, and optional total-map dir (default existing `ilc/inputs_nside2048_npipe` for storage only).

`skies.py`: resolve `sky_CMB_tSZ_kSZ_CIB_npipe_split{A|B}_{freq}GHz_nside2048_K.fits`; load all six channels; convert K→µK; return ordered array or dict.

- [ ] **Step 4: Run test — expect pass**

- [ ] **Step 5: Commit** (only if user requests)

---

### Task 2: Tile cutouts + masks

**Files:**
- Create: `src/flamingo_mock/szifi/tiles.py`
- Test: `tests/test_szifi_tiles.py`

**Interfaces:**
- Consumes: `load_total_maps_uK`, PR4 mask path
- Produces: `select_pilot_tile_ids(n=8, |b|_min=40) -> list[int]`
- Produces: `prepare_tile(field_id, maps_uK, gal, ps) -> writes tmap.npy + mask.npy`

- [ ] **Step 1: Write test for pilot tile selection (synthetic nside=8)**

```python
def test_pilot_tiles_high_latitude():
    from flamingo_mock.szifi.tiles import select_pilot_tile_ids
    ids = select_pilot_tile_ids(n=4, b_min_deg=40.0)
    assert len(ids) == 4
    import healpy as hp
    for i in ids:
        _, b = hp.pix2ang(8, i, lonlat=True)
        assert abs(b) >= 40.0
```

- [ ] **Step 2: Implement cutout + mask preparation**

Use `szifi.get_cutout` / `szifi.sphere.get_cutout`. Stack freqs into `(nx,nx,6)` float32 µK. Project GAL (binarise soft edges), PS, and tile boundary mask; save Planck-like `*_tmap.npy` / `*_mask.npy`.

- [ ] **Step 3: Smoke-test one real tile cutout (script or pytest mark)**

Verify shapes `(1024,1024,6)` and masks `(3,1024,1024)`.

---

### Task 3: Survey adapter + SZiFi run + catalogue

**Files:**
- Create: `src/flamingo_mock/szifi/survey.py`
- Create: `src/flamingo_mock/szifi/run.py`
- Create: `notebooks/szifi_pilot.ipynb`

**Interfaces:**
- Produces: survey file readable by SZiFi (`input_data_survey`)
- Produces: `run_mmf(field_ids, mmf_type, split) -> catalogue`
- Produces: merged `q>5` catalogues on disk

- [ ] **Step 1: Implement `survey.py` mirroring `data_planck.py`**

Load prepared npy; maps already µK (no 545/857 Jy hacks); `experiment_name="Planck_simple"`; set `exp.FWHM = beam_fwhm_vec_arcmin()`.

- [ ] **Step 2: Implement `run.py`**

iMMF: `mmf_type="standard"`. sciMMF: `mmf_type="spectrally_constrained"`, `deproject_cib=["cib"]`, `integrate_bandpass=False`, `beam="gaussian"`, `array_backend="jax"`. Compute coupling matrices if missing. Merge with `merge_detections`, threshold `q_th_final=5`.

- [ ] **Step 3: Pilot notebook**

Prepare 4 tiles split A → run both MMFs → save under `.../szifi/pilot/`. Print detection counts.

- [ ] **Step 4: Verify**

`SZIFI_ARRAY_BACKEND=jax` run completes; catalogues have `q_opt >= 5`.

---

### Task 4: CLI packaging (Phase 1)

**Files:**
- Create: `src/flamingo_mock/szifi/cli.py`
- Modify: `pyproject.toml` → `flamingo-szifi = flamingo_mock.szifi.cli:main`

- [ ] **Step 1: CLI subcommands `prepare`, `run`, `catalogue`**
- [ ] **Step 2: Scale prepare/run to all tiles with `min_ftile` cut**
- [ ] **Step 3: Repeat for split B**

---

### Task 5: Spec coverage check

- [ ] Both iMMF and sciMMF
- [ ] Gaussian Table I beams
- [ ] Noise not beam-smoothed (use existing total maps / rebuild recipe)
- [ ] PR4 GAL+PS masks
- [ ] Output under `/rds/.../szifi/`
- [ ] No ILC algorithm import
