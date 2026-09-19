# HILC figure layout

## Per prescription (`L1_m9`, `fgas-8sigma`, `Mstar-1sigma`, `LS8`)

Each prescription has one subdirectory per **deprojection case**:

| Subfolder | `N_deproj` | Deprojected components |
|-----------|------------|------------------------|
| `nodeproj/` | 0 | — |
| `deproj_cib/` | 1 | CIB |
| `deproj_cib_dbeta/` | 2 | CIB, δβ |
| `deproj_cib_dbeta_cmb/` | 3 | CIB, δβ, CMB |
| `deproj_moments/` | 3 | CIB, δβ, δT |

Within each: `r1xr2_fig9_{fullsky,q5masked}.png`, `y_vs_truth_{fullsky,q5masked}.png`.

Masked runs use `szifi_homog/<prescription>/catalogues/fullsky_splitA_immf_q5.npz`
(cluster count differs per case).

## Combined overlays

`combined/{deproj}/` — four-panel stacks and prescription overlays for that deproj case.

## Legacy L1_m9-only plots

`L1_m9/legacy_l1m9_scripts/` — earlier single-prescription deproj suite from
`scripts/plot_hilc_homog_*.py` (auto Fig. 9, weight comparisons, etc.).

## Regenerate

```bash
# write YAMLs + run missing HILC + plot all deproj cases
bash scripts/run_hilc_all_prescriptions_deproj.sh

# plot only (skip HILC execution)
python scripts/plot_hilc_homog_prescriptions.py
python scripts/plot_hilc_homog_prescriptions.py --deproj deproj_cib deproj_moments
```
