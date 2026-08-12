# HILC deprojection suite metrics (vs truth Compton-y)

| Deprojection | std_y | corr (beamed) | ρ_50–500 | T_mid | summary_ok |
|--------------|------:|--------------:|---------:|------:|:----------:|
| No deprojection | 1.807e-06 | 0.473 | 0.654 | 0.909 | True |
| CMB | 1.817e-06 | 0.470 | 0.655 | 0.910 | True |
| CIB | 1.838e-06 | 0.496 | 0.671 | 0.960 | True |
| CIB + CMB | 1.850e-06 | 0.493 | 0.672 | 0.959 | True |
| CIB + δβ | 1.948e-06 | 0.463 | 0.614 | 0.949 | True |
| CIB + δβ + CMB | 1.968e-06 | 0.458 | 0.615 | 0.948 | True |
| CIB + δβ + δT | 4.194e-06 | 0.226 | 0.279 | 1.000 | True |
| CIB + δβ + δT + CMB | 5.998e-06 | 0.159 | 0.271 | 0.995 | True |

Configs: `configs/deproj_suite/hilc_splitA_*.yml`
Figures: `figures/deproj_suite_hilc_{auto_power,cross_truth,transfer}.png`
