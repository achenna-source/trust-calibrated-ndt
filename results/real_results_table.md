Colosseum O-RAN COMMAG data (real); twin trained on scheduling_policy 0, evaluated on 2; 10 seeds.

### CQI-region (primary)
base harmful=0.551 | conformal coverage=0.907 (nom 0.90) | LOCALIZABILITY AUC=0.508

| Condition | Catch | False-alarm | SLA-violation | Commit |
|---|---|---|---|---|
| C0 (commit all) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.551 ± 0.002 | 1.000 |
| C1 (validity only) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.551 ± 0.002 | 1.000 |
| C2 (validity + TFS) | 0.493 ± 0.005 | 0.507 ± 0.004 | 0.558 ± 0.002 | 0.501 |
| C3 (random block) | 0.501 ± 0.005 | 0.501 ± 0.005 | 0.551 ± 0.002 | 0.499 |

### predicted-magnitude region (pre-specified variant)
base harmful=0.551 | conformal coverage=0.902 (nom 0.90) | LOCALIZABILITY AUC=0.338

| Condition | Catch | False-alarm | SLA-violation | Commit |
|---|---|---|---|---|
| C0 (commit all) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.551 ± 0.002 | 1.000 |
| C1 (validity only) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.551 ± 0.002 | 1.000 |
| C2 (validity + TFS) | 0.338 ± 0.003 | 0.689 ± 0.007 | 0.723 ± 0.004 | 0.505 |
| C3 (random block) | 0.497 ± 0.005 | 0.497 ± 0.004 | 0.551 ± 0.002 | 0.503 |

### LEARNED error-predictor region
base harmful=0.551 | conformal coverage=0.897 (nom 0.90) | LOCALIZABILITY AUC=0.911

| Condition | Catch | False-alarm | SLA-violation | Commit |
|---|---|---|---|---|
| C0 (commit all) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.551 ± 0.002 | 1.000 |
| C1 (validity only) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.551 ± 0.002 | 1.000 |
| C2 (validity + TFS) | 0.828 ± 0.003 | 0.101 ± 0.004 | 0.190 ± 0.002 | 0.499 |
| C3 (random block) | 0.502 ± 0.006 | 0.503 ± 0.004 | 0.551 ± 0.002 | 0.497 |
