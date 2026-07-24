controller=rule, data_is_real=False, seed=42, n=120, h=5.0, alpha=0.1

| Condition | Catch rate | False-alarm rate | Live SLA-violation | Commit frac |
|---|---|---|---|---|
| C0 (commit all) | 0.000 | 0.000 | 0.233 | 1.000 |
| C1 (validity only) | 0.000 | 0.000 | 0.233 | 1.000 |
| C2 (validity + TFS) | 1.000 | 0.011 | 0.000 | 0.758 |
| C3 (random block ctrl) | 0.107 | 0.304 | 0.281 | 0.742 |
