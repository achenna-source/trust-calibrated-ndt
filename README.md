# Trust-Calibrated Network Digital Twins — reproducibility artifact

Code and reference outputs for the case study in **"Can You Trust the Twin? Trust-Calibrated
Network Digital Twins as a Validation-and-Rollback Substrate for AI-Native 5G/6G Network
Management"** (submitted to IEEE Communications Surveys & Tutorials).

The artifact reproduces every number and figure in **Section VIII** and **Appendix A** of the
paper. It is zero-cost by design: it runs on a single CPU in a free notebook, needs no GPU, and
uses only open-source libraries and a public dataset.

## What is here

| File | Produces (paper) |
|---|---|
| `ndt_trust_experiment.py` | Synthetic controlled demonstration — the gate catches unsafe commits when twin optimism is localizable (Appendix A.1–A.3, Table 12) |
| `coverage_check.py` | Empirical vs nominal conformal coverage (Fig. 4) |
| `sensitivity_sweep.py` | Operating characteristic across twin-optimism magnitude (Appendix A.3, Table 13, Fig. 5) |
| `ndt_real_experiment.py` | Real-data study on Colosseum traces: CQI, predicted-magnitude, and learned regions (Section VIII.B, Tables 8–9) |
| `extra_synth.py`, `extra_real.py` | Supporting sweeps (continuous localizability, diffuse null, direct-harm baseline, split-sensitivity) |
| `PREREG_region_variant.md` | Pre-specification of the one region variant fixed before the real-data run |
| `results/` | Reference tables and figures to diff your re-run against |

## Reproducibility is two-tier

- **Synthetic (needs no external data).** `ndt_trust_experiment.py`, `coverage_check.py`,
  `sensitivity_sweep.py`, `extra_synth.py` use a documented synthetic surrogate and regenerate
  their numbers on any machine.
- **Real data (needs the public Colosseum dataset).** `ndt_real_experiment.py` reads the
  Colosseum O-RAN KPM traces. That dataset is third-party and is **not shipped here**; see
  [DATA.md](DATA.md) to obtain it and where to place it.

The learned region function is **exploratory**: it was devised after the two pre-specified
region functions failed on the real data. The paper states this explicitly, and so should any
use of this code.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
bash reproduce.sh                                    # runs the synthetic parts; runs the real part if data is present
```

Windows without bash: run the four scripts in `reproduce.sh` individually.

## Reference numbers (what a correct re-run reproduces)

Real-data study, ten seeds (from `ndt_real_experiment.py`; matches Tables 8–9 of the paper):

| Region function | Localizability Λ | C2 catch | C2 false-alarm | C2 SLA-violation | conformal coverage |
|---|---|---|---|---|---|
| CQI (primary) | 0.508 | 0.493 | 0.507 | 0.558 | 0.907 |
| Predicted-magnitude (pre-specified) | 0.338 | 0.338 | 0.689 | 0.723 | 0.902 |
| **Learned error-predictor** | **0.911** | **0.828** | **0.101** | **0.190** | 0.897 |

In every region the validity-only condition C1 is identical to the commit-all baseline C0
(catch 0.000), by construction: a commit is counted harmful only when the twin rated it
acceptable, so the action-validity gate blocks nothing. The informative comparison is C2 against
the compute-matched random control C3. Nominal conformal coverage is 0.90 throughout.

## Data

See [DATA.md](DATA.md). The Colosseum dataset is public but third-party; obtain and cite it from
its source rather than from this repository.

## License

Code is released under the MIT License ([LICENSE](LICENSE)); set the copyright holder before
release. The dataset is governed by its own terms at the source.

## Citation

If you use this artifact, please cite the paper (bibliographic details to be completed on
acceptance) and the Colosseum / OpenRAN Gym dataset it builds on.
