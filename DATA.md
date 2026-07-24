# Data provenance

The real-data study uses the **Colosseum O-RAN COMMAG / OpenRAN Gym key-performance-measurement
(KPM) dataset** (Bonati et al.). It is a third-party public dataset and is **not redistributed
here**; please obtain it from its original source and cite it, not this repository.

- OpenRAN Gym / Colosseum: <https://openrangym.com/>
- Reference: L. Bonati et al., "OpenRAN Gym: AI/ML development, data collection, and testing
  for O-RAN on PAWR platforms," *Computer Networks*, 2022. See also the Colosseum O-RAN
  digital-twin description (2024).

## What the code expects

`ndt_real_experiment.py` reads every `*_metrics.csv` it finds in a folder named
`colosseum_data/` beside the script:

```
artifact/
├── ndt_real_experiment.py
└── colosseum_data/
    ├── slice_mixed__..._metrics.csv
    ├── ...
```

Each CSV is a per-run KPM trace with (at least) the columns used as features and target in the
script (downlink CQI, allocation/PRB fields, and the downlink buffer occupancy used as the
congestion proxy). Column names are stripped of whitespace on load; no other preprocessing is
applied. Place the KPM CSVs from the public dataset into `colosseum_data/` and run.

## Reproducibility without the dataset

The **synthetic demonstration** (`ndt_trust_experiment.py`, and the sweeps in `extra_synth.py`)
is fully self-contained: it uses a documented synthetic surrogate, needs no external data, and
regenerates the Appendix-A numbers on any machine. Only the **real-data numbers** (paper
Section VIII.B, Tables 8–9) require the Colosseum CSVs above.
