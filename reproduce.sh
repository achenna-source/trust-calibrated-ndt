#!/usr/bin/env bash
# Reproduce the numbers in the paper. The synthetic parts need no external data;
# the real-data part needs the Colosseum KPM CSVs in ./colosseum_data (see DATA.md).
set -e
echo "== 1/4  Synthetic controlled demonstration (Appendix A.3) =="
python ndt_trust_experiment.py --smoke

echo "== 2/4  Coverage check (empirical vs nominal, Fig. 4) =="
python coverage_check.py

echo "== 3/4  Twin-optimism sweep (Appendix A.3, Table 13 / Fig. 5) =="
python sensitivity_sweep.py

if ls colosseum_data/*_metrics.csv >/dev/null 2>&1; then
  echo "== 4/4  Real-data study on Colosseum traces (Section VIII.B, Tables 8-9) =="
  python ndt_real_experiment.py
else
  echo "== 4/4  SKIPPED: no colosseum_data/*_metrics.csv found. See DATA.md to obtain the public dataset. =="
fi
echo "Done. Generated tables/figures are written next to the scripts; reference copies are in results/."
