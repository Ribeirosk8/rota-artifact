#!/usr/bin/env bash
# Reproduce every published number and figure from the raw counter data.
#
# This is ANALYSIS-ONLY reproduction: it recomputes the aggregates from the
# raw NVML counter CSVs in data/ and regenerates Figures 1-3. It runs
# offline, deterministically, on any machine with Python 3 + numpy +
# matplotlib. It does NOT re-run the measurements: the measured operating
# point (per-device SM clocks reported in Section II) is not reproducible
# on demand, since every NVML clock-control interface returns NO_PERMISSION
# on the measured system (Section IV). Re-measuring is possible with the
# scripts in code/, but absolute values will reflect the then-current clock
# regime.
set -euo pipefail
cd "$(dirname "$0")"

echo "== [1/2] Verifying figs/curve_data.json against the raw counters =="
python3 verify_aggregates.py

echo "== [2/2] Regenerating Figures 1-3 from curve_data.json =="
( cd figs
  python3 fig1_divergence.py
  python3 fig2_model.py
  python3 fig3_heatmap.py )

for f in figs/fig1_divergence.pdf figs/fig2_model.pdf figs/fig3_heatmap.pdf; do
    [ -s "$f" ] || { echo "MISSING OR EMPTY: $f" >&2; exit 1; }
done
echo "DONE: aggregates verified, 3 figures regenerated."
