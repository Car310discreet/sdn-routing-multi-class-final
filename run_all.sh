#!/usr/bin/env bash
# run_all.sh - reproduce the results of the report.
#   offline (no root): tests, dataset profiling, training + cross-validation of both ensembles, the AIRSDN
#                      Logistic Regression model, the offline compute benchmark
#   live (root, Floodlight running): the five routing methods on GEANT2 and on the fat-tree, then every figure
#                      and table (routing/final_results.py) and the report's figures (report/make_report_figures.py)
#
#     ./run_all.sh                   # everything: 5 scenarios per topology, about 4-5 hours
#     SCENARIOS=1 ./run_all.sh       # quick end-to-end check (about 1 hour); figures go to results/quick_check_<ts>/
#     LIVE=0 ./run_all.sh            # offline steps only; results/ rebuilt from the recorded runs and benchmark
#     AIRSDN=1 ./run_all.sh          # also re-run the base-paper reproduction on NSFNET (about 17 minutes)
#     COLLECT=8 ./run_all.sh         # also collect N new dataset rows per topology into data/collected/
# The live part asks for the sudo password once, up front, and keeps the machine awake until it finishes.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
SCENARIOS=${SCENARIOS:-5}
LIVE=${LIVE:-1}
AIRSDN=${AIRSDN:-0}
COLLECT=${COLLECT:-0}
# a full run replaces the report's results; a quick check writes its own folder; LIVE=0 keeps the recorded runs
# and the recorded compute benchmark (timings vary between runs and machines)
if [ "$LIVE" = 1 ] && [ "$SCENARIOS" != 5 ]; then
    out=results/quick_check_$(date +%Y%m%d_%H%M%S)
else
    out=results
fi

echo "== offline =="
$PY -m pytest -q tests
$PY -m pytest -q airsdn_reproduction/tests
$PY routing/profile_datasets.py
for topology in geant2 fattree; do
    $PY routing/train_models.py --topology "$topology"
    $PY routing/cross_validate.py --topology "$topology"
    if [ "$LIVE" = 1 ]; then
        $PY routing/run_experiment.py --benchmark-only --topology "$topology" --benchmark-dir "$out/benchmark"
    fi
done
$PY airsdn_reproduction/train_lr.py

if [ "$LIVE" != 1 ]; then
    $PY routing/final_results.py                        # newest runs in results/runs/ (the report's runs)
    $PY report/make_report_figures.py
    exit 0
fi

echo "== live =="
curl -sf http://127.0.0.1:8080/wm/core/health/json >/dev/null \
    || { echo "Floodlight is not reachable on :8080 - start ./start_floodlight.sh first" >&2; exit 1; }
sudo -v
sudo systemd-inhibit --what=sleep:idle:handle-lid-switch --who="sdn-multimedia-routing" --why="Mininet runs" \
    bash -c "
        set -e
        for topology in geant2 fattree; do
            mn -c >/dev/null 2>&1 || true
            if [ '$COLLECT' -gt 0 ]; then
                '$PY' routing/collect_dataset.py --topology \"\$topology\" --samples '$COLLECT'
                mn -c >/dev/null 2>&1 || true
            fi
            '$PY' routing/run_experiment.py --topology \"\$topology\" --scenarios '$SCENARIOS' --congest-default-path
        done
        if [ '$AIRSDN' = 1 ]; then
            mn -c >/dev/null 2>&1 || true
            '$PY' airsdn_reproduction/run_experiment.py
        fi
        mn -c >/dev/null 2>&1 || true"

geant2_run=$(ls -td results/runs/geant2_* | head -1)
fattree_run=$(ls -td results/runs/fattree_* | head -1)
$PY routing/final_results.py --geant2-run "$geant2_run" --fattree-run "$fattree_run" --out "$out" \
    --benchmark-dir "$out/benchmark"
if [ "$out" = results ]; then
    $PY report/make_report_figures.py
fi
if [ "$AIRSDN" = 1 ]; then
    $PY airsdn_reproduction/visualize_benchmarks.py
fi
echo "Results: $out  (runs: $geant2_run, $fattree_run)"
