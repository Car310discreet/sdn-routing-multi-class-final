# Multi-Metric Candidate Discovery with Multi-Class Ensemble Selection for Multimedia Routing in SDN

Code, data and results of the project report [report/final-report.pdf](report/final-report.pdf): a two-stage SDN
router evaluated on GEANT2 and a k=4 fat-tree, compared with the base paper AIRSDN (İpek et al., *Computer
Communications* 240, 2025).

- **Stage 1** finds the five cheapest paths with Yen's algorithm over a normalised composite cost of delay, jitter,
  inverse available bandwidth and loss (α = 0.3, β = 0.1, γ = 0.4, δ = 0.2).
- **Stage 2** probes each candidate for 11 features and classifies all five into four congestion classes in one
  batched call to a soft-voting ensemble (XGBoost, Random Forest, Gradient Boosting, SVM). It installs the
  candidate with the lowest class.

Everything runs on Mininet + Open vSwitch + Floodlight, with a real 66 s H.264 video.

## Repository layout

```
routing/                       the proposed router and its evaluation
  topology_builder.py            GEANT2 and k=4 fat-tree networks; four-class background traffic
  path_discovery.py              Stage 1: hop / latency / widest / composite top-K (Yen, hop-bounded)
  telemetry_and_model.py         Floodlight client, link telemetry, 11-feature probe, ensemble, Stage 2 selection
  multimedia_evaluator.py        video transfer; per-second throughput, RTT, jitter, loss; PSNR and SSIM
  udp_probe.py                   UDP probe for per-second loss and RFC 3550 jitter
  collect_dataset.py             builds a topology's labelled path dataset on the live network
  train_models.py                trains the four models and the soft-voting ensemble per topology
  cross_validate.py              10 x 5-fold cross-validation of the models and the ensemble
  run_experiment.py              live evaluation of the five methods; offline compute benchmark
  profile_datasets.py            ydata-profiling reports of both datasets
  final_results.py               every figure and table of the report, each next to its data
  visualize_benchmarks.py        plotting helpers (style, per-run figures)
airsdn_reproduction/           the base paper reproduced on NSFNET (binary Logistic Regression, 19 features)
data/                          geant2_dataset.csv (500 paths), fattree_dataset.csv (896; 912 before balancing)
models/                        geant2_bundle.joblib, fattree_bundle.joblib (four models + ensemble each)
assets/surreal.ts              the test video (1920x1048, 60 fps, 4,007 frames, ~4.7 Mbit/s)
results/                       the report's results (see below)
report/                        final-report.pdf and the script that draws its figures
tests/                         offline tests (no Mininet or Floodlight needed)
```

## Where each part of the report comes from

| Report | Code | Output |
|---|---|---|
| §V-C composite cost (1), Stage 1 | `routing/path_discovery.py` | – |
| §V-C ensemble (2), selection rule | `routing/telemetry_and_model.py`, `routing/train_models.py` | `models/` |
| §V-C cross-validation of the ensemble | `routing/cross_validate.py` | `results/training/<topology>/cross_validation_summary.csv` |
| §VI topologies, Table III | `routing/topology_builder.py` | `report/figures/topologies.png` |
| §VI datasets | `routing/collect_dataset.py` | `data/` |
| §VII-B base paper reproduction | `airsdn_reproduction/` | `airsdn_reproduction/results/` |
| §VII-C dataset profiling, Fig. 3 | `routing/profile_datasets.py` | `results/0_dataset_profiling/` |
| Table IV, Fig. 4 classifier accuracy | `routing/train_models.py` | `results/1_training/` |
| Tables V, VI, Figs. 5–8 routing results | `routing/run_experiment.py`, `routing/final_results.py` | `results/2_path_selection/`, `3_video_timeseries/`, `5_paths/` |
| Figs. 9–10 overhead and batching | `routing/run_experiment.py` (timings, `--benchmark-only`) | `results/4_overhead_geant2/` |

`results/README.md` lists every figure with its data file and the mean results per method. `results/runs/` holds
the two live runs the report is built on (`geant2_20261021_224332`, `fattree_20261021_235820`). Each run has
`decision.json`, `timeseries.csv`, `frames.csv` and `summary.json` per scenario and method; the received videos
are not kept. The five methods are named `hop_count`, `latency`, `bandwidth`, `baseline` (hop-count + ML) and
`proposed` (combined + ML).

## Setup (Ubuntu 24.04)

```bash
./setup.sh          # apt packages, Floodlight (~/floodlight, JDK 8), .venv with the exact Python packages
```

## Running

```bash
./start_floodlight.sh                         # terminal 1, keep it running

./run_all.sh                                  # terminal 2: everything, 5 scenarios per topology (~4-5 h)
SCENARIOS=1 ./run_all.sh                      # quick end-to-end check (~1 h) -> results/quick_check_<timestamp>/
LIVE=0 ./run_all.sh                           # offline steps only (~2 min); rebuilds results/ from results/runs/
AIRSDN=1 ./run_all.sh                         # also re-run the NSFNET base-paper reproduction
COLLECT=8 ./run_all.sh                        # also collect 8 new dataset rows per topology into data/collected/
```

The live part needs root, so the script asks for the sudo password once. It keeps the machine awake until both
topologies are done. Individual steps:

```bash
.venv/bin/python -m pytest tests                                           # offline tests
.venv/bin/python routing/train_models.py --topology fattree                # retrain one topology's ensemble
.venv/bin/python routing/run_experiment.py --benchmark-only --topology geant2
sudo .venv/bin/python routing/run_experiment.py --topology geant2 --scenarios 5 --congest-default-path
sudo .venv/bin/python routing/collect_dataset.py --topology fattree --samples 1000
.venv/bin/python routing/collect_dataset.py --trim data/collected/fattree_dataset.csv
.venv/bin/python routing/final_results.py                                  # figures from the newest runs
```

## Reproducibility notes

- Retraining with `train_models.py` reproduces Table IV exactly (seed 40). Trimming
  `data/fattree_dataset_full.csv` reproduces `data/fattree_dataset.csv` exactly. `airsdn_reproduction/train_lr.py`
  reproduces the base paper's Logistic Regression model.
- Live runs depend on timing in the emulator, so a new run gives similar but not identical numbers. The report's
  runs stay in `results/runs/`. A full `./run_all.sh` replaces `results/` with the new run, while a quick check
  writes to its own folder.
- Compute timings (Fig. 10) depend on the machine. `LIVE=0` keeps the recorded benchmark in `results/benchmark/`.
- Video quality is scored with the system FFmpeg, which has the native H.264 decoder on Ubuntu.
