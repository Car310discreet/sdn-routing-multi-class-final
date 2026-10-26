# AIRSDN reproduction (report Section VII-B)

The base paper (İpek et al., *AIRSDN*, Computer Communications 240, 2025) reproduced in its own scenario. It
keeps the paper's scope:
- the 14-switch NSFNET, with 25 ms / 10 Mbit/s links, 20 hosts and 9 pinned background flows;
- the 5 hop-count candidates (Yen);
- 19 path features (ping, iperf3, NFStream flow statistics);
- a binary Logistic Regression model.

| File | Role |
|---|---|
| `topology_builder.py` | NSFNET in Mininet and the paper's background traffic |
| `path_discovery.py` | Yen's 5 shortest paths by hop count |
| `telemetry_and_model.py` | path probing (19 features), the Logistic Regression model, static flow rules |
| `multimedia_evaluator.py` | 60 s RTT and throughput monitoring, video transfer, PSNR / SSIM |
| `run_experiment.py` | hop-count vs QoS-aware vs AIRSDN routing (needs root and Floodlight) |
| `visualize_benchmarks.py` | the comparison figures |
| `train_lr.py` | trains the classifiers on `data/result_data.csv` and saves `models/logistic_regression_model.pkl` |

```bash
.venv/bin/python airsdn_reproduction/train_lr.py
.venv/bin/python -m pytest airsdn_reproduction/tests
sudo .venv/bin/python airsdn_reproduction/run_experiment.py        # about 17 minutes
.venv/bin/python airsdn_reproduction/visualize_benchmarks.py
```

## Recorded result (`results/`)

| Approach | Mean RTT | Mean throughput | PSNR | SSIM |
|---|---|---|---|---|
| Hop-count (Floodlight default) | 1830 ms | 0.21 Mbit/s | 20.40 dB | 0.874 |
| QoS-aware (Floodlight latency) | 1896 ms | 6.64 Mbit/s | 38.11 dB | 0.997 |
| AIRSDN (hop-count candidates + LR) | 351 ms | 6.52 Mbit/s | 38.11 dB | 0.997 |

The paper reports 2063 ms / 21.45 dB / 0.927 for hop-count, 1039 ms / 21.57 dB / 0.929 for QoS-aware, and
354.05 ms / 37.27 dB / 0.997 for AIRSDN. As in the paper's Fig. 15, AIRSDN chose candidate 3,
s0-s3-s4-s5-s12-s9.
