
## Metric definitions (verbatim from `runner.py`)

| Anchor / interval | Definition |
|---|---|
| `t_leak`         | step at which MONITOR → LEAK (gas appears) |
| `t_first_det`    | step at which first robot's own sensor sets `has_detected = True` |
| `t_consensus`    | step at which LEAK → REACHING CONSENSUS (≥ 50% have detected) |
| `t_reported`     | step at which REACHING CONSENSUS → REPORTED (σ stable) |
| `patrol_steps`     | `t_first_det − t_leak` — passive detection latency |
| `investigate_steps`| `t_consensus − t_first_det` — scout-led convergence |
| `fuse_steps`       | `t_reported − t_consensus` — gossip + DKF lock-in |
| `total_steps`      | `t_reported − t_leak` — end-to-end |
| `err_x`, `err_y`   | `agreed_px − source_pxs[0]` (signed, px) |
| `err`              | `‖(err_x, err_y)‖₂` |

## How to run

```bash

python3 experiments/exp1_mode1_accuracy.py --runs 100
python3 experiments/exp2_comm_radius.py --runs 15
python3 experiments/exp3_explorers.py --runs 30
python3 experiments/exp4_degradation.py --runs 10
python3 experiments/exp5_ablation.py --runs 15

python3 experiments/plot_results.py

```

Each script writes a CSV and a text summary to `experiments/results/`.
Plots read those CSVs and write PNGs to the same folder.

Seeds are deterministic and disjoint per experiment:
- exp1: 2000…2099
- exp2: 3000…3000+N-1 (replayed for every r_c)
- exp3: 4000…4000+N-1 (replayed for both strategies)
- exp4: 5000…5000+N-1 (replayed for baseline / k=1 / k=2)
