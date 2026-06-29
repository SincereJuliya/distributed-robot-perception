# Experiments (revised pipeline)

This folder replaces the old `experiments/` folder. The goal of the
revision is to make the measurement methodology unambiguous and
identical across all four experiments, so that numbers from any two
tables can be compared directly.

## Why the old pipeline was replaced

The previous pipeline had three issues that made the reported numbers
hard to interpret:

1. **Three scripts measured "detection time" differently.** One used
   the LEAK → CONSENSUS phase transition, another used LEAK first-entry
   minus LEAK exit. The same column name meant different things.
2. **The scatter plot of `(e_x, e_y)` used a fabricated angle.** The
   CSV only stored the scalar `‖e‖`, so the plotting script drew points
   on a circle of the correct radius with `np.random.uniform(0, 2π)`.
   Distances were honest but directions were not.
3. **No confidence intervals.** With N=15 or N=30, comparing two means
   that differ by 2 px without quoting any spread is not informative.

## What the new pipeline does

- `runner.py` exposes a single `run_one_cycle(seed, …)` function that
  every experiment script imports. All metrics come from one definition.
- The four phase anchors are tracked separately: `t_leak`,
  `t_first_det`, `t_consensus`, `t_reported`. The four interesting
  intervals (`patrol_steps`, `investigate_steps`, `fuse_steps`,
  `total_steps`) are computed by subtraction in one place.
- Errors are stored as the signed vector `(err_x, err_y)` and the
  norm `err`. The scatter plot reads the signed vector directly.
- `stats.py` provides percentile bootstrap CIs and Welch / Mann–Whitney
  tests. All summary tables print `mean [lo, hi]` instead of bare means.
- Experiments use **paired seeds** wherever a comparison is made (radii
  sweep, Lloyd vs RandomWalk, degradation arms), so the variance from
  source position cancels out of the contrasts.

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
# Experiment 1 — Mode-1 accuracy (default N=100, r_c=500)
python experiments/exp1_mode1_accuracy.py

# Experiment 2 — comm-radius sweep (paired seeds across radii)
python experiments/exp2_comm_radius.py --runs 15

# Experiment 3 — Lloyd vs RandomWalk (paired)
python experiments/exp3_explorers.py --runs 30

# Experiment 4 — Mode-2 degradation (paired baseline / k=1 / k=2)
python experiments/exp4_degradation.py --runs 10

# All figures
python experiments/plot_results.py
```

Each script writes a CSV and a text summary to `experiments/results/`.
Plots read those CSVs and write PNGs to the same folder.

Seeds are deterministic and disjoint per experiment:
- exp1: 2000…2099
- exp2: 3000…3000+N-1 (replayed for every r_c)
- exp3: 4000…4000+N-1 (replayed for both strategies)
- exp4: 5000…5000+N-1 (replayed for baseline / k=1 / k=2)
