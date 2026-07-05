# Distributed Perception — Multi-Robot Hazard Localization

Three-mode simulation of a homogeneous team of ground robots performing
distributed environmental monitoring **without any central controller**.

## Modes

| Key | Mode | Algorithms |
|---|---|---|
| **1** | Hazard Localization | Gossip + DKF + Lloyd–Voronoi + λ₂ |
| **2** | Sensor Degradation & Trust | + trust-weighted consensus, outlier resilience |
| **3** | Formation Control | Consensus-based formation tracking (circle / line / V) |

## Project structure

```
distributed_perception/
├── main.py                         ← launch the live visualiser
├── config/default.py               ← all parameters
├── core/                           ← swappable algorithm classes
│   ├── consensus.py                  Weighted Gossip Consensus
│   ├── kalman.py                     Distributed Kalman Filter (info form)
│   ├── coverage.py                   Lloyd–Voronoi
│   ├── graph_metrics.py              Comm graph + λ₂
│   ├── trust.py                      Trust / reputation system
│   └── formation.py                  Formation control
├── environment/
│   ├── gas_field.py                  Multi-source gas dynamics
│   └── map.py                        Free space, obstacles, spawn helpers
├── agents/
│   └── robot.py                      Robot state, sensing, movement
├── scenario/
│   ├── manager.py                    Top-level mode dispatcher
│   └── modes/
│       ├── localization.py
│       ├── degradation.py
│       └── formation.py
├── simulation/
│   ├── simulation.py                 Orchestrator (one step = one tick)
│   └── event_log.py                  In-memory + .log + .csv logging
├── visualisation/
│   └── visualizer.py                 White-theme matplotlib UI
├── experiments/
│   ├── batch_run.py                  N runs → CSV
│   ├── plot_results.py               CSV → report figures
│   └── results/
└── logs/                             Auto-generated per-run logs
```

### To change something, edit only one file

| What | Where |
|---|---|
| Algorithm internals | `core/<thing>.py` |
| Robot model | `agents/robot.py` |
| Gas / map | `environment/` |
| Mode logic | `scenario/modes/<mode>.py` |
| UI / theme | `visualisation/visualizer.py` |
| Parameters | `config/default.py` |

## Run

```bash
pip install -r requirements.txt
python3 main.py
```

### Keyboard controls

| Key | Action | Mode |
|---|---|---|
| `1` `2` `3` | Switch mode | any |
| `L` | Force new leak | 1, 2 |
| `D` | Degrade random robot | 2 |
| `0` … `4` | Degrade specific robot | 2 |
| `A` | Toggle auto-degradation | 2 |
| `F` | Cycle formation shape | 3 |
| `K` | Kill random robot | any |
| `Space` | Pause / resume | any |
| `R` | Reset simulation | any |
| `Q` / `Esc` | Quit | any |
