"""
experiments/exp5_ablation.py
--------------------------------
Experiment 5 — Per-robot ablation.

Goal: demonstrate that the distributed perception pipeline has no
critical agent. We remove each robot i ∈ {0,1,2,3,4} in turn (force
it dead before the leak starts) and measure the resulting localization
error and total time. If perception were secretly centralised on one
"lead" robot, removing that robot would dramatically degrade the
system. If it is genuinely distributed, removing any single robot
should yield similar degradation.

Design: paired seeds 6000..6000+N-1 are replayed under N+1 conditions:
  - baseline    : all 5 robots alive
  - drop robot 0
  - drop robot 1
  - drop robot 2
  - drop robot 3
  - drop robot 4

Outputs:
  results/exp5_ablation.csv
  results/exp5_summary.txt
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runner import run_one_cycle, FIELDS
from stats  import summarise, fmt_ci, bootstrap_ci

# Local helper — same as run_one_cycle but with a "drop robot" hook.
# We re-implement it here as a thin wrapper rather than adding yet
# another parameter to the canonical runner.

import random
import numpy as np
import config
from simulation import Simulation

_PHASE_MONITOR   = "MONITORING"
_PHASE_LEAK      = "LEAK DETECTED"
_PHASE_CONSENSUS = "REACHING CONSENSUS..."
_PHASE_REPORTED  = "REPORTED — patrolling map"


def run_with_drop(seed: int, drop_id: int | None,
                  comm_radius: int = 500, max_steps: int = 3000) -> dict:
    """
    Run a leak cycle with robot `drop_id` forced dead from step 0.
    If drop_id is None, no robot is dropped (baseline).
    """
    config.COMM_RADIUS = comm_radius
    config.EXPLORATION_STRATEGY = "lloyd"
    random.seed(seed); np.random.seed(seed)

    sim = Simulation()
    if drop_id is not None and 0 <= drop_id < len(sim.robots):
        sim.robots[drop_id].is_alive = False

    t_leak = t_first_det = t_consensus = t_reported = None
    lam2_log = []
    prev = sim.scenario.phase

    for step in range(max_steps):
        sim.step()
        ph = sim.scenario.phase

        if prev == _PHASE_MONITOR and ph == _PHASE_LEAK:
            t_leak = step; lam2_log = []
        if t_leak is not None and t_first_det is None:
            if any(r.has_detected for r in sim.robots if r.is_alive):
                t_first_det = step
        if prev == _PHASE_LEAK and ph == _PHASE_CONSENSUS:
            t_consensus = step
        if prev == _PHASE_CONSENSUS and ph == _PHASE_REPORTED:
            t_reported = step
            break

        if t_leak is not None and sim.scenario.lambda2_history:
            lam2_log.append(sim.scenario.lambda2_history[-1])
        prev = ph

    err_x = err_y = err = None
    if sim.scenario.agreed_px is not None and sim.scenario.source_pxs:
        ag  = np.asarray(sim.scenario.agreed_px, dtype=float)
        src = np.asarray(sim.scenario.source_pxs[0], dtype=float)
        d = ag - src
        err_x, err_y = float(d[0]), float(d[1])
        err = float(np.linalg.norm(d))

    sim.log.close()
    return {
        "seed":      seed,
        "drop_id":   -1 if drop_id is None else drop_id,
        "completed": t_reported is not None,
        "t_leak":      t_leak,
        "t_first_det": t_first_det,
        "t_consensus": t_consensus,
        "t_reported":  t_reported,
        "err_x": err_x, "err_y": err_y, "err": err,
        "in_tolerance_50": err is not None and err < 50,
        "total_steps":
            (t_reported - t_leak) if (t_reported is not None and t_leak is not None) else None,
        "lam2_mean": float(np.mean(lam2_log)) if lam2_log else None,
        "n_alive": sum(1 for r in sim.robots if r.is_alive),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=15)
    p.add_argument("--r-c",  type=int, default=500)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--out-dir", default="experiments/results")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    conditions = [("baseline", None)] + [(f"drop_R{i}", i) for i in range(5)]

    rows = []
    for label, drop in conditions:
        print(f"\n=== {label} ===")
        for i in range(args.runs):
            seed = 6000 + i
            rec = run_with_drop(seed, drop, args.r_c, args.max_steps)
            rec["condition"] = label
            rows.append(rec)
            ok = "✓" if rec["completed"] else "✗"
            err = f"{rec['err']:.1f}" if rec["err"] is not None else " - "
            print(f"  seed={seed}  {ok}  err={err:>5} px  total={rec['total_steps']}")

    fields = ["seed", "condition", "drop_id", "completed",
              "t_leak", "t_first_det", "t_consensus", "t_reported",
              "err_x", "err_y", "err", "in_tolerance_50",
              "total_steps", "lam2_mean", "n_alive"]
    csv_path = os.path.join(args.out_dir, "exp5_ablation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\n✓ CSV → {csv_path}")

    lines = [
        f"Experiment 5 — Per-robot ablation (r_c={args.r_c}, N={args.runs} per arm)",
        "",
        f"{'Condition':<14} {'done':<7} {'err mean [CI]':<22} "
        f"{'within ρ_m':<12} {'total mean [CI]':<22}",
        "-" * 78,
    ]
    for label, _ in conditions:
        sub  = [r for r in rows if r["condition"] == label]
        done = [r for r in sub if r["completed"]]
        s_err = summarise([r["err"]         for r in done])
        s_tot = summarise([r["total_steps"] for r in done])
        within = sum(1 for r in done if r["in_tolerance_50"])
        lines.append(
            f"{label:<14} {len(done)}/{len(sub):<5} "
            f"{fmt_ci(s_err['mean'], s_err['ci95_lo'], s_err['ci95_hi']):<22} "
            f"{within}/{len(done):<7} "
            f"{fmt_ci(s_tot['mean'], s_tot['ci95_lo'], s_tot['ci95_hi'], 0):<22}"
        )

    # Robustness summary: how much does the error spread across robot identities?
    drop_means = []
    for label, drop in conditions:
        if drop is None: continue
        done = [r for r in rows if r["condition"] == label and r["completed"]]
        errs = [r["err"] for r in done if r["err"] is not None]
        if errs:
            drop_means.append((label, sum(errs) / len(errs)))
    if drop_means:
        m_lo = min(m for _, m in drop_means)
        m_hi = max(m for _, m in drop_means)
        lines += [
            "",
            f"Spread of mean error across robot identities: "
            f"min = {m_lo:.1f} px, max = {m_hi:.1f} px, range = {m_hi - m_lo:.1f} px",
            "",
            "Interpretation: small range across drop_id means no robot is critical;",
            "perception is genuinely distributed.",
        ]
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.out_dir, "exp5_summary.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
