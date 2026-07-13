"""
experiments/exp3_explorers.py
-----------------------------
Experiment 3 - Lloyd-Voronoi vs Random Walk exploration

Outputs:
  results/exp3_explorers.csv  - one row per (seed, strategy)
  results/exp3_summary.txt    - per-strategy table + mean paired difference
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runner import run_one_cycle, FIELDS
from stats import summarise, fmt_pm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--r-c",  type=int, default=500)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--out-dir", default="experiments/results")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for i in range(args.runs):
        seed = 4000 + i
        for strat in ("lloyd", "random_walk"):
            rec = run_one_cycle(seed,
                                comm_radius=args.r_c,
                                exploration=strat,
                                max_steps=args.max_steps)
            rows.append(rec)
            ok = "OK" if rec["completed"] else "--"
            err = f"{rec['err']:.1f}" if rec["err"] is not None else " - "
            print(f"  seed={seed} {strat:11s} {ok}  "
                  f"err={err:>5}  total={rec['total_steps']}")

    csv_path = os.path.join(args.out_dir, "exp3_explorers.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"\nCSV -> {csv_path}")

    # Per-strategy summary (mean +- std)
    lloyd = [r for r in rows if r["exploration"] == "lloyd"       and r["completed"]]
    rw    = [r for r in rows if r["exploration"] == "random_walk" and r["completed"]]

    lines = [
        f"Experiment 3 - Lloyd vs Random Walk (paired, r_c={args.r_c}, N={args.runs})",
        "",
        f"{'Metric':<22} {'Lloyd':<20} {'Random Walk':<20}",
        "-" * 64,
    ]
    for key, label in [
        ("err",          "Localization err px"),
        ("patrol_steps", "Patrol-to-detect"),
        ("total_steps",  "Total leak-report"),
    ]:
        sL = summarise([r[key] for r in lloyd])
        sR = summarise([r[key] for r in rw])
        lines.append(
            f"{label:<22} "
            f"{fmt_pm(sL['mean'], sL['std']):<20} "
            f"{fmt_pm(sR['mean'], sR['std']):<20}")

    lines.append("")
    lines.append(f"Completion:  Lloyd {len(lloyd)}/{args.runs}   "
                 f"RandomWalk {len(rw)}/{args.runs}")

    # Paired difference (Lloyd - RandomWalk), mean +- std over shared seeds.
    # The design is paired (same seed per strategy), so we report the paired
    # difference directly rather than an unpaired test.
    L_by_seed = {int(r["seed"]): r for r in lloyd}
    R_by_seed = {int(r["seed"]): r for r in rw}
    seeds = sorted(set(L_by_seed) & set(R_by_seed))

    lines.append("")
    lines.append("Mean paired difference (Lloyd - RandomWalk):")
    for key, label in [("err",          "error px"),
                       ("patrol_steps", "patrol steps"),
                       ("total_steps",  "total steps")]:
        pairs = [(L_by_seed[s][key], R_by_seed[s][key]) for s in seeds
                 if L_by_seed[s][key] is not None and R_by_seed[s][key] is not None]
        if len(pairs) < 2:
            continue
        diff = np.array([a - b for a, b in pairs], dtype=float)
        lines.append(
            f"  {label:<14} {diff.mean():+.2f} +- {diff.std(ddof=1):.2f}"
            f"   [N pairs = {len(pairs)}]")

    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.out_dir, "exp3_summary.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()