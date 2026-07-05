"""
experiments/exp4_degradation.py
-----------------------------------
Experiment 4 — Mode-2 sensor degradation robustness.

Design: paired-seed three-arm comparison.
  baseline    : all 5 robots healthy
  k=1 degraded: robot 2 forced into degradation at step 80
  k=2 degraded: robots 2 and 3 forced into degradation at step 80

The Pasqualetti–Bicchi–Bullo bound says n=5 non-colluding faulty agents
tolerate up to k < n/2 = 2 simultaneously faulty; we test both k=1 and
k=2 to verify the bound holds in our system.

Outputs:
  results/exp4_degradation.csv
  results/exp4_summary.txt
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runner import run_one_cycle, FIELDS
from stats import summarise, fmt_ci


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--r-c",  type=int, default=350)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--out-dir", default="experiments/results")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    conditions = [
        ("baseline",    ()),
        ("k=1 degraded", (2,)),
        ("k=2 degraded", (2, 3)),
        ("k=3 degraded", (2, 3, 4)),
        ("k=4 degraded", (1, 2, 3, 4)),
    ]
    for label, degraded in conditions:
        print(f"\n=== {label} ===")
        for i in range(args.runs):
            seed = 5000 + i
            rec = run_one_cycle(seed,
                                comm_radius=args.r_c,
                                exploration="lloyd",
                                degrade_robots=degraded,
                                max_steps=args.max_steps)
            rec["condition"] = label
            rows.append(rec)
            ok = "✓" if rec["completed"] else "✗"
            err = f"{rec['err']:.1f}" if rec["err"] is not None else " - "
            print(f"  seed={seed}  {ok}  err={err:>5}  total={rec['total_steps']}")

    csv_path = os.path.join(args.out_dir, "exp4_degradation.csv")
    fields = FIELDS + ["condition"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\n✓ CSV → {csv_path}")

    lines = [
        f"Experiment 4 — Mode-2 degradation (r_c={args.r_c}, N={args.runs} per arm)",
        "",
        f"{'Condition':<14} {'done':<7} {'err mean [CI]':<22} "
        f"{'within 50px':<12} {'total mean [CI]':<22}",
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
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.out_dir, "exp4_summary.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
