"""
experiments/exp1_mode1_accuracy.py
----------------------------------
Experiment 1 - Mode-1 localization accuracy at r_c = 500 px

Goal: characterize the steady-state distribution of localization error when the communication network is essentially never disconnected, so
that the measurement reflects DKF + gossip behaviour and is not dominated by topology effects

Sample size: N = 100 independent seeds (2000..2099)
Outputs:
  results/exp1_mode1.csv     - one row per run, all FIELDS from runner
  results/exp1_summary.txt   - mean +- std summary table
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runner import run_one_cycle, FIELDS
from stats import summarise, fmt_pm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--r-c", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=2500)
    p.add_argument("--out-dir", default="experiments/results")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for i in range(args.runs):
        seed = 2000 + i
        rec = run_one_cycle(seed,
                            comm_radius=args.r_c,
                            exploration="lloyd",
                            max_steps=args.max_steps)
        rows.append(rec)
        ok  = "OK" if rec["completed"] else "--"
        err = f"{rec['err']:.1f}" if rec["err"] is not None else " - "
        print(f"  seed={seed}  {ok}  err={err:>5} px  total={rec['total_steps']}")

    # CSV
    csv_path = os.path.join(args.out_dir, "exp1_mode1.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"\nCSV -> {csv_path}")

    # Summary (mean +- std, matching the report)
    done = [r for r in rows if r["completed"]]
    lines = [
        "Experiment 1 - Mode-1 accuracy",
        f"  r_c = {args.r_c} px,  runs = {args.runs},  completed = {len(done)}/{len(rows)}",
        "",
        f"{'Metric':<22} {'mean +- std':<20} {'median':>8} {'max':>7}",
        "-" * 62,
    ]
    for key, label in [
        ("err",               "Localization error px"),
        ("patrol_steps",      "Patrol-to-detect"),
        ("investigate_steps", "Detect-to-consensus"),
        ("fuse_steps",        "Consensus-to-report"),
        ("total_steps",       "Total (leak-report)"),
        ("lam2_mean",         "Mean lambda_2 / cycle"),
    ]:
        s = summarise([r[key] for r in done], label)
        if s["n"] == 0:
            continue
        lines.append(
            f"{label:<22} "
            f"{fmt_pm(s['mean'], s['std']):<20} "
            f"{s['median']:>8.1f} {s['max']:>7.1f}")

    within50 = sum(1 for r in done if r["in_tolerance_50"])
    within25 = sum(1 for r in done if r["in_tolerance_25"])
    lines += [
        "",
        f"Within rho_m = 50 px:  {within50}/{len(done)}",
        f"Within        25 px:  {within25}/{len(done)}",
    ]
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.out_dir, "exp1_summary.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()