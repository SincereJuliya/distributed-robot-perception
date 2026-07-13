"""
experiments/exp2_comm_radius.py
-------------------------------
Experiment 2 - Effect of communication radius r_c

Design: paired-seeds sweep. The SAME seeds 3000..3000+N-1 are replayed under every r_c value, so the variance attributable to the leak source
position cancels out when comparing radii (block design)

Sweep: r_c in {200, 250, 300, 350, 500} px
N: --runs seeds per radius (default 15)

Outputs:
  results/exp2_comm_radius.csv    - one row per (seed, r_c)
  results/exp2_summary.txt        - per-radius table, mean +- std
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
    p.add_argument("--runs",  type=int, default=15)
    p.add_argument("--radii", type=int, nargs="+",
                   default=[200, 250, 300, 350, 500])
    p.add_argument("--max-steps", type=int, default=3500)
    p.add_argument("--out-dir", default="experiments/results")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [3000 + i for i in range(args.runs)]
    rows  = []
    for rc in args.radii:
        print(f"\n=== r_c = {rc} px ===")
        for seed in seeds:
            rec = run_one_cycle(seed,
                                comm_radius=rc,
                                exploration="lloyd",
                                max_steps=args.max_steps)
            rows.append(rec)
            ok = "OK" if rec["completed"] else "--"
            err = f"{rec['err']:.1f}" if rec["err"] is not None else " - "
            print(f"  rc={rc:3d}  seed={seed}  {ok}  "
                  f"err={err:>5} px  total={rec['total_steps']}  "
                  f"lam2={rec['lam2_mean']:.3f}" if rec['lam2_mean'] is not None else "lam2=n/a")

    # CSV
    csv_path = os.path.join(args.out_dir, "exp2_comm_radius.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"\nCSV -> {csv_path}")

    # Per-radius summary (mean +- std, matching the report)
    lines = [
        "Experiment 2 - Communication radius sweep",
        f"  seeds = {args.runs} (paired across radii)",
        "",
        f"{'r_c':>5} {'done':>6} {'err mean +- std':<20} "
        f"{'total mean +- std':<22} {'lam2 mean +- std':<20} {'lam2=0 %':>9}",
        "-" * 88,
    ]
    for rc in args.radii:
        sub  = [r for r in rows if r["comm_radius"] == rc]
        done = [r for r in sub if r["completed"]]
        s_err  = summarise([r["err"]           for r in done])
        s_tot  = summarise([r["total_steps"]   for r in done])
        s_lam  = summarise([r["lam2_mean"]     for r in done])
        s_zero = summarise([r["lam2_zero_pct"] for r in done])
        if s_err["n"] == 0:
            lines.append(f"{rc:>5} {len(done)}/{len(sub):<3}  (no completed runs)")
            continue
        lines.append(
            f"{rc:>5} {len(done)}/{len(sub):<3} "
            f"{fmt_pm(s_err['mean'], s_err['std']):<20} "
            f"{fmt_pm(s_tot['mean'], s_tot['std'], 0):<22} "
            f"{fmt_pm(s_lam['mean'], s_lam['std'], 3):<20} "
            f"{s_zero['mean']:>8.1f}%"
        )
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.out_dir, "exp2_summary.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()