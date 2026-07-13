"""
experiments/exp5_ablation.py
----------------------------
Experiment 5 - Per-robot ablation.

Goal: demonstrate the distributed perception pipeline has no critical
agent. We remove each robot i in {0,1,2,3,4} in turn (force it dead
before the leak starts) and measure localization error and total time.
If perception were secretly centralised on one "lead" robot, removing
that robot would sharply degrade the system; if it is genuinely
distributed, removing any single robot yields similar, small degradation.

Design: paired seeds 6000..6000+N-1 replayed under N+1 conditions:
  baseline, drop R0, drop R1, drop R2, drop R3, drop R4.

Uses the shared runner (run_one_cycle with drop_robots), so every CSV
column has the same definition as the other experiments.

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
from stats  import summarise, fmt_pm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=15)
    p.add_argument("--r-c",  type=int, default=500)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--out-dir", default="experiments/results")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    conditions = [("baseline", ())] + [(f"drop_R{i}", (i,)) for i in range(5)]

    rows = []
    for label, drop in conditions:
        print(f"\n=== {label} ===")
        for i in range(args.runs):
            seed = 6000 + i
            rec = run_one_cycle(seed,
                                comm_radius=args.r_c,
                                exploration="lloyd",
                                drop_robots=drop,
                                max_steps=args.max_steps)
            rec["condition"] = label
            rows.append(rec)
            ok = "OK" if rec["completed"] else "--"
            err = f"{rec['err']:.1f}" if rec["err"] is not None else " - "
            print(f"  seed={seed}  {ok}  err={err:>5} px  total={rec['total_steps']}")

    fields = FIELDS + ["condition"]
    csv_path = os.path.join(args.out_dir, "exp5_ablation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\nCSV -> {csv_path}")

    # Per-condition summary (mean +- std)
    lines = [
        f"Experiment 5 - Per-robot ablation (r_c={args.r_c}, N={args.runs} per arm)",
        "",
        f"{'Condition':<14} {'done':<7} {'err mean +- std':<20} "
        f"{'within rho_m':<13} {'total mean +- std':<20}",
        "-" * 76,
    ]
    for label, _ in conditions:
        sub  = [r for r in rows if r["condition"] == label]
        done = [r for r in sub if r["completed"]]
        if not done:
            lines.append(f"{label:<14} 0/{len(sub):<5} (no completed runs)")
            continue
        s_err = summarise([r["err"]         for r in done])
        s_tot = summarise([r["total_steps"] for r in done])
        within = sum(1 for r in done if r["in_tolerance_50"])
        lines.append(
            f"{label:<14} {len(done)}/{len(sub):<5} "
            f"{fmt_pm(s_err['mean'], s_err['std']):<20} "
            f"{within}/{len(done):<11} "
            f"{fmt_pm(s_tot['mean'], s_tot['std'], 0):<20}"
        )

    # Robustness: spread of mean error across the five drop identities
    drop_means = []
    for label, drop in conditions:
        if not drop:
            continue
        done = [r for r in rows if r["condition"] == label and r["completed"]]
        errs = [r["err"] for r in done if r["err"] is not None]
        if errs:
            drop_means.append((label, sum(errs) / len(errs)))
    if drop_means:
        m_lo = min(m for _, m in drop_means)
        m_hi = max(m for _, m in drop_means)
        base = [r for r in rows if r["condition"] == "baseline" and r["completed"]]
        base_std = summarise([r["err"] for r in base])["std"] if base else float("nan")
        lines += [
            "",
            f"Spread of mean error across drop identities: "
            f"min={m_lo:.1f} px, max={m_hi:.1f} px, range={m_hi - m_lo:.1f} px",
            f"Baseline error std: {base_std:.1f} px",
            "",
            "A range smaller than the baseline std means no single robot is",
            "critical: perception is genuinely distributed.",
        ]
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.out_dir, "exp5_summary.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()