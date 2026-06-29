"""
experiments/exp3_explorers.py
---------------------------------
Experiment 3 — Lloyd–Voronoi vs Random Walk exploration.

Design: paired-seed comparison. The same N seeds are run under both
strategies. We then run Welch's t-test on the means and a non-parametric
Mann–Whitney U on the medians (the error distribution is skewed).

Outputs:
  results/exp3_explorers.csv  — one row per (seed, strategy)
  results/exp3_summary.txt    — summary + significance tests
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runner import run_one_cycle, FIELDS
from stats import (summarise, fmt_ci, paired_t_test, wilcoxon_signed_rank)


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
            ok = "✓" if rec["completed"] else "✗"
            err = f"{rec['err']:.1f}" if rec["err"] is not None else " - "
            print(f"  seed={seed} {strat:11s} {ok}  "
                  f"err={err:>5}  total={rec['total_steps']}")

    csv_path = os.path.join(args.out_dir, "exp3_explorers.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"\n✓ CSV → {csv_path}")

    # Summary
    lloyd = [r for r in rows if r["exploration"] == "lloyd"     and r["completed"]]
    rw    = [r for r in rows if r["exploration"] == "random_walk" and r["completed"]]

    lines = [
        f"Experiment 3 — Lloyd vs Random Walk (paired, r_c={args.r_c}, N={args.runs})",
        "",
        f"{'Metric':<22} {'Lloyd':<26} {'Random Walk':<26}",
        "-" * 76,
    ]
    for key, label in [
        ("err",          "Localization err px"),
        ("patrol_steps", "Patrol-to-detect"),
        ("total_steps",  "Total leak→report"),
    ]:
        sL = summarise([r[key] for r in lloyd])
        sR = summarise([r[key] for r in rw])
        lines.append(
            f"{label:<22} "
            f"{fmt_ci(sL['mean'], sL['ci95_lo'], sL['ci95_hi']):<26} "
            f"{fmt_ci(sR['mean'], sR['ci95_lo'], sR['ci95_hi']):<26}")

    lines.append("")
    lines.append(f"Completion:  Lloyd {len(lloyd)}/{args.runs}   "
                 f"RandomWalk {len(rw)}/{args.runs}")
    lines.append("")

    # The design is PAIRED (same seeds for both strategies), so we use
    # paired t-test and Wilcoxon signed-rank — NOT Welch / Mann-Whitney,
    # which assume independent samples.
    # Build paired arrays indexed by seed.
    seeds = sorted({int(r["seed"]) for r in lloyd} & {int(r["seed"]) for r in rw})
    L_by_seed = {int(r["seed"]): r for r in lloyd}
    R_by_seed = {int(r["seed"]): r for r in rw}

    lines.append("")
    for key, label in [("err",          "error"),
                       ("patrol_steps", "patrol_steps"),
                       ("total_steps",  "total_steps")]:
        a = [L_by_seed[s][key] for s in seeds if L_by_seed[s][key] is not None
                                              and R_by_seed[s][key] is not None]
        b = [R_by_seed[s][key] for s in seeds if L_by_seed[s][key] is not None
                                              and R_by_seed[s][key] is not None]
        t, p_t = paired_t_test(a, b)
        w, p_w = wilcoxon_signed_rank(a, b)
        lines.append(
            f"{label:<14}  paired t = {t:+.2f} (p = {p_t:.4f})  "
            f"Wilcoxon W = {w:.0f} (p = {p_w:.4f})  [N pairs = {len(a)}]"
        )
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.out_dir, "exp3_summary.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
