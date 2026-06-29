"""
experiments/reanalyse.py
----------------------------
Run AFTER all four experiments are done. Re-reads the CSVs and computes
extras that we didn't ask for in the first pass:

  • Paired t-test and Wilcoxon signed-rank for Lloyd vs RandomWalk
    (correct test for the paired-seed design).
  • Mean wind-direction bias in the Mode-1 error scatter:
    (mean err_x, mean err_y) and a one-sample t-test against zero.
  • Correlations between mean λ₂ and total time / error in the radius sweep.

No simulations are re-run. Pure post-hoc analysis from CSVs.
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import (bootstrap_ci, paired_t_test, wilcoxon_signed_rank,
                   summarise, fmt_ci)

RESULTS = "experiments/results"


def _load(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if v in ("", "None", None):
                    row[k] = None
                elif v in ("True", "False"):
                    row[k] = (v == "True")
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
            out.append(row)
    return out


# ── 1. exp1: wind-direction bias ───────────────────────────────────────────
print("=" * 64)
print("EXP1 — wind-direction bias check")
print("=" * 64)
e1 = _load(f"{RESULTS}/exp1_mode1.csv")
done = [r for r in e1 if r["completed"]]
exs = [r["err_x"] for r in done if r["err_x"] is not None]
eys = [r["err_y"] for r in done if r["err_y"] is not None]

mx, mxlo, mxhi = bootstrap_ci(exs)
my, mylo, myhi = bootstrap_ci(eys)
print(f"mean err_x = {fmt_ci(mx, mxlo, mxhi)}  px")
print(f"mean err_y = {fmt_ci(my, mylo, myhi)}  px")

# One-sample t-test against zero
try:
    from scipy import stats
    tx, px = stats.ttest_1samp(exs, 0.0)
    ty, py = stats.ttest_1samp(eys, 0.0)
    print(f"H0: mean err_x = 0  →  t = {tx:+.3f},  p = {px:.4f}")
    print(f"H0: mean err_y = 0  →  t = {ty:+.3f},  p = {py:.4f}")
except ImportError:
    pass

# Direction of bias relative to wind vector
print()
print("WIND_VEC in config is approximately (+0.06, +0.02) — downwind = +x mostly.")
print("If estimate is downwind of true source, err_x = est_x − src_x should be > 0.")


# ── 2. exp3: paired tests, correctly ───────────────────────────────────────
print()
print("=" * 64)
print("EXP3 — Lloyd vs RandomWalk, PAIRED tests")
print("=" * 64)
e3 = _load(f"{RESULTS}/exp3_explorers.csv")
L = {int(r["seed"]): r for r in e3 if r["exploration"] == "lloyd" and r["completed"]}
R = {int(r["seed"]): r for r in e3 if r["exploration"] == "random_walk" and r["completed"]}
seeds = sorted(set(L) & set(R))

for key, label in [("err",          "Localization error"),
                   ("patrol_steps", "Patrol time"),
                   ("total_steps",  "Total time")]:
    a = [L[s][key] for s in seeds]
    b = [R[s][key] for s in seeds]
    diffs = [ai - bi for ai, bi in zip(a, b)]
    md, mlo, mhi = bootstrap_ci(diffs)
    t, p_t = paired_t_test(a, b)
    w, p_w = wilcoxon_signed_rank(a, b)
    print(f"\n  {label}:  N_pairs = {len(a)}")
    print(f"    Lloyd  − RandomWalk  diff = {fmt_ci(md, mlo, mhi)}")
    print(f"    paired t   = {t:+.3f},  p = {p_t:.4f}")
    print(f"    Wilcoxon W = {w:.0f},   p = {p_w:.4f}")


# ── 3. exp2: λ₂ vs total time correlation ──────────────────────────────────
print()
print("=" * 64)
print("EXP2 — λ₂ correlation analysis")
print("=" * 64)
e2 = _load(f"{RESULTS}/exp2_comm_radius.csv")
done = [r for r in e2 if r["completed"]
        and r["lam2_mean"] is not None and r["total_steps"] is not None]
lam = np.array([r["lam2_mean"]   for r in done])
tot = np.array([r["total_steps"] for r in done])
err = np.array([r["err"]         for r in done if r["err"] is not None])
lam_for_err = np.array([r["lam2_mean"] for r in done if r["err"] is not None])

try:
    from scipy import stats
    rho1, p1 = stats.spearmanr(lam, tot)
    rho2, p2 = stats.spearmanr(lam_for_err, err)
    print(f"Spearman ρ (λ₂̄, T_tot) = {rho1:+.3f},  p = {p1:.4g}  (N = {len(lam)})")
    print(f"Spearman ρ (λ₂̄, ‖e‖)  = {rho2:+.3f},  p = {p2:.4g}  (N = {len(err)})")
    print()
    print("Negative correlation expected: better connectivity → faster, more accurate.")
except ImportError:
    print("scipy not available — skipping correlation")
