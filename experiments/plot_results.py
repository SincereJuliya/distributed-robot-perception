"""
experiments/plot_results.py
-------------------------------
Produce report-ready figures from the CSVs written by exp1..exp5.

All plots draw from REAL data only - no synthetic angles, no fabricated
points. The Fig.4-style scatter uses the (err_x, err_y) vector that
runner.run_one_cycle stores explicitly.

Outputs (130 dpi PNG):
  fig1_error_scatter.png      - (err_x, err_y) scatter with rho_m circle
  fig1_error_histogram.png    - histogram of ||e|| for exp1
  fig2_radius_sweep.png       - error / lambda_2 / time vs r_c with error bars
  fig3_explorers_box.png      - boxplot Lloyd vs RandomWalk
  fig4_degradation_box.png    - boxplot baseline / k=1 / k=2

Usage:
  python experiments/plot_results.py
"""

import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS = "experiments/results"


def _load(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rec = {}
            for k, v in r.items():
                if v in ("", "None", None):
                    rec[k] = None
                elif v in ("True", "False"):
                    rec[k] = (v == "True")
                else:
                    try:
                        rec[k] = float(v)
                    except ValueError:
                        rec[k] = v
            out.append(rec)
    return out


#  Figure 1a: real (err_x, err_y) scatter 
def fig_error_scatter(rows, out_path, rho_m: float = 50):
    pts = [(r["err_x"], r["err_y"])
           for r in rows
           if r["completed"] and r["err_x"] is not None]
    if not pts:
        print("skip scatter - no data"); return
    xs, ys = zip(*pts)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    ax.scatter(xs, ys, s=28, color="#5a8b5a", edgecolors="#1a1a2e",
               linewidths=0.6, alpha=0.85)
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(rho_m * np.cos(th), rho_m * np.sin(th),
            "r--", lw=1.5, label=f"rho_m = {rho_m:.0f} px")
    ax.axhline(0, color="#cccccc", lw=0.5)
    ax.axvline(0, color="#cccccc", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("e_x [px]"); ax.set_ylabel("e_y [px]")
    ax.set_title(f"Localization error vector  (N = {len(pts)})")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()
    print(f"OK {out_path}")


def fig_error_histogram(rows, out_path):
    errs = [r["err"] for r in rows if r["completed"] and r["err"] is not None]
    if not errs: return
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    ax.hist(errs, bins=20, color="#3b7a8c", edgecolor="#1a1a2e", alpha=0.85)
    ax.axvline(np.mean(errs),   color="#b04040", lw=2,
               label=f"mean = {np.mean(errs):.1f}")
    ax.axvline(np.median(errs), color="#c89000", lw=2, ls="--",
               label=f"median = {np.median(errs):.1f}")
    ax.axvline(50, color="red", ls=":", lw=1.5, label="rho_m = 50 px")
    ax.set_xlabel("||e|| [px]"); ax.set_ylabel("Frequency")
    ax.set_title(f"Localization error distribution (N = {len(errs)})")
    ax.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()
    print(f"OK {out_path}")


#  Figure 2: radius sweep with bootstrap error bars 
def fig_radius_sweep(rows, out_path):
    radii = sorted({int(r["comm_radius"]) for r in rows})
    err_pt, err_sd = [], []
    tot_pt, tot_sd = [], []
    lam_pt, lam_sd = [], []
    zero_mean = []
    def _ms(vals):
        a = np.array([v for v in vals if v is not None], dtype=float)
        return (float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0)
    for rc in radii:
        sub  = [r for r in rows
                if int(r["comm_radius"]) == rc and r["completed"]]
        m, s = _ms([r["err"]         for r in sub]); err_pt.append(m); err_sd.append(s)
        m, s = _ms([r["total_steps"] for r in sub]); tot_pt.append(m); tot_sd.append(s)
        m, s = _ms([r["lam2_mean"]   for r in sub]); lam_pt.append(m); lam_sd.append(s)
        zero_mean.append(np.mean([r["lam2_zero_pct"] for r in sub]))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="white")
    axes[0].errorbar(radii, err_pt, yerr=err_sd,
                     fmt="o-", color="#3b7a8c", capsize=4)
    axes[0].axhline(50, color="red", ls="--", lw=1, label="rho_m = 50 px")
    axes[0].set_xlabel("r_c [px]"); axes[0].set_ylabel("Localization error [px]")
    axes[0].set_title("Localization error (mean +- std)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.errorbar(radii, lam_pt, yerr=lam_sd,
                 fmt="s-", color="#c89000", capsize=4, label="mean lambda_2")
    ax2b = ax2.twinx()
    ax2b.plot(radii, zero_mean, "v--", color="#b04040", label="% disconnected")
    ax2.set_xlabel("r_c [px]"); ax2.set_ylabel("Mean lambda_2")
    ax2b.set_ylabel("% steps with lambda_2 = 0")
    ax2.set_title("Connectivity"); ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left"); ax2b.legend(loc="upper right")

    axes[2].errorbar(radii, tot_pt, yerr=tot_sd,
                     fmt="^-", color="#5a8b5a", capsize=4)
    axes[2].set_xlabel("r_c [px]"); axes[2].set_ylabel("Steps from leak to report")
    axes[2].set_title("Total time (mean +- std)"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()
    print(f"OK {out_path}")


#  Figure 3: Lloyd vs RandomWalk boxplots 
def fig_explorers_box(rows, out_path):
    L = [r["err"]         for r in rows if r["exploration"] == "lloyd"       and r["completed"]]
    R = [r["err"]         for r in rows if r["exploration"] == "random_walk" and r["completed"]]
    LT = [r["total_steps"] for r in rows if r["exploration"] == "lloyd"       and r["completed"]]
    RT = [r["total_steps"] for r in rows if r["exploration"] == "random_walk" and r["completed"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="white")
    axes[0].boxplot([L, R], patch_artist=True,
                    boxprops=dict(facecolor="#3b7a8c", alpha=0.6))
    axes[0].set_xticks([1, 2]); axes[0].set_xticklabels(["Lloyd", "RandomWalk"])
    axes[0].set_ylabel("Localization error [px]")
    axes[0].set_title(f"Error  (N = {len(L)} per arm)")
    axes[0].axhline(50, color="red", ls="--", lw=1)
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot([LT, RT], patch_artist=True,
                    boxprops=dict(facecolor="#c89000", alpha=0.6))
    axes[1].set_xticks([1, 2]); axes[1].set_xticklabels(["Lloyd", "RandomWalk"])
    axes[1].set_ylabel("Total time [steps]")
    axes[1].set_title("Leak -> report")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()
    print(f"OK {out_path}")


#  Figure 4: degradation boxplot 
def fig_degradation_box(rows, out_path):
    conds = ["baseline", "k=1 degraded", "k=2 degraded"]
    data = [[r["err"] for r in rows
             if r.get("condition") == c and r["completed"]] for c in conds]
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
    ax.boxplot(data, patch_artist=True,
               boxprops=dict(facecolor="#7a5a8c", alpha=0.6))
    ax.set_xticks(range(1, len(conds) + 1)); ax.set_xticklabels(conds)
    ax.axhline(50, color="red", ls="--", lw=1, label="rho_m")
    ax.set_ylabel("Localization error [px]")
    ax.set_title(f"Mode-2 sensor degradation  (N = {len(data[0])} per arm)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()
    print(f"OK {out_path}")


def fig_ablation_box(rows, out_path):
    conds = ["baseline", "drop_R0", "drop_R1", "drop_R2", "drop_R3", "drop_R4"]
    data = [[r["err"] for r in rows
             if r.get("condition") == c and r["completed"]
             and r["err"] is not None] for c in conds]
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    ax.boxplot(data, patch_artist=True,
               boxprops=dict(facecolor="#3b7a8c", alpha=0.55))
    ax.set_xticks(range(1, len(conds) + 1)); ax.set_xticklabels(conds)
    ax.axhline(50, color="red", ls="--", lw=1, label="rho_m = 50 px")
    ax.set_ylabel("Localization error [px]")
    ax.set_title(f"Per-robot ablation (N = {len(data[0])} per arm)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()
    print(f"OK {out_path}")


def main():
    os.makedirs(RESULTS, exist_ok=True)

    e1 = _load(f"{RESULTS}/exp1_mode1.csv")
    if e1:
        fig_error_scatter  (e1, f"{RESULTS}/fig1_error_scatter.png")
        fig_error_histogram(e1, f"{RESULTS}/fig1_error_histogram.png")

    e2 = _load(f"{RESULTS}/exp2_comm_radius.csv")
    if e2:
        fig_radius_sweep(e2, f"{RESULTS}/fig2_radius_sweep.png")

    e3 = _load(f"{RESULTS}/exp3_explorers.csv")
    if e3:
        fig_explorers_box(e3, f"{RESULTS}/fig3_explorers_box.png")

    e4 = _load(f"{RESULTS}/exp4_degradation.csv")
    if e4:
        fig_degradation_box(e4, f"{RESULTS}/fig4_degradation_box.png")

    e5 = _load(f"{RESULTS}/exp5_ablation.csv")
    if e5:
        fig_ablation_box(e5, f"{RESULTS}/fig5_ablation_box.png")


if __name__ == "__main__":
    main()