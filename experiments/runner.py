"""
experiments/core.py
-----------------------
Single-source-of-truth runner for ONE leak cycle.

Returns a flat dict with all metrics; every experiment script consumes this
function so that all CSVs share the same column semantics.

Metric definitions (all values in simulation steps unless stated):
  t_leak       — step at which MONITOR → LEAK DETECTED.
                 (gas appears; sources are sampled)
  t_first_det  — step at which the FIRST robot sets has_detected = True.
                 (its own sensor saw a real in-radius peak)
                 None if no robot ever detects within max_steps.
  t_consensus  — step at which LEAK DETECTED → REACHING CONSENSUS.
                 (>= 50% of alive robots have has_detected = True)
  t_reported   — step at which REACHING CONSENSUS → REPORTED.
                 (σ < CONSENSUS_THRESH for CONSENSUS_STABLE_STEPS steps)

Derived times (only meaningful if all four anchors exist):
  patrol_steps      = t_first_det - t_leak       (passive detection latency)
  investigate_steps = t_consensus - t_first_det  (scout-mediated convergence)
  fuse_steps        = t_reported  - t_consensus  (gossip + DKF lock-in)
  total_steps       = t_reported  - t_leak       (end-to-end)

Error metrics:
  err_x, err_y — agreed_px - source_pxs[0]  (signed, in px)
  err          — Euclidean norm of (err_x, err_y)

Connectivity:
  lam2_mean, lam2_min, lam2_zero_pct over [t_leak, t_reported].

Status:
  completed   — True iff t_reported is set within max_steps
"""

from __future__ import annotations
import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from simulation import Simulation


# Phase string constants — must match scenario/modes/localisation.py:Phase
_PHASE_MONITOR   = "MONITORING"
_PHASE_LEAK      = "LEAK DETECTED"
_PHASE_CONSENSUS = "REACHING CONSENSUS..."
_PHASE_REPORTED  = "REPORTED — patrolling map"


def run_one_cycle(
    seed: int,
    *,
    comm_radius: int | None = None,
    exploration: str | None = None,
    degrade_robots: tuple[int, ...] = (),
    degrade_after: int = 80,
    max_steps: int = 3500,
) -> dict:
    """
    Run a single leak-localization cycle and return a metric dict.

    Parameters
    ----------
    seed : int
        Seed for both `random` and `np.random` — fully determines run.
    comm_radius : int, optional
        If given, overrides config.COMM_RADIUS BEFORE Simulation() is built.
    exploration : {"lloyd", "random_walk"}, optional
        If given, overrides config.EXPLORATION_STRATEGY BEFORE build.
    degrade_robots : tuple[int]
        IDs of robots to degrade (Mode-2 conditions). Empty => baseline.
    degrade_after : int
        Step at which to switch to Mode 2 and apply degradation.
    max_steps : int
        Hard cap; run is marked incomplete if t_reported not reached.
    """
    # ── Config overrides (must happen BEFORE Simulation() reads config) ──
    if comm_radius is not None:
        config.COMM_RADIUS = int(comm_radius)
    if exploration is not None:
        config.EXPLORATION_STRATEGY = exploration

    random.seed(seed)
    np.random.seed(seed)

    sim = Simulation()

    # ── Tracking state ──
    t_leak = t_first_det = t_consensus = t_reported = None
    lam2_log: list[float] = []
    prev_phase = sim.scenario.phase

    for step in range(max_steps):
        sim.step()
        ph = sim.scenario.phase

        # Apply Mode-2 degradation at the requested step
        if degrade_robots and step == degrade_after:
            sim.scenario.switch_to("2", sim.step_count)
            for rid in degrade_robots:
                sim.scenario.active.degrade_specific(sim.step_count, rid)

        # Anchor 1: MONITOR → LEAK
        if prev_phase == _PHASE_MONITOR and ph == _PHASE_LEAK:
            t_leak = step
            lam2_log = []  # start fresh window from leak onset

        # Anchor 2: first robot with has_detected=True
        if t_leak is not None and t_first_det is None:
            if any(r.has_detected for r in sim.robots if r.is_alive):
                t_first_det = step

        # Anchor 3: LEAK → CONSENSUS
        if prev_phase == _PHASE_LEAK and ph == _PHASE_CONSENSUS:
            t_consensus = step

        # Anchor 4: CONSENSUS → REPORTED
        if prev_phase == _PHASE_CONSENSUS and ph == _PHASE_REPORTED:
            t_reported = step
            break

        # λ₂ accumulation (only after leak appears)
        if t_leak is not None and sim.scenario.lambda2_history:
            lam2_log.append(sim.scenario.lambda2_history[-1])

        prev_phase = ph

    # ── Compute error vector ──
    err_x = err_y = err = None
    if (sim.scenario.agreed_px is not None
            and sim.scenario.source_pxs):
        ag  = np.asarray(sim.scenario.agreed_px, dtype=float)
        src = np.asarray(sim.scenario.source_pxs[0], dtype=float)
        d = ag - src
        err_x = float(d[0])
        err_y = float(d[1])
        err   = float(np.linalg.norm(d))

    completed = t_reported is not None

    # λ₂ summary over [leak, reported] window
    if lam2_log:
        lam2_mean = float(np.mean(lam2_log))
        lam2_min  = float(np.min(lam2_log))
        lam2_zero_pct = 100.0 * sum(1 for v in lam2_log if v <= 1e-3) / len(lam2_log)
    else:
        lam2_mean = lam2_min = lam2_zero_pct = None

    # Derived intervals
    def _diff(a, b): return (a - b) if (a is not None and b is not None) else None

    sim.log.close()

    return {
        "seed":        seed,
        "completed":   completed,
        # raw anchors
        "t_leak":         t_leak,
        "t_first_det":    t_first_det,
        "t_consensus":    t_consensus,
        "t_reported":     t_reported,
        # derived intervals
        "patrol_steps":     _diff(t_first_det, t_leak),
        "investigate_steps":_diff(t_consensus, t_first_det),
        "fuse_steps":       _diff(t_reported,  t_consensus),
        "total_steps":      _diff(t_reported,  t_leak),
        # error
        "err_x": err_x,
        "err_y": err_y,
        "err":   err,
        "in_tolerance_50": (err is not None and err < 50),
        "in_tolerance_25": (err is not None and err < 25),
        # connectivity
        "lam2_mean":     lam2_mean,
        "lam2_min":      lam2_min,
        "lam2_zero_pct": lam2_zero_pct,
        # context
        "comm_radius":  config.COMM_RADIUS,
        "exploration":  getattr(config, "EXPLORATION_STRATEGY", "lloyd"),
        "n_degraded":   len(degrade_robots),
        "robots_alive_end": sum(1 for r in sim.robots if r.is_alive),
    }


FIELDS = [
    "seed", "completed",
    "t_leak", "t_first_det", "t_consensus", "t_reported",
    "patrol_steps", "investigate_steps", "fuse_steps", "total_steps",
    "err_x", "err_y", "err", "in_tolerance_50", "in_tolerance_25",
    "lam2_mean", "lam2_min", "lam2_zero_pct",
    "comm_radius", "exploration", "n_degraded", "robots_alive_end",
]
