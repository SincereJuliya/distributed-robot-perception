"""
scenario/modes/degradation.py
-----------------------------
Mode 2 — Sensor Degradation & Trust-Weighted Consensus.

Same localization pipeline as Mode 1, but the gossip / DKF fusion uses
TrustReputation weights. When a robot's sensor degrades (either organically
or via manual D / 0-4 keys), its DKF belief drifts from the network mean.
Through the trust-update rule, neighbours lower their trust in the bad
robot — and its influence on the fused estimate shrinks automatically.

This demonstrates *robust* distributed estimation in the presence of
unreliable sensors.
"""

import numpy as np

import config
from environment.map import random_leak_point


class Phase:
    MONITOR   = "MONITORING"
    LEAK      = "LEAK DETECTED"
    CONSENSUS = "REACHING CONSENSUS..."
    REPORTED  = "REPORTED — patrolling map"


class DegradationMode:
    """Mode 2 — same flow as Mode 1, plus trust mechanics."""

    name = "Sensor Degradation & Trust"

    def __init__(self, robots, gas, lloyd, dkf, trust, free_space, event_log):
        self.robots     = robots
        self.gas        = gas
        self.lloyd      = lloyd
        self.dkf        = dkf
        self.trust      = trust
        self.free_space = free_space
        self.log        = event_log

        self.phase             = Phase.MONITOR
        self.phase_ticks       = config.MONITOR_STEPS
        self.source_pxs        = []
        self.agreed_px         = None
        self.consensus_history = []
        self.lambda2_history   = []
        self.leak_count        = 0
        self._consensus_stable = 0

        # Auto-degradation toggle
        self.auto_degrade        = False
        self._auto_degrade_timer = 0

    def enter(self, step):
        self.log.info(step, "MODE", f"entered {self.name}")
        self.phase = Phase.MONITOR
        self.phase_ticks = config.MONITOR_STEPS

    def exit(self, step):
        self.log.info(step, "MODE", f"exiting {self.name}")
        self.gas.clear()
        self.agreed_px = None
        # Reset DKF beliefs and manual-degrade flags
        for r in self.robots:
            r.dkf_mu = None; r.dkf_P = None
            r.has_detected = False
            r.manual_degradation = False

    # ── Manual degradation hooks ──────────────────────────────────────────────

    def degrade_random(self, step):
        import random
        alive = [r for r in self.robots if r.is_alive and not r.manual_degradation]
        if not alive: return None
        victim = random.choice(alive)
        victim.manual_degradation = True
        self.log.robot_event(step, victim.id,
                             "manual degradation triggered", "warn")
        return victim.id

    def degrade_specific(self, step, robot_id):
        for r in self.robots:
            if r.id == robot_id and r.is_alive:
                r.manual_degradation = True
                self.log.robot_event(step, r.id,
                                     "manual degradation triggered", "warn")
                return r.id
        return None

    def toggle_auto(self, step):
        self.auto_degrade = not self.auto_degrade
        msg = "auto-degradation ON" if self.auto_degrade else "auto-degradation OFF"
        self.log.info(step, "DEGRADE", msg, "event")

    def _maybe_auto_degrade(self, step):
        if not self.auto_degrade: return
        self._auto_degrade_timer += 1
        if self._auto_degrade_timer >= 220:
            self._auto_degrade_timer = 0
            self.degrade_random(step)

    # ── Main step ─────────────────────────────────────────────────────────────

    def update(self, step):
        self._maybe_auto_degrade(step)

        ph = self.phase
        if ph == Phase.MONITOR:
            self.phase_ticks -= 1
            self.lloyd.update_targets_uniform()
            if self.phase_ticks <= 0:
                self._start_leak(step)

        elif ph == Phase.LEAK:
            # Robots patrol via Lloyd until their OWN sensor detects gas.
            self.lloyd.update_targets_uniform()

            # ── SCOUT ALERT — critical for Mode 2 demonstration ─────────────
            # has_detected is the reliable alert signal (requires real peak
            # in sensing radius), not self_observed (which can fire on noise).
            # Even a degraded robot's POSITION is accurate (GPS), so others
            # can investigate by heading toward the scout's location.
            alive_robots = [r for r in self.robots if r.is_alive]
            scouts       = [r for r in alive_robots if r.has_detected]

            for r in alive_robots:
                if r.has_detected and r.dkf_mu is not None:
                    if r.manual_degradation:
                        # DEGRADED scout = beacon: stay in place
                        r.target = r.position.copy()
                    else:
                        d = r.position - r.dkf_mu
                        n = np.linalg.norm(d)
                        if n > 1e-3:
                            r.target = r.dkf_mu + (d / n) * 60
                        else:
                            r.target = r.dkf_mu.copy()
                elif scouts:
                    nearest = None
                    nearest_d = float("inf")
                    for s in scouts:
                        if s.id == r.id: continue
                        d = float(np.linalg.norm(s.position - r.position))
                        if d <= r.comm_r and d < nearest_d:
                            nearest = s; nearest_d = d
                    if nearest is not None:
                        r.target = nearest.position.copy()

            alive = alive_robots
            n_obs = sum(1 for r in alive if r.has_detected)
            threshold = max(2, int(len(alive) * 0.5 + 0.5))
            if n_obs >= threshold:
                self.log.phase_transition(step, "LEAK", "CONSENSUS")
                self.phase = Phase.CONSENSUS

        elif ph == Phase.CONSENSUS:
            self._consensus_step(step)

        elif ph == Phase.REPORTED:
            self.gas.stop_all()
            self.lloyd.update_targets_uniform()
            self._reported_ticks = getattr(self, "_reported_ticks", 0) + 1
            hold = getattr(config, "REPORT_HOLD_STEPS", 180)
            if self.gas.is_clear() and self._reported_ticks > hold:
                self.log.phase_transition(step, "REPORTED", "MONITOR")
                self._reset_for_next_cycle()

    def _start_leak(self, step):
        self.log.phase_transition(step, "MONITOR", "LEAK")
        self.gas.clear()
        self.source_pxs = []
        for _ in range(max(1, config.NUM_GAS_SOURCES)):
            p = random_leak_point(self.robots, self.free_space)
            if p is not None:
                self.source_pxs.append(p)
                self.gas.add_source(p)
                self.log.leak_started(step, p)
        # Clear stale DKF beliefs from any previous cycle
        for r in self.robots:
            r.dkf_mu = None; r.dkf_P = None
            r.has_detected = False
            r.dkf_seed_mu = None
            r.self_observed = False
        self.agreed_px = None
        self.phase = Phase.LEAK
        self.leak_count += 1
        self._consensus_stable = 0
        self.consensus_history.clear()

    def _consensus_step(self, step):
        # Scout-attraction in CONSENSUS: robots without their own belief
        # head toward the scout cluster, even if outside comm range.
        # This is critical when multiple robots are degraded — isolated
        # healthy robots need to reach the gas area to participate.
        alive_robots = [r for r in self.robots if r.is_alive]
        scouts       = [r for r in alive_robots if r.has_detected]
        scout_centroid = (np.mean([s.position for s in scouts], axis=0)
                          if scouts else None)

        for r in alive_robots:
            if r.dkf_mu is not None:
                if r.manual_degradation:
                    # DEGRADED scout: stay in place as a beacon
                    r.target = r.position.copy()
                else:
                    d = r.position - r.dkf_mu
                    n = np.linalg.norm(d)
                    if n > 1e-3:
                        r.target = r.dkf_mu + (d / n) * 80
                    else:
                        r.target = r.dkf_mu.copy()
            elif scouts:
                nearest = None
                nearest_d = float("inf")
                for s in scouts:
                    if s.id == r.id: continue
                    d = float(np.linalg.norm(s.position - r.position))
                    if d <= r.comm_r and d < nearest_d:
                        nearest = s; nearest_d = d
                if nearest is not None:
                    r.target = nearest.position.copy()
                elif scout_centroid is not None:
                    r.target = scout_centroid.copy()
            else:
                cell = self.lloyd.cells.get(r.id) if hasattr(self.lloyd, 'cells') else None
                if cell and not cell.is_empty:
                    c = cell.centroid
                    r.target = np.array([c.x, c.y])

        # Use seed beliefs (not post-gossip current beliefs) — see
        # comment in localization.py for full explanation.
        sigma_display = self.dkf.estimate_disagreement_selfobs()
        if sigma_display >= 998:
            sigma_display = self.dkf.estimate_disagreement_all()
        self.consensus_history.append(sigma_display)
        sigma_lock = sigma_display

        alive = alive_robots
        n_obs_healthy = sum(1 for r in alive
                            if r.has_detected and not r.manual_degradation)
        n_obs_total   = sum(1 for r in alive if r.has_detected)
        majority = (n_obs_healthy >= 2 and
                    n_obs_total >= max(2, int(len(alive) * 0.5 + 0.5)))

        if majority and sigma_lock < config.CONSENSUS_THRESH:
            self._consensus_stable += 1
        else:
            self._consensus_stable = max(0, self._consensus_stable - 1)

        if step % 10 == 0:
            lam2 = self.lambda2_history[-1] if self.lambda2_history else 0.0
            self.log.info(step, "CONSENSUS",
                f"σ={sigma_display:.1f}px  obs={n_obs_healthy}H/{n_obs_total}T  "
                f"stable={self._consensus_stable}/{config.CONSENSUS_STABLE_STEPS}  "
                f"λ₂={lam2:.3f}",
                severity="event")

        if self._consensus_stable >= config.CONSENSUS_STABLE_STEPS:
            self._lock_consensus(step)

    def _lock_consensus(self, step):
        mean_mu, _ = self.dkf.network_belief()
        if mean_mu is not None:
            self.agreed_px = tuple(mean_mu)
            if self.source_pxs:
                dists = [np.linalg.norm(np.array(s) - mean_mu)
                         for s in self.source_pxs]
                idx = int(np.argmin(dists))
                self.log.leak_resolved(step, self.agreed_px, dists[idx])
        self.log.phase_transition(step, "CONSENSUS", "REPORTED")
        self.phase = Phase.REPORTED

    def _reset_for_next_cycle(self):
        self.gas.clear()
        self.phase = Phase.MONITOR
        self.phase_ticks = config.MONITOR_STEPS
        self.agreed_px = None
        self._reported_ticks = 0
        for r in self.robots:
            r.dkf_mu = None; r.dkf_P = None
            r.has_detected = False
            r.self_observed = False
            r._patrol_waypoint = None

    def force_new_leak(self, step):
        if self.phase in (Phase.MONITOR, Phase.REPORTED):
            self._start_leak(step)

    def progress_frac(self) -> float:
        if self.phase != Phase.CONSENSUS:
            return 0.0
        return self._consensus_stable / config.CONSENSUS_STABLE_STEPS
