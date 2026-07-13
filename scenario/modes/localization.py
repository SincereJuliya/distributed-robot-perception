"""
scenario/modes/localization.py
------------------------------
Mode 1 - Hazard Localization

Phases:
  MONITOR   -> uniform Lloyd patrol; no gas
  LEAK      -> gas active; robots steer toward strongest reading
  CONSENSUS -> DKF beliefs fuse; sigma falls below threshold (stable)
  REPORTED  -> coordinates displayed; gas dissipates; back to MONITOR
"""

import numpy as np

import config
from environment.map import random_leak_point


class Phase:
    MONITOR   = "MONITORING"
    LEAK      = "LEAK DETECTED"
    CONSENSUS = "REACHING CONSENSUS..."
    REPORTED  = "REPORTED — patrolling map"


class LocalisationMode:
    """State machine for the hazard-localization scenario."""

    name = "Hazard Localization"

    def __init__(self, robots, gas, lloyd, dkf, free_space, event_log):
        self.robots     = robots
        self.gas        = gas
        self.lloyd      = lloyd
        self.dkf        = dkf
        self.free_space = free_space
        self.log        = event_log

        self.phase                  = Phase.MONITOR
        self.phase_ticks            = config.MONITOR_STEPS
        self._reported_ticks        = 0
        self.source_pxs: list       = []    # list of (x,y) ground-truth leaks
        self.agreed_px              = None
        self.consensus_history      = []    # sigma over time
        self.lambda2_history        = []    # lambda_2 over time
        self.leak_count             = 0
        self._consensus_stable      = 0

    #  Activation / deactivation 

    def enter(self, step):
        self.log.info(step, "MODE", f"entered {self.name}")
        # Reset to a clean MONITORING phase
        self.phase = Phase.MONITOR
        self.phase_ticks = config.MONITOR_STEPS

    def exit(self, step):
        self.log.info(step, "MODE", f"exiting {self.name}")
        self.gas.clear()
        self.agreed_px = None
        for r in self.robots:
            r.dkf_mu = None; r.dkf_P = None
            r.has_detected = False

    #  Main step 

    def update(self, step):
        ph = self.phase

        if ph == Phase.MONITOR:
            self.phase_ticks -= 1
            self.lloyd.update_targets_uniform()
            if self.phase_ticks <= 0:
                self._start_leak(step)

        elif ph == Phase.LEAK:
            # Robots do NOT know where the gas is.
            # Each robot continues its Lloyd-Voronoi patrol until its OWN local sensor reads above the detection threshold.
            # Only AFTER self-detection does a robot move toward the gas.

            # First: Lloyd sets patrol targets for ALL alive robots
            self.lloyd.update_targets_uniform()

            # SCOUT ALERT MECHANISM !
            # Any robot with has_detected=True (real peak seen in radius) serves as a reliable scout
            # Its POSITION is always accurate (GPS), even if its DKF belief is biased 
            # Nearby robots break their Lloyd patrol and head to the scout's position to VERIFY with their own clean sensors
            # We use has_detected (not self_observed) because self_observed can fire from sensor noise far from gas; has_detected requires a real peak in the sensing radius
            alive_robots = [r for r in self.robots if r.is_alive]
            scouts       = [r for r in alive_robots if r.has_detected]

            for r in alive_robots:
                if r.has_detected and r.dkf_mu is not None:
                    if r.manual_degradation:
                        # DEGRADED scout: STAY in place as a beacon
                        # Don't follow biased belief - clean robots need a stable point to converge to
                        r.target = r.position.copy()
                    else:
                        # HEALTHY scout: move toward own belief stand-off
                        d = r.position - r.dkf_mu
                        n = np.linalg.norm(d)
                        if n > 1e-3:
                            r.target = r.dkf_mu + (d / n) * 60
                        else:
                            r.target = r.dkf_mu.copy()
                elif scouts:
                    # I don't have a peak yet - but a scout does
                    # Head toward the nearest scout's POSITION (reliable) within comm range
                    nearest = None
                    nearest_d = float("inf")
                    for s in scouts:
                        if s.id == r.id: continue
                        d = float(np.linalg.norm(s.position - r.position))
                        if d <= r.comm_r and d < nearest_d:
                            nearest = s; nearest_d = d
                    if nearest is not None:
                        r.target = nearest.position.copy()
                # else: keep Lloyd patrol

            # Transition to CONSENSUS only when MAJORITY of robots have made their OWN observation by physically being near the gas
            alive = alive_robots
            n_sniff  = sum(1 for r in alive if r.self_observed)
            n_detect = sum(1 for r in alive if r.has_detected)
            threshold = max(2, int(len(alive) * 0.5 + 0.5))

            # Log detection only when has_detected count changes OR every 60 steps as heartbeat
            # We log ONLY has_detected (real peak in sensor radius)
            last_n_detect = getattr(self, "_last_n_detect", -1)
            count_changed = (n_detect != last_n_detect)
            if (count_changed or step % 60 == 0) and (n_sniff > 0 or n_detect > 0):
                lam2 = self.lambda2_history[-1] if self.lambda2_history else 0.0
                self.log.info(step, "DETECT",
                    f"{n_detect}/{threshold} located the leak  lambda_2={lam2:.2f}",
                    severity="warn")
                self._last_n_detect = n_detect

            if n_detect >= threshold:
                self.log.phase_transition(step, "LEAK", "CONSENSUS")
                self.phase = Phase.CONSENSUS

        elif ph == Phase.CONSENSUS:
            self._consensus_step(step)

        elif ph == Phase.REPORTED:
            self.gas.stop_all()
            self.lloyd.update_targets_uniform()
            # Stay in REPORTED for a generous window so the user can see the robots return to their Voronoi cells and resume patrolling
            # Exit only when (a) gas has cleared AND (b) at least REPORT_HOLD_STEPS have elapsed since entering REPORTED
            self._reported_ticks = getattr(self, "_reported_ticks", 0) + 1
            hold = getattr(config, "REPORT_HOLD_STEPS", 180)
            if self.gas.is_clear() and self._reported_ticks > hold:
                self.log.phase_transition(step, "REPORTED", "MONITOR")
                self._reset_for_next_cycle()

    #  Helpers 

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
        # CRITICAL: clear stale DKF beliefs from any previous cycle(otherwise consensus instantly "locks" on the old leak location)
        for r in self.robots:
            r.dkf_mu = None; r.dkf_P = None
            r.dkf_seed_mu = None
            r.has_detected = False
            r.self_observed = False
        self.agreed_px = None
        self.phase = Phase.LEAK
        self.leak_count += 1
        self._consensus_stable = 0
        self.consensus_history.clear()

    def _consensus_step(self, step):
        # Apply the same scout-attraction logic as in LEAK phase: robots without their own belief should head toward a scout 
        # within or out of comm range - once in CONSENSUS we use the broadest possible attraction so isolated robots reconnect
        alive_robots = [r for r in self.robots if r.is_alive]
        scouts       = [r for r in alive_robots if r.has_detected]
        # Centroid of all scouts - used as fallback for robots out of comm range of any individual scout
        if scouts:
            scout_centroid = np.mean([s.position for s in scouts], axis=0)
        else:
            scout_centroid = None

        for r in alive_robots:
            # Bugfix: stand-off behaviour requires an OWN detection
            # (has_detected), not merely a belief received via gossip -
            # same condition as in the LEAK phase. Otherwise a robot that
            # never sensed the gas could stop 80 px short of it and never
            # contribute the independent observation the consensus gate
            # counts.
            if r.has_detected and r.dkf_mu is not None:
                if r.manual_degradation:
                    # DEGRADED scout: stay in place as a beacon
                    r.target = r.position.copy()
                else:
                    # HEALTHY scout: stand off from own belief
                    d = r.position - r.dkf_mu
                    n = np.linalg.norm(d)
                    if n > 1e-3:
                        r.target = r.dkf_mu + (d / n) * 80
                    else:
                        r.target = r.dkf_mu.copy()
            elif scouts:
                # No own belief but there are scouts - head toward nearest scout if within comm range, else toward the scout centroid to break out of isolation
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
                    # Isolated - go toward the scout cluster anyway
                    r.target = scout_centroid.copy()
            else:
                # Should not happen in CONSENSUS phase, but fallback
                cell = self.lloyd.cells.get(r.id) if hasattr(self.lloyd, 'cells') else None
                if cell and not cell.is_empty:
                    c = cell.centroid
                    r.target = np.array([c.x, c.y])

        # sigma MUST be computed from SEED beliefs (the initial estimate each robot made from its OWN sensor)
        sigma_display = self.dkf.estimate_disagreement_selfobs()
        
        # Fallback: if not enough healthy seeds yet, fall back to the spread of ALL current beliefs so we at least show a number in the panel
        if sigma_display >= 998:
            sigma_display = self.dkf.estimate_disagreement_all()
        self.consensus_history.append(sigma_display)
        sigma_lock = sigma_display

        alive = alive_robots
        # Count healthy robots with belief
        n_obs_healthy = sum(1 for r in alive
                            if r.has_detected and not r.manual_degradation)
        n_obs_total   = sum(1 for r in alive if r.has_detected)

        # Require at least 2 healthy observers for a valid consensus
        majority = (n_obs_healthy >= 2 and
                    n_obs_total >= max(2, int(len(alive) * 0.5 + 0.5)))

        if majority and sigma_lock < config.CONSENSUS_THRESH:
            self._consensus_stable += 1
        else:
            self._consensus_stable = max(0, self._consensus_stable - 1)

        # Log metrics every 10 steps so they appear in the event log
        if step % 10 == 0:
            lam2 = self.lambda2_history[-1] if self.lambda2_history else 0.0
            self.log.info(step, "CONSENSUS",
                f"sigma={sigma_display:.1f}px  obs={n_obs_healthy}H/{n_obs_total}T  "
                f"stable={self._consensus_stable}/{config.CONSENSUS_STABLE_STEPS}  "
                f"lambda_2={lam2:.3f}",
                severity="event")

        if self._consensus_stable >= config.CONSENSUS_STABLE_STEPS:
            self._lock_consensus(step)

    def _lock_consensus(self, step):
        mean_mu, _ = self.dkf.network_belief()
        if mean_mu is not None:
            self.agreed_px = tuple(mean_mu)
            # Closest true source (in case of multiple)
            if self.source_pxs:
                dists = [np.linalg.norm(np.array(s) - mean_mu)
                         for s in self.source_pxs]
                idx = int(np.argmin(dists))
                err = dists[idx]
                self.log.leak_resolved(step, self.agreed_px, err)
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
            # Clear patrol waypoint so Lloyd picks a fresh one inside the (new) Voronoi cell
            r._patrol_waypoint = None

    #  External hooks 
    def force_new_leak(self, step):
        """Triggered by L key - immediate new leak."""
        if self.phase in (Phase.MONITOR, Phase.REPORTED):
            self._start_leak(step)

    #  Metrics for the right panel 
    def progress_frac(self) -> float:
        if self.phase != Phase.CONSENSUS:
            return 0.0
        return self._consensus_stable / config.CONSENSUS_STABLE_STEPS