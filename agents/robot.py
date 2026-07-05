"""
agents/robot.py
---------------
Ground robot. Carries state used by all three modes:

  • Mode 1 (Localization): dkf_mu, dkf_P, has_detected
  • Mode 2 (Trust):        trust_in_others[other_id], own_reputation
  • Mode 3 (Formation):    formation_slot (assigned index in the formation)

Movement is constant-speed toward `target`, respecting the free-space polygon.
"""

import collections
import numpy as np
from shapely.geometry import Point

import config


class Stage:
    HEALTHY = 0
    NOISY   = 1
    FAILING = 2
    DEAD    = 3


STAGE_NAMES = {Stage.HEALTHY: "healthy",
               Stage.NOISY:   "noisy",
               Stage.FAILING: "failing",
               Stage.DEAD:    "dead"}


class Robot:
    """Homogeneous ground robot."""

    def __init__(self, robot_id, position, free_space=None):
        # Identity
        self.id    = robot_id
        self.color = config.PALETTE[robot_id % len(config.PALETTE)]

        # Physical
        self.position   = np.array(position, dtype=float)
        self.target     = self.position.copy()
        self.speed      = config.ROBOT_SPEED
        self.sense_r    = config.SENSE_RADIUS
        self.comm_r     = config.COMM_RADIUS
        self.free_space = free_space   # shapely polygon, None = unbounded

        # Health
        self.is_alive       = True
        self.exposure       = 0.0
        self.stage          = Stage.HEALTHY
        self.base_quality   = config.SENSOR_QUALITY
        self.sensor_quality = self.base_quality

        # Manual degradation flag (Mode 2 — "D" key)
        self.manual_degradation = False

        # Sensing
        self.local_gas    = 0.0
        self.has_detected = False    # legacy: any belief (own or received)
        self.self_observed = False   # True only when OWN sensor saw the gas
                                     # (used by the consensus criterion so the
                                     # team can't lock on a single robot's
                                     # estimate propagated via gossip)

        # Belief state (Mode 1 — DKF)
        self.position_estimates: dict = {self.id: self.position.copy()}
        self.dkf_mu: np.ndarray | None = None
        self.dkf_P:  np.ndarray | None = None
        self.dkf_seed_mu: np.ndarray | None = None
        # raw seed before gossip; _last_peak_obs = this step's own peak
        # observation (z, dist), consumed by the DKF update step in the
        # simulation loop (course notes, Sec. 18.2).
        self._last_peak_obs = None
        self.gossip_updated = False

        # Trust (Mode 2)
        # trust_in_others[other_id] ∈ [TRUST_MIN, TRUST_MAX]
        self.trust_in_others: dict = {}
        self.own_reputation = 1.0   # mean trust other robots place in us

        # Formation (Mode 3)
        self.formation_slot = None   # int, set by FormationControl

        # Movement support
        self._smooth_centroid = None
        self.trail = collections.deque(maxlen=config.TRAIL_LEN)

    # ── Sensing ─

    def sense(self, gas):
        """
        Sample local gas concentration with distance-dependent noise.

        Sensor model (after Facinelli et al. 2019, eq. 4):
            h_i(xi) = sat( C_i(xi) + noise )
            noise_std ∝ distance_to_peak²

        The noise increases with distance from the source, mimicking the
        behaviour of a real PID (photoionisation detector) sensor whose
        accuracy degrades with distance.
        """
        # True concentration at robot position (mean over sensing radius).
        # In the Pasquill–Gifford model this directly evaluates Eq. (2)
        # at the receptor; in the diffusion model it samples the grid.
        true_val = gas.sample_at_px(self.position[0], self.position[1],
                                    self.sense_r)

        # ── MiniPID 2 sensor noise (Facinelli et al. 2019, eq. 3–4) ─────────
        # σ_ν(d) = σ_0 · (1 + (d / ρ)²), where:
        #   σ_0 = noise floor at zero distance, calibrated from the
        #         MiniPID 2 datasheet (noise floor ≈ 10 ppb, max range
        #         ≈ 100 ppb → σ_0 ≈ 0.01 in normalised units)
        #   ρ   = effective sensor range (SENSE_RADIUS, 70 px ≈ 7 m)
        #   d   = distance to nearest source
        # Stage multiplier amplifies σ_ν when the sensor is degraded.
        dist_to_source = config.MAP_W
        if gas.sources:
            dist_to_source = min(
                float(np.linalg.norm(self.position - np.array(s)))
                for s in gas.sources)
        stage_mult = {Stage.HEALTHY: 1.0, Stage.NOISY: 4.0,
                      Stage.FAILING: 10.0, Stage.DEAD: 0.0}[self.stage]
        dist_norm  = min(dist_to_source / max(self.sense_r, 1), 3.0)
        sigma_0    = getattr(config, "MINIPID2_SIGMA_0", 0.01)
        noise_std  = sigma_0 * stage_mult * (1.0 + dist_norm ** 2)
        self.local_gas = float(
            np.clip(true_val + np.random.normal(0, noise_std), 0, 1))

        # Use the NOISY reading for detection — more realistic.
        # A robot smells gas if its sensor (including noise) exceeds the threshold.
        detected_gas = self.local_gas if self.stage != Stage.DEAD else 0.0

        # self_observed = robot's sensor read something above threshold.
        # This triggers CONSENSUS when enough robots have reached the cloud.
        if detected_gas > config.DETECT_THRESH:
            self.self_observed = True

        # DKF seed: only when the robot can actually see a clear peak
        # (i.e., the gas concentration peak is visible in the sensing radius).
        # This prevents robots far from the source from seeding with their
        # own position (which would give a wildly wrong estimate).
        if not self.has_detected:
            local_peak = gas.peak_in_radius_px(
                self.position[0], self.position[1], self.sense_r)
            if local_peak is not None:
                seed = np.array(local_peak, dtype=float)
                # Small observation noise (distance-scaled)
                dist = float(np.linalg.norm(seed - self.position))
                noise = max(2.0, dist * 0.02)
                if self.manual_degradation:
                    noise = max(noise, 80.0)   # degraded sensor: huge bias
                seed = seed + np.random.normal(0, noise, size=2)
                self.dkf_mu      = seed.copy()
                self.dkf_P       = np.eye(2) * config.DKF_INIT_COV
                self.has_detected = True
                self.dkf_seed_mu  = seed.copy()
                # Expose the raw observation for the DKF update step
                # (Ch. 18.2 of the course notes): z_i = peak location,
                # observed directly (H = I).
                self._last_peak_obs = (seed.copy(), dist)
        elif self.has_detected and not self.manual_degradation:
            # Refresh the seed estimate as the robot moves closer to the
            # source. A closer view gives a less noisy peak measurement
            # (distance-scaled noise), so the seed gradually improves.
            # This is what makes σ_seed (the displayed disagreement)
            # DECREASE over time as the team converges on the gas.
            # Degraded robots are skipped — their seeds should stay biased
            # to make the fault visible.
            local_peak = gas.peak_in_radius_px(
                self.position[0], self.position[1], self.sense_r)
            if local_peak is not None:
                new_seed = np.array(local_peak, dtype=float)
                dist = float(np.linalg.norm(new_seed - self.position))
                noise = max(2.0, dist * 0.02)
                new_seed = new_seed + np.random.normal(0, noise, size=2)
                # Smooth update — α=0.15 — so the curve doesn't jump
                self.dkf_seed_mu = 0.85 * self.dkf_seed_mu + 0.15 * new_seed
                # Raw (unsmoothed) observation for the DKF update step
                self._last_peak_obs = (new_seed.copy(), dist)

        # Continuous belief perturbation for a degraded robot.
        # Even after the seed, a broken sensor keeps drifting the estimate.
        # This makes the degradation effect VISIBLE: the bad robot's × marker
        # wanders away from the network consensus, the σ-curve stays high,
        # and trust in this robot keeps falling.
        if (self.manual_degradation and self.dkf_mu is not None
                and self.stage != Stage.DEAD):
            self.dkf_mu = self.dkf_mu + np.random.uniform(-3.5, 3.5, size=2)
            # Inflate covariance so its belief is "less confident" → DKF
            # information weighting also down-weights it naturally.
            self.dkf_P = self.dkf_P * 1.02 if self.dkf_P is not None else None

        # Exposure from gas
        if true_val > config.SAFE_THRESHOLD:
            self.exposure += config.DEGRADE_RATE * true_val

        # Manual degradation does NOT add exposure (so the robot stays alive
        # and demonstrates the degraded-sensor effect). It just forces the
        # stage to NOISY in _update_stage().
        self._update_stage()

        # Sensor quality depends on stage
        qf = {Stage.HEALTHY: 1.0,  Stage.NOISY: 0.55,
              Stage.FAILING: 0.18, Stage.DEAD: 0.0}[self.stage]
        self.sensor_quality = self.base_quality * qf

    def _update_stage(self):
        prev_stage = self.stage
        if self.exposure >= config.DEGRADE_FATAL:
            self.stage    = Stage.DEAD
            self.is_alive = False
        elif self.exposure >= config.DEGRADE_STAGE2:
            self.stage = Stage.FAILING
        elif self.exposure >= config.DEGRADE_STAGE1:
            self.stage = Stage.NOISY
        else:
            self.stage = Stage.HEALTHY

        # Manual degradation overrides natural stage progression:
        # keep the robot ALIVE but force NOISY behaviour.
        if self.manual_degradation and self.is_alive:
            self.stage = Stage.NOISY

        self._stage_changed = prev_stage != self.stage

    # Communication 

    def neighbours(self, all_robots):
        return [r for r in all_robots
                if r.is_alive and r.id != self.id
                and np.linalg.norm(self.position - r.position) < self.comm_r]

    # ── Movement 

    def update_smooth_centroid(self, c):
        if self._smooth_centroid is None:
            self._smooth_centroid = c.copy()
        else:
            a = config.LLOYD_ALPHA
            self._smooth_centroid = a * c + (1 - a) * self._smooth_centroid

    def move(self):
        """Constant-speed step toward target, constrained to free space."""
        self.trail.append(self.position.copy())
        delta = self.target - self.position
        dist  = np.linalg.norm(delta)
        if dist < 0.5:
            return
        step    = min(self.speed, dist)
        new_pos = self.position + step * (delta / dist)
        new_pos[0] = np.clip(new_pos[0], 5, config.MAP_W - 5)
        new_pos[1] = np.clip(new_pos[1], 5, config.MAP_H - 5)

        if self.free_space is None or self.free_space.contains(Point(new_pos)):
            self.position = new_pos
        else:
            # Slide along x or y axis only
            for cand in [np.array([new_pos[0], self.position[1]]),
                         np.array([self.position[0], new_pos[1]])]:
                if self.free_space.contains(Point(cand)):
                    self.position = cand
                    break

    def kill(self):
        self.is_alive = False
        self.stage    = Stage.DEAD

    def dist_to(self, px):
        return float(np.linalg.norm(self.position - np.array(px)))
