"""
agents/robot.py
---------------
Ground robot. State used by all three modes:

  Mode 1 (Localization): dkf_mu, dkf_P, has_detected
  Mode 2 (Trust):        trust_in_others[other_id], own_reputation
  Mode 3 (Formation):    formation_slot
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

        # Manual degradation flag (Mode 2)
        self.manual_degradation = False

        # Sensing
        self.local_gas     = 0.0
        self.has_detected  = False   # robot holds a belief seeded from a real peak; gates scout-alert and n_obs (Eq.14)
        self.self_observed = False   # own sensor reading crossed the threshold; used by the sigma criterion (Eq.15)

        # Belief state (DKF)
        self.position_estimates: dict = {self.id: self.position.copy()}
        self.dkf_mu: np.ndarray | None = None
        self.dkf_P:  np.ndarray | None = None
        self.dkf_seed_mu: np.ndarray | None = None
        self._last_peak_obs = None    # (z, dist) for this step's DKF update
        self.gossip_updated = False

        # Trust (Mode 2)
        self.trust_in_others: dict = {}   # trust_in_others[j] in [TRUST_MIN, TRUST_MAX]
        self.own_reputation = 1.0         # mean trust others place in us

        # Formation (Mode 3)
        self.formation_slot = None

        # Movement support
        self._smooth_centroid = None
        self.trail = collections.deque(maxlen=config.TRAIL_LEN)

    # Sensing
    def sense(self, gas):
        """Sample local gas concentration with distance-dependent noise."""
        true_val = gas.sample_at_px(self.position[0], self.position[1],
                                    self.sense_r)

        # Sensor noise std: sigma(d) = sigma_0 * stage_mult * (1 + (d/rho)^2)
        #   sigma_0 = noise floor
        #   d = distance to nearest source, rho = SENSE_RADIUS
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

        detected_gas = self.local_gas if self.stage != Stage.DEAD else 0.0
        if detected_gas > config.DETECT_THRESH:
            self.self_observed = True

        # DKF seed: only when a real peak is visible in the sensing radius, so distant robots dont seed with their own position
        if not self.has_detected:
            local_peak = gas.peak_in_radius_px(
                self.position[0], self.position[1], self.sense_r)
            if local_peak is not None:
                seed = np.array(local_peak, dtype=float)
                dist = float(np.linalg.norm(seed - self.position))
                noise = max(2.0, dist * 0.02)     # observation noise
                if self.manual_degradation:
                    noise = max(noise, 80.0)      # degraded sensor: large bias
                seed = seed + np.random.normal(0, noise, size=2)
                self.dkf_mu       = seed.copy()
                self.dkf_P        = np.eye(2) * config.DKF_INIT_COV
                self.has_detected = True
                self.dkf_seed_mu  = seed.copy()
                # z = peak location for the update
                self._last_peak_obs = (seed.copy(), dist)
                
        elif self.has_detected and not self.manual_degradation:
            # Refresh seed as the robot moves closer (less noisy peak), which makes sigma_seed shrink as the team converges
            local_peak = gas.peak_in_radius_px(
                self.position[0], self.position[1], self.sense_r)
            if local_peak is not None:
                new_seed = np.array(local_peak, dtype=float)
                dist = float(np.linalg.norm(new_seed - self.position))
                noise = max(2.0, dist * 0.02)
                new_seed = new_seed + np.random.normal(0, noise, size=2)
                self.dkf_seed_mu = 0.85 * self.dkf_seed_mu + 0.15 * new_seed  # EMA smooth
                self._last_peak_obs = (new_seed.copy(), dist)

        # Degraded robot: keep drifting the belief and inflating covariance, so its marker diverges, sigma stays high, and trust in it falls
        if (self.manual_degradation and self.dkf_mu is not None
                and self.stage != Stage.DEAD):
            self.dkf_mu = self.dkf_mu + np.random.uniform(-3.5, 3.5, size=2)
            self.dkf_P = self.dkf_P * 1.02 if self.dkf_P is not None else None

        # Exposure accumulates only from real gas, not manual degradation
        if true_val > config.SAFE_THRESHOLD:
            self.exposure += config.DEGRADE_RATE * true_val
        self._update_stage()

        # Gossip weight q by stage
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

        # Manual degradation forces NOISY but keeps the robot alive - цwe used it just for the shortcut demo
        if self.manual_degradation and self.is_alive:
            self.stage = Stage.NOISY

        self._stage_changed = prev_stage != self.stage

    # Communication
    def neighbours(self, all_robots):
        return [r for r in all_robots
                if r.is_alive and r.id != self.id
                and np.linalg.norm(self.position - r.position) < self.comm_r]

    # # Movement
    # def update_smooth_centroid(self, c):
    #     if self._smooth_centroid is None:
    #         self._smooth_centroid = c.copy()
    #     else:
    #         a = config.LLOYD_ALPHA
    #         self._smooth_centroid = a * c + (1 - a) * self._smooth_centroid

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
            # Blocked: slide along one axis
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