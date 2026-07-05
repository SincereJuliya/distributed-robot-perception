"""
core/trust.py
-------------
Trust-based reputation system for Mode 2.

Each robot maintains a trust score for every other robot:
    trust_in_others[other_id] ∈ [TRUST_MIN, TRUST_MAX]

Update rule (each step where both robots have a DKF belief):
    disagreement_ij = ‖μ_i − μ_j‖
    target_trust    = exp(−disagreement_ij / TRUST_DISAGREE_PX)
    new_trust       = (1 − α) · old_trust + α · target_trust

Effect:
  • If R_i's belief consistently disagrees with R_j's, R_j lowers trust in R_i.
  • A degraded robot ends up trusted less by the entire network.
  • Gossip / DKF fusion uses these trusts as additional weights.

References:
  • Pasqualetti, Bicchi, Bullo (2012) — Consensus on Misbehaving Robots
  • Bullo, Lectures on Network Systems — chapter on robust consensus
"""

import math
import numpy as np

import config


class TrustReputation:
    """
    Pairwise trust scores between robots.

    Trust is held per-robot in robot.trust_in_others.
    This class is the *update rule* that maintains those scores
    based on observed disagreement.
    """

    def __init__(self, robots):
        self.robots = robots
        self._init_trust_tables()

    def _init_trust_tables(self):
        ids = [r.id for r in self.robots]
        for r in self.robots:
            for other_id in ids:
                if other_id != r.id:
                    r.trust_in_others.setdefault(other_id, 1.0)

    def update_step(self):
        """
        Update trust scores from OBSERVABLE signals only. No robot
        reads `manual_degradation` directly here — the trust mechanism
        must discover faulty agents from their behaviour, just like a
        real distributed fault-detection-and-isolation scheme would
        (cf.\\ Pasqualetti, Bicchi, Bullo 2012). Two signals are used:

          1) DKF belief disagreement (when both robots have beliefs)
          2) Position-estimate disagreement (always available via gossip)

        A degraded robot's belief naturally drifts due to its noisy
        sensor and biased seed (see robot.py); neighbours observe this
        drift and lower their trust accordingly. After ~80 steps the
        faulty robot is collectively identified by the network without
        any central supervisor and without reading any internal flag.
        """
        alive = [r for r in self.robots if r.is_alive]
        if len(alive) < 2:
            return

        alpha = config.TRUST_LEARNING_RATE
        for ri in alive:
            for rj in ri.neighbours(alive):
                target = 1.0

                # Signal 1: DKF belief disagreement.
                # Bugfix: compare the SEED beliefs (raw independent
                # observations) rather than the post-fusion dkf_mu —
                # gossip merges everyone's dkf_mu toward a common value
                # each step, so post-fusion disagreement is ~0 by
                # construction and the signal could never fire.
                if ri.dkf_seed_mu is not None and rj.dkf_seed_mu is not None:
                    disagree = float(np.linalg.norm(
                        ri.dkf_seed_mu - rj.dkf_seed_mu))
                    target = min(target,
                                 math.exp(-disagree / config.TRUST_DISAGREE_PX))

                # Signal 2: position-estimate disagreement
                if rj.id in ri.position_estimates:
                    est_disagree = float(np.linalg.norm(
                        ri.position_estimates[rj.id] - rj.position))
                    target = min(target,
                                 math.exp(-est_disagree /
                                          config.TRUST_DISAGREE_PX))

                old = ri.trust_in_others.get(rj.id, 1.0)
                new = (1 - alpha) * old + alpha * target
                new = max(config.TRUST_MIN, min(config.TRUST_MAX, new))
                ri.trust_in_others[rj.id] = new

        # Own reputation = mean trust others place in us
        for r in alive:
            others = [o for o in alive if o.id != r.id]
            if others:
                r.own_reputation = float(np.mean(
                    [o.trust_in_others.get(r.id, 1.0) for o in others]))
            else:
                r.own_reputation = 1.0

    # ── Public accessors ──────────────────────────────────────────────────────

    def pair_weights(self) -> dict:
        """
        Return {(observer_id, target_id) → trust} for use by DKF fusion.

        observer_id = robot whose trust we are using
        target_id   = robot being weighted
        """
        out = {}
        for r in self.robots:
            if not r.is_alive: continue
            for other_id, t in r.trust_in_others.items():
                out[(r.id, other_id)] = t
        return out

    def lowest_reputation_robot(self):
        """Robot the network trusts least — useful for visualisation."""
        alive = [r for r in self.robots if r.is_alive]
        if not alive: return None
        return min(alive, key=lambda r: r.own_reputation)
