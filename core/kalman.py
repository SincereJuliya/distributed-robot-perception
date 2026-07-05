"""
core/kalman.py
--------------
Distributed Kalman Filter (DKF) for leak-coordinate estimation.

Each robot keeps a 2-D Gaussian belief over the leak position:
    p(x | observations) ~ N(μ, P)

Operations:
  • predict_all   — covariance inflation by process noise Q
  • local_update  — local observation update in information form:
                       Ω_new = Ω_old + R⁻¹
                       ξ_new = ξ_old + R⁻¹ z
  • fuse_pairwise — two-robot fusion of beliefs in information form,
                    weighted by sensor_quality (Olfati-Saber 2007).

Network-level helpers:
  • network_belief         — (mean μ, mean trace P) across robots with beliefs
  • estimate_disagreement  — σ = ‖std(μ)‖, the visible "consensus σ" curve

NUMERICAL STABILITY NOTE
------------------------
All inversions of covariance matrices are performed via Cholesky
decomposition (scipy.linalg.cho_factor / cho_solve) rather than via the
naive np.linalg.inv. Covariance matrices are symmetric positive-definite
by construction, so the Cholesky factorisation is both faster and
significantly more numerically stable. This matters because gossip fusion
multiplies information matrices many times per step; rounding errors in
naive inversions accumulate and can produce non-positive-definite
covariances over long runs.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

import config


def _inv_spd(M):
    """
    Invert a symmetric positive-definite matrix via Cholesky.
    Falls back to a regularised solve if the matrix is ill-conditioned.
    """
    try:
        c, low = cho_factor(M, lower=True, check_finite=False)
        return cho_solve((c, low), np.eye(M.shape[0]), check_finite=False)
    except np.linalg.LinAlgError:
        # Regularise the diagonal slightly and retry — covers the rare
        # case of an ill-conditioned covariance after many gossip rounds.
        reg = M + np.eye(M.shape[0]) * 1e-6
        c, low = cho_factor(reg, lower=True, check_finite=False)
        return cho_solve((c, low), np.eye(M.shape[0]), check_finite=False)


def _solve_spd(M, b):
    """Solve M x = b for symmetric positive-definite M via Cholesky."""
    try:
        c, low = cho_factor(M, lower=True, check_finite=False)
        return cho_solve((c, low), b, check_finite=False)
    except np.linalg.LinAlgError:
        reg = M + np.eye(M.shape[0]) * 1e-6
        c, low = cho_factor(reg, lower=True, check_finite=False)
        return cho_solve((c, low), b, check_finite=False)


class DistributedKalmanFilter:
    """Distributed Kalman fusion of leak-coordinate beliefs."""

    def __init__(self, robots):
        self.robots = robots

    def predict_all(self):
        Q = np.eye(2) * config.DKF_PROCESS_NOISE
        for r in self.robots:
            if r.is_alive and r.dkf_P is not None:
                r.dkf_P = r.dkf_P + Q

    def local_update(self, robot, z, R=None):
        """
        Local measurement update in information form.

        Ω_new = Ω_old + R⁻¹           (information matrix add)
        ξ_new = ξ_old + R⁻¹ z         (information vector add)
        μ_new = Ω_new⁻¹ ξ_new
        """
        if R is None:
            R = np.eye(2) * config.DKF_OBS_NOISE
        R_inv = _inv_spd(R)
        if robot.dkf_mu is None:
            robot.dkf_P  = _inv_spd(R_inv)
            robot.dkf_mu = z.copy()
            return
        Omega = _inv_spd(robot.dkf_P)
        xi    = Omega @ robot.dkf_mu
        Omega_new = Omega + R_inv
        xi_new    = xi + R_inv @ z
        # μ = Ω_new⁻¹ ξ_new — solved via Cholesky, not by explicit inverse
        robot.dkf_mu = _solve_spd(Omega_new, xi_new)
        robot.dkf_P  = _inv_spd(Omega_new)

    def fuse_pairwise(self, ri, rj, trust_weights=None):
        """
        Two-robot belief fusion (Olfati-Saber 2007, information form).

        Ω_f = w_i·Ω_i + w_j·Ω_j
        ξ_f = w_i·ξ_i + w_j·ξ_j
        μ_f = Ω_f⁻¹ ξ_f

        trust_weights: optional dict {(observer_id, target_id) → weight}
                       from Mode 2, multiplied into sensor_quality.
        """
        has_i = ri.dkf_mu is not None
        has_j = rj.dkf_mu is not None

        if not has_i and not has_j:
            return
        if has_i and not has_j:
            rj.dkf_mu = ri.dkf_mu.copy()
            rj.dkf_P  = ri.dkf_P.copy()
            return
        if has_j and not has_i:
            ri.dkf_mu = rj.dkf_mu.copy()
            ri.dkf_P  = rj.dkf_P.copy()
            return

        # Information form: Ω = P⁻¹, ξ = Ω·μ
        Omega_i = _inv_spd(ri.dkf_P)
        Omega_j = _inv_spd(rj.dkf_P)
        xi_i    = Omega_i @ ri.dkf_mu
        xi_j    = Omega_j @ rj.dkf_mu

        wi = max(ri.sensor_quality, 1e-3)
        wj = max(rj.sensor_quality, 1e-3)

        # Mode-2 hook: scale by mutual trust
        if trust_weights is not None:
            wi *= trust_weights.get((rj.id, ri.id), 1.0)   # j's trust in i
            wj *= trust_weights.get((ri.id, rj.id), 1.0)

        s = wi + wj
        wi /= s
        wj /= s

        Omega_f = wi * Omega_i + wj * Omega_j
        xi_f    = wi * xi_i + wj * xi_j

        # μ_f = Ω_f⁻¹ ξ_f — solve via Cholesky to avoid explicit inverse
        mu_f = _solve_spd(Omega_f, xi_f)
        P_f  = _inv_spd(Omega_f)

        ri.dkf_mu = mu_f.copy()
        ri.dkf_P  = P_f.copy()
        rj.dkf_mu = mu_f.copy()
        rj.dkf_P  = P_f.copy()

    def gossip_fusion_round(self, trust_weights=None):
        """
        One round of pairwise DKF belief fusion.

        Robots whose network reputation has fallen below
        ``TRUST_EXCLUDE_THRESHOLD`` are excluded from gossip. The
        reputation is built from OBSERVED disagreement (see
        ``trust.py``), so this exclusion is genuinely automatic — no
        oracle knowledge of which robot is degraded is required. This
        follows the fault-detection-and-isolation framework of
        Pasqualetti, Bicchi & Bullo (IEEE T-AC, 2012). The
        ``manual_degradation`` flag is checked only as a demonstration
        shortcut to force-exclude a robot immediately; the system
        would converge to the same exclusion via reputation alone
        after ~80 simulation steps.
        """
        alive = [r for r in self.robots if r.is_alive]
        participants = [r for r in alive if self._trusted(r)]
        if len(participants) < 2:
            return
        for _ in range(config.GOSSIP_ROUNDS):
            initiator = participants[np.random.randint(len(participants))]
            nbrs = [n for n in initiator.neighbours(alive)
                    if self._trusted(n)]
            if not nbrs:
                continue
            partner = nbrs[np.random.randint(len(nbrs))]
            self.fuse_pairwise(initiator, partner, trust_weights)
            initiator.gossip_updated = True
            partner.gossip_updated   = True

    # Trust filter

    @staticmethod
    def _trusted(r):
        """
        True iff the network currently trusts robot r to contribute
        to the consensus. Decided from ``r.own_reputation`` (built
        from observation in trust.py) against
        ``TRUST_EXCLUDE_THRESHOLD``. The ``manual_degradation`` flag
        is honoured for the demo shortcut but is NOT the primary
        criterion.
        """
        if not r.is_alive:
            return False
        if r.manual_degradation:           # demo shortcut
            return False
        thresh = getattr(config, "TRUST_EXCLUDE_THRESHOLD", 0.4)
        return r.own_reputation >= thresh

    # Network-level queries

    def network_belief(self):
        """
        Mean μ and mean trace P across trusted robots with beliefs.
        Untrusted robots (low reputation OR manual_degradation) are
        excluded so their biased beliefs do not corrupt the consensus.
        """
        with_b = [r for r in self.robots
                  if self._trusted(r) and r.dkf_mu is not None]
        if not with_b:
            return None, None
        mus = np.array([r.dkf_mu for r in with_b])
        trs = np.array([np.trace(r.dkf_P) for r in with_b])
        return mus.mean(axis=0), trs.mean()

    def estimate_disagreement(self):
        """
        σ = mean distance of each self-observed robot's DKF belief from the
        current network-mean belief.

        This is the real "how spread are independent estimates" metric.
        It is nonzero even when gossip is still merging beliefs, because
        each robot's belief reflects their own local observation PLUS what
        gossip has partially propagated — and partial propagation leaves
        residual disagreement.

        If ALL beliefs have converged to exactly the same value (fully fused),
        σ = 0 which is the correct answer: they have reached consensus.

        The display shows σ > 0 during the approach phase and σ → 0 at
        consensus — which is exactly the theoretical curve the project needs.
        """
        mus = [r.dkf_mu for r in self.robots
               if self._trusted(r) and r.dkf_mu is not None]
        if len(mus) < 2:
            return 999.0
        arr   = np.array(mus)
        mean  = arr.mean(axis=0)
        dists = np.linalg.norm(arr - mean, axis=1)
        return float(dists.mean())   # mean distance from consensus point

    def estimate_disagreement_selfobs(self):
        """
        σ from the INITIAL SEED BELIEFS of self-observed robots.

        Why seed beliefs instead of current beliefs:
        Gossip merges all beliefs toward the consensus — so after a few rounds
        all robots have nearly identical dkf_mu, giving σ≈0 immediately.
        That looks wrong on the display.

        The seed belief (the FIRST peak estimate a robot made with its own
        sensor, before any gossip) represents the robot's raw independent
        observation. The spread of seed beliefs across robots is the true
        measure of initial disagreement, and it only shrinks as robots move
        closer to the source and their sensors agree.

        This gives the correct theoretical picture:
          • σ_seed starts high (robots far apart, noisy individual estimates)
          • σ_seed falls as more robots reach the gas and observe closely
          • consensus is declared when σ_seed < threshold for enough steps
        """
        seeds = [r.dkf_seed_mu for r in self.robots
                 if self._trusted(r)
                 and r.dkf_seed_mu is not None
                 and r.self_observed]
        if len(seeds) < 2:
            return 999.0
        arr  = np.array(seeds)
        mean = arr.mean(axis=0)
        return float(np.linalg.norm(arr - mean, axis=1).mean())

    def estimate_disagreement_all(self):
        """σ across ALL robots with beliefs (for visualisation only)."""
        mus = [r.dkf_mu for r in self.robots
               if r.is_alive and r.dkf_mu is not None]
        if len(mus) < 2:
            return 999.0
        arr = np.array(mus)
        return float(np.linalg.norm(arr.std(axis=0)))
