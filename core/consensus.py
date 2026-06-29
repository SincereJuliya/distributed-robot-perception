"""
core/consensus.py
-----------------
Distributed Weighted Gossip Consensus on robot position estimates.

Standard pairwise gossip (Boyd et al. 2006, Olfati-Saber 2007):
each round, a random alive robot picks a random alive neighbour and the pair
averages their position-estimate dictionaries with weights = sensor_quality.

Asymptotic convergence speed is governed by the algebraic connectivity λ₂
of the comm-graph Laplacian (see core/graph_metrics.py).
"""

import numpy as np

import config


class GossipConsensus:
    """Pairwise weighted gossip on robot position estimates."""

    def __init__(self, robots):
        self.robots = robots

    def run_one_step(self):
        alive = [r for r in self.robots if r.is_alive]
        if len(alive) < 2:
            return

        # Refresh each robot's self-estimate with its true position
        for r in alive:
            r.position_estimates[r.id] = r.position.copy()
            r.gossip_updated = False

        for _ in range(config.GOSSIP_ROUNDS):
            initiator  = alive[np.random.randint(len(alive))]
            neighbours = initiator.neighbours(alive)
            if not neighbours:
                continue
            partner = neighbours[np.random.randint(len(neighbours))]
            self._pairwise_merge(initiator, partner)

    @staticmethod
    def _pairwise_merge(ri, rj):
        wi = max(ri.sensor_quality, 1e-3)
        wj = max(rj.sensor_quality, 1e-3)

        for rid in set(ri.position_estimates) | set(rj.position_estimates):
            hi = rid in ri.position_estimates
            hj = rid in rj.position_estimates
            if hi and hj:
                m = (wi * ri.position_estimates[rid] +
                     wj * rj.position_estimates[rid]) / (wi + wj)
                ri.position_estimates[rid] = m.copy()
                rj.position_estimates[rid] = m.copy()
            elif hi:
                rj.position_estimates[rid] = ri.position_estimates[rid].copy()
            else:
                ri.position_estimates[rid] = rj.position_estimates[rid].copy()
