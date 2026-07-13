"""
core/random_walk.py
-------------------
Random Walk exploration strategy Facinelli "Cooperative UAVs Gas Monitoring using Distributed Consensus"

Each robot moves with velocity v_m along a heading theta_i(t) that is perturbed at every step by a uniform random angular increment:

    theta_i(t) = theta_i(t - dt) + nu,   nu ~ U(-theta_M, theta_M)

When a robot reaches the boundary of the area of interest, its heading is reset to the inward normal direction, perturbed by a bounded random offset:

    theta_i(t) = theta_in + nu_r * theta_r,   nu_r ~ U(-1, 1)

where theta_in is the inward normal direction at the border and theta_r is a fixed reflection constant (report Eq. (9), theta_r = 75 deg).

Collision avoidance: if ||xi_i - xi_j|| <= d_m the two robots receive opposite repulsive headings.
"""

import math
import numpy as np

import config


class RandomWalkExplorer:
    """
    Random-walk patrol controller that sets robot.target each step
    """

    def __init__(self, robots, free_space=None,
                 theta_max_deg: float = 10.0,
                 theta_reflect_deg: float = 75.0,
                 collision_dist: float = 25.0,
                 margin: int = 60):
        self.robots          = robots
        self.free_space      = free_space
        self.theta_max       = math.radians(theta_max_deg)
        self.theta_reflect   = math.radians(theta_reflect_deg)
        self.d_m             = collision_dist
        self.margin          = margin
        # Per-robot heading state (random initial direction)
        self.heading = {
            r.id: np.random.uniform(-math.pi, math.pi) for r in robots
        }
        # Tiny phantom Lloyd cells dict so external code can still query
        # `lloyd.cells.get(r.id)` and get None gracefully
        self.cells = {}

    #  Public API expected by the scenario state machine 

    def compute_cells(self):
        """No-op: random walk does not compute Voronoi cells."""
        pass

    def update_targets_uniform(self):
        """Pick a new target ahead of each robot along its heading."""
        for r in self.robots:
            if not r.is_alive:
                continue
            self._step_one(r)

    #  Internal: one robot's heading update 

    def _step_one(self, r):
        # 1. Heading perturbation
        nu = np.random.uniform(-self.theta_max, self.theta_max)
        self.heading[r.id] = self.heading[r.id] + nu

        # 2. Boundary reflection
        x, y = r.position
        m = self.margin
        if x < m:
            self.heading[r.id] = self._reflect_toward(0.0)
        elif x > config.MAP_W - m:
            self.heading[r.id] = self._reflect_toward(math.pi)
        elif y < m:
            self.heading[r.id] = self._reflect_toward(math.pi / 2)
        elif y > config.MAP_H - m:
            self.heading[r.id] = self._reflect_toward(-math.pi / 2)

        # 3. Collision avoidance with other robots
        for other in self.robots:
            if other.id == r.id or not other.is_alive:
                continue
            d = r.position - other.position
            n = np.linalg.norm(d)
            if 0 < n < self.d_m:
                # Repulsive heading away from other
                self.heading[r.id] = math.atan2(d[1], d[0])
                break

        # 4. Move target one step-length ahead in the heading direction
        theta = self.heading[r.id]
        step  = 1.5 * config.ROBOT_SPEED
        tx = r.position[0] + step * math.cos(theta)
        ty = r.position[1] + step * math.sin(theta)
        # Clip into map
        tx = float(np.clip(tx, 5, config.MAP_W - 5))
        ty = float(np.clip(ty, 5, config.MAP_H - 5))
        r.target = np.array([tx, ty], dtype=float)

    def _reflect_toward(self, inward_theta: float) -> float:
        """Reflect heading toward `inward_theta` +- theta_r jitter."""
        nu = np.random.uniform(-1.0, 1.0)
        return inward_theta + nu * self.theta_reflect