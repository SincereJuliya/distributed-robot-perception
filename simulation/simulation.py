"""
simulation/simulation.py
------------------------
Top-level orchestrator. Wires together all algorithm modules + scenario.

One `step()` advances the world by one tick:
  1. gas dynamics
  2. local sensing (each robot)
  3. communication graph snapshot + λ₂
  4. distributed estimation (gossip + DKF)
  5. trust update (Mode 2 only)
  6. scenario state-machine sets per-robot targets
  7. robots move
  8. metrics logged to CSV
"""

import config
from environment import GasField, build_free_space, random_spawn_points
from agents       import Robot
from core         import (GossipConsensus, DistributedKalmanFilter,
                          LloydVoronoi, CommGraph,
                          TrustReputation, FormationControl)
from scenario     import ScenarioManager
from simulation.event_log import EventLog


class Simulation:
    """The full distributed perception simulation."""

    def __init__(self):
        self.free_space, self.obstacles = build_free_space()

        # World
        # Gas field — choose model based on config
        if getattr(config, "PLUME_MODEL", "diffusion") == "pasquill_gifford":
            from environment.pasquill_gifford import PasquillGiffordPlume
            self.gas = PasquillGiffordPlume(
                stability=config.PASQUILL_STABILITY)
        else:
            self.gas = GasField()

        # Spawn robots in free space
        spawn_pts = random_spawn_points(config.NUM_ROBOTS, self.free_space,
                                        min_sep=90)
        self.robots = [Robot(i, pt, self.free_space)
                       for i, pt in enumerate(spawn_pts)]

        # Core algorithm modules (swappable)
        self.gossip         = GossipConsensus(self.robots)
        self.dkf            = DistributedKalmanFilter(self.robots)
        # Exploration strategy — Lloyd–Voronoi (default) or Random Walk
        if getattr(config, "EXPLORATION_STRATEGY", "lloyd") == "random_walk":
            from core.random_walk import RandomWalkExplorer
            self.lloyd = RandomWalkExplorer(self.robots, self.free_space)
        else:
            self.lloyd = LloydVoronoi(self.robots, self.free_space)
        self.comm_graph     = CommGraph(self.robots)
        self.trust          = TrustReputation(self.robots)
        self.formation_ctrl = FormationControl(self.robots)

        # Logging
        self.log = EventLog()
        self.log.info(0, "INIT",
                      f"{config.NUM_ROBOTS} robots, map "
                      f"{config.MAP_W}x{config.MAP_H}, "
                      f"obstacles={'on' if config.USE_OBSTACLES else 'off'}, "
                      f"sources={config.NUM_GAS_SOURCES}")

        # Scenario / mode manager
        self.scenario = ScenarioManager(
            self.robots, self.gas, self.lloyd, self.dkf, self.trust,
            self.formation_ctrl, self.free_space, self.log)

        self.step_count = 0

    # ── Main loop ─────────────────────────────────────────────────────────────

    def step(self):
        # 1. environment dynamics
        self.gas.step()

        # 2. local sensing
        for r in self.robots:
            if r.is_alive:
                r.sense(self.gas)

        # 3. communication graph + λ₂
        self.comm_graph.update()
        lam2 = self.comm_graph.lambda_2()
        self.scenario.lambda2_history.append(lam2)

        # 4. distributed estimation
        self.gossip.run_one_step()
        self.dkf.predict_all()

        # In Mode 2 we use trust-weighted fusion; in other modes plain fusion
        if self.scenario.active_key == ScenarioManager.DEGRADATION:
            self.trust.update_step()
            self.dkf.gossip_fusion_round(trust_weights=self.trust.pair_weights())
        else:
            self.dkf.gossip_fusion_round()

        # 5. coverage cells
        self.lloyd.compute_cells()

        # 6. scenario sets targets
        self.scenario.update(self.step_count)

        # 7. motion
        for r in self.robots:
            if r.is_alive:
                r.move()

        # 8. CSV log
        sigma = (self.scenario.consensus_history[-1]
                 if self.scenario.consensus_history else None)
        self.log.metric(
            self.step_count,
            phase        = self.scenario.phase,
            mode         = self.scenario.active.name,
            robots_alive = self.alive_count,
            sigma        = sigma,
            lambda2      = lam2,
            leak_px      = self.scenario.source_pxs[0]
                           if self.scenario.source_pxs else None,
            agreed_px    = self.scenario.agreed_px,
        )

        self.step_count += 1

    # ── Public introspection ──────────────────────────────────────────────────

    @property
    def alive_count(self):
        return sum(1 for r in self.robots if r.is_alive)

    def kill_random_robot(self):
        import random
        alive = [r for r in self.robots if r.is_alive]
        if len(alive) <= 1:
            return None
        victim = random.choice(alive)
        victim.kill()
        self.log.robot_event(self.step_count, victim.id, "killed (manual)",
                             "critical")
        return victim.id
