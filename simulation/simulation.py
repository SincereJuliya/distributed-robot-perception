"""
simulation/simulation.py
------------------------
Top-level orchestrator

One step() by one tick:
  1. gas dynamics
  2. local sensing (each robot)
  3. communication graph snapshot + lambda_2
  4. distributed estimation (gossip + DKF)
  5. trust update (Mode 2 only)
  6. scenario state-machine sets per-robot targets
  7. robots move
  8. metrics logged to CSV
"""

import numpy as np
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
        self.free_space = build_free_space()

        # World
        # Gas field - choose model based on config
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
        # Exploration strategy - Lloyd-Voronoi (default) or Random Walk
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
                      f"sources={config.NUM_GAS_SOURCES}")

        # Scenario / mode manager
        self.scenario = ScenarioManager(
            self.robots, self.gas, self.lloyd, self.dkf, self.trust,
            self.formation_ctrl, self.free_space, self.log)

        self.step_count = 0

    #  Main loop 

    def step(self):
        # 1. environment dynamics
        self.gas.step()

        # 2. local sensing
        for r in self.robots:
            if r.is_alive:
                r.sense(self.gas)

        # 3. communication graph + lambda_2
        self.comm_graph.update()
        lam2 = self.comm_graph.lambda_2()
        self.scenario.lambda2_history.append(lam2)

        # 4. distributed estimation
        self.gossip.run_one_step()
        self.dkf.predict_all()

        # DKF update step: each healthy robot that observed a gas peak this step incorporates it as a local measurement in information form,
        #   Omega <- Omega + R^-1,   xi <- xi + R^-1 z, with H = I 
        # and measurement noise scaled with the distance to the observed peak, 
        #   R = max(R_min, R0 (d/rho)^2) I
        
        for _r in self.robots:
            obs = getattr(_r, "_last_peak_obs", None)
            _r._last_peak_obs = None
            if (obs is None or not _r.is_alive or _r.manual_degradation
                    or _r.dkf_mu is None):
                continue
            _z, _dist = obs
            _rho = max(_r.sense_r, 1)
            _Rvar = max(100.0, config.DKF_OBS_NOISE * (_dist / _rho) ** 2)
            self.dkf.local_update(_r, _z, R=np.eye(2) * _Rvar)

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

        # 7. motion - every mode goes through the robot's own dynamics
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

    #  Public introspection 

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
    
    # """
# simulation/simulation.py
# ------------------------
# Top-level orchestrator

# One step() by one tick:
#   1. gas dynamics
#   2. local sensing (each robot)
#   3. communication graph snapshot + lambda_2
#   4. distributed estimation (gossip + DKF)
#   5. trust update (Mode 2 only)
#   6. scenario state-machine sets per-robot targets
#   7. robots move
#   8. metrics logged to CSV
# """

# import numpy as np
# import config
# from environment import GasField, build_free_space, random_spawn_points
# from agents       import Robot
# from core         import (GossipConsensus, DistributedKalmanFilter,
#                           LloydVoronoi, CommGraph,
#                           TrustReputation, FormationControl)
# from scenario     import ScenarioManager
# from simulation.event_log import EventLog


# class Simulation:
#     """The full distributed perception simulation."""

#     def __init__(self):
#         self.free_space = build_free_space()

#         # World
#         # Gas field - choose model based on config
#         if getattr(config, "PLUME_MODEL", "diffusion") == "pasquill_gifford":
#             from environment.pasquill_gifford import PasquillGiffordPlume
#             self.gas = PasquillGiffordPlume(
#                 stability=config.PASQUILL_STABILITY)
#         else:
#             self.gas = GasField()

#         # Spawn robots in free space
#         spawn_pts = random_spawn_points(config.NUM_ROBOTS, self.free_space,
#                                         min_sep=90)
#         self.robots = [Robot(i, pt, self.free_space)
#                        for i, pt in enumerate(spawn_pts)]

#         # Core algorithm modules (swappable)
#         self.gossip         = GossipConsensus(self.robots)
#         self.dkf            = DistributedKalmanFilter(self.robots)
#         # Exploration strategy - Lloyd-Voronoi (default) or Random Walk
#         if getattr(config, "EXPLORATION_STRATEGY", "lloyd") == "random_walk":
#             from core.random_walk import RandomWalkExplorer
#             self.lloyd = RandomWalkExplorer(self.robots, self.free_space)
#         else:
#             self.lloyd = LloydVoronoi(self.robots, self.free_space)
#         self.comm_graph     = CommGraph(self.robots)
#         self.trust          = TrustReputation(self.robots)
#         self.formation_ctrl = FormationControl(self.robots)

#         # Logging
#         self.log = EventLog()
#         self.log.info(0, "INIT",
#                       f"{config.NUM_ROBOTS} robots, map "
#                       f"{config.MAP_W}x{config.MAP_H}, "
#                       f"sources={config.NUM_GAS_SOURCES}")

#         # Scenario / mode manager
#         self.scenario = ScenarioManager(
#             self.robots, self.gas, self.lloyd, self.dkf, self.trust,
#             self.formation_ctrl, self.free_space, self.log)

#         self.step_count = 0

#     #  Main loop 

#     def step(self):
#         # 1. environment dynamics
#         self.gas.step()

#         # 2. local sensing
#         for r in self.robots:
#             if r.is_alive:
#                 r.sense(self.gas)

#         # 3. communication graph + lambda_2
#         self.comm_graph.update()
#         lam2 = self.comm_graph.lambda_2()
#         self.scenario.lambda2_history.append(lam2)

#         # 4. distributed estimation
#         self.gossip.run_one_step()
#         self.dkf.predict_all()

#         # DKF update step: each healthy robot that observed a gas peak this step incorporates it as a local measurement in information form,
#         #   Omega <- Omega + R^-1,   xi <- xi + R^-1 z, with H = I 
#         # and measurement noise scaled with the distance to the observed peak, 
#         #   R = max(R_min, R0 (d/rho)^2) I
        
#         for _r in self.robots:
#             obs = getattr(_r, "_last_peak_obs", None)
#             _r._last_peak_obs = None
#             if (obs is None or not _r.is_alive or _r.manual_degradation
#                     or _r.dkf_mu is None):
#                 continue
#             _z, _dist = obs
#             _rho = max(_r.sense_r, 1)
#             _Rvar = max(100.0, config.DKF_OBS_NOISE * (_dist / _rho) ** 2)
#             self.dkf.local_update(_r, _z, R=np.eye(2) * _Rvar)

#         # In Mode 2 we use trust-weighted fusion; in other modes plain fusion
#         if self.scenario.active_key == ScenarioManager.DEGRADATION:
#             self.trust.update_step()
#             self.dkf.gossip_fusion_round(trust_weights=self.trust.pair_weights())
#         else:
#             self.dkf.gossip_fusion_round()

#         # 5. coverage cells
#         self.lloyd.compute_cells()

#         # 6. scenario sets targets
#         self.scenario.update(self.step_count)

#         # 7. motion
#         if self.scenario.active_key != ScenarioManager.FORMATION:
#             for r in self.robots:
#                 if r.is_alive:
#                     r.move()

#         # 8. CSV log
#         sigma = (self.scenario.consensus_history[-1]
#                  if self.scenario.consensus_history else None)
#         self.log.metric(
#             self.step_count,
#             phase        = self.scenario.phase,
#             mode         = self.scenario.active.name,
#             robots_alive = self.alive_count,
#             sigma        = sigma,
#             lambda2      = lam2,
#             leak_px      = self.scenario.source_pxs[0]
#                            if self.scenario.source_pxs else None,
#             agreed_px    = self.scenario.agreed_px,
#         )

#         self.step_count += 1

#     #  Public introspection 

#     @property
#     def alive_count(self):
#         return sum(1 for r in self.robots if r.is_alive)

#     def kill_random_robot(self):
#         import random
#         alive = [r for r in self.robots if r.is_alive]
#         if len(alive) <= 1:
#             return None
#         victim = random.choice(alive)
#         victim.kill()
#         self.log.robot_event(self.step_count, victim.id, "killed (manual)",
#                              "critical")
#         return victim.id