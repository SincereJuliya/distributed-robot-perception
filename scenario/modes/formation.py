"""
scenario/modes/formation.py
---------------------------
Mode 3

Robots form a geometric formation (circle / line / V) and the formation center moves along a leader path.
If a robot is killed, the remaining robots redistribute themselves so the formation persists.
"""

import config


class FormationMode:
    """Mode 3 - coordinated formation patrol."""

    name = "Formation Control"

    def __init__(self, robots, formation_ctrl, free_space, event_log):
        self.robots         = robots
        self.formation_ctrl = formation_ctrl
        self.free_space     = free_space
        self.log            = event_log

        # No gas / no consensus in this mode - empty histories for the panel
        self.consensus_history = []
        self.lambda2_history   = []
        self.leak_count        = 0
        self.agreed_px         = None

    @property
    def phase(self) -> str:
        return f"FORMATION: {self.formation_ctrl.shape.upper()}"

    #  Activation / deactivation 

    def enter(self, step):
        self.log.info(step, "MODE", f"entered {self.name}")
        self.formation_ctrl.assign_slots()
        # Reset per-robot smoothing state so the formation forms cleanly
        for r in self.robots:
            r._smooth_centroid = None

    def exit(self, step):
        self.log.info(step, "MODE", f"exiting {self.name}")

    #  Main step 

    def update(self, step):
        self.formation_ctrl.step_center()
        self.formation_ctrl.update_targets()

    #  External hooks 

    def cycle_shape(self, step):
        new_shape = self.formation_ctrl.cycle_shape()
        self.log.info(step, "FORMATION", f"shape -> {new_shape}", "event")
        return new_shape

    def set_shape(self, step, shape):
        """Directly select a shape (C/L/V/T hotkeys)."""
        new = self.formation_ctrl.set_shape(shape)
        if new is not None:
            self.log.info(step, "FORMATION", f"shape -> {new}", "event")
        return new

    def kill_random(self, step):
        import random
        alive = [r for r in self.robots if r.is_alive]
        if len(alive) <= 1: return None
        victim = random.choice(alive)
        victim.kill()
        self.formation_ctrl.assign_slots()
        self.log.robot_event(step, victim.id, "killed (formation rebalanced)",
                             "critical")
        return victim.id

    def progress_frac(self) -> float:
        return 0.0