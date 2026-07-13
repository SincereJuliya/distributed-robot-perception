"""
scenario/manager.py
-------------------
Top-level scenario manager
"""

from scenario.modes import LocalisationMode, DegradationMode, FormationMode


class ScenarioManager:
    """Owns the three modes and delegates per-step calls to the active one."""

    LOCALISATION = "1"
    DEGRADATION  = "2"
    FORMATION    = "3"

    def __init__(self, robots, gas, lloyd, dkf, trust,
                 formation_ctrl, free_space, event_log):
        self.event_log = event_log

        self.modes = {
            self.LOCALISATION: LocalisationMode(
                robots, gas, lloyd, dkf, free_space, event_log),
            self.DEGRADATION:  DegradationMode(
                robots, gas, lloyd, dkf, trust, free_space, event_log),
            self.FORMATION:    FormationMode(
                robots, formation_ctrl, free_space, event_log),
        }
        self.active_key = self.LOCALISATION
        self.active     = self.modes[self.active_key]
        self.active.enter(0)

    # Mode switching

    def switch_to(self, key: str, step: int):
        if key not in self.modes or key == self.active_key:
            return False
        self.event_log.mode_change(step, self.active.name,
                                   self.modes[key].name)
        self.active.exit(step)
        self.active_key = key
        self.active     = self.modes[key]
        self.active.enter(step)
        return True

    # Delegation

    def update(self, step):
        self.active.update(step)

    @property
    def phase(self) -> str:
        return getattr(self.active, "phase", "—")

    @property
    def agreed_px(self):
        return getattr(self.active, "agreed_px", None)

    @property
    def source_pxs(self):
        return getattr(self.active, "source_pxs", [])

    @property
    def consensus_history(self):
        return self.active.consensus_history

    @property
    def lambda2_history(self):
        return self.active.lambda2_history

    @property
    def leak_count(self):
        return self.active.leak_count

    def progress_frac(self) -> float:
        return self.active.progress_frac()
