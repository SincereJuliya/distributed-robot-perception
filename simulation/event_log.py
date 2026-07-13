"""
simulation/event_log.py
-----------------------
Structured event logging for the simulation.

Two outputs:
  In-memory ring buffer  — last N events for on-screen display
  Files                  — logs/sim_<TIMESTAMP>.log (text)
                              logs/sim_<TIMESTAMP>.csv (numerical)

Usage:
  log = EventLog()
  log.info(t, "PHASE", "MONITORING → LEAK")
  log.metric(t, sigma=8.3, lambda2=0.42)
"""

import collections
import csv
import os
import time


class EventLog:
    """Simulation event log — on-screen ring buffer + persistent files."""

    SEVERITY = ("info", "event", "warn", "critical")

    def __init__(self, max_onscreen: int = 60, log_dir: str = "logs"):
        # In-memory ring buffer for on-screen display
        self.events = collections.deque(maxlen=max_onscreen)

        os.makedirs(log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.text_path   = os.path.join(log_dir, f"sim_{stamp}.log")
        self.csv_path    = os.path.join(log_dir, f"sim_{stamp}.csv")

        self._text_f = open(self.text_path, "w", buffering=1)  # line-buffered
        self._text_f.write(f"# Simulation started at {time.ctime()}\n")

        self._csv_f  = open(self.csv_path,  "w", newline="")
        self._csv_w  = csv.writer(self._csv_f)
        self._csv_w.writerow([
            "step", "phase", "mode", "robots_alive",
            "sigma", "lambda2", "leak_x", "leak_y",
            "agreed_x", "agreed_y",
        ])

    def info(self, step, tag, message, severity="info"):
        """Record a human-readable event. Shown on screen + written to text log."""
        record = {"step": step, "tag": tag, "msg": message, "sev": severity}
        self.events.append(record)
        self._text_f.write(f"[t={step:5d}] {tag:18s} {message}\n")

    def metric(self, step, phase=None, mode=None, robots_alive=None,
               sigma=None, lambda2=None, leak_px=None, agreed_px=None):
        """Append a numerical row to the CSV log."""
        leak_x, leak_y = (leak_px or (None, None))
        ag_x,  ag_y    = (agreed_px or (None, None))
        self._csv_w.writerow([step, phase, mode, robots_alive,
                              sigma, lambda2, leak_x, leak_y, ag_x, ag_y])

    #  Convenience helpers

    def phase_transition(self, step, frm, to):
        self.info(step, "PHASE", f"{frm} → {to}", severity="event")

    def mode_change(self, step, frm, to):
        self.info(step, "MODE", f"{frm} → {to}", severity="event")

    def leak_started(self, step, source_px):
        x, y = source_px
        self.info(step, "LEAK_START",
                  f"new leak at ({x:.0f}, {y:.0f})", severity="warn")

    def leak_resolved(self, step, agreed_px, error):
        x, y = agreed_px
        self.info(step, "REPORTED",
                  f"agreed ({x:.0f}, {y:.0f}) err={error:.1f}px",
                  severity="event")

    def robot_event(self, step, robot_id, what, severity="info"):
        self.info(step, "ROBOT", f"R{robot_id}: {what}", severity=severity)

    def gossip_event(self, step, ri_id, rj_id):
        # NOTE: gossip is high-frequency; we log to text only every Nth call
        # by callers that want it.  Here is the formatting helper.
        self.info(step, "GOSSIP", f"R{ri_id} ↔ R{rj_id}", severity="info")

    def close(self):
        try: self._text_f.close()
        except Exception: pass
        try: self._csv_f.close()
        except Exception: pass

    def __del__(self):
        self.close()
