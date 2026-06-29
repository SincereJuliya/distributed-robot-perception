"""
core/formation.py
-----------------
Formation control for Mode 3.

Each robot is assigned a slot in a formation defined by a shape function
that maps slot_index → offset relative to the formation center:

    target_i = formation_center + offset(slot_i, shape)

Supported shapes (config.FORMATION_SHAPE):
  • "circle"  — robots on a circle of radius FORMATION_RADIUS
  • "line"    — robots in a horizontal line, spacing FORMATION_SPACING
  • "v"       — V-formation like bird flocks

Distributed update rule (consensus-based):
    target_i ← (1 − k) · target_i + k · (formation_center + offset_i)

with k = FORMATION_GAIN. This is the standard linear consensus law applied
to formation tracking (Olfati-Saber, Fax, Murray 2004).

Leader path (config.LEADER_PATH_TYPE):
  • "circle"           — formation centre moves on a large circle
  • "lissajous"        — figure-8 path
  • "random_waypoints" — slow random walk
"""

import math
import numpy as np

import config


class FormationControl:
    """Assigns slots and computes target positions for each robot."""

    SHAPES = ("circle", "line", "v", "triangle")

    def __init__(self, robots, shape: str = None):
        self.robots = robots
        self.shape  = shape if shape in self.SHAPES else config.FORMATION_SHAPE
        self.center = np.array([config.MAP_W / 2, config.MAP_H / 2])
        self._t     = 0   # internal clock for leader path
        self._waypoint = self.center.copy()
        self._waypoint_age = 0
        # Rotation of the formation around its own center (rad).
        # Set to non-zero for shapes that should "dance" (circle).
        self.rotation = 0.0
        self.assign_slots()

    # ── Slot assignment ──────────────────────────────────────────────────────

    def assign_slots(self):
        """Assign formation_slot to each alive robot."""
        alive = [r for r in self.robots if r.is_alive]
        for i, r in enumerate(alive):
            r.formation_slot = i
        for r in self.robots:
            if not r.is_alive:
                r.formation_slot = None

    def set_shape(self, shape: str):
        """Explicitly set formation shape (used by hotkeys C/L/V/T)."""
        if shape in self.SHAPES:
            self.shape = shape
            return shape
        return None

    def cycle_shape(self):
        """Cycle to next formation shape (F key)."""
        idx = self.SHAPES.index(self.shape)
        self.shape = self.SHAPES[(idx + 1) % len(self.SHAPES)]
        return self.shape

    # ── Leader path ───────────────────────────────────────────────────────────

    def step_center(self):
        """Advance the formation center along the chosen leader path."""
        self._t += 1
        # Rotation: the circle "dances" — it rotates around its own center
        if self.shape == "circle":
            self.rotation += 0.02   # ~20°/sec at 30 FPS
        else:
            self.rotation = 0.0

        cx0, cy0 = config.MAP_W / 2, config.MAP_H / 2
        path = config.LEADER_PATH_TYPE

        # All paths tuned to ≤ 1.5 px/step so robots (speed 2.0) can keep up
        if path == "circle":
            r = 180
            theta = self._t * 0.004     # slower
            self.center = np.array([cx0 + r * math.cos(theta),
                                     cy0 + r * math.sin(theta)])
        elif path == "lissajous":
            a, b = 180, 110
            theta = self._t * 0.005     # slower — robots can follow
            self.center = np.array([cx0 + a * math.sin(theta * 1.0),
                                     cy0 + b * math.sin(theta * 2.0)])
        elif path == "random_waypoints":
            # Pick a new waypoint either when the timer expires OR when we
            # arrive close to it. Avoids long stationary pauses where the
            # formation just sits in place.
            d = self._waypoint - self.center
            n = np.linalg.norm(d)
            if self._waypoint_age <= 0 or n < 12:
                margin = 120
                self._waypoint = np.array([
                    np.random.uniform(margin, config.MAP_W - margin),
                    np.random.uniform(margin, config.MAP_H - margin),
                ])
                self._waypoint_age = 250
                d = self._waypoint - self.center
                n = np.linalg.norm(d)
            self._waypoint_age -= 1
            if n > 1:
                self.center = self.center + (d / n) * 0.8

        m = 80
        self.center[0] = np.clip(self.center[0], m, config.MAP_W - m)
        self.center[1] = np.clip(self.center[1], m, config.MAP_H - m)

    # ── Offset for a given slot ──────────────────────────────────────────────

    def slot_offset(self, slot, n_total):
        """Offset vector from center for slot index `slot` (of n_total).
           Result is then rotated by self.rotation."""
        offset = self._raw_offset(slot, n_total)
        # Apply rotation (the "dance")
        if abs(self.rotation) > 1e-6:
            c, s = math.cos(self.rotation), math.sin(self.rotation)
            offset = np.array([c * offset[0] - s * offset[1],
                               s * offset[0] + c * offset[1]])
        return offset

    def _raw_offset(self, slot, n_total):
        """Shape-specific offset before rotation."""
        if self.shape == "circle":
            theta = 2 * math.pi * slot / max(n_total, 1)
            r = config.FORMATION_RADIUS
            return np.array([r * math.cos(theta), r * math.sin(theta)])
        if self.shape == "line":
            offset = (slot - (n_total - 1) / 2) * config.FORMATION_SPACING
            return np.array([offset, 0.0])
        if self.shape == "v":
            # Classic V (bird-flock): apex is slot 0 at the front (+x),
            # then pairs spread back-left and back-right.
            #
            #   slot 0            →  apex          offset = (0, 0)
            #   slot 1            →  left 1        offset = (-sp, -sp)
            #   slot 2            →  right 1       offset = (-sp, +sp)
            #   slot 3            →  left 2        offset = (-2sp, -2sp)
            #   slot 4            →  right 2       offset = (-2sp, +2sp)
            #
            sp = config.FORMATION_SPACING
            if slot == 0:
                return np.array([0.0, 0.0])
            rank = (slot + 1) // 2          # 1, 1, 2, 2, 3, 3 …
            side = -1 if slot % 2 == 1 else 1   # odd → left, even → right
            return np.array([-rank * sp, side * rank * sp])
        if self.shape == "triangle":
            # Distribute n_total robots evenly over the three edges of an
            # equilateral triangle inscribed in a circle of FORMATION_RADIUS.
            r = config.FORMATION_RADIUS
            # 3 vertices at 90°, 210°, 330° (pointing up)
            verts = [np.array([r * math.cos(math.radians(a)),
                               r * math.sin(math.radians(a))])
                     for a in (90, 210, 330)]
            # Map slot to a point along the perimeter
            t = slot / max(n_total, 1)        # 0..1 around the triangle
            edge = int(t * 3)                  # which edge: 0, 1, or 2
            edge = min(edge, 2)
            local = (t * 3) - edge             # 0..1 along that edge
            v0 = verts[edge]
            v1 = verts[(edge + 1) % 3]
            return v0 + local * (v1 - v0)
        return np.zeros(2)

    # ── Target update ────────────────────────────────────────────────────────

    def update_targets(self):
        alive = [r for r in self.robots if r.is_alive]
        n = len(alive)
        # Refresh slot assignment if anyone died
        if any(r.formation_slot is None for r in alive):
            self.assign_slots()

        # Set targets DIRECTLY — no smoothing. Formation must respond crisply
        # so the geometry is recognisable. (The Lloyd EMA from other modes
        # would make robots lag and the formation looks chaotic.)
        for r in alive:
            offset = self.slot_offset(r.formation_slot, n)
            desired = self.center + offset
            r.target = desired
            # Reset smooth_centroid so a later mode-switch back to Lloyd
            # starts clean.
            r._smooth_centroid = None

    # ── Formation error metric ───────────────────────────────────────────────

    def formation_error(self) -> float:
        """Total deviation of robots from their assigned formation slots."""
        alive = [r for r in self.robots if r.is_alive]
        n = len(alive)
        if n == 0: return 0.0
        err = 0.0
        for r in alive:
            offset = self.slot_offset(r.formation_slot, n)
            desired = self.center + offset
            err += float(np.linalg.norm(r.position - desired))
        return err / n
