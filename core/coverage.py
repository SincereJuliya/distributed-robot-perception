"""
core/coverage.py
----------------
Voronoi tessellation + Lloyd centroid update.

Two density modes:
  • uniform — geometric centroid → even spread (Mode 1 idle, Mode 2 baseline)
  • gas     — gas-concentration weighted centroid → robots crowd the cloud
              (used by Mode 1 during LEAK / CONSENSUS phases)

Lloyd iteration on these centroids drives the team toward a Centroidal
Voronoi Tessellation that minimises the coverage cost
    H(p) = Σᵢ ∫_Vᵢ φ(q) ||q − pᵢ||² dq
(Cortes-Bullo et al. 2004).
"""

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import box, Point
from shapely.geometry import Polygon as ShPoly

import config


class LloydVoronoi:
    """Voronoi cells clipped to free space + Lloyd centroid targets."""

    SAMPLES = 100   # Monte-Carlo samples per cell for weighted centroid

    def __init__(self, robots, free_space=None):
        self.robots     = robots
        self.free_space = free_space if free_space is not None else \
                          box(0, 0, config.MAP_W, config.MAP_H)
        self.cells: dict = {}

    def compute_cells(self):
        alive = [r for r in self.robots if r.is_alive]
        self.cells = {}
        if not alive:
            return
        if len(alive) == 1:
            self.cells[alive[0].id] = self.free_space
            return

        positions = np.array([r.position for r in alive])
        ids       = [r.id for r in alive]
        augmented = np.vstack([positions, self._mirror_points(positions)])
        vor       = Voronoi(augmented)

        for idx, rid in enumerate(ids):
            region = vor.regions[vor.point_region[idx]]
            if -1 in region or not region:
                cell = self.free_space
            else:
                raw = ShPoly(vor.vertices[region])
                cell = raw.intersection(self.free_space)
            if not cell.is_empty:
                self.cells[rid] = cell

    def update_targets_uniform(self, exclude_ids=None):
        """
        Lloyd-Voronoi patrol with active coverage.

        Instead of standing at the cell centroid (which would freeze the
        robot after convergence), each robot picks a random waypoint
        INSIDE its own Voronoi cell. When it reaches the waypoint
        (within 15 px), a new random waypoint is chosen. This way the
        robot patrols its region of responsibility instead of
        stagnating at the centroid.

        Two consequences:
          • Robots keep moving even after Lloyd converges
          • A gas leak appearing anywhere in a robot's cell will
            eventually be detected (probability → 1 with patrol time)
          • The Voronoi cell still defines each robot's region of
            responsibility, so the area-partition property holds
        """
        exclude_ids = exclude_ids or set()
        for r in self.robots:
            if not r.is_alive or r.id in exclude_ids:
                continue
            cell = self.cells.get(r.id)
            if cell is None or cell.is_empty:
                continue

            # Initialise patrol waypoint if missing
            if not hasattr(r, "_patrol_waypoint") or r._patrol_waypoint is None:
                r._patrol_waypoint = self._random_point_in_cell(cell)

            # If close to waypoint, pick a new one inside the cell
            dist = float(np.linalg.norm(r.position - r._patrol_waypoint))
            if dist < 15:
                r._patrol_waypoint = self._random_point_in_cell(cell)

            # Re-check waypoint is still inside the (possibly updated) cell
            if not cell.contains(Point(*r._patrol_waypoint)):
                r._patrol_waypoint = self._random_point_in_cell(cell)

            r.target = r._patrol_waypoint.copy()

    def _random_point_in_cell(self, cell):
        """Sample a uniform random point inside a Voronoi cell."""
        b = cell.bounds   # (xmin, ymin, xmax, ymax)
        for _ in range(40):
            x = np.random.uniform(b[0], b[2])
            y = np.random.uniform(b[1], b[3])
            if cell.contains(Point(x, y)):
                return np.array([x, y], dtype=float)
        # Fallback: centroid
        c = cell.centroid
        return np.array([c.x, c.y], dtype=float)

    def update_targets_gas_weighted(self, gas_grid, exclude_ids=None):
        exclude_ids = exclude_ids or set()
        for r in self.robots:
            if not r.is_alive or r.id in exclude_ids:
                continue
            cell = self.cells.get(r.id)
            if cell is None or cell.is_empty:
                continue
            centroid = self._weighted_centroid(cell, gas_grid)
            r.update_smooth_centroid(centroid)
            r.target = r._smooth_centroid.copy()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _weighted_centroid(self, cell, density_grid):
        """
        Gas-weighted centroid via vectorised Monte-Carlo sampling.

        Generates SAMPLES random points in the cell bounding box at once
        (np.random.uniform with size=), filters by Shapely containment,
        and computes the weighted mean in NumPy without an inner loop.
        """
        xmin, ymin, xmax, ymax = cell.bounds

        # Vectorised candidate sampling — generate all points at once
        xs = np.random.uniform(xmin, xmax, size=self.SAMPLES)
        ys = np.random.uniform(ymin, ymax, size=self.SAMPLES)

        # Containment check — single pass through Shapely
        in_cell = np.array([cell.contains(Point(x, y))
                            for x, y in zip(xs, ys)])
        if not in_cell.any():
            c = cell.centroid
            return np.array([c.x, c.y])

        xs, ys = xs[in_cell], ys[in_cell]

        # Vectorised grid lookup
        cx = np.clip((xs / config.MAP_W * config.GRID_W).astype(int),
                     0, config.GRID_W - 1)
        cy = np.clip((ys / config.MAP_H * config.GRID_H).astype(int),
                     0, config.GRID_H - 1)
        ws = density_grid[cy, cx] + 1e-4

        if ws.max() < 0.005:
            c = cell.centroid
            return np.array([c.x, c.y])

        pts = np.column_stack([xs, ys])
        return (pts * ws[:, None]).sum(axis=0) / ws.sum()

    @staticmethod
    def _mirror_points(pts):
        W, H = config.MAP_W, config.MAP_H
        m = []
        for p in pts:
            m += [[-p[0], p[1]], [2*W-p[0], p[1]],
                  [p[0], -p[1]], [p[0], 2*H-p[1]]]
        return np.array(m)
