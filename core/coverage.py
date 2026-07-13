"""
core/coverage.py
----------------
Voronoi tessellation + Lloyd centroid update with active patrol.

Uniform density (phi = 1): each robot patrols random waypoints inside its own Voronoi cell. Lloyd iteration drives the team toward a Centroidal Voronoi Tessellation minimising the coverage cost
    H(p) = sum_i integral_{V_i} phi(q) ||q - p_i||^2 dq
"""

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import box, Point
from shapely.geometry import Polygon as ShPoly

import config


class LloydVoronoi:
    """Voronoi cells clipped to free space + Lloyd patrol targets"""

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
        # Mirror points across the four walls so cells stay bounded
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
        Active-patrol coverage: instead of standing at the cell centroid,
        each robot targets a random waypoint inside its own Voronoi cell,
        and picks a new one on arrival (within 15 px). 
        The cell still defines each robot's region, so the area partition holds
        """
        exclude_ids = exclude_ids or set()
        for r in self.robots:
            if not r.is_alive or r.id in exclude_ids:
                continue
            cell = self.cells.get(r.id)
            if cell is None or cell.is_empty:
                continue

            if not hasattr(r, "_patrol_waypoint") or r._patrol_waypoint is None:
                r._patrol_waypoint = self._random_point_in_cell(cell)

            # Arrived: pick a new waypoint
            dist = float(np.linalg.norm(r.position - r._patrol_waypoint))
            if dist < 15:
                r._patrol_waypoint = self._random_point_in_cell(cell)

            # Cells move as robots move; resample if the waypoint fell outside
            if not cell.contains(Point(*r._patrol_waypoint)):
                r._patrol_waypoint = self._random_point_in_cell(cell)

            r.target = r._patrol_waypoint.copy()

    def _random_point_in_cell(self, cell):
        """Uniform random point inside a cell (rejection sampling)."""
        b = cell.bounds   # (xmin, ymin, xmax, ymax)
        for _ in range(40):
            x = np.random.uniform(b[0], b[2])
            y = np.random.uniform(b[1], b[3])
            if cell.contains(Point(x, y)):
                return np.array([x, y], dtype=float)
        c = cell.centroid   # fallback
        return np.array([c.x, c.y], dtype=float)

    @staticmethod
    def _mirror_points(pts):
        W, H = config.MAP_W, config.MAP_H
        m = []
        for p in pts:
            m += [[-p[0], p[1]], [2*W-p[0], p[1]],
                  [p[0], -p[1]], [p[0], 2*H-p[1]]]
        return np.array(m)